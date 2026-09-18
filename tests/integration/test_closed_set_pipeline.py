from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import re
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import pytest
import torch
from herb_authentication.config import load_config
from herb_authentication.data import (
    HerbImageDataset,
    ManifestSource,
    apply_split,
    build_closed_set_manifest,
    build_label_space,
    create_group_splits,
    create_herb_dataloader,
    validate_manifest_pair,
    write_split_artifacts,
)
from herb_authentication.inference.calibration import CalibrationBundle
from herb_authentication.inference.service import InferenceService
from herb_authentication.models import MultiTaskHerbModel
from herb_authentication.training.checkpoints import CheckpointContract, save_checkpoint_atomic
from herb_authentication.training.promotion import (
    DEFAULT_PROMOTION_THRESHOLDS,
    PromotionRequest,
    promote_candidate,
)
from herb_authentication.training.run_artifacts import sha256_file
from herb_authentication.training.trainer import Trainer
from PIL import Image
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLAIMED_HERB = "Drynaria_fortunei"
THRESHOLDS = {
    "macro_f1": 0.80,
    "balanced_accuracy": 0.80,
    "counterfeit_recall": 0.80,
}

_before_evaluation: Callable[[Path], None] | None = None


@dataclass(frozen=True, slots=True)
class TinyPipelineArtifacts:
    training_ids: frozenset[str]
    validation_ids: frozenset[str]
    test_ids: frozenset[str]
    calibration_ids: frozenset[str]
    evaluation_ids: frozenset[str]
    response: Mapping[str, object]


class TinyEncoder(nn.Module):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        means = images.mean(dim=(2, 3))
        return torch.cat((means, means[:, :1] - means[:, 2:3]), dim=1)


def _model_factory(_metadata: Mapping[str, object] | None = None) -> MultiTaskHerbModel:
    model = MultiTaskHerbModel(
        TinyEncoder(),
        feature_dim=4,
        num_claimed=1,
        num_species=1,
        dropout=0.0,
    )
    with torch.no_grad():
        model.authenticity_head.weight.copy_(
            torch.tensor([[4.0, 0.0, -4.0, 0.0], [-4.0, 0.0, 4.0, 0.0]])
        )
        model.authenticity_head.bias.zero_()
    return model


def _to_tensor(image: Image.Image) -> torch.Tensor:
    width, height = image.size
    pixels = torch.frombuffer(bytearray(image.convert("RGB").tobytes()), dtype=torch.uint8)
    return pixels.reshape(height, width, 3).permute(2, 0, 1).float().div(255.0)


def _load_script(name: str) -> ModuleType:
    path = PROJECT_ROOT / "项目" / "Files" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"tiny_pipeline_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_pattern_jpeg(path: Path, identity: str, *, authentic: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(identity.encode("ascii")).digest()
    low, high = ((180, 80, 80), (255, 160, 160)) if authentic else ((80, 80, 180), (160, 160, 255))
    pixels = [high if digest[index // 8] & (1 << (index % 8)) else low for index in range(64)]
    image = Image.new("RGB", (8, 8))
    image.putdata(pixels)
    image.save(path, format="JPEG", quality=100, subsampling=0)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _ids(rows: list[dict[str, str]]) -> frozenset[str]:
    return frozenset(row["sample_id"] for row in rows)


def _collect_logits(
    checkpoint: Path,
    rows: list[dict[str, str]],
    data_root: Path,
) -> tuple[torch.Tensor, torch.Tensor]:
    service = InferenceService.from_checkpoint(
        checkpoint,
        device="cpu",
        model_factory=_model_factory,
        preprocess_factory=lambda _metadata: _to_tensor,
    )
    logits = []
    targets = []
    for row in rows:
        output = service.predict_logits(data_root / row["image_path"])
        logits.append(output.authenticity.squeeze(0))
        targets.append(0 if row["authenticity"] == "authentic" else 1)
    return torch.stack(logits), torch.tensor(targets, dtype=torch.long)


def run_tiny_pipeline(root: Path) -> TinyPipelineArtifacts:
    torch.manual_seed(42)
    references = root / "references"
    raw = references / "raw"
    authentic_dir = raw / "Drynaria_fortunei_authentic"
    counterfeit_dir = raw / "Drynaria_fortunei_fake"
    for index in range(1, 7):
        _write_pattern_jpeg(
            authentic_dir / f"Drynaria-fortunei_AUTHENTIC_{index:04d}.jpg",
            f"authentic-{index}",
            authentic=True,
        )
        _write_pattern_jpeg(
            counterfeit_dir / f"Drynaria-fortunei_FAKE_{index:04d}.jpg",
            f"counterfeit-{index}",
            authentic=False,
        )

    data_dir = root / "data"
    manifest = data_dir / "metadata.csv"
    build_closed_set_manifest(
        (
            ManifestSource(
                directory=authentic_dir,
                filename_pattern=re.compile(r"Drynaria-fortunei_AUTHENTIC_(\d{4})\.jpg"),
                id_prefix="authentic",
                authenticity="authentic",
                actual_species=CLAIMED_HERB,
                species_annotation_status="verified",
            ),
            ManifestSource(
                directory=counterfeit_dir,
                filename_pattern=re.compile(r"Drynaria-fortunei_FAKE_(\d{4})\.jpg"),
                id_prefix="counterfeit",
                authenticity="counterfeit",
                actual_species=None,
                species_annotation_status="unverified",
            ),
        ),
        references,
        manifest,
        CLAIMED_HERB,
    )
    validated = validate_manifest_pair(
        manifest,
        unknown_manifest_path=None,
        data_root=references,
        report_path=data_dir / "manifest_validation.json",
        encoding="utf-8-sig",
        perceptual_hash_distance=4,
    )
    assignment = create_group_splits(
        validated.known,
        ratios=(0.70, 0.15, 0.15),
        seed=42,
        perceptual_hash_distance=4,
    )
    partitioned = apply_split(validated.known, assignment)
    split_artifacts = write_split_artifacts(validated.known, assignment, data_dir)
    rows_by_partition = {
        name: _read_rows(getattr(split_artifacts, f"{name}_csv"))
        for name in ("train", "val", "test")
    }
    all_rows = [row for rows in rows_by_partition.values() for row in rows]
    assert len(all_rows) == 12
    assert len({row["sample_id"] for row in all_rows}) == 12
    assert len({row["group_id"] for row in all_rows}) == 12
    assert all(
        {row["authenticity"] for row in rows} == {"authentic", "counterfeit"}
        for rows in rows_by_partition.values()
    )

    labels = build_label_space(partitioned)
    datasets = {
        name: HerbImageDataset(
            tuple(item for item in partitioned if item.partition == name),
            labels,
            _to_tensor,
        )
        for name in ("train", "val", "test")
    }
    for dataset in datasets.values():
        for index, item in enumerate(dataset):
            if dataset.rows[index].row.authenticity == "counterfeit":
                assert item["actual_species_mask"].item() is False
    train_loader = create_herb_dataloader(
        datasets["train"], batch_size=4, shuffle=True, seed=42, num_workers=0
    )
    val_loader = create_herb_dataloader(
        datasets["val"], batch_size=2, shuffle=False, seed=42, num_workers=0
    )

    training_ids = frozenset(
        metadata["sample_id"] for batch in train_loader for metadata in batch["metadata"]
    )
    validation_ids = _ids(rows_by_partition["val"])
    test_ids = _ids(rows_by_partition["test"])
    assert training_ids == _ids(rows_by_partition["train"])
    assert training_ids.isdisjoint(validation_ids | test_ids)

    config = root / "config.yaml"
    config.write_text(
        "\n".join(
            [
                'project_root: "."',
                "data:",
                '  manifest_path: "data/metadata.csv"',
                "  unknown_manifest_path: null",
                '  data_root: "references"',
                '  provenance: "synthetic"',
                '  text_encoding: "utf-8-sig"',
                "  image_size: 8",
                "  batch_size: 4",
                "  num_workers: 0",
                "  train_ratio: 0.70",
                "  val_ratio: 0.15",
                "  test_ratio: 0.15",
                "  seed: 42",
                "  max_image_bytes: 1000000",
                "  max_image_pixels: 1000000",
                "  perceptual_hash_distance: 4",
                "  use_weighted_sampler: false",
                "model:",
                '  backbone: "tiny"',
                "  pretrained: false",
                "  freeze_backbone: false",
                "  unfreeze_epoch: 0",
                "  dropout: 0.0",
                "training:",
                "  epochs: 1",
                "  learning_rate: 0.0001",
                "  weight_decay: 0.0",
                "  claimed_herb_weight: 1.0",
                "  actual_species_weight: 1.0",
                "  authenticity_weight: 1.0",
                "  label_smoothing: 0.0",
                "  early_stop_patience: 1",
                "  mixed_precision: false",
                "  gradient_clip: 1.0",
                "  min_species_supervision_coverage: 0.0",
                "  use_authenticity_class_weights: true",
                '  checkpoint_path: "models/tiny.candidate.pth"',
                '  log_dir: "logs"',
                "calibration:",
                '  mode: "closed_set_temperature"',
                '  algorithm_version: "closed-set-temperature-v1"',
                '  output_checkpoint_path: "models/tiny.calibrated.candidate.pth"',
                "evaluation:",
                '  report_dir: "runs/evaluation"',
                "  bootstrap_samples: 5",
                "inference:",
                '  checkpoint_path: "models/tiny.calibrated.candidate.pth"',
                '  device: "cpu"',
                "  allow_uncalibrated: false",
                "  max_image_bytes: 1000000",
                "  max_image_pixels: 1000000",
                "run:",
                '  root_dir: "runs"',
                "promotion:",
                "  min_macro_f1: 0.80",
                "  min_balanced_accuracy: 0.80",
                "  min_counterfeit_recall: 0.80",
                '  target_checkpoint_path: "release/official.pth"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    typed_config = load_config(config)
    evaluate = _load_script("evaluate")
    source_candidate = root / "models" / "tiny.candidate.pth"
    run_dir = root / "runs" / "tiny-e2e"
    manifest_fingerprint = sha256_file(manifest)
    split_fingerprint = sha256_file(split_artifacts.splits_json)
    config_fingerprint = evaluate._training_config_fingerprint(typed_config)
    validation_fingerprint = sha256_file(split_artifacts.val_csv)
    model = _model_factory()
    trainer = Trainer(
        model,
        epochs=1,
        learning_rate=1e-4,
        freeze_backbone=False,
        mixed_precision=False,
        device="cpu",
        checkpoint_path=source_candidate,
        checkpoint_contract=CheckpointContract(
            schema_version=1,
            backbone="tiny",
            claimed_herb_classes=labels.claimed_classes,
            actual_species_classes=labels.actual_species_classes,
            authenticity_classes=labels.authenticity_classes,
            image_size=8,
            preprocessing_version="tiny-e2e-v1",
            data_fingerprint=assignment.input_fingerprint,
            split_fingerprint=split_fingerprint,
            training_config_fingerprint=config_fingerprint,
        ),
        checkpoint_metadata={
            "run_id": run_dir.name,
            "model_version": "tiny-e2e-v1",
            "counterfeit_species_verified": False,
            "split_artifact_fingerprint": split_fingerprint,
            "split_input_fingerprint": assignment.input_fingerprint,
            "fingerprints": {
                "manifest": manifest_fingerprint,
                "locked_split": split_fingerprint,
                "known_validation": validation_fingerprint,
            },
            "calibration": CalibrationBundle.uncalibrated().to_dict(),
        },
        authenticity_class_weight=torch.ones(2),
        min_species_supervision_coverage=0.0,
    )
    trainer.fit(train_loader, val_loader)
    run_dir.mkdir(parents=True)
    (run_dir / "run_metadata.json").write_text(
        json.dumps({"checkpoint": str(source_candidate.resolve())}),
        encoding="utf-8",
    )

    calibration_ids: frozenset[str] = frozenset()

    def calibration_loader(**kwargs):
        nonlocal calibration_ids
        known_rows = kwargs["known_rows"]
        calibration_ids = _ids(known_rows)
        return _collect_logits(source_candidate, known_rows, references)

    calibrated_candidate = root / "models" / "tiny.calibrated.candidate.pth"
    calibrate = _load_script("calibrate")
    assert (
        calibrate.main(
            [
                "--mode",
                "closed-set-temperature",
                "--checkpoint",
                str(source_candidate),
                "--known-val",
                str(split_artifacts.val_csv),
                "--output",
                str(calibrated_candidate),
                "--data-root",
                str(references),
                "--device",
                "cpu",
            ],
            calibration_data_loader=calibration_loader,
        )
        == 0
    )

    if _before_evaluation is not None:
        _before_evaluation(calibrated_candidate)

    evaluation_ids: frozenset[str] = frozenset()
    reservation = run_dir / "locked_test" / "reservation.json"

    def collector(**kwargs):
        nonlocal evaluation_ids
        partition_manifest = kwargs["manifest"]
        checkpoint = kwargs["checkpoint"]
        data_root = kwargs["data_root"]
        expected_partition = kwargs["expected_partition"]
        assert kwargs["split"] == split_artifacts.splits_json.resolve()
        rows = _read_rows(partition_manifest)
        if expected_partition == "test":
            assert reservation.is_file()
            evaluation_ids = _ids(rows)
        return _collect_logits(checkpoint, rows, data_root)

    def evaluate_partition(partition: str) -> dict[str, object]:
        code = evaluate.main(
            [
                "--config",
                str(config),
                "--checkpoint",
                str(calibrated_candidate),
                "--partition",
                partition,
                "--run-dir",
                str(run_dir),
            ],
            closed_set_logits_collector=collector,
        )
        if code != 0:
            raise RuntimeError(f"{partition} evaluate.main failed")
        directory = "validation" if partition == "validation" else "locked_test"
        return json.loads((run_dir / directory / "evaluation.json").read_text(encoding="utf-8"))

    validation_report = evaluate_partition("validation")
    locked_report = evaluate_partition("locked-test")
    validation_json = run_dir / "validation" / "evaluation.json"
    locked_json = run_dir / "locked_test" / "evaluation.json"
    assert validation_report["experimental_closed_set"] is True
    assert validation_report["open_set_calibrated"] is False
    assert locked_report["experimental_closed_set"] is True
    assert locked_report["open_set_calibrated"] is False
    assert validation_report["promotion_gate"]["passed"] is True
    assert locked_report["thresholds_met"]["passed"] is True
    service = InferenceService.from_checkpoint(
        calibrated_candidate,
        device="cpu",
        model_factory=_model_factory,
        preprocess_factory=lambda _metadata: _to_tensor,
    )
    smoke_image = datasets["test"].rows[0].row.image_path
    path_response = service.predict_experimental_closed_set(smoke_image).to_dict()
    bytes_response = service.predict_experimental_closed_set(smoke_image.read_bytes()).to_dict()
    assert path_response == bytes_response
    counterfeit_image = next(
        item.row.image_path
        for item in datasets["test"].rows
        if item.row.authenticity == "counterfeit"
    )
    counterfeit_response = service.predict_experimental_closed_set(counterfeit_image).to_dict()
    counterfeit_confidence = counterfeit_response["authenticity"]["confidence"]
    assert counterfeit_response["actual_species_verified"] is False
    assert counterfeit_response["message"] == (
        f"实际药材无法确定，有{counterfeit_confidence:.1%}的可能该药材为伪品，请人工复核。"
    )
    sentinel = root / "must-not-change.txt"
    sentinel.write_bytes(b"untouched")
    before = {
        "candidate": sha256_file(calibrated_candidate),
        "validation": sha256_file(validation_json),
        "locked": sha256_file(locked_json),
        "sentinel": sha256_file(sentinel),
    }
    target = root / "release" / "official.pth"
    request = PromotionRequest(
        candidate=calibrated_candidate,
        target=target,
        validation_report=validation_json,
        locked_test_report=locked_json,
        run_dir=run_dir,
    )
    record = promote_candidate(
        request,
        DEFAULT_PROMOTION_THRESHOLDS,
        smoke_image,
        model_factory=_model_factory,
        preprocess_factory=lambda _metadata: _to_tensor,
    )
    assert record.target == target.resolve()
    assert sha256_file(target) == before["candidate"]
    assert sha256_file(calibrated_candidate) == before["candidate"]
    assert sha256_file(validation_json) == before["validation"]
    assert sha256_file(locked_json) == before["locked"]
    assert sha256_file(sentinel) == before["sentinel"]
    return TinyPipelineArtifacts(
        training_ids=training_ids,
        validation_ids=validation_ids,
        test_ids=test_ids,
        calibration_ids=calibration_ids,
        evaluation_ids=evaluation_ids,
        response=path_response,
    )


def test_closed_set_pipeline_keeps_train_val_test_roles_isolated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_load_script = _load_script
    evaluate_main_partitions: list[str] = []

    def tracked_load_script(name: str) -> ModuleType:
        module = original_load_script(name)
        if name == "evaluate":
            original_main = module.main

            def tracked_main(argv=None, **kwargs):
                arguments = list(argv or ())
                partition_index = arguments.index("--partition") + 1
                evaluate_main_partitions.append(arguments[partition_index])
                return original_main(argv, **kwargs)

            module.main = tracked_main
        return module

    monkeypatch.setattr(sys.modules[__name__], "_load_script", tracked_load_script)
    artifacts = run_tiny_pipeline(tmp_path)

    assert artifacts.training_ids.isdisjoint(artifacts.validation_ids)
    assert artifacts.training_ids.isdisjoint(artifacts.test_ids)
    assert artifacts.validation_ids.isdisjoint(artifacts.test_ids)
    assert artifacts.calibration_ids == artifacts.validation_ids
    assert artifacts.evaluation_ids == artifacts.test_ids
    assert evaluate_main_partitions == ["validation", "locked-test"]
    assert artifacts.response["schema_version"] == 1
    assert artifacts.response["experimental_closed_set"] is True
    assert artifacts.response["open_set_calibrated"] is False
    assert artifacts.response["scope_warning"] == "仅适用于本数据集定义的骨碎补正品/伪品范围"
    assert list(artifacts.response["probabilities"]) == ["authentic", "counterfeit"]
    assert artifacts.response["authenticity"]["label"] == "authentic"
    confidence = artifacts.response["authenticity"]["confidence"]
    assert artifacts.response["message"] == (
        f"此药品为正品，有{confidence:.1%}的可能该药材为正品。"
    )


def test_closed_set_pipeline_rejects_tampered_checkpoint_manifest_fingerprint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def tamper_manifest_fingerprint(checkpoint: Path) -> None:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        metadata = dict(state["metadata"])
        fingerprints = dict(metadata["fingerprints"])
        fingerprints["manifest"] = "0" * 64
        metadata["fingerprints"] = fingerprints
        save_checkpoint_atomic(
            checkpoint,
            dict(state) | {"metadata": metadata},
            protected_paths=(),
        )

    monkeypatch.setattr(
        sys.modules[__name__],
        "_before_evaluation",
        tamper_manifest_fingerprint,
        raising=False,
    )
    with pytest.raises(RuntimeError, match="validation evaluate.main failed"):
        run_tiny_pipeline(tmp_path)

    assert "checkpoint manifest 指纹与当前工件不一致" in capsys.readouterr().err
