from __future__ import annotations

import argparse
import importlib.util
import json
import re
from pathlib import Path
from types import SimpleNamespace

import herb_authentication.config as config_module
import herb_authentication.data as data_module
import herb_authentication.environment as environment_module
import herb_authentication.models as models_module
import herb_authentication.training.run_artifacts as run_artifacts_module
import herb_authentication.training.trainer as trainer_module
import pytest
import torch
from herb_authentication.data import ManifestSource, build_closed_set_manifest
from herb_authentication.data.manifest import KnownManifestRow, ValidatedKnownRecord
from herb_authentication.data.splitting import PartitionedImageRecord
from PIL import Image


def _load_train_script():
    path = Path(__file__).resolve().parents[2] / "项目" / "Files" / "train.py"
    spec = importlib.util.spec_from_file_location("train_assembly_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _partitioned_row(
    root: Path,
    sample_id: str,
    partition: str,
    authenticity: str,
    species_annotation_status: str,
) -> PartitionedImageRecord:
    row = KnownManifestRow(
        image_path=root / f"{sample_id}.jpg",
        group_id=f"group-{sample_id}",
        sample_id=sample_id,
        claimed_herb_name="骨碎补",
        authenticity=authenticity,
        actual_species=("Drynaria_fortunei",) if species_annotation_status == "verified" else (),
        species_annotation_status=species_annotation_status,
        batch_id=None,
        origin=None,
        capture_device=None,
        collected_at=None,
        annotator=None,
        notes=None,
        source_line=1,
    )
    return PartitionedImageRecord(
        record=ValidatedKnownRecord(
            row=row,
            sha256="0" * 64,
            perceptual_hash="0" * 16,
            image_format="JPEG",
            width=8,
            height=8,
            file_size=1,
        ),
        partition=partition,
    )


def test_train_split_loader_uses_public_payload_parser(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = object()
    observed: dict[str, object] = {}

    def parse(payload):
        observed["payload"] = payload
        return expected

    monkeypatch.setattr(data_module, "split_assignment_from_payload", parse, raising=False)
    split = tmp_path / "splits.json"
    split.write_text('{"schema_version": 1, "marker": "v2"}', encoding="utf-8")

    assert _load_train_script()._load_split_assignment(split) is expected
    assert observed == {"payload": {"schema_version": 1, "marker": "v2"}}


def test_training_assembly_uses_public_label_space_contract(tmp_path: Path, monkeypatch) -> None:
    config = SimpleNamespace(
        data=SimpleNamespace(
            manifest_path=tmp_path / "metadata.csv",
            unknown_manifest_path=None,
            data_root=tmp_path,
            text_encoding=None,
            max_image_bytes=1000,
            max_image_pixels=1000,
            perceptual_hash_distance=4,
            train_ratio=0.7,
            val_ratio=0.2,
            test_ratio=0.1,
            seed=7,
            image_size=8,
            batch_size=2,
            num_workers=0,
            provenance="synthetic",
        ),
        model=SimpleNamespace(
            backbone="tiny",
            pretrained=False,
            dropout=0.0,
            freeze_backbone=False,
            unfreeze_epoch=None,
        ),
        training=SimpleNamespace(
            checkpoint_path=tmp_path / "default.pth",
            log_dir=tmp_path / "logs",
            epochs=1,
            learning_rate=0.01,
            weight_decay=0.0,
            claimed_herb_weight=1.0,
            actual_species_weight=1.0,
            authenticity_weight=1.0,
            label_smoothing=0.0,
            gradient_clip=None,
            mixed_precision=False,
            early_stop_patience=None,
            min_species_supervision_coverage=0.2,
            use_authenticity_class_weights=True,
        ),
        run=SimpleNamespace(root_dir=tmp_path / "runs"),
    )
    rows = (
        _partitioned_row(tmp_path, "train-authentic", "train", "authentic", "verified"),
        _partitioned_row(tmp_path, "train-counterfeit", "train", "counterfeit", "unverified"),
        _partitioned_row(tmp_path, "val-authentic", "val", "authentic", "verified"),
    )
    config.data.manifest_path.write_text("manifest", encoding="utf-8")

    def write_artifacts(_rows, _assignment, output_dir):
        val = output_dir / "val.csv"
        split = output_dir / "splits.json"
        val.write_text("partition\\nval\\n", encoding="utf-8")
        split.write_text("{}", encoding="utf-8")
        return SimpleNamespace(val_csv=val, splits_json=split)

    labels = SimpleNamespace(
        claimed_classes=("当归",),
        actual_species_classes=("当归", "欧当归"),
        authenticity_classes=("authentic", "counterfeit"),
    )
    monkeypatch.setattr(config_module, "load_config", lambda _path: config)
    monkeypatch.setattr(
        data_module,
        "validate_manifest_pair",
        lambda *_args, **_kwargs: SimpleNamespace(known=(object(),), known_encoding="utf-8"),
    )
    monkeypatch.setattr(data_module, "write_split_artifacts", write_artifacts)
    monkeypatch.setattr(
        data_module,
        "create_group_splits",
        lambda *_args, **_kwargs: SimpleNamespace(
            input_fingerprint="data-fingerprint",
            algorithm_version="group-stratified-v1",
            seed=7,
            ratios=(0.7, 0.2, 0.1),
            by_group={
                "group-a": "train",
                "group-b": "val",
                "group-c": "test",
            },
        ),
    )
    monkeypatch.setattr(data_module, "apply_split", lambda *_args: rows)
    monkeypatch.setattr(data_module, "build_label_space", lambda _rows: labels)
    monkeypatch.setattr(data_module, "build_train_transform", lambda _size: object())
    monkeypatch.setattr(data_module, "build_eval_transform", lambda _size: object())
    monkeypatch.setattr(data_module, "HerbImageDataset", lambda *_args: object())
    monkeypatch.setattr(data_module, "create_herb_dataloader", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        models_module,
        "build_torchvision_encoder",
        lambda *_args, **_kwargs: SimpleNamespace(encoder=object(), feature_dim=4),
    )
    monkeypatch.setattr(models_module, "MultiTaskHerbModel", lambda *_args, **_kwargs: object())
    run_root = tmp_path / "runs" / "assembly-run"
    run_root.mkdir(parents=True)
    parent_checkpoint = tmp_path / "parent.pth"
    torch.save(
        {"metadata": {"run_id": "parent-run"}, "training": {"epoch": 2}},
        parent_checkpoint,
    )
    parent_sha256 = run_artifacts_module.sha256_file(parent_checkpoint)
    monkeypatch.setattr(
        run_artifacts_module,
        "create_run_context",
        lambda _config, requested_run_id=None: SimpleNamespace(
            run_id=requested_run_id or "assembly-run",
            root=run_root,
            config_snapshot=run_root / "config.yaml",
        ),
    )
    written_metadata = {}
    monkeypatch.setattr(
        run_artifacts_module,
        "write_run_metadata",
        lambda _context, payload: written_metadata.update(payload),
    )

    class FakeTrainer:
        def __init__(self, _model, **kwargs):
            contract = kwargs["checkpoint_contract"]
            assert contract.claimed_herb_classes == ("当归",)
            assert contract.data_fingerprint == "data-fingerprint"
            assert len(contract.split_fingerprint) == 64
            assert len(contract.training_config_fingerprint) == 64
            assert contract.split_fingerprint != contract.data_fingerprint
            assert kwargs["authenticity_class_weight"].tolist() == [1.0, 1.0]
            assert kwargs["checkpoint_metadata"]["run_id"] == "assembly-run"
            assert kwargs["checkpoint_metadata"]["selection_formula"] == (
                "(authenticity_macro_f1 + authenticity_balanced_accuracy) / 2.0"
            )
            assert kwargs["checkpoint_metadata"]["counterfeit_species_verified"] is False
            assert kwargs["checkpoint_metadata"]["resume_from"] == {
                "checkpoint_path": str(parent_checkpoint.resolve()),
                "checkpoint_sha256": parent_sha256,
                "parent_run_id": "parent-run",
                "start_epoch": 3,
            }

        def fit(self, _train, _val, resume_from=None):
            assert resume_from == parent_checkpoint
            metrics = SimpleNamespace(
                claimed_macro_f1=0.5,
                authenticity_macro_f1=0.75,
                authenticity_balanced_accuracy=0.8,
                species_macro_f1=None,
                species_supervision_coverage=0.0,
                selection_score=0.775,
                selection_tasks=(
                    "authenticity_macro_f1",
                    "authenticity_balanced_accuracy",
                ),
            )
            epoch = SimpleNamespace(
                epoch=0,
                training=True,
                total_loss=1.0,
                claimed_herb_loss=0.4,
                actual_species_loss=0.3,
                authenticity_loss=0.3,
                species_supervised_rows=0,
                batches=1,
                samples=2,
                freeze_stage="full_model",
                learning_rates=(0.01,),
                duration_seconds=0.01,
                entity_metrics=metrics,
            )
            validation = SimpleNamespace(**(epoch.__dict__ | {"training": False}))
            return SimpleNamespace(
                train_epochs=(epoch,),
                validation_epochs=(validation,),
                device="cpu",
                amp_enabled=False,
                best_epoch=0,
                best_selection_score=0.775,
                stopped_early=False,
            )

    monkeypatch.setattr(trainer_module, "Trainer", FakeTrainer)
    module = _load_train_script()

    result = module._run_training(
        argparse.Namespace(
            config=tmp_path / "config.yaml",
            output=tmp_path / "new.pth",
            resume=parent_checkpoint,
            device="cpu",
            run_id="assembly-run",
        )
    )

    assert result == 0
    assert written_metadata["run_id"] == "assembly-run"
    assert written_metadata["resume_from"] == {
        "checkpoint_path": str(parent_checkpoint.resolve()),
        "checkpoint_sha256": parent_sha256,
        "parent_run_id": "parent-run",
        "start_epoch": 3,
    }
    history = json.loads((run_root / "training_history.json").read_text(encoding="utf-8"))
    assert history["best_epoch"] == 0
    assert history["early_stop_reason"] == "epochs_completed"
    assert history["authenticity_class_counts"] == {"authentic": 1, "counterfeit": 1}
    assert history["authenticity_class_weights"] == [1.0, 1.0]
    assert history["epochs"][0]["validation"]["diagnostic_metrics"] == {
        "claimed_macro_f1": 0.5,
        "authenticity_macro_f1": 0.75,
        "authenticity_balanced_accuracy": 0.8,
        "species_macro_f1": None,
        "species_supervision_coverage": 0.0,
        "selection_score": 0.775,
        "selection_tasks": [
            "authenticity_macro_f1",
            "authenticity_balanced_accuracy",
        ],
    }


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
def test_training_history_rejects_nonfinite_numeric_values(
    tmp_path: Path,
    invalid: float,
) -> None:
    module = _load_train_script()
    summary = SimpleNamespace(
        train_epochs=(),
        validation_epochs=(),
        best_epoch=None,
        best_selection_score=invalid,
        stopped_early=False,
    )

    with pytest.raises(ValueError, match="非有限"):
        module._write_training_history(
            tmp_path,
            summary,
            class_counts={"authentic": 1, "counterfeit": 1},
            class_weights=[1.0, 1.0],
        )

    assert not (tmp_path / "training_history.json").exists()


def test_cli_reports_noop_resume_as_failure_without_success_json(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    module = _load_train_script()
    monkeypatch.setattr(
        environment_module,
        "run_bootstrap_preflight",
        lambda: SimpleNamespace(python_supported=True),
    )

    def fail_noop(_args) -> int:
        raise ValueError("resume start_epoch=1 必须小于 configured epochs=1")

    monkeypatch.setattr(module, "_run_training", fail_noop)
    output = tmp_path / "child.pth"

    result = module.main(
        [
            "--config",
            str(tmp_path / "config.yaml"),
            "--resume",
            str(tmp_path / "parent.pth"),
            "--output",
            str(output),
        ]
    )

    captured = capsys.readouterr()
    assert result == 2
    assert captured.out == ""
    assert "start_epoch=1" in captured.err
    assert not output.exists()


def test_training_v2_missing_split_fails_before_run_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = SimpleNamespace(
        data=SimpleNamespace(
            manifest_path=tmp_path / "data" / "metadata.csv",
            split_algorithm_version="group-stratified-near-duplicate-components-v2",
        )
    )
    monkeypatch.setattr(config_module, "load_config", lambda _path: config)

    def forbidden_run_creation(*_args, **_kwargs):
        raise AssertionError("run context must not be created before split preflight")

    monkeypatch.setattr(run_artifacts_module, "create_run_context", forbidden_run_creation)
    module = _load_train_script()

    with pytest.raises(ValueError, match="prepare_closed_set.py"):
        module._run_training(
            argparse.Namespace(
                config=tmp_path / "config.yaml",
                output=None,
                resume=None,
                device="cpu",
                run_id="must-not-exist",
            )
        )

    assert not (tmp_path / "data" / "splits.json").exists()


@pytest.mark.parametrize(
    ("configured", "recorded"),
    (
        ("group-stratified-v1", "group-stratified-near-duplicate-components-v2"),
        ("group-stratified-near-duplicate-components-v2", "group-stratified-v1"),
    ),
)
def test_training_rejects_both_split_version_mismatch_directions(
    tmp_path: Path, configured: str, recorded: str
) -> None:
    module = _load_train_script()
    config = SimpleNamespace(data=SimpleNamespace(split_algorithm_version=configured))
    assignment = SimpleNamespace(algorithm_version=recorded)

    with pytest.raises(ValueError, match="切分算法版本"):
        module._require_configured_split_version(config, assignment, tmp_path / "splits.json")


@pytest.mark.parametrize(
    ("configured", "recorded"),
    (
        ("group-stratified-v1", "group-stratified-near-duplicate-components-v2"),
        ("group-stratified-near-duplicate-components-v2", "group-stratified-v1"),
    ),
)
def test_training_version_mismatch_fails_before_run_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    configured: str,
    recorded: str,
) -> None:
    split_path = tmp_path / "data" / "splits.json"
    split_path.parent.mkdir()
    split_path.write_text("{}", encoding="utf-8")
    config = SimpleNamespace(
        data=SimpleNamespace(
            manifest_path=tmp_path / "data" / "metadata.csv",
            split_algorithm_version=configured,
        )
    )
    module = _load_train_script()
    monkeypatch.setattr(config_module, "load_config", lambda _path: config)
    monkeypatch.setattr(
        module,
        "_load_split_assignment",
        lambda _path: SimpleNamespace(algorithm_version=recorded),
    )

    def forbidden_run_creation(*_args, **_kwargs):
        raise AssertionError("run context must not be created before version preflight")

    monkeypatch.setattr(run_artifacts_module, "create_run_context", forbidden_run_creation)

    with pytest.raises(ValueError, match="切分算法版本"):
        module._run_training(
            argparse.Namespace(
                config=tmp_path / "config.yaml",
                output=None,
                resume=None,
                device="cpu",
                run_id="must-not-exist",
            )
        )


@pytest.mark.parametrize(
    "version",
    ("group-stratified-v1", "group-stratified-near-duplicate-components-v2"),
)
def test_training_accepts_matching_split_versions(tmp_path: Path, version: str) -> None:
    module = _load_train_script()
    config = SimpleNamespace(data=SimpleNamespace(split_algorithm_version=version))
    assignment = SimpleNamespace(algorithm_version=version)

    assert (
        module._require_configured_split_version(config, assignment, tmp_path / "splits.json")
        is assignment
    )


def _fingerprint_config(split_algorithm_version: str):
    return SimpleNamespace(
        data=SimpleNamespace(
            image_size=224,
            batch_size=8,
            num_workers=0,
            seed=42,
            split_algorithm_version=split_algorithm_version,
        ),
        model=SimpleNamespace(
            backbone="resnet18",
            pretrained=False,
            dropout=0.1,
            freeze_backbone=False,
            unfreeze_epoch=None,
        ),
        training=SimpleNamespace(
            epochs=2,
            learning_rate=0.001,
            weight_decay=0.0,
            claimed_herb_weight=1.0,
            actual_species_weight=1.0,
            authenticity_weight=1.0,
            label_smoothing=0.0,
            gradient_clip=None,
            mixed_precision=False,
            early_stop_patience=None,
            min_species_supervision_coverage=0.2,
            use_authenticity_class_weights=True,
        ),
    )


def test_train_and_evaluate_config_fingerprints_match_and_bind_split_version() -> None:
    train_module = _load_train_script()
    evaluate_path = Path(__file__).resolve().parents[2] / "项目" / "Files" / "evaluate.py"
    spec = importlib.util.spec_from_file_location("evaluate_config_fingerprint", evaluate_path)
    assert spec is not None and spec.loader is not None
    evaluate_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluate_module)
    v1 = _fingerprint_config("group-stratified-v1")
    v2 = _fingerprint_config("group-stratified-near-duplicate-components-v2")

    assert train_module._training_config_fingerprint(v1) == (
        evaluate_module._training_config_fingerprint(v1)
    )
    assert train_module._training_config_fingerprint(v2) == (
        evaluate_module._training_config_fingerprint(v2)
    )
    assert train_module._training_config_fingerprint(v1) != (
        train_module._training_config_fingerprint(v2)
    )


def test_train_main_real_manifest_split_reaches_lightweight_trainer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    references = tmp_path / "references"
    authentic_dir = references / "raw" / "authentic"
    counterfeit_dir = references / "raw" / "counterfeit"
    for index, (directory, prefix) in enumerate(
        (
            (authentic_dir, "AUTHENTIC"),
            (authentic_dir, "AUTHENTIC"),
            (authentic_dir, "AUTHENTIC"),
            (counterfeit_dir, "FAKE"),
            (counterfeit_dir, "FAKE"),
            (counterfeit_dir, "FAKE"),
        )
    ):
        directory.mkdir(parents=True, exist_ok=True)
        image = Image.new("L", (8, 8), 0)
        image.putpixel((index, 0), 255)
        image.convert("RGB").save(
            directory / f"{prefix}_{index % 3 + 1:04d}.jpg",
            format="JPEG",
            quality=100,
            subsampling=0,
        )

    manifest = tmp_path / "data" / "metadata.csv"
    build_closed_set_manifest(
        (
            ManifestSource(
                directory=authentic_dir,
                filename_pattern=re.compile(r"AUTHENTIC_(\d{4})\.jpg"),
                id_prefix="authentic",
                authenticity="authentic",
                actual_species="Drynaria_fortunei",
                species_annotation_status="verified",
            ),
            ManifestSource(
                directory=counterfeit_dir,
                filename_pattern=re.compile(r"FAKE_(\d{4})\.jpg"),
                id_prefix="counterfeit",
                authenticity="counterfeit",
                actual_species=None,
                species_annotation_status="unverified",
            ),
        ),
        references,
        manifest,
        "Drynaria_fortunei",
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        """
project_root: "."
data:
  manifest_path: "data/metadata.csv"
  unknown_manifest_path: null
  data_root: "references"
  provenance: "synthetic"
  text_encoding: "utf-8-sig"
  image_size: 8
  batch_size: 2
  num_workers: 0
  train_ratio: 0.3333333333333333
  val_ratio: 0.3333333333333333
  test_ratio: 0.3333333333333333
  seed: 42
  max_image_bytes: 1000000
  max_image_pixels: 1000000
  perceptual_hash_distance: 0
  split_algorithm_version: "group-stratified-v1"
  use_weighted_sampler: false
model:
  backbone: "tiny"
  pretrained: false
  freeze_backbone: false
  unfreeze_epoch: 0
  dropout: 0.0
training:
  epochs: 1
  learning_rate: 0.001
  weight_decay: 0.0
  claimed_herb_weight: 0.0
  actual_species_weight: 0.25
  authenticity_weight: 1.0
  label_smoothing: 0.0
  early_stop_patience: 1
  mixed_precision: false
  gradient_clip: 1.0
  min_species_supervision_coverage: 0.0
  use_authenticity_class_weights: true
  checkpoint_path: "models/candidate.pth"
  log_dir: "logs"
calibration:
  mode: "closed_set_temperature"
  algorithm_version: "closed-set-temperature-v1"
  output_checkpoint_path: "models/calibrated.candidate.pth"
evaluation:
  report_dir: "evaluation"
  bootstrap_samples: 5
inference:
  checkpoint_path: "models/calibrated.candidate.pth"
  device: "cpu"
  allow_uncalibrated: false
  max_image_bytes: 1000000
  max_image_pixels: 1000000
run:
  root_dir: "runs"
promotion:
  min_macro_f1: 0.80
  min_balanced_accuracy: 0.80
  min_counterfeit_recall: 0.80
  target_checkpoint_path: "release/official.pth"
""".lstrip(),
        encoding="utf-8",
    )

    observed: dict[str, object] = {}

    class LightweightTrainer:
        def __init__(self, _model, **kwargs):
            observed["constructed"] = True
            observed["class_weights"] = kwargs["authenticity_class_weight"].tolist()
            observed["counterfeit_species_verified"] = kwargs["checkpoint_metadata"][
                "counterfeit_species_verified"
            ]

        def fit(self, _train, _val, resume_from=None):
            observed["fit"] = True
            assert resume_from is None
            metrics = SimpleNamespace(
                claimed_macro_f1=0.0,
                authenticity_macro_f1=0.0,
                authenticity_balanced_accuracy=0.0,
                species_macro_f1=None,
                species_supervision_coverage=0.0,
                selection_score=0.0,
                selection_tasks=(
                    "authenticity_macro_f1",
                    "authenticity_balanced_accuracy",
                ),
            )
            epoch = SimpleNamespace(
                epoch=0,
                training=True,
                total_loss=1.0,
                claimed_herb_loss=0.0,
                actual_species_loss=0.0,
                authenticity_loss=1.0,
                species_supervised_rows=0,
                batches=1,
                samples=2,
                freeze_stage="full_model",
                learning_rates=(0.001,),
                duration_seconds=0.01,
                entity_metrics=metrics,
            )
            return SimpleNamespace(
                train_epochs=(epoch,),
                validation_epochs=(SimpleNamespace(**(epoch.__dict__ | {"training": False})),),
                device="cpu",
                amp_enabled=False,
                best_epoch=0,
                best_selection_score=0.0,
                stopped_early=False,
            )

    monkeypatch.setattr(
        models_module,
        "build_torchvision_encoder",
        lambda *_args, **_kwargs: SimpleNamespace(encoder=object(), feature_dim=4),
    )
    monkeypatch.setattr(models_module, "MultiTaskHerbModel", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(trainer_module, "Trainer", LightweightTrainer)
    module = _load_train_script()

    result = module.main(
        [
            "--config",
            str(config),
            "--device",
            "cpu",
            "--run-id",
            "real-boundary",
        ]
    )

    assert result == 0
    assert observed == {
        "constructed": True,
        "class_weights": [1.0, 1.0],
        "counterfeit_species_verified": False,
        "fit": True,
    }
    assert (tmp_path / "data" / "splits.json").is_file()
    assert (tmp_path / "runs" / "real-boundary" / "run_metadata.json").is_file()
