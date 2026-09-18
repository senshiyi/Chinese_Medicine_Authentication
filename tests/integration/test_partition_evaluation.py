from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVALUATE = PROJECT_ROOT / "项目" / "Files" / "evaluate.py"


def _module():
    spec = importlib.util.spec_from_file_location("partition_evaluation_under_test", EVALUATE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fingerprints() -> dict[str, str]:
    return {
        "manifest": "a" * 64,
        "split": "b" * 64,
        "config": "c" * 64,
        "source_candidate": "d" * 64,
        "calibration": "e" * 64,
    }


def _thresholds() -> dict[str, float]:
    return {
        "macro_f1": 0.8,
        "balanced_accuracy": 0.8,
        "counterfeit_recall": 0.8,
    }


def _write_pattern_jpeg(path: Path, identity: str) -> None:
    digest = hashlib.sha256(identity.encode("ascii")).digest()
    pixels = [255 if digest[index // 8] & (1 << (index % 8)) else 0 for index in range(64)]
    image = Image.new("L", (8, 8))
    image.putdata(pixels)
    image.convert("RGB").save(path, format="JPEG", quality=100, subsampling=0)


def _canonical_closed_set(tmp_path: Path, module):
    from herb_authentication.data import (
        create_group_splits,
        validate_manifest_pair,
        write_split_artifacts,
    )
    from herb_authentication.data.preparation import MANIFEST_FIELDNAMES

    data_root = tmp_path / "images"
    data_root.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    rows = []
    for authenticity in ("authentic", "counterfeit"):
        for index in range(3):
            name = f"{authenticity}-{index}.jpg"
            _write_pattern_jpeg(data_root / name, f"{authenticity}-{index}")
            rows.append(
                {
                    "image_path": (data_root / name).resolve().as_posix(),
                    "group_id": f"{authenticity}-{index}",
                    "sample_id": f"{authenticity}-{index}",
                    "claimed_herb_name": "Drynaria_fortunei",
                    "authenticity": authenticity,
                    "actual_species": ("Drynaria_fortunei" if authenticity == "authentic" else ""),
                    "species_annotation_status": (
                        "verified" if authenticity == "authentic" else "unverified"
                    ),
                    "batch_id": "",
                    "origin": "",
                    "capture_device": "",
                    "collected_at": "",
                    "annotator": "",
                    "notes": "",
                }
            )
    manifest = data_dir / "metadata.csv"
    with manifest.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    bundle = validate_manifest_pair(
        manifest,
        unknown_manifest_path=None,
        data_root=data_root,
        report_path=tmp_path / "manifest-validation.json",
        encoding="utf-8-sig",
        max_image_bytes=1_000_000,
        max_image_pixels=1_000_000,
        perceptual_hash_distance=0,
    )
    assignment = create_group_splits(
        bundle.known,
        ratios=(0.70, 0.15, 0.15),
        seed=42,
        perceptual_hash_distance=0,
    )
    artifacts = write_split_artifacts(bundle.known, assignment, data_dir)
    source = tmp_path / "candidate.pth"
    source.write_bytes(b"candidate")
    checkpoint = tmp_path / "calibrated.pth"
    checkpoint.write_bytes(b"calibrated")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run_metadata.json").write_text(
        json.dumps({"checkpoint": str(source)}), encoding="utf-8"
    )
    config = SimpleNamespace(
        data=SimpleNamespace(
            manifest_path=manifest,
            data_root=data_root,
            text_encoding="utf-8-sig",
            seed=42,
            image_size=224,
            batch_size=2,
            num_workers=0,
            max_image_bytes=1_000_000,
            max_image_pixels=1_000_000,
            perceptual_hash_distance=0,
        ),
        model=SimpleNamespace(
            backbone="resnet18",
            pretrained=False,
            dropout=0.0,
            freeze_backbone=False,
            unfreeze_epoch=0,
        ),
        training=SimpleNamespace(),
        promotion=SimpleNamespace(
            min_macro_f1=0.8,
            min_balanced_accuracy=0.8,
            min_counterfeit_recall=0.8,
        ),
        evaluation=SimpleNamespace(bootstrap_samples=2),
        inference=SimpleNamespace(device="cpu"),
    )
    config_fingerprint = module._training_config_fingerprint(config)
    manifest_fingerprint = hashlib.sha256(manifest.read_bytes()).hexdigest()
    split_fingerprint = hashlib.sha256(artifacts.splits_json.read_bytes()).hexdigest()
    validation_fingerprint = hashlib.sha256(artifacts.val_csv.read_bytes()).hexdigest()
    source_fingerprint = hashlib.sha256(source.read_bytes()).hexdigest()
    metadata = {
        "authenticity_classes": ("authentic", "counterfeit"),
        "data_fingerprint": assignment.input_fingerprint,
        "split_fingerprint": split_fingerprint,
        "training_config_fingerprint": config_fingerprint,
        "split_input_fingerprint": assignment.input_fingerprint,
        "split_artifact_fingerprint": split_fingerprint,
        "experimental_closed_set": True,
        "open_set_calibrated": False,
        "fingerprints": {
            "manifest": manifest_fingerprint,
            "known_validation": validation_fingerprint,
            "locked_split": split_fingerprint,
        },
        "closed_set_calibration": {
            "temperature": 1.0,
            "algorithm_version": "closed-set-temperature-v1",
            "validation_fingerprint": validation_fingerprint,
            "checkpoint_fingerprint": source_fingerprint,
            "nll_before": 0.5,
            "nll_after": 0.4,
            "brier_before": 0.3,
            "brier_after": 0.2,
            "ece_before": 0.1,
            "ece_after": 0.05,
        },
    }
    train_image = next(
        record.row.image_path
        for record in bundle.known
        if assignment.by_group[record.row.group_id] == "train"
    )
    return config, metadata, checkpoint, run_dir, artifacts, train_image


def _assert_locked_preflight_rejects(
    module, config, metadata, checkpoint: Path, run_dir: Path, monkeypatch
) -> None:
    locked_dir = run_dir / "locked_test"
    locked_dir.mkdir()
    json_path = locked_dir / "evaluation.json"
    csv_path = locked_dir / "evaluation.csv"
    json_path.write_bytes(b"old-json")
    csv_path.write_bytes(b"old-csv")
    calls = 0

    def collect(**_kwargs):
        nonlocal calls
        calls += 1
        return torch.tensor([[3.0, 0.0], [0.0, 3.0]]), torch.tensor([0, 1])

    monkeypatch.setattr(module, "_collect_closed_set_logits", collect)
    from herb_authentication.errors import HerbAuthenticationError

    with pytest.raises((HerbAuthenticationError, ValueError)):
        module._evaluate_closed_set_cli(
            config=config,
            checkpoint=checkpoint,
            metadata=metadata,
            partition="locked-test",
            run_dir=run_dir,
        )
    assert calls == 0
    assert not (locked_dir / "reservation.json").exists()
    assert json_path.read_bytes() == b"old-json"
    assert csv_path.read_bytes() == b"old-csv"


def test_partition_execution_selects_only_requested_csv(tmp_path: Path) -> None:
    module = _module()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "val.csv").write_text("partition\nval\n", encoding="utf-8")
    (data_dir / "test.csv").write_text("partition\ntest\n", encoding="utf-8")
    seen: list[Path] = []

    def collect(path: Path):
        seen.append(path)
        return torch.tensor([[3.0, 0.0], [0.0, 3.0]]), torch.tensor([0, 1])

    report, _ = module._execute_closed_set_partition(
        partition="validation",
        data_dir=data_dir,
        run_dir=tmp_path / "run",
        fingerprint_bundle=_fingerprints(),
        temperature=2.0,
        bootstrap_samples=5,
        seed=7,
        thresholds=_thresholds(),
        logits_collector=collect,
    )

    assert seen == [data_dir.resolve() / "val.csv"]
    assert report["partition"] == "validation"
    assert report["run_id"] == "run"
    assert "promotion_gate" in report


def test_locked_test_reserves_before_inference_and_preserves_first_reports(
    tmp_path: Path,
) -> None:
    module = _module()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "test.csv").write_text("partition\ntest\n", encoding="utf-8")
    run_dir = tmp_path / "run"
    calls = 0

    def collect(path: Path):
        nonlocal calls
        calls += 1
        assert path == data_dir.resolve() / "test.csv"
        assert (run_dir / "locked_test" / "reservation.json").is_file()
        reservation = json.loads(
            (run_dir / "locked_test" / "reservation.json").read_text(encoding="utf-8")
        )
        assert reservation["run_id"] == "run"
        return torch.tensor([[3.0, 0.0], [0.0, 3.0]]), torch.tensor([0, 1])

    report, artifacts = module._execute_closed_set_partition(
        partition="locked-test",
        data_dir=data_dir,
        run_dir=run_dir,
        fingerprint_bundle=_fingerprints(),
        temperature=2.0,
        bootstrap_samples=5,
        seed=7,
        thresholds=_thresholds(),
        logits_collector=collect,
    )
    before_json = artifacts.json_path.read_bytes()
    before_csv = artifacts.csv_path.read_bytes()

    with pytest.raises(ValueError, match="该 run 已执行 locked test"):
        module._execute_closed_set_partition(
            partition="locked-test",
            data_dir=data_dir,
            run_dir=run_dir,
            fingerprint_bundle=_fingerprints(),
            temperature=2.0,
            bootstrap_samples=5,
            seed=7,
            thresholds=_thresholds(),
            logits_collector=collect,
        )

    assert calls == 1
    assert report["partition"] == "locked-test"
    assert "promotion_gate" not in report
    assert artifacts.json_path.read_bytes() == before_json
    assert artifacts.csv_path.read_bytes() == before_csv


def test_real_collector_uses_checkpoint_validated_class_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json
    from types import SimpleNamespace

    from herb_authentication.inference import service as service_module

    module = _module()
    manifest = tmp_path / "test.csv"
    manifest.write_text(
        "image_path,group_id,authenticity,partition\na.jpg,g1,authentic,test\nb.jpg,g2,counterfeit,test\n",
        encoding="utf-8",
    )
    split = tmp_path / "splits.json"
    split.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "algorithm_version": "group-stratified-v1",
                "seed": 42,
                "ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
                "input_fingerprint": "f" * 64,
                "by_group": {"g1": "test", "g2": "test"},
            }
        ),
        encoding="utf-8",
    )

    class FakeService:
        def predict_logits(self, image: Path):
            logits = [[3.0, 0.0]] if image.name == "a.jpg" else [[0.0, 3.0]]
            return SimpleNamespace(authenticity=torch.tensor(logits))

    monkeypatch.setattr(
        service_module.InferenceService,
        "from_checkpoint",
        lambda *_args, **_kwargs: FakeService(),
    )

    logits, targets = module._collect_closed_set_logits(
        checkpoint=tmp_path / "calibrated.pth",
        manifest=manifest,
        data_root=tmp_path,
        encoding="utf-8",
        device="cpu",
        split=split,
        expected_partition="test",
    )

    assert logits.tolist() == [[3.0, 0.0], [0.0, 3.0]]
    assert targets.tolist() == [0, 1]


def test_locked_cli_subprocess_is_one_shot_and_preserves_first_reports(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "test.csv").write_text("partition\ntest\n", encoding="utf-8")
    checkpoint = tmp_path / "calibrated.pth"
    checkpoint.write_bytes(b"checkpoint")
    run_dir = tmp_path / "run"
    counter = tmp_path / "collector-count.txt"
    wrapper = tmp_path / "invoke_evaluate.py"
    wrapper.write_text(
        f"""from __future__ import annotations
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
import torch
EVALUATE = Path({str(EVALUATE)!r})
spec = importlib.util.spec_from_file_location("locked_cli_subprocess", EVALUATE)
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
data_dir, run_dir, checkpoint, counter = map(Path, sys.argv[1:])
bundle = {{
 "manifest": "a" * 64,
 "split": "b" * 64,
 "config": "c" * 64,
 "source_candidate": "d" * 64,
 "calibration": "e" * 64,
}}
config = SimpleNamespace(
 calibration=SimpleNamespace(mode="closed_set_temperature"),
 data=SimpleNamespace(manifest_path=data_dir/"metadata.csv",data_root=data_dir,text_encoding="utf-8",seed=7,max_image_bytes=1000000,max_image_pixels=1000000),
 promotion=SimpleNamespace(min_macro_f1=.8,min_balanced_accuracy=.8,min_counterfeit_recall=.8),
 evaluation=SimpleNamespace(bootstrap_samples=2), inference=SimpleNamespace(device="cpu"))
import herb_authentication.config as config_module
config_module.load_config = lambda _path: config
module._checkpoint_metadata = lambda _path: {{}}
module._source_candidate_from_run = lambda _path: checkpoint
module._training_config_fingerprint = lambda _config: "c"*64
module._validate_closed_set_fingerprints = lambda **_kwargs: (
 bundle,
 SimpleNamespace(temperature=1.0),
)
def collect(**_kwargs):
 count = int(counter.read_text() or "0") if counter.exists() else 0
 counter.write_text(str(count + 1))
 return torch.tensor([[3.,0.],[0.,3.]]), torch.tensor([0,1])
module._collect_closed_set_logits = collect
raise SystemExit(module.main([
 "--config", str(data_dir / "config.yaml"),
 "--partition", "locked-test",
 "--run-dir", str(run_dir),
 "--checkpoint", str(checkpoint),
]))
""",
        encoding="utf-8",
    )
    command = [
        sys.executable,
        "-X",
        "utf8",
        str(wrapper),
        str(data_dir),
        str(run_dir),
        str(checkpoint),
        str(counter),
    ]
    first = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert first.returncode == 0, first.stderr
    json_path = run_dir / "locked_test" / "evaluation.json"
    csv_path = run_dir / "locked_test" / "evaluation.csv"
    before = (json_path.read_bytes(), csv_path.read_bytes())
    second = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert first.returncode == 0
    assert second.returncode == 2
    assert "该 run 已执行 locked test" in second.stderr
    assert counter.read_text() == "1"
    assert (json_path.read_bytes(), csv_path.read_bytes()) == before


def test_locked_preflight_rejects_replaced_nonselected_image_before_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    config, metadata, checkpoint, run_dir, _artifacts, train_image = _canonical_closed_set(
        tmp_path, module
    )
    _write_pattern_jpeg(train_image, "replacement-bytes")

    _assert_locked_preflight_rejects(module, config, metadata, checkpoint, run_dir, monkeypatch)


def test_locked_preflight_rejects_legal_but_wrong_partition_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    config, metadata, checkpoint, run_dir, artifacts, _train_image = _canonical_closed_set(
        tmp_path, module
    )
    with artifacts.test_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames is not None
        rows = list(reader)
        fieldnames = tuple(reader.fieldnames)
    rows[0]["sha256"] = ("0" if rows[0]["sha256"][0] != "0" else "1") + rows[0]["sha256"][1:]
    rows[0]["perceptual_hash"] = ("0" if rows[0]["perceptual_hash"][0] != "0" else "1") + rows[0][
        "perceptual_hash"
    ][1:]
    with artifacts.test_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    _assert_locked_preflight_rejects(module, config, metadata, checkpoint, run_dir, monkeypatch)
