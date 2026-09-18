from __future__ import annotations

import csv
import importlib.util
import subprocess
import sys
from hashlib import sha256
from pathlib import Path
from types import ModuleType

import torch
from herb_authentication.inference.calibration import (
    KnownValidationData,
    UnknownCalibrationData,
)
from herb_authentication.training.checkpoints import save_checkpoint_atomic

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CALIBRATE = PROJECT_ROOT / "项目" / "Files" / "calibrate.py"
LEGACY_MODELS = (
    PROJECT_ROOT / "项目" / "Models" / "best_model.pth",
    PROJECT_ROOT / "项目" / "Models" / "epoch_001.pth",
)


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("calibrate_cli_under_test", CALIBRATE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _manifests(tmp_path: Path, *, unknown_role: str = "calibration") -> tuple[Path, Path]:
    known = tmp_path / "known_val.csv"
    unknown = tmp_path / "unknown.csv"
    _write_csv(
        known,
        ("image_path", "dataset_role"),
        [{"image_path": "known.png", "dataset_role": "validation"}],
    )
    _write_csv(
        unknown,
        ("image_path", "dataset_role"),
        [{"image_path": "unknown.png", "dataset_role": unknown_role}],
    )
    return known, unknown


def _checkpoint_state() -> dict[str, object]:
    return {
        "schema_version": 1,
        "metadata": {
            "schema_version": 1,
            "backbone": "tiny",
            "claimed_herb_classes": ("当归", "黄芪"),
            "actual_species_classes": ("当归", "欧当归"),
            "authenticity_classes": ("authentic", "counterfeit"),
            "image_size": 8,
            "preprocessing_version": "test-v1",
            "calibration": {
                "is_calibrated": False,
                "method": None,
                "parameters": {},
                "thresholds": None,
            },
        },
        "training": {"epoch": 1},
        "model_state": {"weight": torch.tensor([1.0])},
        "optimizer_state": {"state": {}, "param_groups": []},
    }


def _calibration_data() -> tuple[KnownValidationData, UnknownCalibrationData]:
    return (
        KnownValidationData(
            claimed_logits=torch.tensor([[4.0, 0.0], [0.0, 4.0]]),
            claimed_targets=torch.tensor([0, 1]),
            species_logits=torch.tensor([[4.0, -4.0], [-4.0, 4.0]]),
            species_targets=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            species_mask=torch.tensor([True, True]),
            authenticity_logits=torch.tensor([[4.0, 0.0], [0.0, 4.0]]),
            authenticity_targets=torch.tensor([0, 1]),
        ),
        UnknownCalibrationData(
            claimed_logits=torch.tensor([[0.0, 0.0]]),
            species_logits=torch.tensor([[0.0, 0.0]]),
            authenticity_logits=torch.tensor([[0.0, 0.0]]),
        ),
    )


def test_help_is_utf8_and_does_not_import_torch() -> None:
    completed = subprocess.run(
        [sys.executable, "-I", "-X", "utf8", "-X", "importtime", str(CALIBRATE), "--help"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
        timeout=30,
    )

    assert completed.returncode == 0
    assert "--mode" in completed.stdout
    assert "--known-val" in completed.stdout
    assert "--unknown" in completed.stdout
    assert "torch" not in completed.stderr


def test_test_role_is_rejected_before_checkpoint_loading(tmp_path: Path) -> None:
    checkpoint = tmp_path / "input.pth"
    checkpoint.write_bytes(b"not-loaded")
    known, unknown = _manifests(tmp_path, unknown_role="test")
    before = sha256(checkpoint.read_bytes()).hexdigest()

    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-X",
            "utf8",
            str(CALIBRATE),
            "--checkpoint",
            str(checkpoint),
            "--known-val",
            str(known),
            "--unknown",
            str(unknown),
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
        timeout=30,
    )

    assert completed.returncode == 2
    assert "test 数据禁止用于校准" in completed.stderr
    assert sha256(checkpoint.read_bytes()).hexdigest() == before


def test_output_cannot_overwrite_input_or_legacy_models(tmp_path: Path, capsys) -> None:
    module = _module()
    checkpoint = tmp_path / "input.pth"
    checkpoint.write_bytes(b"protected")
    known, unknown = _manifests(tmp_path)
    protected_before = {path: sha256(path.read_bytes()).hexdigest() for path in LEGACY_MODELS}

    same_code = module.main(
        [
            "--checkpoint",
            str(checkpoint),
            "--known-val",
            str(known),
            "--unknown",
            str(unknown),
            "--output",
            str(checkpoint),
        ]
    )
    legacy_code = module.main(
        [
            "--checkpoint",
            str(checkpoint),
            "--known-val",
            str(known),
            "--unknown",
            str(unknown),
            "--output",
            str(LEGACY_MODELS[0]),
        ]
    )

    assert same_code == 2 and legacy_code == 2
    assert "不允许覆盖输入 checkpoint" in capsys.readouterr().err
    assert checkpoint.read_bytes() == b"protected"
    assert {
        path: sha256(path.read_bytes()).hexdigest() for path in LEGACY_MODELS
    } == protected_before


def test_success_writes_new_calibrated_checkpoint_atomically(tmp_path: Path) -> None:
    module = _module()
    checkpoint = tmp_path / "input.pth"
    output = tmp_path / "input.calibrated.pth"
    known, unknown = _manifests(tmp_path)
    state = _checkpoint_state()
    state["metadata"]["fingerprints"] = {
        "known_validation": sha256(known.read_bytes()).hexdigest(),
        "unknown_calibration": sha256(unknown.read_bytes()).hexdigest(),
    }
    save_checkpoint_atomic(checkpoint, state, protected_paths=())
    input_before = sha256(checkpoint.read_bytes()).hexdigest()
    legacy_before = {path: sha256(path.read_bytes()).hexdigest() for path in LEGACY_MODELS}

    code = module.main(
        [
            "--checkpoint",
            str(checkpoint),
            "--known-val",
            str(known),
            "--unknown",
            str(unknown),
            "--output",
            str(output),
            "--max-unknown-accept-rate",
            "0",
            "--min-known-selective-accuracy",
            "1",
        ],
        calibration_data_loader=lambda **_kwargs: _calibration_data(),
    )

    loaded = torch.load(output, map_location="cpu", weights_only=True)
    assert code == 0
    assert loaded["metadata"]["calibration"]["is_calibrated"] is True
    assert loaded["metadata"]["calibration"]["thresholds"] is not None
    assert "closed_set_calibration" not in loaded["metadata"]
    assert sha256(checkpoint.read_bytes()).hexdigest() == input_before
    assert (
        loaded["metadata"]["fingerprints"]["known_validation"]
        == sha256(known.read_bytes()).hexdigest()
    )
    assert (
        loaded["metadata"]["fingerprints"]["unknown_calibration"]
        == sha256(unknown.read_bytes()).hexdigest()
    )
    assert loaded["metadata"]["calibration"]["constraints"]["max_unknown_accept_rate"] == 0
    assert {path: sha256(path.read_bytes()).hexdigest() for path in LEGACY_MODELS} == legacy_before


def test_fingerprint_mismatch_stops_before_data_loading(tmp_path: Path, capsys) -> None:
    module = _module()
    checkpoint = tmp_path / "input.pth"
    output = tmp_path / "calibrated.pth"
    known, unknown = _manifests(tmp_path)
    state = _checkpoint_state()
    state["metadata"]["fingerprints"] = {"known_validation": "0" * 64}
    save_checkpoint_atomic(checkpoint, state, protected_paths=())
    loader_called = False

    def loader(**_kwargs):
        nonlocal loader_called
        loader_called = True
        return _calibration_data()

    code = module.main(
        [
            "--checkpoint",
            str(checkpoint),
            "--known-val",
            str(known),
            "--unknown",
            str(unknown),
            "--output",
            str(output),
        ],
        calibration_data_loader=loader,
    )

    assert code == 2
    assert "数据指纹" in capsys.readouterr().err
    assert loader_called is False
    assert not output.exists()


def test_no_feasible_threshold_does_not_create_output(tmp_path: Path, capsys) -> None:
    module = _module()
    checkpoint = tmp_path / "input.pth"
    output = tmp_path / "calibrated.pth"
    known_manifest, unknown_manifest = _manifests(tmp_path)
    state = _checkpoint_state()
    state["metadata"]["fingerprints"] = {
        "known_validation": sha256(known_manifest.read_bytes()).hexdigest(),
        "unknown_calibration": sha256(unknown_manifest.read_bytes()).hexdigest(),
    }
    save_checkpoint_atomic(checkpoint, state, protected_paths=())
    known, unknown = _calibration_data()
    known.claimed_targets[:] = torch.tensor([1, 0])

    code = module.main(
        [
            "--checkpoint",
            str(checkpoint),
            "--known-val",
            str(known_manifest),
            "--unknown",
            str(unknown_manifest),
            "--output",
            str(output),
            "--max-unknown-accept-rate",
            "0",
            "--min-known-selective-accuracy",
            "1",
        ],
        calibration_data_loader=lambda **_kwargs: (known, unknown),
    )

    assert code == 2
    assert "没有满足安全约束" in capsys.readouterr().err
    assert not output.exists()
