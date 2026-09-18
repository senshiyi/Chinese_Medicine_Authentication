from __future__ import annotations

import csv
import importlib.util
import json
from hashlib import sha256
from pathlib import Path
from types import ModuleType

import torch
from herb_authentication.training.checkpoints import save_checkpoint_atomic

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CALIBRATE = PROJECT_ROOT / "项目" / "Files" / "calibrate.py"


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("closed_set_calibrate_cli_under_test", CALIBRATE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _known_validation(path: Path) -> Path:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("image_path", "dataset_role", "authenticity"),
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "image_path": "authentic.jpg",
                    "dataset_role": "validation",
                    "authenticity": "authentic",
                },
                {
                    "image_path": "counterfeit.jpg",
                    "dataset_role": "validation",
                    "authenticity": "counterfeit",
                },
            ]
        )
    return path


def _checkpoint_state(validation_fingerprint: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "metadata": {
            "schema_version": 1,
            "backbone": "tiny",
            "claimed_herb_classes": ("骨碎补",),
            "actual_species_classes": ("Drynaria_fortunei",),
            "authenticity_classes": ("authentic", "counterfeit"),
            "image_size": 8,
            "preprocessing_version": "test-v1",
            "fingerprints": {"known_validation": validation_fingerprint},
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


def test_closed_set_mode_writes_isolated_authenticity_calibration(
    tmp_path: Path,
    capsys,
) -> None:
    module = _module()
    known = _known_validation(tmp_path / "validation.csv")
    checkpoint = tmp_path / "candidate.pth"
    output = tmp_path / "candidate.closed-set.pth"
    validation_fingerprint = sha256(known.read_bytes()).hexdigest()
    state = _checkpoint_state(validation_fingerprint)
    original_open_set_calibration = dict(state["metadata"]["calibration"])
    save_checkpoint_atomic(checkpoint, state, protected_paths=())
    checkpoint_fingerprint = sha256(checkpoint.read_bytes()).hexdigest()

    code = module.main(
        [
            "--mode",
            "closed-set-temperature",
            "--checkpoint",
            str(checkpoint),
            "--known-val",
            str(known),
            "--output",
            str(output),
        ],
        calibration_data_loader=lambda **_kwargs: (
            torch.tensor([[3.0, 0.0], [2.0, 0.0], [0.0, 3.0], [0.0, 2.0]]),
            torch.tensor([0, 1, 1, 0]),
        ),
    )

    loaded = torch.load(output, map_location="cpu", weights_only=True)
    block = loaded["metadata"]["closed_set_calibration"]
    printed = json.loads(capsys.readouterr().out)
    assert code == 0
    assert loaded["metadata"]["experimental_closed_set"] is True
    assert loaded["metadata"]["open_set_calibrated"] is False
    assert loaded["metadata"]["calibration"] == original_open_set_calibration
    assert block["validation_fingerprint"] == validation_fingerprint
    assert block["checkpoint_fingerprint"] == checkpoint_fingerprint
    assert set(block) == {
        "temperature",
        "algorithm_version",
        "validation_fingerprint",
        "checkpoint_fingerprint",
        "nll_before",
        "nll_after",
        "brier_before",
        "brier_after",
        "ece_before",
        "ece_after",
    }
    assert printed["closed_set_calibration"] == block
    assert printed["experimental_closed_set"] is True
    assert printed["open_set_calibrated"] is False


def test_closed_set_mode_rejects_unknown_manifest_before_checkpoint_loading(
    tmp_path: Path,
    capsys,
) -> None:
    module = _module()
    known = _known_validation(tmp_path / "validation.csv")
    unknown = tmp_path / "unknown.csv"
    unknown.write_text("dataset_role\ncalibration\n", encoding="utf-8")
    checkpoint = tmp_path / "candidate.pth"
    checkpoint.write_bytes(b"must-not-load")

    code = module.main(
        [
            "--mode",
            "closed-set-temperature",
            "--checkpoint",
            str(checkpoint),
            "--known-val",
            str(known),
            "--unknown",
            str(unknown),
        ]
    )

    assert code == 2
    assert "--unknown" in capsys.readouterr().err


def test_closed_set_mode_rejects_validation_fingerprint_mismatch_before_data_loading(
    tmp_path: Path,
    capsys,
) -> None:
    module = _module()
    known = _known_validation(tmp_path / "validation.csv")
    checkpoint = tmp_path / "candidate.pth"
    output = tmp_path / "candidate.closed-set.pth"
    save_checkpoint_atomic(checkpoint, _checkpoint_state("0" * 64), protected_paths=())
    loader_called = False

    def loader(**_kwargs):
        nonlocal loader_called
        loader_called = True
        return torch.ones(2, 2), torch.tensor([0, 1])

    code = module.main(
        [
            "--mode",
            "closed-set-temperature",
            "--checkpoint",
            str(checkpoint),
            "--known-val",
            str(known),
            "--output",
            str(output),
        ],
        calibration_data_loader=loader,
    )

    assert code == 2
    assert "数据指纹" in capsys.readouterr().err
    assert loader_called is False
    assert not output.exists()
