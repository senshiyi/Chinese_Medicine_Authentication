from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
from herb_authentication.models import MultiTaskLogits


def _load_calibrate_script():
    path = Path(__file__).resolve().parents[2] / "项目" / "Files" / "calibrate.py"
    spec = importlib.util.spec_from_file_location("calibrate_adapter_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PublicOnlyService:
    def predict_logits(self, image: Path) -> MultiTaskLogits:
        assert image == Path("药材.png")
        return MultiTaskLogits(
            claimed_herb=torch.tensor([[2.0, 0.0]]),
            actual_species=torch.tensor([[1.0, -1.0]]),
            authenticity=torch.tensor([[0.0, 2.0]]),
        )


def test_calibration_adapter_uses_public_logits_interface() -> None:
    values = _load_calibrate_script()._predict_logits(
        PublicOnlyService(), Path("药材.png"), torch
    )

    assert tuple(tuple(value.shape) for value in values) == ((2,), (2,), (2,))
