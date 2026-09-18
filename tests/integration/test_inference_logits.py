from __future__ import annotations

from io import BytesIO

import torch
from herb_authentication.inference.calibration import CalibrationBundle
from herb_authentication.inference.service import InferenceService
from herb_authentication.models import MultiTaskLogits
from PIL import Image
from torch import nn


class FixedModel(nn.Module):
    def forward(self, images: torch.Tensor) -> MultiTaskLogits:
        batch = images.shape[0]
        return MultiTaskLogits(
            claimed_herb=torch.tensor([[2.0, 0.0]]).expand(batch, -1),
            actual_species=torch.tensor([[1.0, -1.0]]).expand(batch, -1),
            authenticity=torch.tensor([[0.0, 2.0]]).expand(batch, -1),
        )


def _png() -> bytes:
    output = BytesIO()
    Image.new("RGB", (8, 8)).save(output, format="PNG")
    return output.getvalue()


def test_public_predict_logits_reuses_safe_decode_and_returns_three_heads() -> None:
    service = InferenceService(
        model=FixedModel(),
        preprocess=lambda _image: torch.zeros(3, 8, 8),
        claimed_herb_classes=("当归", "黄芪"),
        actual_species_classes=("当归", "欧当归"),
        calibration=CalibrationBundle.uncalibrated(),
        model_version="tiny-v1",
        device="cpu",
    )

    logits = service.predict_logits(_png())

    assert logits.claimed_herb.shape == (1, 2)
    assert logits.actual_species.shape == (1, 2)
    assert logits.authenticity.shape == (1, 2)
