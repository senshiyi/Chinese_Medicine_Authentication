from __future__ import annotations

from hashlib import sha256
from io import BytesIO
from pathlib import Path

import pytest
import torch
from herb_authentication.errors import CheckpointCompatibilityError, InputImageError
from herb_authentication.inference.calibration import CalibrationBundle, SelectedThresholds
from herb_authentication.inference.closed_set import ClosedSetCalibration
from herb_authentication.inference.service import InferenceService
from herb_authentication.models.multitask import MultiTaskLogits
from PIL import Image
from torch import nn


class TinyModel(nn.Module):
    def __init__(
        self,
        *,
        claimed: tuple[float, ...] = (5.0, 0.0),
        species: tuple[float, ...] = (5.0, -5.0),
        authenticity: tuple[float, ...] = (5.0, 0.0),
    ) -> None:
        super().__init__()
        self.register_buffer("claimed_logits", torch.tensor([claimed]))
        self.register_buffer("species_logits", torch.tensor([species]))
        self.register_buffer("authenticity_logits", torch.tensor([authenticity]))
        self.calls = 0

    def forward(self, images: torch.Tensor) -> MultiTaskLogits:
        self.calls += 1
        batch = images.shape[0]
        return MultiTaskLogits(
            claimed_herb=self.claimed_logits.expand(batch, -1),
            actual_species=self.species_logits.expand(batch, -1),
            authenticity=self.authenticity_logits.expand(batch, -1),
        )


def calibrated_bundle() -> CalibrationBundle:
    return CalibrationBundle(
        is_calibrated=True,
        method="temperature_scaling",
        temperatures={"claimed_herb": 1.0, "actual_species": 1.0, "authenticity": 1.0},
        thresholds=SelectedThresholds(0.8, 0.8, 0.8, 0.8, 0.96, 0.04),
    )


def png_bytes(size: tuple[int, int] = (8, 8)) -> bytes:
    stream = BytesIO()
    Image.new("RGB", size, color=(80, 120, 40)).save(stream, format="PNG")
    return stream.getvalue()


def service(
    model: nn.Module,
    *,
    calibration: CalibrationBundle | None = None,
    max_image_bytes: int = 1024 * 1024,
    max_pixels: int = 1024,
    max_decoded_bytes: int = 4096,
) -> InferenceService:
    return InferenceService(
        model=model,
        preprocess=lambda image: torch.zeros(3, image.height, image.width),
        claimed_herb_classes=("当归", "黄芪"),
        actual_species_classes=("当归", "欧当归"),
        calibration=calibration or calibrated_bundle(),
        model_version="tiny-v1",
        device="cpu",
        max_image_bytes=max_image_bytes,
        max_pixels=max_pixels,
        max_decoded_bytes=max_decoded_bytes,
    )


def test_predict_bytes_returns_authentic_result_without_rebuilding_model() -> None:
    model = TinyModel()
    inference = service(model)

    first = inference.predict_image(png_bytes())
    second = inference.predict_image(png_bytes())

    assert first.message == "此药品为正品（置信度：99.3%）。"
    assert first.rejected is False
    assert first.actual_species == (first.actual_species[0],)
    assert first.actual_species[0].label == "当归"
    assert model.calls == 2
    assert second.model_version == "tiny-v1"


def test_predict_path_returns_counterfeit_with_actual_species(tmp_path: Path) -> None:
    path = tmp_path / "药材.png"
    path.write_bytes(png_bytes())
    inference = service(TinyModel(species=(-5.0, 5.0), authenticity=(0.0, 5.0)))

    result = inference.predict_image(path)

    assert result.message == ("此药品为伪品（置信度：99.3%），声称为‘当归’，实际应为‘欧当归’。")
    assert result.to_dict()["actual_species"] == [
        {"label": "欧当归", "confidence": pytest.approx(0.993307)}
    ]


def test_unknown_actual_species_rejects_entire_prediction() -> None:
    result = service(TinyModel(species=(0.0, 0.0))).predict_image(png_bytes())

    assert result.rejected is True
    assert "actual_species_unknown" in result.rejection_reasons
    assert result.actual_species == ()
    assert result.message == "无法判断该药品的实际药材及真伪，请人工复核。"
    assert "%" not in result.message and "正品" not in result.message


def test_uncalibrated_service_fails_closed_without_percentage() -> None:
    result = service(TinyModel(), calibration=CalibrationBundle.uncalibrated()).predict_image(
        png_bytes()
    )

    assert result.rejected is True
    assert result.message == "模型尚未完成真实数据校准，请人工复核。"
    assert result.rejection_reasons == ("model_uncalibrated",)
    assert "%" not in result.message


@pytest.mark.parametrize(
    ("payload", "overrides", "match"),
    [
        (b"not-an-image", {}, "解码"),
        (png_bytes(), {"max_image_bytes": 10}, "字节"),
        (png_bytes((20, 20)), {"max_pixels": 100}, "像素"),
        (png_bytes((20, 20)), {"max_decoded_bytes": 1000}, "解码"),
    ],
)
def test_invalid_or_oversized_input_raises_domain_error(
    payload: bytes, overrides: dict[str, int], match: str
) -> None:
    with pytest.raises(InputImageError, match=match):
        service(TinyModel(), **overrides).predict_image(payload)


def test_engineering_failure_is_not_disguised_as_model_rejection() -> None:
    def broken_preprocess(_image: Image.Image) -> torch.Tensor:
        raise RuntimeError("pipeline bug")

    inference = service(TinyModel())
    inference.preprocess = broken_preprocess

    with pytest.raises(RuntimeError, match="pipeline bug"):
        inference.predict_image(png_bytes())


def _closed_set_checkpoint_state(*, include_experimental_metadata: bool) -> dict[str, object]:
    model = TinyModel()
    metadata: dict[str, object] = {
        "backbone": "tiny",
        "claimed_herb_classes": ("当归", "黄芪"),
        "actual_species_classes": ("当归", "欧当归"),
        "authenticity_classes": ("authentic", "counterfeit"),
        "image_size": 8,
        "preprocessing_version": "test-v1",
        "model_version": "tiny-v1",
        "calibration": calibrated_bundle().to_dict(),
    }
    if include_experimental_metadata:
        metadata.update(
            {
                "closed_set_calibration": ClosedSetCalibration(
                    temperature=2.0,
                    algorithm_version="closed-set-temperature-v1",
                    validation_fingerprint="d" * 64,
                    checkpoint_fingerprint="a" * 64,
                    nll_before=0.8,
                    nll_after=0.7,
                    brier_before=0.4,
                    brier_after=0.3,
                    ece_before=0.2,
                    ece_after=0.1,
                ).to_dict(),
                "counterfeit_species_verified": False,
                "experimental_closed_set": True,
                "open_set_calibrated": False,
                "data_fingerprint": "b" * 64,
                "split_fingerprint": "c" * 64,
            }
        )
    return {
        "schema_version": 1,
        "metadata": metadata,
        "model_state": model.state_dict(),
    }


def _load_tiny_checkpoint(path: Path) -> InferenceService:
    return InferenceService.from_checkpoint(
        path,
        device="cpu",
        model_factory=lambda _metadata: TinyModel(),
        preprocess_factory=lambda _metadata: lambda _image: torch.zeros(3, 8, 8),
    )


def test_old_checkpoint_metadata_remains_loadable_for_default_prediction(tmp_path: Path) -> None:
    checkpoint = tmp_path / "old-compatible.pth"
    torch.save(_closed_set_checkpoint_state(include_experimental_metadata=False), checkpoint)

    loaded = _load_tiny_checkpoint(checkpoint)

    assert loaded.predict_image(png_bytes()).model_version == "tiny-v1"
    with pytest.raises(CheckpointCompatibilityError, match="实验闭集"):
        loaded.predict_experimental_closed_set(png_bytes())


def test_checkpoint_closed_set_metadata_enables_experimental_prediction(tmp_path: Path) -> None:
    checkpoint = tmp_path / "closed-set.pth"
    torch.save(_closed_set_checkpoint_state(include_experimental_metadata=True), checkpoint)

    result = _load_tiny_checkpoint(checkpoint).predict_experimental_closed_set(png_bytes())

    assert result.experimental_closed_set is True
    assert result.open_set_calibrated is False
    assert result.model_fingerprint == sha256(checkpoint.read_bytes()).hexdigest()
    assert result.data_fingerprint == "b" * 64
    assert result.split_fingerprint == "c" * 64
