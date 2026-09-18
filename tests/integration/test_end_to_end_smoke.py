from __future__ import annotations

from io import BytesIO
from pathlib import Path

import torch
from herb_authentication.inference.calibration import CalibrationBundle, SelectedThresholds
from herb_authentication.inference.service import InferenceService
from herb_authentication.models import MultiTaskHerbModel
from herb_authentication.training.checkpoints import save_checkpoint_atomic
from herb_authentication.training.trainer import Trainer
from PIL import Image
from torch import nn


def _model() -> MultiTaskHerbModel:
    encoder = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 4), nn.ReLU())
    return MultiTaskHerbModel(encoder, 4, num_claimed=2, num_species=2, dropout=0.0)


def _batch() -> dict[str, torch.Tensor]:
    return {
        "image": torch.zeros(2, 3, 8, 8),
        "claimed_target": torch.tensor([0, 1]),
        "actual_species_target": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        "actual_species_mask": torch.tensor([True, True]),
        "authenticity_target": torch.tensor([0, 1]),
    }


def _image_bytes() -> bytes:
    stream = BytesIO()
    Image.new("RGB", (8, 8), color=(90, 60, 30)).save(stream, format="PNG")
    return stream.getvalue()


def _state(
    model: MultiTaskHerbModel,
    optimizer_state: dict[str, object],
    calibration: CalibrationBundle,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "metadata": {
            "backbone": "tiny",
            "claimed_herb_classes": ("当归", "黄芪"),
            "actual_species_classes": ("当归", "欧当归"),
            "authenticity_classes": ("authentic", "counterfeit"),
            "image_size": 8,
            "preprocessing_version": "test-v1",
            "model_version": "tiny-v1",
            "dropout": 0.0,
            "calibration": calibration.to_dict(),
        },
        "training": {"epoch": 0, "global_step": 1, "freeze_stage": "full_model"},
        "model_state": model.state_dict(),
        "optimizer_state": optimizer_state,
        "scheduler_state": None,
        "scaler_state": None,
        "rng_state": None,
        "dataloader_generator_state": None,
    }


def _load(path: Path) -> InferenceService:
    return InferenceService.from_checkpoint(
        path,
        device="cpu",
        model_factory=lambda _metadata: _model(),
        preprocess_factory=lambda _metadata: lambda _image: torch.zeros(3, 8, 8),
    )


def test_tiny_train_save_load_predict_pipeline(tmp_path: Path) -> None:
    model = _model()
    trainer = Trainer(
        model,
        epochs=1,
        learning_rate=0.01,
        freeze_backbone=False,
        device="cpu",
    )
    summary = trainer.fit([_batch()], [_batch()])
    assert summary.train_epochs[0].batches == 1

    checkpoint = tmp_path / "tiny.pth"
    save_checkpoint_atomic(
        checkpoint,
        _state(model, trainer.optimizer.state_dict(), CalibrationBundle.uncalibrated()),
        protected_paths=(),
    )
    result = _load(checkpoint).predict_image(_image_bytes())

    assert result.schema_version == 1
    assert result.rejected is True
    assert result.message == "模型尚未完成真实数据校准，请人工复核。"


def test_calibrated_counterfeit_pipeline_outputs_exact_message(tmp_path: Path) -> None:
    model = _model()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.claimed_head.bias.copy_(torch.tensor([5.0, 0.0]))
        model.species_head.bias.copy_(torch.tensor([-5.0, 5.0]))
        model.authenticity_head.bias.copy_(torch.tensor([0.0, torch.logit(torch.tensor(0.881))]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    calibration = CalibrationBundle(
        is_calibrated=True,
        method="temperature_scaling",
        temperatures={"claimed_herb": 1.0, "actual_species": 1.0, "authenticity": 1.0},
        thresholds=SelectedThresholds(0.8, 0.8, 0.8, 1.0, 1.0, 0.0),
    )
    checkpoint = tmp_path / "calibrated.pth"
    save_checkpoint_atomic(
        checkpoint,
        _state(model, optimizer.state_dict(), calibration),
        protected_paths=(),
    )

    result = _load(checkpoint).predict_image(_image_bytes())

    assert result.is_calibrated is True
    assert result.rejected is False
    assert result.message == (
        "此药品为伪品（置信度：88.1%），声称为‘当归’，实际应为‘欧当归’。"
    )
