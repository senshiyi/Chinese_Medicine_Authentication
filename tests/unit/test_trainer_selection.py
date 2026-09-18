from __future__ import annotations

from types import MethodType

import pytest
import torch
from herb_authentication.models import MultiTaskHerbModel
from herb_authentication.training.metrics import EntityMetrics
from herb_authentication.training.trainer import EpochResult, Trainer
from torch import nn


class TinyEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(4, 3)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.projection(images)


def model() -> MultiTaskHerbModel:
    return MultiTaskHerbModel(TinyEncoder(), 3, num_claimed=2, num_species=2, dropout=0.0)


def loader_with_metadata() -> list[dict[str, object]]:
    return [
        {
            "image": torch.tensor([[1.0, 0.0, 0.5, -0.5], [0.0, 1.0, -0.5, 0.5]]),
            "claimed_target": torch.tensor([0, 1]),
            "actual_species_target": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            "authenticity_target": torch.tensor([0, 1]),
            "actual_species_mask": torch.tensor([True, True]),
            "metadata": (
                {"sample_id": "s1", "group_id": "g1"},
                {"sample_id": "s2", "group_id": "g2"},
            ),
        }
    ]


def test_epoch_records_entity_metrics_when_dataset_supplies_metadata() -> None:
    trainer = Trainer(model(), epochs=1, learning_rate=0.01, freeze_backbone=False)

    result = trainer.run_epoch(0, loader_with_metadata(), training=False)

    assert result.entity_metrics is not None
    assert result.entity_metrics.selection_tasks == (
        "authenticity_macro_f1",
        "authenticity_balanced_accuracy",
    )
    assert 0.0 <= result.entity_metrics.selection_score <= 1.0


def test_fit_early_stops_and_records_best_authenticity_score() -> None:
    trainer = Trainer(
        model(),
        epochs=5,
        learning_rate=0.01,
        freeze_backbone=False,
        early_stop_patience=1,
    )
    validation_scores = iter((0.8, 0.7, 0.6))

    def fake_run_epoch(self, epoch, _loader, *, training):
        score = 0.5 if training else next(validation_scores)
        metrics = EntityMetrics(
            score,
            score,
            score,
            score,
            1.0,
            score,
            (
                "authenticity_macro_f1",
                "authenticity_balanced_accuracy",
            ),
        )
        return EpochResult(
            epoch=epoch,
            training=training,
            total_loss=1.0,
            claimed_herb_loss=1.0,
            actual_species_loss=1.0,
            authenticity_loss=1.0,
            species_supervised_rows=2,
            batches=1,
            samples=2,
            freeze_stage="full_model",
            learning_rates=(0.01,),
            duration_seconds=0.01,
            entity_metrics=metrics,
        )

    trainer.run_epoch = MethodType(fake_run_epoch, trainer)
    summary = trainer.fit([], [])

    assert summary.stopped_early is True
    assert summary.best_epoch == 0
    assert summary.best_selection_score == 0.8
    assert len(summary.validation_epochs) == 2


def test_validation_missing_an_authenticity_class_fails_closed() -> None:
    trainer = Trainer(model(), epochs=1, learning_rate=0.01, freeze_backbone=False)
    loader = loader_with_metadata()
    loader[0]["authenticity_target"] = torch.tensor([0, 0])

    with pytest.raises(ValueError, match="验证集.*authenticity"):
        trainer.run_epoch(0, loader, training=False)
