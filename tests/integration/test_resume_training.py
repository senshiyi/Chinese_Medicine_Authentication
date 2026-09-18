from __future__ import annotations

import random

import numpy as np
import torch
from herb_authentication.models import MultiTaskHerbModel
from herb_authentication.training.checkpoints import (
    CheckpointContract,
    capture_rng_state,
    restore_rng_state,
)
from herb_authentication.training.trainer import Trainer
from torch import nn


def consume_random_step() -> tuple[float, float, torch.Tensor]:
    return random.random(), float(np.random.random()), torch.rand(3)


def test_rng_restore_matches_continuous_execution() -> None:
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    consume_random_step()
    saved = capture_rng_state()
    expected = [consume_random_step() for _ in range(3)]

    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    restore_rng_state(saved)
    resumed = [consume_random_step() for _ in range(3)]

    for expected_step, resumed_step in zip(expected, resumed, strict=True):
        assert expected_step[:2] == resumed_step[:2]
        assert torch.equal(expected_step[2], resumed_step[2])


class TinyEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(4, 3)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.projection(images)


def make_model() -> MultiTaskHerbModel:
    return MultiTaskHerbModel(
        TinyEncoder(),
        feature_dim=3,
        num_claimed=2,
        num_species=2,
        dropout=0.0,
    )


def make_loader() -> list[dict[str, object]]:
    return [
        {
            "image": torch.tensor(
                [[1.0, 0.0, 0.5, -0.5], [0.0, 1.0, -0.5, 0.5]],
                dtype=torch.float32,
            ),
            "claimed_target": torch.tensor([0, 1]),
            "actual_species_target": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            "authenticity_target": torch.tensor([0, 1]),
            "actual_species_mask": torch.tensor([True, True]),
        }
    ]


def checkpoint_contract() -> CheckpointContract:
    return CheckpointContract(
        schema_version=1,
        backbone="tiny",
        claimed_herb_classes=("当归", "黄芪"),
        actual_species_classes=("当归", "欧当归"),
        authenticity_classes=("authentic", "counterfeit"),
        image_size=2,
        preprocessing_version="test-v1",
        data_fingerprint="data",
        split_fingerprint="locked-split",
        training_config_fingerprint="training",
    )


def test_resume_restores_complete_epoch_history_into_summary_and_checkpoint(tmp_path) -> None:
    checkpoint_path = tmp_path / "resume-history.pth"
    first = Trainer(
        make_model(),
        epochs=1,
        learning_rate=1e-2,
        freeze_backbone=False,
        checkpoint_path=checkpoint_path,
        checkpoint_contract=checkpoint_contract(),
        checkpoint_metadata={"run_id": "parent-run"},
    )
    first.fit(make_loader(), make_loader())

    resumed = Trainer(
        make_model(),
        epochs=2,
        learning_rate=1e-2,
        freeze_backbone=False,
        checkpoint_path=checkpoint_path,
        checkpoint_contract=checkpoint_contract(),
        checkpoint_metadata={"run_id": "child-run"},
    )
    summary = resumed.fit(make_loader(), make_loader(), resume_from=checkpoint_path)

    assert [row.epoch for row in summary.train_epochs] == [0, 1]
    assert [row.epoch for row in summary.validation_epochs] == [0, 1]
    assert summary.best_epoch in {row.epoch for row in summary.validation_epochs}
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert [row["epoch"] for row in state["training"]["history"]["train_epochs"]] == [0, 1]
    assert [row["epoch"] for row in state["training"]["history"]["validation_epochs"]] == [
        0,
        1,
    ]
