from __future__ import annotations

import pytest
import torch
from herb_authentication.evaluation import collect_known_predictions
from herb_authentication.models import MultiTaskLogits
from torch import nn


class FixedModel(nn.Module):
    def forward(self, images: torch.Tensor) -> MultiTaskLogits:
        assert images.shape == (2, 3, 4, 4)
        return MultiTaskLogits(
            claimed_herb=torch.tensor([[2.0, 0.0], [0.0, 2.0]]),
            actual_species=torch.tensor([[2.0, -2.0], [-2.0, 2.0]]),
            authenticity=torch.tensor([[2.0, 0.0], [0.0, 2.0]]),
        )


def test_collect_known_predictions_uses_temperatures_and_metadata() -> None:
    loader = [
        {
            "image": torch.zeros(2, 3, 4, 4),
            "claimed_target": torch.tensor([0, 1]),
            "actual_species_target": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            "actual_species_mask": torch.tensor([True, True]),
            "authenticity_target": torch.tensor([0, 1]),
            "metadata": (
                {"sample_id": "s1", "group_id": "g1"},
                {"sample_id": "s2", "group_id": "g2"},
            ),
        }
    ]

    predictions = collect_known_predictions(
        FixedModel(),
        loader,
        device="cpu",
        temperatures={"claimed_herb": 2.0, "actual_species": 2.0, "authenticity": 2.0},
    )

    assert predictions.sample_ids == ("s1", "s2")
    assert predictions.group_ids == ("g1", "g2")
    assert predictions.claimed_probabilities[0, 0].item() == pytest.approx(0.731059)
    assert predictions.species_probabilities[0, 0].item() == pytest.approx(0.731059)
