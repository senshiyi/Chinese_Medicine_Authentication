from __future__ import annotations

import pytest
import torch
from herb_authentication.training.metrics import (
    ImageLevelPredictions,
    aggregate_by_sample,
    compute_entity_metrics,
    group_bootstrap_selection_score,
    harmonic_mean,
)


def make_predictions() -> ImageLevelPredictions:
    probabilities = torch.tensor([[0.9, 0.1], [0.7, 0.3], [0.2, 0.8], [0.1, 0.9]])
    return ImageLevelPredictions(
        claimed_probabilities=probabilities,
        species_probabilities=probabilities,
        authenticity_probabilities=probabilities,
        claimed_targets=torch.tensor([0, 0, 1, 1]),
        species_targets=torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]),
        authenticity_targets=torch.tensor([0, 0, 1, 1]),
        species_mask=torch.tensor([True, True, True, True]),
        sample_ids=("sample-a", "sample-a", "sample-b", "sample-c"),
        group_ids=("group-a", "group-a", "group-b", "group-c"),
    )


def test_aggregate_by_sample_averages_probabilities() -> None:
    aggregated = aggregate_by_sample(make_predictions())

    assert aggregated.sample_ids == ("sample-a", "sample-b", "sample-c")
    assert torch.allclose(aggregated.claimed_probabilities[0], torch.tensor([0.8, 0.2]))
    assert aggregated.group_ids == ("group-a", "group-b", "group-c")


def test_selection_score_uses_only_authenticity_macro_f1_and_balanced_accuracy() -> None:
    predictions = aggregate_by_sample(make_predictions())
    predictions.claimed_probabilities[:] = predictions.claimed_probabilities.flip(1)
    predictions.species_probabilities[:] = predictions.species_probabilities.flip(1)
    metrics = compute_entity_metrics(predictions, min_species_supervision_coverage=0.0)
    assert metrics.authenticity_macro_f1 == 1.0
    assert metrics.authenticity_balanced_accuracy == 1.0
    assert metrics.selection_score == 1.0
    assert metrics.selection_tasks == (
        "authenticity_macro_f1",
        "authenticity_balanced_accuracy",
    )


def test_species_below_coverage_remains_a_diagnostic() -> None:
    aggregated = aggregate_by_sample(make_predictions())
    aggregated.species_mask[:] = torch.tensor([True, False, False])

    metrics = compute_entity_metrics(aggregated, min_species_supervision_coverage=0.5)

    assert metrics.species_supervision_coverage == pytest.approx(1 / 3)
    assert metrics.species_macro_f1 == pytest.approx(0.5)
    assert metrics.selection_tasks == (
        "authenticity_macro_f1",
        "authenticity_balanced_accuracy",
    )


def test_harmonic_mean_fails_closed_on_zero_and_empty_values() -> None:
    assert harmonic_mean((1.0, 0.5)) == pytest.approx(2 / 3)
    assert harmonic_mean((1.0, 0.0)) == 0.0
    with pytest.raises(ValueError, match="至少一个"):
        harmonic_mean(())


def test_group_bootstrap_is_deterministic() -> None:
    aggregated = aggregate_by_sample(make_predictions())
    first = group_bootstrap_selection_score(
        aggregated, min_species_supervision_coverage=0.5, samples=20, seed=7
    )
    second = group_bootstrap_selection_score(
        aggregated, min_species_supervision_coverage=0.5, samples=20, seed=7
    )

    assert first == second
    assert first.samples == 20
    assert 0.0 <= first.lower <= first.median <= first.upper <= 1.0
