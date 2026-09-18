from __future__ import annotations

import pytest
import torch
from herb_authentication.inference.calibration import (
    CalibrationBundle,
    KnownValidationData,
    ThresholdSearchData,
    UnknownCalibrationData,
    fit_calibrators,
    select_thresholds,
    species_exact_match,
)


def test_inference_package_exposes_stable_core_api() -> None:
    from herb_authentication import inference

    assert inference.CalibrationBundle is CalibrationBundle
    assert inference.fit_calibrators is fit_calibrators
    assert inference.select_thresholds is select_thresholds


def test_species_exact_match_rejects_partial_match() -> None:
    assert species_exact_match({"当归"}, {"当归"}) is True
    assert species_exact_match({"当归"}, {"当归", "欧当归"}) is False


def test_threshold_selection_uses_required_deterministic_tie_break() -> None:
    data = ThresholdSearchData(
        known_claimed_confidences=torch.tensor([0.9, 0.8, 0.7]),
        known_species_probabilities=torch.tensor([[0.9], [0.8], [0.7]]),
        known_authenticity_confidences=torch.tensor([0.9, 0.8, 0.7]),
        known_claimed_correct=torch.tensor([True, True, False]),
        known_species_targets=torch.tensor([[True], [True], [True]]),
        known_authenticity_correct=torch.tensor([True, True, False]),
        unknown_claimed_confidences=torch.tensor([0.75]),
        unknown_species_probabilities=torch.tensor([[0.75]]),
        unknown_authenticity_confidences=torch.tensor([0.75]),
    )

    first = select_thresholds(
        data,
        max_unknown_accept_rate=0.0,
        min_known_selective_accuracy=1.0,
    )
    second = select_thresholds(
        data,
        max_unknown_accept_rate=0.0,
        min_known_selective_accuracy=1.0,
    )

    assert first == second
    assert first is not None
    assert first.claimed_herb == pytest.approx(0.8)
    assert first.actual_species == pytest.approx(0.8)
    assert first.authenticity == pytest.approx(0.8)
    assert first.known_coverage == pytest.approx(2 / 3)
    assert first.known_selective_accuracy == pytest.approx(1.0)
    assert first.unknown_accept_rate == pytest.approx(0.0)


def test_threshold_selection_returns_none_when_constraints_are_impossible() -> None:
    data = ThresholdSearchData(
        known_claimed_confidences=torch.tensor([0.9]),
        known_species_probabilities=torch.tensor([[0.9]]),
        known_authenticity_confidences=torch.tensor([0.9]),
        known_claimed_correct=torch.tensor([False]),
        known_species_targets=torch.tensor([[True]]),
        known_authenticity_correct=torch.tensor([True]),
        unknown_claimed_confidences=torch.tensor([0.9]),
        unknown_species_probabilities=torch.tensor([[0.9]]),
        unknown_authenticity_confidences=torch.tensor([0.9]),
    )

    selected = select_thresholds(
        data,
        max_unknown_accept_rate=0.0,
        min_known_selective_accuracy=1.0,
    )

    assert selected is None


def test_threshold_constraints_include_exact_fractional_boundaries() -> None:
    known_count = 20
    unknown_count = 20
    data = ThresholdSearchData(
        known_claimed_confidences=torch.full((known_count,), 0.9),
        known_species_probabilities=torch.full((known_count, 1), 0.9),
        known_authenticity_confidences=torch.full((known_count,), 0.9),
        known_claimed_correct=torch.tensor([True] * 19 + [False]),
        known_species_targets=torch.ones((known_count, 1), dtype=torch.bool),
        known_authenticity_correct=torch.ones(known_count, dtype=torch.bool),
        unknown_claimed_confidences=torch.tensor([0.9] + [0.1] * 19),
        unknown_species_probabilities=torch.tensor([[0.9]] + [[0.1]] * 19),
        unknown_authenticity_confidences=torch.tensor([0.9] + [0.1] * 19),
    )

    selected = select_thresholds(
        data,
        max_unknown_accept_rate=1 / unknown_count,
        min_known_selective_accuracy=19 / known_count,
    )

    assert selected is not None
    assert selected.known_coverage == pytest.approx(1.0)
    assert selected.known_selective_accuracy == pytest.approx(0.95)
    assert selected.unknown_accept_rate == pytest.approx(0.05)


def test_fit_calibrators_is_deterministic_and_returns_serializable_bundle() -> None:
    known = KnownValidationData(
        claimed_logits=torch.tensor([[4.0, 0.0], [0.0, 4.0]]),
        claimed_targets=torch.tensor([0, 1]),
        species_logits=torch.tensor([[4.0, -4.0], [-4.0, 4.0]]),
        species_targets=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        species_mask=torch.tensor([True, True]),
        authenticity_logits=torch.tensor([[4.0, 0.0], [0.0, 4.0]]),
        authenticity_targets=torch.tensor([0, 1]),
    )
    unknown = UnknownCalibrationData(
        claimed_logits=torch.tensor([[0.0, 0.0]]),
        species_logits=torch.tensor([[0.0, 0.0]]),
        authenticity_logits=torch.tensor([[0.0, 0.0]]),
    )

    first = fit_calibrators(
        known,
        unknown,
        max_unknown_accept_rate=0.0,
        min_known_selective_accuracy=1.0,
    )
    second = fit_calibrators(
        known,
        unknown,
        max_unknown_accept_rate=0.0,
        min_known_selective_accuracy=1.0,
    )

    assert first == second
    assert first.is_calibrated is True
    assert first.thresholds is not None
    assert all(value > 0 for value in first.temperatures.values())
    assert CalibrationBundle.from_dict(first.to_dict()) == first


def test_fit_calibrators_without_feasible_threshold_fails_closed() -> None:
    known = KnownValidationData(
        claimed_logits=torch.tensor([[4.0, 0.0]]),
        claimed_targets=torch.tensor([1]),
        species_logits=torch.tensor([[4.0]]),
        species_targets=torch.tensor([[1.0]]),
        species_mask=torch.tensor([True]),
        authenticity_logits=torch.tensor([[4.0, 0.0]]),
        authenticity_targets=torch.tensor([0]),
    )
    unknown = UnknownCalibrationData(
        claimed_logits=torch.tensor([[4.0, 0.0]]),
        species_logits=torch.tensor([[4.0]]),
        authenticity_logits=torch.tensor([[4.0, 0.0]]),
    )

    bundle = fit_calibrators(
        known,
        unknown,
        max_unknown_accept_rate=0.0,
        min_known_selective_accuracy=1.0,
    )

    assert bundle == CalibrationBundle.uncalibrated()


def test_calibration_rejects_unverified_or_non_finite_inputs() -> None:
    known = KnownValidationData(
        claimed_logits=torch.tensor([[float("nan"), 0.0]]),
        claimed_targets=torch.tensor([0]),
        species_logits=torch.tensor([[1.0]]),
        species_targets=torch.tensor([[1.0]]),
        species_mask=torch.tensor([True]),
        authenticity_logits=torch.tensor([[1.0, 0.0]]),
        authenticity_targets=torch.tensor([0]),
    )
    unknown = UnknownCalibrationData(
        claimed_logits=torch.tensor([[0.0, 0.0]]),
        species_logits=torch.tensor([[0.0]]),
        authenticity_logits=torch.tensor([[0.0, 0.0]]),
    )

    with pytest.raises(ValueError, match="有限"):
        fit_calibrators(known, unknown)


def test_threshold_search_uses_exact_species_set_correctness() -> None:
    data = ThresholdSearchData(
        known_claimed_confidences=torch.tensor([0.9]),
        known_species_probabilities=torch.tensor([[0.9, 0.8]]),
        known_authenticity_confidences=torch.tensor([0.9]),
        known_claimed_correct=torch.tensor([True]),
        known_species_targets=torch.tensor([[True, False]]),
        known_authenticity_correct=torch.tensor([True]),
        unknown_claimed_confidences=torch.tensor([0.1]),
        unknown_species_probabilities=torch.tensor([[0.1, 0.1]]),
        unknown_authenticity_confidences=torch.tensor([0.1]),
    )

    selected = select_thresholds(
        data,
        max_unknown_accept_rate=0.0,
        min_known_selective_accuracy=1.0,
    )

    assert selected is not None
    assert selected.actual_species == pytest.approx(0.9)
