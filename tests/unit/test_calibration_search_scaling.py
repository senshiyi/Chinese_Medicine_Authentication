from __future__ import annotations

import math

import pytest
import torch
from herb_authentication.inference import calibration


def test_threshold_search_avoids_large_three_axis_cartesian_product(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample_count = 32
    species_count = 6
    known_claimed = torch.linspace(0.55, 0.95, sample_count)
    known_authenticity = torch.linspace(0.56, 0.96, sample_count)
    known_species = torch.linspace(
        0.51,
        0.99,
        sample_count * species_count,
    ).reshape(sample_count, species_count)
    unknown_claimed = torch.linspace(0.05, 0.45, sample_count)
    unknown_authenticity = torch.linspace(0.06, 0.46, sample_count)
    unknown_species = torch.linspace(
        0.01,
        0.49,
        sample_count * species_count,
    ).reshape(sample_count, species_count)
    raw_candidate_count = math.prod(
        torch.cat(pair).unique().numel() + 2
        for pair in (
            (known_claimed, unknown_claimed),
            (known_species.reshape(-1), unknown_species.reshape(-1)),
            (known_authenticity, unknown_authenticity),
        )
    )
    assert raw_candidate_count > 1_000_000
    axes = (
        calibration._threshold_candidates(known_claimed, unknown_claimed),
        calibration._threshold_candidates(known_species, unknown_species),
        calibration._threshold_candidates(known_authenticity, unknown_authenticity),
    )
    assert all(len(axis) <= calibration.MAX_THRESHOLD_CANDIDATES for axis in axes)

    data = calibration.ThresholdSearchData(
        known_claimed_confidences=known_claimed,
        known_species_probabilities=known_species,
        known_authenticity_confidences=known_authenticity,
        known_claimed_correct=torch.ones(sample_count, dtype=torch.bool),
        known_species_targets=known_species >= 0.75,
        known_authenticity_correct=torch.ones(sample_count, dtype=torch.bool),
        unknown_claimed_confidences=unknown_claimed,
        unknown_species_probabilities=unknown_species,
        unknown_authenticity_confidences=unknown_authenticity,
    )

    def reject_cartesian_product(*_axes: object) -> object:
        pytest.fail("阈值搜索不得枚举三条候选轴的笛卡尔积")

    monkeypatch.setattr(
        calibration,
        "product",
        reject_cartesian_product,
        raising=False,
    )

    original_intersections = calibration._pair_intersection_counts
    intersection_shapes: list[tuple[int, int]] = []

    def bounded_intersections(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        result = original_intersections(left, right)
        intersection_shapes.append(tuple(result.shape))
        return result

    monkeypatch.setattr(
        calibration,
        "_pair_intersection_counts",
        bounded_intersections,
    )

    selected = calibration.select_thresholds(
        data,
        max_unknown_accept_rate=0.0,
        min_known_selective_accuracy=0.0,
    )

    assert selected is not None
    assert len(intersection_shapes) <= 3 * calibration.MAX_THRESHOLD_CANDIDATES
    assert all(
        rows <= calibration.MAX_THRESHOLD_CANDIDATES
        and columns <= calibration.MAX_THRESHOLD_CANDIDATES
        for rows, columns in intersection_shapes
    )
    assert selected.unknown_accept_rate == 0.0
