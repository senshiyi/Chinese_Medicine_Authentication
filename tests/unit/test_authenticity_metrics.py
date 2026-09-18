from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from herb_authentication.evaluation import (
    bootstrap_authenticity_metrics,
    build_closed_set_evaluation_report,
    compute_authenticity_metrics,
    reserve_locked_test,
)


def test_authenticity_metrics_report_confusion_per_class_and_probability_quality() -> None:
    probabilities = torch.tensor([[0.9, 0.1], [0.4, 0.6], [0.2, 0.8], [0.7, 0.3]])
    targets = torch.tensor([0, 0, 1, 1])

    result = compute_authenticity_metrics(probabilities, targets, bins=2)

    assert result.confusion_matrix == ((1, 1), (1, 1))
    assert result.macro_f1 == pytest.approx(0.5)
    assert result.balanced_accuracy == pytest.approx(0.5)
    assert tuple(result.per_class) == ("authentic", "counterfeit")
    assert result.per_class["counterfeit"].recall == pytest.approx(0.5)
    assert result.brier == pytest.approx(0.45)
    assert result.ece == pytest.approx(0.25)
    assert result.nll == pytest.approx(0.612192, rel=1e-5)
    assert 0.0 <= result.auroc <= 1.0
    assert 0.0 <= result.pr_auc <= 1.0


def test_authenticity_bootstrap_is_deterministic_and_covers_every_scalar() -> None:
    probabilities = torch.tensor(
        [
            [0.9, 0.1],
            [0.8, 0.2],
            [0.4, 0.6],
            [0.3, 0.7],
            [0.2, 0.8],
            [0.7, 0.3],
        ]
    )
    targets = torch.tensor([0, 0, 0, 1, 1, 1])

    first = bootstrap_authenticity_metrics(probabilities, targets, samples=40, seed=9)
    second = bootstrap_authenticity_metrics(probabilities, targets, samples=40, seed=9)

    expected = {
        "macro_f1",
        "balanced_accuracy",
        "authentic_recall",
        "authentic_f1",
        "counterfeit_recall",
        "counterfeit_f1",
        "auroc",
        "pr_auc",
        "brier",
        "ece",
        "nll",
    }
    assert first == second
    assert set(first) == expected
    for interval in first.values():
        assert interval.samples == 40
        assert interval.seed == 9
        assert interval.lower <= interval.median <= interval.upper


@pytest.mark.parametrize(
    ("probabilities", "targets", "match"),
    [
        (torch.tensor([[0.9, 0.1], [0.8, 0.2]]), torch.tensor([0, 0]), "两个类别"),
        (
            torch.tensor([[0.9, 0.1], [0.2, 0.8]]),
            torch.tensor([0.5, 1.0]),
            "0 或 1",
        ),
        (
            torch.tensor([[float("nan"), 0.0], [0.2, 0.8]]),
            torch.tensor([0, 1]),
            "有限",
        ),
        (
            torch.tensor([[0.9, 0.1], [0.2, float("inf")]]),
            torch.tensor([0, 1]),
            "有限",
        ),
    ],
)
def test_authenticity_metrics_reject_invalid_inputs(
    probabilities: torch.Tensor, targets: torch.Tensor, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        compute_authenticity_metrics(probabilities, targets)


def test_locked_test_reservation_is_one_shot_even_for_identical_bundle(
    tmp_path: Path,
) -> None:
    bundle = {
        "manifest": "a" * 64,
        "split": "b" * 64,
        "config": "c" * 64,
        "source_candidate": "d" * 64,
        "calibration": "e" * 64,
    }

    reservation = reserve_locked_test(tmp_path, bundle)
    before = reservation.read_bytes()

    assert reservation == tmp_path.resolve() / "locked_test" / "reservation.json"
    assert json.loads(before)["run_id"] == tmp_path.name
    with pytest.raises(ValueError, match="该 run 已执行 locked test"):
        reserve_locked_test(tmp_path, bundle)
    assert reservation.read_bytes() == before


def test_only_validation_report_contains_promotion_gate() -> None:
    probabilities = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
    targets = torch.tensor([0, 1])
    metrics = compute_authenticity_metrics(probabilities, targets)
    intervals = bootstrap_authenticity_metrics(probabilities, targets, samples=5, seed=3)
    fingerprints = {
        "manifest": "a" * 64,
        "split": "b" * 64,
        "config": "c" * 64,
        "source_candidate": "d" * 64,
        "calibration": "e" * 64,
    }
    thresholds = {
        "macro_f1": 0.8,
        "balanced_accuracy": 0.8,
        "counterfeit_recall": 0.8,
    }

    validation = build_closed_set_evaluation_report(
        run_id="run-123",
        partition="validation",
        uncalibrated=metrics,
        calibrated=metrics,
        bootstrap=intervals,
        fingerprints=fingerprints,
        thresholds=thresholds,
    )
    locked = build_closed_set_evaluation_report(
        run_id="run-123",
        partition="locked-test",
        uncalibrated=metrics,
        calibrated=metrics,
        bootstrap=intervals,
        fingerprints=fingerprints,
        thresholds=thresholds,
    )

    assert validation["schema_version"] == 2
    assert validation["run_id"] == "run-123"
    assert validation["evaluation_scope"] == "closed_set_authenticity"
    assert validation["experimental_closed_set"] is True
    assert validation["open_set_calibrated"] is False
    assert validation["metrics"]["class_order"] == ["authentic", "counterfeit"]
    assert validation["promotion_gate"]["passed"] is True
    assert "promotion_gate" not in locked
    assert locked["thresholds_met"]["passed"] is True
    assert locked["run_id"] == "run-123"


@pytest.mark.parametrize("class_names", [("counterfeit", "authentic"), ("genuine", "fake")])
def test_public_metrics_reject_noncanonical_class_names(class_names: tuple[str, str]) -> None:
    probabilities = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
    targets = torch.tensor([0, 1])
    with pytest.raises(ValueError, match="authentic.*counterfeit"):
        compute_authenticity_metrics(probabilities, targets, class_names=class_names)
    with pytest.raises(ValueError, match="authentic.*counterfeit"):
        bootstrap_authenticity_metrics(
            probabilities, targets, class_names=class_names, samples=2, seed=1
        )
