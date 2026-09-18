from __future__ import annotations

import pytest
from herb_authentication.inference.calibration import CalibrationBundle, SelectedThresholds
from herb_authentication.inference.policy import (
    DISCLAIMER,
    PredictionScores,
    ScoredLabel,
    apply_rejection_policy,
)


@pytest.fixture
def calibrated_bundle() -> CalibrationBundle:
    return CalibrationBundle(
        is_calibrated=True,
        method="temperature_scaling",
        temperatures={"claimed_herb": 1.0, "actual_species": 1.0, "authenticity": 1.0},
        thresholds=SelectedThresholds(
            claimed_herb=0.8,
            actual_species=0.7,
            authenticity=0.75,
            known_coverage=0.8,
            known_selective_accuracy=0.96,
            unknown_accept_rate=0.04,
        ),
    )


def scores(
    *,
    claimed_confidence: float = 0.9,
    actual_species: tuple[ScoredLabel, ...] = (ScoredLabel("当归", 0.9),),
    authenticity_label: str = "authentic",
    authenticity_confidence: float = 0.923,
) -> PredictionScores:
    return PredictionScores(
        claimed_herb=ScoredLabel("当归", claimed_confidence),
        actual_species=actual_species,
        authenticity=ScoredLabel(authenticity_label, authenticity_confidence),
    )


def test_missing_calibration_fails_closed_without_percentage() -> None:
    decision = apply_rejection_policy(scores(), CalibrationBundle.uncalibrated())

    assert decision.rejected is True
    assert decision.reason_codes == ("model_uncalibrated",)
    assert decision.message == "模型尚未完成真实数据校准，请人工复核。"
    assert "%" not in decision.message
    assert decision.disclaimer == DISCLAIMER


def test_incomplete_calibrated_bundle_also_fails_closed() -> None:
    incomplete = CalibrationBundle(
        is_calibrated=True,
        method="temperature_scaling",
        temperatures={},
        thresholds=SelectedThresholds(0.8, 0.7, 0.75, 0.8, 0.96, 0.04),
    )

    decision = apply_rejection_policy(scores(), incomplete)

    assert decision.rejected is True
    assert decision.reason_codes == ("model_uncalibrated",)
    assert decision.message == "模型尚未完成真实数据校准，请人工复核。"


def test_non_finite_calibration_temperature_fails_closed(
    calibrated_bundle: CalibrationBundle,
) -> None:
    calibrated_bundle.temperatures["authenticity"] = float("nan")

    decision = apply_rejection_policy(scores(), calibrated_bundle)

    assert decision.rejected is True
    assert decision.reason_codes == ("model_uncalibrated",)
    assert decision.message == "模型尚未完成真实数据校准，请人工复核。"


@pytest.mark.parametrize(
    ("candidate", "reason"),
    [
        (scores(claimed_confidence=0.79), "claimed_herb_low_confidence"),
        (scores(authenticity_confidence=0.74), "authenticity_low_confidence"),
    ],
)
def test_any_low_confidence_head_rejects_entire_prediction(
    calibrated_bundle: CalibrationBundle,
    candidate: PredictionScores,
    reason: str,
) -> None:
    decision = apply_rejection_policy(candidate, calibrated_bundle)

    assert decision.rejected is True
    assert reason in decision.reason_codes
    assert decision.message == "无法确定此药品的种类或真伪，请人工复核。"
    assert "正品" not in decision.message
    assert "伪品" not in decision.message
    assert "%" not in decision.message


@pytest.mark.parametrize(
    "actual_species",
    [(), (ScoredLabel("当归", 0.69),)],
)
def test_unknown_actual_species_rejects_without_authenticity_judgement(
    calibrated_bundle: CalibrationBundle,
    actual_species: tuple[ScoredLabel, ...],
) -> None:
    decision = apply_rejection_policy(
        scores(actual_species=actual_species),
        calibrated_bundle,
    )

    assert decision.rejected is True
    assert "actual_species_unknown" in decision.reason_codes
    assert decision.message == "无法判断该药品的实际药材及真伪，请人工复核。"
    assert "正品" not in decision.message
    assert "伪品" not in decision.message
    assert "%" not in decision.message


def test_authentic_result_only_displays_authenticity_and_confidence(
    calibrated_bundle: CalibrationBundle,
) -> None:
    decision = apply_rejection_policy(scores(), calibrated_bundle)

    assert decision.rejected is False
    assert decision.reason_codes == ()
    assert decision.message == "此药品为正品（置信度：92.3%）。"
    assert "声称" not in decision.message
    assert "实际" not in decision.message
    assert decision.disclaimer == "模型结果不能替代专业鉴定。"


def test_counterfeit_result_displays_claimed_and_actual_species(
    calibrated_bundle: CalibrationBundle,
) -> None:
    decision = apply_rejection_policy(
        scores(
            actual_species=(ScoredLabel("欧当归", 0.847),),
            authenticity_label="counterfeit",
            authenticity_confidence=0.881,
        ),
        calibrated_bundle,
    )

    assert decision.rejected is False
    assert decision.message == ("此药品为伪品（置信度：88.1%），声称为‘当归’，实际应为‘欧当归’。")
    assert decision.disclaimer == DISCLAIMER


def test_species_below_threshold_are_not_listed_in_counterfeit_message(
    calibrated_bundle: CalibrationBundle,
) -> None:
    decision = apply_rejection_policy(
        scores(
            actual_species=(ScoredLabel("欧当归", 0.9), ScoredLabel("未知伴随物", 0.2)),
            authenticity_label="counterfeit",
            authenticity_confidence=0.9,
        ),
        calibrated_bundle,
    )

    assert decision.message == ("此药品为伪品（置信度：90.0%），声称为‘当归’，实际应为‘欧当归’。")


def test_invalid_authenticity_label_fails_explicitly(
    calibrated_bundle: CalibrationBundle,
) -> None:
    with pytest.raises(ValueError, match="authentic|counterfeit"):
        apply_rejection_policy(
            scores(authenticity_label="unknown"),
            calibrated_bundle,
        )
