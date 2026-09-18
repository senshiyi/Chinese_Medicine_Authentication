from __future__ import annotations

import json

from herb_authentication.inference.formatter import ResultFormatter
from herb_authentication.inference.policy import DISCLAIMER, ScoredLabel
from herb_authentication.inference.results import PredictionResult


def result(
    *,
    calibrated: bool = True,
    rejected: bool = False,
    reasons: tuple[str, ...] = (),
    authenticity: str = "authentic",
    confidence: float = 0.9234,
    claimed: str = "当归",
    species: tuple[ScoredLabel, ...] = (ScoredLabel("当归", 0.847),),
) -> PredictionResult:
    return PredictionResult(
        model_version="multitask-v1",
        is_calibrated=calibrated,
        claimed_herb=ScoredLabel(claimed, 0.923),
        actual_species=species,
        authenticity=ScoredLabel(authenticity, confidence),
        rejected=rejected,
        rejection_reasons=reasons,
        message="",
    )


def test_authentic_message_contains_only_verdict_and_confidence() -> None:
    message = ResultFormatter().format(result())

    assert message == "此药品为正品（置信度：92.3%）。"
    assert "声称" not in message and "实际" not in message


def test_counterfeit_message_names_claimed_and_actual() -> None:
    message = ResultFormatter().format(
        result(
            authenticity="counterfeit",
            confidence=0.881,
            species=(ScoredLabel("欧当归", 0.847),),
        )
    )

    assert message == "此药品为伪品（置信度：88.1%），声称为‘当归’，实际应为‘欧当归’。"


def test_unknown_actual_species_never_leaks_verdict_or_percentage() -> None:
    message = ResultFormatter().format(
        result(rejected=True, reasons=("actual_species_unknown",), species=())
    )

    assert message == "无法判断该药品的实际药材及真伪，请人工复核。"
    assert "正品" not in message and "伪品" not in message and "%" not in message


def test_uncalibrated_and_generic_rejection_have_stable_messages() -> None:
    formatter = ResultFormatter()

    assert formatter.format(result(calibrated=False, rejected=True)) == (
        "模型尚未完成真实数据校准，请人工复核。"
    )
    assert (
        formatter.format(result(rejected=True, reasons=("authenticity_low_confidence",)))
        == "无法确定此药品的种类或真伪，请人工复核。"
    )


def test_prediction_result_schema_v1_is_json_serializable() -> None:
    prediction = result(
        authenticity="counterfeit",
        confidence=0.881,
        species=(ScoredLabel("欧当归", 0.847),),
    )
    prediction = prediction.with_message(ResultFormatter().format(prediction))

    payload = prediction.to_dict()

    assert payload == {
        "schema_version": 1,
        "model_version": "multitask-v1",
        "is_calibrated": True,
        "message": "此药品为伪品（置信度：88.1%），声称为‘当归’，实际应为‘欧当归’。",
        "disclaimer": DISCLAIMER,
        "claimed_herb": {"label": "当归", "confidence": 0.923},
        "actual_species": [{"label": "欧当归", "confidence": 0.847}],
        "authenticity": {"label": "counterfeit", "confidence": 0.881},
        "rejected": False,
        "rejection_reasons": [],
    }
    assert "当归" in json.dumps(payload, ensure_ascii=False)


def test_disclaimer_is_a_separate_fixed_field() -> None:
    prediction = result().with_message("此药品为正品（置信度：92.3%）。")

    assert prediction.disclaimer == "模型结果不能替代专业鉴定。"
    assert prediction.disclaimer not in prediction.message
