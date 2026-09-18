"""面向用户的 fail-closed 拒识决策和中文结论。"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .calibration import CalibrationBundle

DISCLAIMER = "模型结果不能替代专业鉴定。"


@dataclass(frozen=True, slots=True)
class ScoredLabel:
    label: str
    confidence: float


@dataclass(frozen=True, slots=True)
class PredictionScores:
    claimed_herb: ScoredLabel
    actual_species: tuple[ScoredLabel, ...]
    authenticity: ScoredLabel


@dataclass(frozen=True, slots=True)
class RejectionDecision:
    rejected: bool
    reason_codes: tuple[str, ...]
    message: str
    disclaimer: str = DISCLAIMER


def apply_rejection_policy(
    scores: PredictionScores,
    bundle: CalibrationBundle,
) -> RejectionDecision:
    """只有三头均达到已校准阈值时才给出真伪结论。"""

    _validate_scores(scores)
    complete_bundle = (
        bundle.is_calibrated
        and bundle.method == "temperature_scaling"
        and all(math.isfinite(value) and value > 0 for value in bundle.temperatures.values())
        and bundle.thresholds is not None
        and set(bundle.temperatures) == {"claimed_herb", "actual_species", "authenticity"}
    )
    if not complete_bundle:
        return RejectionDecision(
            rejected=True,
            reason_codes=("model_uncalibrated",),
            message="模型尚未完成真实数据校准，请人工复核。",
        )

    thresholds = bundle.thresholds
    selected_species = tuple(
        sorted(
            (
                item
                for item in scores.actual_species
                if item.confidence >= thresholds.actual_species
            ),
            key=lambda item: (-item.confidence, item.label),
        )
    )
    reasons: list[str] = []
    if scores.claimed_herb.confidence < thresholds.claimed_herb:
        reasons.append("claimed_herb_low_confidence")
    if not selected_species:
        reasons.append("actual_species_unknown")
    if scores.authenticity.confidence < thresholds.authenticity:
        reasons.append("authenticity_low_confidence")
    if reasons:
        return RejectionDecision(
            rejected=True,
            reason_codes=tuple(reasons),
            message=(
                "无法判断该药品的实际药材及真伪，请人工复核。"
                if "actual_species_unknown" in reasons
                else "无法确定此药品的种类或真伪，请人工复核。"
            ),
        )

    confidence = scores.authenticity.confidence
    if scores.authenticity.label == "authentic":
        message = f"此药品为正品（置信度：{confidence:.1%}）。"
    else:
        species_list = "、".join(item.label for item in selected_species)
        message = (
            f"此药品为伪品（置信度：{confidence:.1%}），"
            f"声称为‘{scores.claimed_herb.label}’，实际应为‘{species_list}’。"
        )
    return RejectionDecision(rejected=False, reason_codes=(), message=message)


def _validate_scores(scores: PredictionScores) -> None:
    if scores.authenticity.label not in {"authentic", "counterfeit"}:
        raise ValueError("authenticity label 只允许 authentic 或 counterfeit")
    values = [
        ("claimed_herb", scores.claimed_herb),
        ("authenticity", scores.authenticity),
        *(("actual_species", item) for item in scores.actual_species),
    ]
    for name, item in values:
        if not isinstance(item.label, str) or not item.label.strip():
            raise ValueError(f"{name} label 必须是非空字符串")
        if (
            isinstance(item.confidence, bool)
            or not isinstance(item.confidence, int | float)
            or not math.isfinite(item.confidence)
            or not 0 <= item.confidence <= 1
        ):
            raise ValueError(f"{name} confidence 必须是 [0, 1] 内的有限概率")
