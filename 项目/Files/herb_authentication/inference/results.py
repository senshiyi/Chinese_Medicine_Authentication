"""未来 CLI 与 Web 共用的版本化推理返回对象。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from .policy import DISCLAIMER, ScoredLabel

SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class PredictionResult:
    """一次图片预测的稳定 schema v1。"""

    model_version: str
    is_calibrated: bool
    claimed_herb: ScoredLabel
    actual_species: tuple[ScoredLabel, ...]
    authenticity: ScoredLabel
    rejected: bool
    rejection_reasons: tuple[str, ...]
    message: str
    disclaimer: str = DISCLAIMER
    schema_version: int = SCHEMA_VERSION

    def with_message(self, message: str) -> PredictionResult:
        return replace(self, message=message)

    def to_dict(self) -> dict[str, Any]:
        """只返回 JSON 原生类型，中文保持为 Unicode 字符串。"""

        return {
            "schema_version": self.schema_version,
            "model_version": self.model_version,
            "is_calibrated": self.is_calibrated,
            "message": self.message,
            "disclaimer": self.disclaimer,
            "claimed_herb": _label_to_dict(self.claimed_herb),
            "actual_species": [_label_to_dict(item) for item in self.actual_species],
            "authenticity": _label_to_dict(self.authenticity),
            "rejected": self.rejected,
            "rejection_reasons": list(self.rejection_reasons),
        }


def _label_to_dict(value: ScoredLabel) -> dict[str, str | float]:
    return {"label": value.label, "confidence": float(value.confidence)}
