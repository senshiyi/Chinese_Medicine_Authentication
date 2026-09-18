"""仅针对实验闭集真伪二分类的温度校准。"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import torch
from torch.nn import functional

from .calibration import fit_positive_temperature
from .policy import ScoredLabel

CLOSED_SET_ALGORITHM_VERSION = "closed-set-temperature-v1"
CLOSED_SET_SCOPE_WARNING = "仅适用于本数据集定义的骨碎补正品/伪品范围"
AUTHENTICITY_LABELS = ("authentic", "counterfeit")

CALIBRATION_FIELDS = {
    "temperature",
    "algorithm_version",
    "validation_fingerprint",
    "checkpoint_fingerprint",
    "nll_before",
    "nll_after",
    "brier_before",
    "brier_after",
    "ece_before",
    "ece_after",
}


@dataclass(frozen=True, slots=True)
class ClosedSetCalibration:
    temperature: float
    algorithm_version: str
    validation_fingerprint: str
    checkpoint_fingerprint: str
    nll_before: float
    nll_after: float
    brier_before: float
    brier_after: float
    ece_before: float
    ece_after: float

    def __post_init__(self) -> None:
        _positive_finite(self.temperature, "temperature")
        if not 1e-3 <= self.temperature <= 1e3:
            raise ValueError("temperature 必须在 [1e-3, 1e3] 范围内")
        if self.algorithm_version != CLOSED_SET_ALGORITHM_VERSION:
            raise ValueError("algorithm_version 只支持 closed-set-temperature-v1")
        _sha256(self.validation_fingerprint, "validation_fingerprint")
        _sha256(self.checkpoint_fingerprint, "checkpoint_fingerprint")
        for name in (
            "nll_before",
            "nll_after",
            "brier_before",
            "brier_after",
            "ece_before",
            "ece_after",
        ):
            _nonnegative_finite(getattr(self, name), name)
        for name in ("ece_before", "ece_after"):
            if getattr(self, name) > 1:
                raise ValueError(f"{name} 必须在 [0, 1] 范围内")
        for name in ("brier_before", "brier_after"):
            if getattr(self, name) > 2:
                raise ValueError(f"{name} 必须在 [0, 2] 范围内")
        if self.nll_after > self.nll_before + 1e-9:
            raise ValueError("校准后 NLL 不得高于校准前 NLL")

    def to_dict(self) -> dict[str, float | str]:
        return {name: getattr(self, name) for name in CALIBRATION_FIELDS}

    @classmethod
    def from_dict(
        cls,
        value: Any,
        *,
        checkpoint_fingerprint: str | None = None,
    ) -> ClosedSetCalibration:
        if not isinstance(value, dict) or set(value) != CALIBRATION_FIELDS:
            raise ValueError("closed_set_calibration 字段不完整或包含未知字段")
        calibration = cls(**value)
        if checkpoint_fingerprint is not None:
            expected = _sha256(checkpoint_fingerprint, "checkpoint_fingerprint")
            if calibration.checkpoint_fingerprint != expected:
                raise ValueError("closed_set_calibration checkpoint 指纹与源候选不一致")
        return calibration


@dataclass(frozen=True, slots=True)
class ClosedSetPredictionResult:
    """实验闭集推理的版本化、可审计响应。

    ``model_fingerprint`` 是当前加载的校准后 checkpoint 文件 SHA-256；校准块内的
    ``checkpoint_fingerprint`` 仍表示源候选 checkpoint。``actual_species_verified``
    表示数据集级伪品物种标注能力，不表示本次样本已经人工检验。
    """

    model_version: str
    authenticity: ScoredLabel
    probabilities: Mapping[str, float]
    actual_species_verified: bool
    message: str
    model_fingerprint: str
    data_fingerprint: str
    split_fingerprint: str
    experimental_closed_set: bool = True
    open_set_calibrated: bool = False
    scope_warning: str = CLOSED_SET_SCOPE_WARNING
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.probabilities, Mapping):
            raise ValueError("probabilities 必须是映射")
        snapshot = MappingProxyType(dict(self.probabilities))
        object.__setattr__(self, "probabilities", snapshot)
        _validate_prediction_result(self)

    def to_dict(self) -> dict[str, Any]:
        probabilities = _validate_prediction_result(self)
        return {
            "schema_version": self.schema_version,
            "model_version": self.model_version,
            "experimental_closed_set": self.experimental_closed_set,
            "open_set_calibrated": self.open_set_calibrated,
            "scope_warning": self.scope_warning,
            "message": self.message,
            "authenticity": {
                "label": self.authenticity.label,
                "confidence": float(self.authenticity.confidence),
            },
            "probabilities": dict(probabilities),
            "actual_species_verified": self.actual_species_verified,
            "model_fingerprint": self.model_fingerprint,
            "data_fingerprint": self.data_fingerprint,
            "split_fingerprint": self.split_fingerprint,
        }


def format_closed_set_result(result: ClosedSetPredictionResult) -> str:
    """返回已由推理边界集中生成的闭集用户文案。"""

    if not isinstance(result, ClosedSetPredictionResult):
        raise TypeError("result 必须是 ClosedSetPredictionResult")
    return result.message


def fit_closed_set_temperature(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    validation_fingerprint: str,
    checkpoint_fingerprint: str,
    algorithm_version: str,
) -> ClosedSetCalibration:
    """只用 validation 真伪 logits 拟合单个正温度并记录校准指标。"""

    _validate_authenticity_data(logits, targets)
    validation_fingerprint = _sha256(validation_fingerprint, "validation_fingerprint")
    checkpoint_fingerprint = _sha256(checkpoint_fingerprint, "checkpoint_fingerprint")
    if not isinstance(algorithm_version, str) or not algorithm_version.strip():
        raise ValueError("algorithm_version 必须是非空字符串")

    clean_logits = logits.detach().to(dtype=torch.float64, device="cpu")
    clean_targets = targets.detach().to(dtype=torch.long, device="cpu")
    before = _metrics(clean_logits, clean_targets)
    temperature = fit_positive_temperature(clean_logits, clean_targets)
    after = _metrics(clean_logits / temperature, clean_targets)
    if not math.isfinite(after[0]) or after[0] > before[0] + 1e-9:
        raise ValueError("校准后 NLL 非有限或比校准前更差")
    return ClosedSetCalibration(
        temperature=temperature,
        algorithm_version=algorithm_version,
        validation_fingerprint=validation_fingerprint,
        checkpoint_fingerprint=checkpoint_fingerprint,
        nll_before=before[0],
        nll_after=after[0],
        brier_before=before[1],
        brier_after=after[1],
        ece_before=before[2],
        ece_after=after[2],
    )


def _metrics(logits: torch.Tensor, targets: torch.Tensor) -> tuple[float, float, float]:
    probabilities = torch.softmax(logits, dim=1)
    nll = float(functional.cross_entropy(logits, targets).item())
    expected = functional.one_hot(targets, num_classes=2).to(dtype=torch.float64)
    brier = float(((probabilities - expected) ** 2).sum(dim=1).mean().item())
    confidences, predictions = probabilities.max(dim=1)
    correct = predictions.eq(targets).to(dtype=torch.float64)
    ece = torch.zeros((), dtype=torch.float64)
    boundaries = torch.linspace(0.0, 1.0, steps=11, dtype=torch.float64)
    for index in range(10):
        lower, upper = boundaries[index], boundaries[index + 1]
        selected = (
            (confidences >= lower) & (confidences <= upper)
            if index == 9
            else (confidences >= lower) & (confidences < upper)
        )
        if bool(selected.any()):
            weight = selected.to(dtype=torch.float64).mean()
            ece += weight * (correct[selected].mean() - confidences[selected].mean()).abs()
    return nll, brier, float(ece.item())


def _validate_authenticity_data(logits: torch.Tensor, targets: torch.Tensor) -> None:
    if not isinstance(logits, torch.Tensor) or logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError("真伪 logits 必须是 [N, 2]")
    if logits.shape[0] == 0:
        raise ValueError("真伪 logits 必须是非空 [N, 2]")
    if not isinstance(targets, torch.Tensor) or targets.shape != (logits.shape[0],):
        raise ValueError("真伪 targets 必须是 [N]")
    if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(targets).all()):
        raise ValueError("闭集校准输入必须全部是有限数值")
    if not bool(((targets == 0) | (targets == 1)).all()):
        raise ValueError("真伪 targets 只能是 0 或 1")
    if torch.unique(targets).numel() != 2:
        raise ValueError("闭集校准 targets 必须包含两个类别")


def _sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} 必须是 64 位小写 SHA-256")
    return value


def _positive_finite(value: Any, name: str) -> float:
    parsed = _nonnegative_finite(value, name)
    if parsed <= 0:
        raise ValueError(f"{name} 必须是有限正数")
    return parsed


def _nonnegative_finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} 必须是有限非负数")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError(f"{name} 必须是有限非负数")
    return parsed


def _validate_prediction_result(result: ClosedSetPredictionResult) -> dict[str, float]:
    if not isinstance(result.model_version, str) or not result.model_version.strip():
        raise ValueError("model_version 必须是非空字符串")
    if not isinstance(result.authenticity, ScoredLabel):
        raise ValueError("authenticity 必须是 ScoredLabel")
    if result.authenticity.label not in AUTHENTICITY_LABELS:
        raise ValueError("authenticity label 只允许 authentic 或 counterfeit")
    if not isinstance(result.probabilities, Mapping):
        raise ValueError("probabilities 必须是映射")
    if tuple(result.probabilities) != AUTHENTICITY_LABELS:
        raise ValueError("probabilities 必须按 authentic、counterfeit 顺序包含两个类别")
    probabilities = {
        label: _probability(result.probabilities[label], f"probabilities.{label}")
        for label in AUTHENTICITY_LABELS
    }
    if not math.isclose(sum(probabilities.values()), 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("probabilities 概率和必须和为 1")
    confidence = _probability(result.authenticity.confidence, "authenticity.confidence")
    if not math.isclose(
        probabilities[result.authenticity.label], confidence, rel_tol=0.0, abs_tol=1e-6
    ):
        raise ValueError("authenticity 置信度必须与对应概率一致")
    if not isinstance(result.actual_species_verified, bool):
        raise ValueError("actual_species_verified 必须是布尔值")
    if result.experimental_closed_set is not True or result.open_set_calibrated is not False:
        raise ValueError("实验闭集状态标志不合法")
    if not isinstance(result.message, str) or not result.message:
        raise ValueError("message 必须是非空字符串")
    if result.scope_warning != CLOSED_SET_SCOPE_WARNING:
        raise ValueError("scope_warning 必须使用批准的闭集范围文案")
    if type(result.schema_version) is not int or result.schema_version != 1:
        raise ValueError("schema_version 必须是整数 1")
    _sha256(result.model_fingerprint, "model_fingerprint")
    _sha256(result.data_fingerprint, "data_fingerprint")
    _sha256(result.split_fingerprint, "split_fingerprint")
    return probabilities


def _probability(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} 必须是有限概率")
    parsed = float(value)
    if not math.isfinite(parsed) or not 0 <= parsed <= 1:
        raise ValueError(f"{name} 必须是 [0, 1] 内的有限概率")
    return parsed
