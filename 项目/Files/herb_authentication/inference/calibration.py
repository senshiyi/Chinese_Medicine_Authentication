"""概率校准与只使用校准集的确定性阈值选择。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch.nn import functional as functional

HEAD_NAMES = ("claimed_herb", "actual_species", "authenticity")
MAX_THRESHOLD_CANDIDATES = 64


@dataclass(frozen=True, slots=True)
class KnownValidationData:
    claimed_logits: torch.Tensor
    claimed_targets: torch.Tensor
    species_logits: torch.Tensor
    species_targets: torch.Tensor
    species_mask: torch.Tensor
    authenticity_logits: torch.Tensor
    authenticity_targets: torch.Tensor


@dataclass(frozen=True, slots=True)
class UnknownCalibrationData:
    claimed_logits: torch.Tensor
    species_logits: torch.Tensor
    authenticity_logits: torch.Tensor


@dataclass(frozen=True, slots=True)
class ThresholdSearchData:
    known_claimed_confidences: torch.Tensor
    known_species_probabilities: torch.Tensor
    known_authenticity_confidences: torch.Tensor
    known_claimed_correct: torch.Tensor
    known_species_targets: torch.Tensor
    known_authenticity_correct: torch.Tensor
    unknown_claimed_confidences: torch.Tensor
    unknown_species_probabilities: torch.Tensor
    unknown_authenticity_confidences: torch.Tensor


@dataclass(frozen=True, slots=True)
class SelectedThresholds:
    claimed_herb: float
    actual_species: float
    authenticity: float
    known_coverage: float
    known_selective_accuracy: float
    unknown_accept_rate: float

    def to_dict(self) -> dict[str, float]:
        return {
            "claimed_herb": self.claimed_herb,
            "actual_species": self.actual_species,
            "authenticity": self.authenticity,
            "known_coverage": self.known_coverage,
            "known_selective_accuracy": self.known_selective_accuracy,
            "unknown_accept_rate": self.unknown_accept_rate,
        }

    @classmethod
    def from_dict(cls, value: Any) -> SelectedThresholds:
        if not isinstance(value, dict):
            raise ValueError("calibration.thresholds 必须是映射")
        expected = {
            "claimed_herb",
            "actual_species",
            "authenticity",
            "known_coverage",
            "known_selective_accuracy",
            "unknown_accept_rate",
        }
        if set(value) != expected:
            raise ValueError("calibration.thresholds 字段不完整")
        parsed = {name: _probability(value[name], f"thresholds.{name}") for name in expected}
        return cls(**parsed)


@dataclass(frozen=True, slots=True)
class CalibrationBundle:
    is_calibrated: bool
    method: str | None
    temperatures: dict[str, float]
    thresholds: SelectedThresholds | None

    @classmethod
    def uncalibrated(cls) -> CalibrationBundle:
        return cls(False, None, {}, None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_calibrated": self.is_calibrated,
            "method": self.method,
            "parameters": {"temperatures": dict(self.temperatures)},
            "thresholds": None if self.thresholds is None else self.thresholds.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Any) -> CalibrationBundle:
        if not isinstance(value, dict):
            raise ValueError("calibration 必须是映射")
        is_calibrated = value.get("is_calibrated")
        if not isinstance(is_calibrated, bool):
            raise ValueError("calibration.is_calibrated 必须是布尔值")
        method = value.get("method")
        if method is not None and not isinstance(method, str):
            raise ValueError("calibration.method 必须是字符串或 null")
        parameters = value.get("parameters", {})
        if not isinstance(parameters, dict):
            raise ValueError("calibration.parameters 必须是映射")
        raw_temperatures = parameters.get("temperatures", {})
        if not isinstance(raw_temperatures, dict):
            raise ValueError("calibration temperatures 必须是映射")
        temperatures = {
            str(name): _positive_finite(item, f"temperature.{name}")
            for name, item in raw_temperatures.items()
        }
        thresholds = (
            None
            if value.get("thresholds") is None
            else SelectedThresholds.from_dict(value["thresholds"])
        )
        bundle = cls(is_calibrated, method, temperatures, thresholds)
        if is_calibrated and (
            method != "temperature_scaling"
            or thresholds is None
            or set(temperatures) != set(HEAD_NAMES)
        ):
            raise ValueError("已校准 bundle 缺少方法、三头温度或阈值")
        return bundle


def species_exact_match(predicted: set[str], expected: set[str]) -> bool:
    """物种多标签必须整集合一致，部分命中不算正确。"""

    return predicted == expected


def select_thresholds(
    data: ThresholdSearchData,
    *,
    max_unknown_accept_rate: float = 0.05,
    min_known_selective_accuracy: float = 0.95,
) -> SelectedThresholds | None:
    """搜索校准置信度和边界值，并按规范中的固定排序键选优。"""

    max_unknown_accept_rate = _probability(max_unknown_accept_rate, "max_unknown_accept_rate")
    min_known_selective_accuracy = _probability(
        min_known_selective_accuracy, "min_known_selective_accuracy"
    )
    _validate_threshold_data(data)
    claimed_candidates = _threshold_candidates(
        data.known_claimed_confidences, data.unknown_claimed_confidences
    )
    species_candidates = _threshold_candidates(
        data.known_species_probabilities, data.unknown_species_probabilities
    )
    auth_candidates = _threshold_candidates(
        data.known_authenticity_confidences, data.unknown_authenticity_confidences
    )
    known_claimed_confidences = _cpu_float64(data.known_claimed_confidences)
    known_authenticity_confidences = _cpu_float64(data.known_authenticity_confidences)
    unknown_claimed_confidences = _cpu_float64(data.unknown_claimed_confidences)
    unknown_authenticity_confidences = _cpu_float64(data.unknown_authenticity_confidences)
    known_species_probabilities = _cpu_float64(data.known_species_probabilities)
    unknown_species_probabilities = _cpu_float64(data.unknown_species_probabilities)
    known_claimed_pass = _threshold_pass_matrix(known_claimed_confidences, claimed_candidates)
    known_authenticity_pass = _threshold_pass_matrix(
        known_authenticity_confidences, auth_candidates
    )
    unknown_claimed_pass = _threshold_pass_matrix(unknown_claimed_confidences, claimed_candidates)
    unknown_authenticity_pass = _threshold_pass_matrix(
        unknown_authenticity_confidences, auth_candidates
    )
    known_claimed_correct = data.known_claimed_correct.detach().to(device="cpu")
    known_species_targets = data.known_species_targets.detach().to(device="cpu")
    known_authenticity_correct = data.known_authenticity_correct.detach().to(device="cpu")
    known_count = data.known_claimed_confidences.numel()
    unknown_count = data.unknown_claimed_confidences.numel()
    best: SelectedThresholds | None = None
    best_key: tuple[float, ...] | None = None

    for species_threshold in species_candidates:
        species_selected = known_species_probabilities >= species_threshold
        species_available = species_selected.any(dim=1)
        species_correct = (species_selected == known_species_targets).all(dim=1)
        all_correct = known_claimed_correct & species_correct & known_authenticity_correct
        accepted_counts = _pair_intersection_counts(
            known_claimed_pass & species_available.unsqueeze(0),
            known_authenticity_pass,
        )
        correct_counts = _pair_intersection_counts(
            known_claimed_pass & species_available.unsqueeze(0) & all_correct.unsqueeze(0),
            known_authenticity_pass,
        )
        unknown_species_available = (unknown_species_probabilities >= species_threshold).any(dim=1)
        unknown_counts = _pair_intersection_counts(
            unknown_claimed_pass & unknown_species_available.unsqueeze(0),
            unknown_authenticity_pass,
        )
        valid = accepted_counts > 0
        valid &= (
            correct_counts.to(torch.float64) / accepted_counts.clamp_min(1)
            >= min_known_selective_accuracy
        )
        valid &= unknown_counts.to(torch.float64) / unknown_count <= max_unknown_accept_rate
        if not bool(valid.any()):
            continue

        accepted_count = int(accepted_counts[valid].max().item())
        valid &= accepted_counts == accepted_count
        selectively_correct = int(correct_counts[valid].max().item())
        valid &= correct_counts == selectively_correct
        unknown_accepted_count = int(unknown_counts[valid].min().item())
        valid &= unknown_counts == unknown_accepted_count
        coordinates = valid.nonzero(as_tuple=False)
        claimed_index = int(coordinates[:, 0].max().item())
        coordinates = coordinates[coordinates[:, 0] == claimed_index]
        auth_index = int(coordinates[:, 1].max().item())
        claimed_threshold = claimed_candidates[claimed_index]
        auth_threshold = auth_candidates[auth_index]
        coverage = accepted_count / known_count
        selective_accuracy = selectively_correct / accepted_count
        unknown_accept_rate = unknown_accepted_count / unknown_count
        candidate = SelectedThresholds(
            claimed_threshold,
            species_threshold,
            auth_threshold,
            coverage,
            selective_accuracy,
            unknown_accept_rate,
        )
        key = (
            -coverage,
            -selective_accuracy,
            unknown_accept_rate,
            -claimed_threshold,
            -species_threshold,
            -auth_threshold,
        )
        if best_key is None or key < best_key:
            best = candidate
            best_key = key
    return best


def _cpu_float64(value: torch.Tensor) -> torch.Tensor:
    return value.detach().to(dtype=torch.float64, device="cpu")


def _threshold_pass_matrix(
    values: torch.Tensor,
    candidates: tuple[float, ...],
) -> torch.Tensor:
    thresholds = torch.tensor(candidates, dtype=torch.float64).unsqueeze(1)
    return values.reshape(1, -1) >= thresholds


def _pair_intersection_counts(
    left_conditions: torch.Tensor,
    right_conditions: torch.Tensor,
) -> torch.Tensor:
    """一次计算两条阈值轴的交集计数，不构造三维候选对象。"""

    return left_conditions.to(torch.int64) @ right_conditions.to(torch.int64).T


def fit_calibrators(
    known_validation: KnownValidationData,
    unknown_calibration: UnknownCalibrationData,
    *,
    max_unknown_accept_rate: float = 0.05,
    min_known_selective_accuracy: float = 0.95,
) -> CalibrationBundle:
    """用已知验证集拟合三头温度，仅用 unknown calibration 选择拒识阈值。"""

    _validate_calibration_data(known_validation, unknown_calibration)
    verified = known_validation.species_mask
    temperatures = {
        "claimed_herb": fit_positive_temperature(
            known_validation.claimed_logits, known_validation.claimed_targets
        ),
        "actual_species": fit_positive_temperature(
            known_validation.species_logits[verified],
            known_validation.species_targets[verified],
            loss_kind="binary_cross_entropy",
        ),
        "authenticity": fit_positive_temperature(
            known_validation.authenticity_logits, known_validation.authenticity_targets
        ),
    }
    claimed_probabilities = torch.softmax(
        known_validation.claimed_logits / temperatures["claimed_herb"], dim=1
    )
    species_probabilities = torch.sigmoid(
        known_validation.species_logits[verified] / temperatures["actual_species"]
    )
    authenticity_probabilities = torch.softmax(
        known_validation.authenticity_logits / temperatures["authenticity"], dim=1
    )
    unknown_claimed = torch.softmax(
        unknown_calibration.claimed_logits / temperatures["claimed_herb"], dim=1
    )
    unknown_species = torch.sigmoid(
        unknown_calibration.species_logits / temperatures["actual_species"]
    )
    unknown_authenticity = torch.softmax(
        unknown_calibration.authenticity_logits / temperatures["authenticity"], dim=1
    )
    thresholds = select_thresholds(
        ThresholdSearchData(
            known_claimed_confidences=claimed_probabilities.max(dim=1).values[verified],
            known_species_probabilities=species_probabilities,
            known_authenticity_confidences=authenticity_probabilities.max(dim=1).values[verified],
            known_claimed_correct=(
                claimed_probabilities.argmax(dim=1) == known_validation.claimed_targets
            )[verified],
            known_species_targets=known_validation.species_targets[verified].bool(),
            known_authenticity_correct=(
                authenticity_probabilities.argmax(dim=1) == known_validation.authenticity_targets
            )[verified],
            unknown_claimed_confidences=unknown_claimed.max(dim=1).values,
            unknown_species_probabilities=unknown_species,
            unknown_authenticity_confidences=unknown_authenticity.max(dim=1).values,
        ),
        max_unknown_accept_rate=max_unknown_accept_rate,
        min_known_selective_accuracy=min_known_selective_accuracy,
    )
    if thresholds is None:
        return CalibrationBundle.uncalibrated()
    return CalibrationBundle(True, "temperature_scaling", temperatures, thresholds)


def fit_positive_temperature(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_kind: str = "cross_entropy",
) -> float:
    """使用 LBFGS 拟合限制在 [1e-3, 1e3] 的单个正温度。"""

    logits = logits.detach().to(dtype=torch.float64, device="cpu")
    targets = targets.detach().to(device="cpu")
    if loss_kind not in {"cross_entropy", "binary_cross_entropy"}:
        raise ValueError("loss_kind 必须是 cross_entropy 或 binary_cross_entropy")
    if logits.numel() == 0 or not bool(torch.isfinite(logits).all()):
        raise ValueError("温度拟合 logits 必须是非空有限 Tensor")
    if not bool(torch.isfinite(targets).all()):
        raise ValueError("温度拟合 targets 必须是有限 Tensor")

    log_temperature = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [log_temperature],
        lr=0.1,
        max_iter=100,
        tolerance_grad=1e-12,
        tolerance_change=1e-15,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(1e-3, 1e3)
        scaled = logits / temperature
        loss = (
            functional.binary_cross_entropy_with_logits(scaled, targets.to(torch.float64))
            if loss_kind == "binary_cross_entropy"
            else functional.cross_entropy(scaled, targets.to(torch.long))
        )
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(log_temperature.detach().exp().clamp(1e-3, 1e3).item())
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("温度拟合未产生有限正温度")
    return temperature


def _threshold_candidates(*values: torch.Tensor) -> tuple[float, ...]:
    flattened = torch.cat(
        [value.detach().to(dtype=torch.float64, device="cpu").reshape(-1) for value in values]
    )
    candidates = tuple(sorted({0.0, 1.0, *(float(item) for item in flattened.tolist())}))
    if len(candidates) <= MAX_THRESHOLD_CANDIDATES:
        return candidates
    indexes = (
        torch.linspace(
            0,
            len(candidates) - 1,
            steps=MAX_THRESHOLD_CANDIDATES,
            dtype=torch.float64,
        )
        .round()
        .to(torch.long)
    )
    return tuple(candidates[int(index)] for index in indexes)


def _validate_calibration_data(known: KnownValidationData, unknown: UnknownCalibrationData) -> None:
    numeric_tensors = (
        known.claimed_logits,
        known.species_logits,
        known.species_targets,
        known.authenticity_logits,
        unknown.claimed_logits,
        unknown.species_logits,
        unknown.authenticity_logits,
    )
    if any(not torch.isfinite(tensor).all().item() for tensor in numeric_tensors):
        raise ValueError("校准输入必须全部是有限数值")
    size = known.claimed_logits.shape[0]
    if size == 0 or known.claimed_logits.ndim != 2 or known.claimed_logits.shape[1] < 2:
        raise ValueError("已知声称药材 logits 形状无效")
    if known.authenticity_logits.shape != (size, 2):
        raise ValueError("已知真伪 logits 必须是 [N, 2]")
    if known.species_logits.ndim != 2 or known.species_logits.shape[0] != size:
        raise ValueError("已知实际物种 logits 形状无效")
    if known.species_targets.shape != known.species_logits.shape:
        raise ValueError("实际物种标签形状必须匹配 logits")
    if known.claimed_targets.shape != (size,) or known.authenticity_targets.shape != (size,):
        raise ValueError("多分类标签形状必须是 [N]")
    if known.species_mask.shape != (size,) or known.species_mask.dtype != torch.bool:
        raise ValueError("species_mask 必须是 [N] bool Tensor")
    if not bool(known.species_mask.any()):
        raise ValueError("校准至少需要一个已核验实际物种样本")
    unknown_size = unknown.claimed_logits.shape[0]
    expected_unknown = (
        (unknown.claimed_logits, known.claimed_logits.shape[1]),
        (unknown.species_logits, known.species_logits.shape[1]),
        (unknown.authenticity_logits, 2),
    )
    if unknown_size == 0 or any(
        tensor.shape != (unknown_size, width) for tensor, width in expected_unknown
    ):
        raise ValueError("unknown calibration logits 不能为空且类别数必须匹配")
    if (
        int(known.claimed_targets.min()) < 0
        or int(known.claimed_targets.max()) >= known.claimed_logits.shape[1]
    ):
        raise ValueError("声称药材标签超出类别范围")
    if int(known.authenticity_targets.min()) < 0 or int(known.authenticity_targets.max()) >= 2:
        raise ValueError("真伪标签超出类别范围")
    verified_targets = known.species_targets[known.species_mask]
    if not bool(((verified_targets == 0) | (verified_targets == 1)).all()):
        raise ValueError("已核验实际物种标签必须是 0/1")


def _validate_threshold_data(data: ThresholdSearchData) -> None:
    known_size = data.known_claimed_confidences.numel()
    unknown_size = data.unknown_claimed_confidences.numel()
    if known_size == 0 or unknown_size == 0:
        raise ValueError("阈值搜索需要非空 known 与 unknown 数据")
    if (
        data.known_claimed_confidences.shape != (known_size,)
        or data.known_authenticity_confidences.shape != (known_size,)
        or data.known_claimed_correct.shape != (known_size,)
        or data.known_authenticity_correct.shape != (known_size,)
    ):
        raise ValueError("known 一维字段必须等长")
    if data.unknown_claimed_confidences.shape != (
        unknown_size,
    ) or data.unknown_authenticity_confidences.shape != (unknown_size,):
        raise ValueError("unknown 一维字段必须等长")
    if (
        data.known_species_probabilities.ndim != 2
        or data.known_species_probabilities.shape[0] != known_size
    ):
        raise ValueError("known species probabilities 形状无效")
    if data.unknown_species_probabilities.shape != (
        unknown_size,
        data.known_species_probabilities.shape[1],
    ):
        raise ValueError("unknown species probabilities 形状无效")
    if data.known_species_targets.shape != data.known_species_probabilities.shape:
        raise ValueError("known species targets 形状无效")
    probabilities = (
        data.known_claimed_confidences,
        data.known_species_probabilities,
        data.known_authenticity_confidences,
        data.unknown_claimed_confidences,
        data.unknown_species_probabilities,
        data.unknown_authenticity_confidences,
    )
    if any(
        not torch.isfinite(tensor).all().item() or not bool(((tensor >= 0) & (tensor <= 1)).all())
        for tensor in probabilities
    ):
        raise ValueError("阈值搜索概率必须是 [0, 1] 内的有限值")
    if any(
        tensor.dtype != torch.bool
        for tensor in (
            data.known_claimed_correct,
            data.known_species_targets,
            data.known_authenticity_correct,
        )
    ):
        raise ValueError("阈值搜索正确性与物种真值必须是 bool Tensor")


def _probability(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise ValueError(f"{name} 必须是 [0, 1] 内的有限概率")
    return float(value)


def _positive_finite(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} 必须是有限正数")
    return float(value)
