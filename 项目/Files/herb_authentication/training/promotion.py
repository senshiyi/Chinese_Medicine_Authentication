"""固定门禁、只读预检和原子候选模型晋升。"""

from __future__ import annotations

import json
import math
import os
import pickle
import re
import shutil
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from itertools import combinations
from pathlib import Path
from typing import Any

import torch

from herb_authentication.errors import HerbAuthenticationError
from herb_authentication.inference.service import InferenceService
from herb_authentication.training.run_artifacts import sha256_file

_HASH = re.compile(r"[0-9a-f]{64}")
_FINGERPRINT_KEYS = frozenset({"manifest", "split", "config", "source_candidate", "calibration"})
_METRIC_KEYS = ("macro_f1", "balanced_accuracy", "counterfeit_recall")


class PromotionError(HerbAuthenticationError):
    """候选模型未满足晋升契约或原子发布失败。"""


@dataclass(frozen=True, slots=True)
class PromotionThresholds:
    macro_f1: float
    balanced_accuracy: float
    counterfeit_recall: float

    def to_dict(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in _METRIC_KEYS}


DEFAULT_PROMOTION_THRESHOLDS = PromotionThresholds(0.80, 0.80, 0.80)


@dataclass(frozen=True, slots=True)
class PromotionRequest:
    candidate: Path
    target: Path
    validation_report: Path
    locked_test_report: Path
    run_dir: Path


@dataclass(frozen=True, slots=True)
class PromotionRecord:
    old_sha256: str | None
    new_sha256: str
    target: Path
    run_id: str
    promoted_at: str

    def to_dict(self) -> dict[str, str | None]:
        return {
            "old_sha256": self.old_sha256,
            "new_sha256": self.new_sha256,
            "target": str(self.target),
            "run_id": self.run_id,
            "promoted_at": self.promoted_at,
        }


@dataclass(frozen=True, slots=True)
class _Preflight:
    request: PromotionRequest
    candidate_sha256: str
    old_sha256: str | None
    run_id: str
    fingerprints: dict[str, str]
    validation_metrics: dict[str, float]
    locked_test_metrics: dict[str, float]
    artifact_sha256: dict[str, str]


def promote_candidate(
    request: PromotionRequest,
    thresholds: PromotionThresholds = DEFAULT_PROMOTION_THRESHOLDS,
    smoke_image: Path | None = None,
    *,
    model_factory: Callable[[Mapping[str, Any]], Any] | None = None,
    preprocess_factory: Callable[[Mapping[str, Any]], Any] | None = None,
) -> PromotionRecord:
    """通过全部只读门禁后，原子替换目标并写入晋升审计记录。"""

    preflight = _preflight(
        request,
        thresholds,
        smoke_image,
        model_factory=model_factory,
        preprocess_factory=preprocess_factory,
    )
    target = preflight.request.target
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copyfile(preflight.request.candidate, temporary)
        temporary_hash = sha256_file(temporary)
        if temporary_hash != preflight.candidate_sha256:
            raise PromotionError("目标临时 checkpoint 哈希与候选不一致")
        _safe_load_service(
            temporary,
            "目标临时 checkpoint",
            model_factory=model_factory,
            preprocess_factory=preprocess_factory,
        )
        _verify_target_unchanged(preflight)
        _verify_artifact_hashes(preflight)
        try:
            os.replace(temporary, target)
        except OSError as exc:
            raise PromotionError(f"模型原子 replace 失败：{exc}") from exc
    finally:
        if temporary.exists():
            temporary.unlink()

    _require_target_hash(target, preflight.candidate_sha256)
    record = PromotionRecord(
        old_sha256=preflight.old_sha256,
        new_sha256=preflight.candidate_sha256,
        target=target,
        run_id=preflight.run_id,
        promoted_at=_utc_now(),
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "run_id": record.run_id,
        "promoted_at": record.promoted_at,
        "candidate": str(preflight.request.candidate),
        "target": str(target),
        "old_sha256": record.old_sha256,
        "new_sha256": record.new_sha256,
        "thresholds": thresholds.to_dict(),
        "validation_metrics": preflight.validation_metrics,
        "locked_test_metrics": preflight.locked_test_metrics,
        "fingerprints": preflight.fingerprints,
        **preflight.artifact_sha256,
    }
    try:
        _require_target_hash(target, preflight.candidate_sha256)
        _verify_artifact_hashes(preflight)
        _write_promotion_record_atomic(preflight.request.run_dir / "promotion.json", payload)
        _require_target_hash(target, preflight.candidate_sha256)
        _verify_artifact_hashes(preflight)
    except (PromotionError, OSError, TypeError, ValueError) as exc:
        raise PromotionError(
            "模型已替换但晋升审计或完整性复核失败；请恢复或补写审计记录："
            f"old_sha256={record.old_sha256}, new_sha256={record.new_sha256}, "
            f"target={target}；原因：{exc}"
        ) from exc
    _require_target_hash(target, preflight.candidate_sha256)
    return record


def _preflight(
    request: PromotionRequest,
    thresholds: PromotionThresholds,
    smoke_image: Path | None,
    *,
    model_factory: Callable[[Mapping[str, Any]], Any] | None = None,
    preprocess_factory: Callable[[Mapping[str, Any]], Any] | None = None,
) -> _Preflight:
    if not isinstance(request, PromotionRequest):
        raise PromotionError("request 必须是 PromotionRequest")
    _require_fixed_thresholds(thresholds)
    resolved = PromotionRequest(
        *(
            Path(getattr(request, field)).expanduser().resolve()
            for field in (
                "candidate",
                "target",
                "validation_report",
                "locked_test_report",
                "run_dir",
            )
        )
    )
    reservation_path = resolved.run_dir / "locked_test" / "reservation.json"
    promotion_path = resolved.run_dir / "promotion.json"
    protected_paths = {
        "candidate": resolved.candidate,
        "target": resolved.target,
        "validation_report": resolved.validation_report,
        "locked_report": resolved.locked_test_report,
        "reservation": reservation_path,
        "run_dir": resolved.run_dir,
        "promotion_json": promotion_path,
    }
    _require_distinct_paths(protected_paths)
    _reject_target_alias(resolved.target, protected_paths)
    expected_validation = resolved.run_dir / "validation" / "evaluation.json"
    expected_locked = resolved.run_dir / "locked_test" / "evaluation.json"
    if resolved.validation_report != expected_validation:
        raise PromotionError("validation report 不在 run_dir 标准位置")
    if resolved.locked_test_report != expected_locked:
        raise PromotionError("locked-test report 不在 run_dir 标准位置")
    if not resolved.candidate.is_file():
        raise PromotionError(f"候选 checkpoint 不存在：{resolved.candidate}")
    smoke = None if smoke_image is None else Path(smoke_image).expanduser().resolve()
    if smoke is None or not smoke.is_file():
        raise PromotionError(f"smoke image 不存在：{smoke}")

    validation, validation_sha = _read_json_object(resolved.validation_report, "validation report")
    locked, locked_sha = _read_json_object(resolved.locked_test_report, "locked-test report")
    reservation, reservation_sha = _read_json_object(
        reservation_path,
        "locked-test reservation",
    )
    run_id = resolved.run_dir.name
    validation_fingerprints, validation_metrics = _validate_report(
        validation, "validation", run_id, thresholds
    )
    locked_fingerprints, locked_metrics = _validate_report(
        locked, "locked-test", run_id, thresholds
    )
    if validation_fingerprints != locked_fingerprints:
        differing = sorted(
            name
            for name in _FINGERPRINT_KEYS
            if validation_fingerprints.get(name) != locked_fingerprints.get(name)
        )
        raise PromotionError(f"validation/locked fingerprints 不一致：{differing}")
    _validate_reservation(reservation, run_id, validation_fingerprints)

    candidate_hash = sha256_file(resolved.candidate)
    if candidate_hash != validation_fingerprints["calibration"]:
        raise PromotionError("候选当前哈希与报告 candidate/calibration 哈希不一致")
    state = _safe_checkpoint_state(resolved.candidate)
    metadata = state.get("metadata")
    if not isinstance(metadata, Mapping):
        raise PromotionError("候选 checkpoint metadata 缺失")
    _validate_candidate_metadata(metadata, run_id, validation_fingerprints)

    service = _safe_load_service(
        resolved.candidate,
        "候选 checkpoint",
        model_factory=model_factory,
        preprocess_factory=preprocess_factory,
    )
    try:
        path_result = service.predict_experimental_closed_set(smoke)
        bytes_result = service.predict_experimental_closed_set(smoke.read_bytes())
        path_json = _json_native_result(path_result)
        bytes_json = _json_native_result(bytes_result)
    except PromotionError:
        raise
    except (HerbAuthenticationError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise PromotionError(f"experimental smoke 失败：{exc}") from exc
    if path_json != bytes_json:
        raise PromotionError("experimental smoke 的 Path 与 bytes JSON 结果不一致")

    old_hash: str | None = None
    if resolved.target.exists():
        if not resolved.target.is_file():
            raise PromotionError(f"target 不是普通文件：{resolved.target}")
        old_hash = sha256_file(resolved.target)
    return _Preflight(
        request=resolved,
        candidate_sha256=candidate_hash,
        old_sha256=old_hash,
        run_id=run_id,
        fingerprints=validation_fingerprints,
        validation_metrics=validation_metrics,
        locked_test_metrics=locked_metrics,
        artifact_sha256={
            "validation_report_sha256": validation_sha,
            "locked_report_sha256": locked_sha,
            "reservation_sha256": reservation_sha,
        },
    )


def _require_fixed_thresholds(thresholds: PromotionThresholds) -> None:
    if not isinstance(thresholds, PromotionThresholds):
        raise PromotionError("thresholds 必须是 PromotionThresholds")
    values = thresholds.to_dict()
    if any(not math.isfinite(value) for value in values.values()) or values != {
        name: 0.80 for name in _METRIC_KEYS
    }:
        raise PromotionError("晋升门槛固定为三个指标均为 0.80")


def _validate_report(
    report: Mapping[str, Any],
    partition: str,
    run_id: str,
    thresholds: PromotionThresholds,
) -> tuple[dict[str, str], dict[str, float]]:
    gate_name = "promotion_gate" if partition == "validation" else "thresholds_met"
    expected_root = {
        "schema_version",
        "run_id",
        "evaluation_scope",
        "partition",
        "experimental_closed_set",
        "open_set_calibrated",
        "limitations",
        "fingerprints",
        "metrics",
        gate_name,
    }
    if set(report) != expected_root:
        raise PromotionError(f"{partition} report schema v2 字段不完整或包含未知字段")
    if report.get("schema_version") != 2:
        raise PromotionError(f"{partition} report schema_version 不是受支持的 v2")
    if report.get("run_id") != run_id:
        raise PromotionError(f"{partition} report run_id 与 run_dir 不一致")
    if report.get("partition") != partition:
        raise PromotionError(f"{partition} report partition 不一致")
    if report.get("evaluation_scope") != "closed_set_authenticity":
        raise PromotionError(f"{partition} report evaluation_scope 不兼容")
    if report.get("experimental_closed_set") is not True:
        raise PromotionError(f"{partition} report experimental_closed_set 必须为 true")
    if report.get("open_set_calibrated") is not False:
        raise PromotionError(f"{partition} report open_set_calibrated 必须为 false")
    limitations = report.get("limitations")
    if (
        not isinstance(limitations, list)
        or not limitations
        or any(not isinstance(item, str) or not item for item in limitations)
    ):
        raise PromotionError(f"{partition} report limitations 类型错误")
    fingerprints = _fingerprints(report.get("fingerprints"), f"{partition} report")
    metrics = _validate_metrics(report.get("metrics"), partition)
    parsed = {
        "macro_f1": metrics["macro_f1"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "counterfeit_recall": metrics["counterfeit_recall"],
    }
    expected_thresholds = thresholds.to_dict()
    gate = report.get(gate_name)
    if not isinstance(gate, Mapping) or set(gate) != {"thresholds", "checks", "passed"}:
        raise PromotionError(f"{partition} report 门禁字段或类型错误")
    _validate_gate_thresholds(gate.get("thresholds"), expected_thresholds, partition)
    computed = {name: parsed[name] >= expected_thresholds[name] for name in _METRIC_KEYS}
    checks = gate.get("checks")
    if (
        not isinstance(checks, Mapping)
        or set(checks) != set(_METRIC_KEYS)
        or any(type(value) is not bool for value in checks.values())
        or dict(checks) != computed
        or type(gate.get("passed")) is not bool
        or gate.get("passed") is not all(computed.values())
    ):
        raise PromotionError(f"{partition} report 门禁声明与指标不一致")
    failed = [name for name, passed in computed.items() if not passed]
    if failed:
        raise PromotionError(f"{partition} 未达到晋升门槛：{', '.join(failed)}")
    return fingerprints, parsed


def _validate_reservation(
    reservation: Mapping[str, Any], run_id: str, fingerprints: Mapping[str, str]
) -> None:
    if reservation.get("schema_version") != 1:
        raise PromotionError("locked-test reservation schema_version 不是受支持的 v1")
    if reservation.get("run_id") != run_id:
        raise PromotionError("locked-test reservation run_id 不一致")
    if _fingerprints(reservation.get("fingerprints"), "locked-test reservation") != fingerprints:
        raise PromotionError("locked-test reservation fingerprints 与报告不一致")


def _validate_candidate_metadata(
    metadata: Mapping[str, Any], run_id: str, fingerprints: Mapping[str, str]
) -> None:
    if metadata.get("run_id") != run_id:
        raise PromotionError("候选 metadata run_id 不一致")
    if metadata.get("experimental_closed_set") is not True:
        raise PromotionError("候选 experimental_closed_set 必须为 true")
    if metadata.get("open_set_calibrated") is not False:
        raise PromotionError("候选 open_set_calibrated 必须为 false")
    recorded = metadata.get("fingerprints")
    if not isinstance(recorded, Mapping):
        raise PromotionError("候选 fingerprints 缺失")
    aliases = {
        "data_fingerprint": metadata.get("data_fingerprint"),
        "split_input_fingerprint": metadata.get("split_input_fingerprint"),
        "fingerprints.manifest": recorded.get("manifest"),
        "split_fingerprint": metadata.get("split_fingerprint"),
        "split_artifact_fingerprint": metadata.get("split_artifact_fingerprint"),
        "fingerprints.locked_split": recorded.get("locked_split"),
        "training_config_fingerprint": metadata.get("training_config_fingerprint"),
        "fingerprints.known_validation": recorded.get("known_validation"),
    }
    for name, value in aliases.items():
        _require_sha256(value, f"候选 {name}")
    expected = {
        "fingerprints.manifest": fingerprints["manifest"],
        "split_fingerprint": fingerprints["split"],
        "split_artifact_fingerprint": fingerprints["split"],
        "fingerprints.locked_split": fingerprints["split"],
        "training_config_fingerprint": fingerprints["config"],
    }
    for name, value in expected.items():
        if aliases[name] != value:
            raise PromotionError(f"候选 {name} 与报告不一致")
    if aliases["data_fingerprint"] != aliases["split_input_fingerprint"]:
        raise PromotionError("候选 data_fingerprint 与 split_input_fingerprint 不一致")
    calibration = metadata.get("closed_set_calibration")
    if not isinstance(calibration, Mapping):
        raise PromotionError("候选 closed_set_calibration 缺失")
    if calibration.get("checkpoint_fingerprint") != fingerprints["source_candidate"]:
        raise PromotionError("calibration source_candidate 指纹链不一致")
    if calibration.get("validation_fingerprint") != aliases["fingerprints.known_validation"]:
        raise PromotionError("calibration validation 指纹链不一致")


def _safe_checkpoint_state(path: Path) -> Mapping[str, Any]:
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except (EOFError, OSError, RuntimeError, TypeError, ValueError, pickle.UnpicklingError) as exc:
        raise PromotionError(f"无法安全加载候选 checkpoint：{path}") from exc
    if not isinstance(state, Mapping) or state.get("schema_version") != 1:
        raise PromotionError("候选 checkpoint schema_version 不是受支持的 v1")
    return state


def _safe_load_service(
    path: Path,
    label: str,
    *,
    model_factory: Callable[[Mapping[str, Any]], Any] | None = None,
    preprocess_factory: Callable[[Mapping[str, Any]], Any] | None = None,
) -> Any:
    try:
        return InferenceService.from_checkpoint(
            path,
            device="cpu",
            model_factory=model_factory,
            preprocess_factory=preprocess_factory,
        )
    except (HerbAuthenticationError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise PromotionError(f"无法安全加载{label}：{path}；{exc}") from exc


def _read_json_object(path: Path, label: str) -> tuple[Mapping[str, Any], str]:
    try:
        payload = path.read_bytes()
        value = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PromotionError(f"{label} 无法读取：{path}") from exc
    if not isinstance(value, Mapping):
        raise PromotionError(f"{label} 根节点必须是 JSON 对象")
    return value, sha256(payload).hexdigest()


def _fingerprints(value: Any, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != _FINGERPRINT_KEYS:
        raise PromotionError(f"{label} fingerprints 字段不完整")
    result = dict(value)
    for name, fingerprint in result.items():
        if not isinstance(fingerprint, str) or _HASH.fullmatch(fingerprint) is None:
            raise PromotionError(f"{label} {name} fingerprint 不是 SHA-256")
    return result


def _finite_probability(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise PromotionError(f"{label} 必须是有限概率")
    return float(value)


def _json_native_result(result: Any) -> dict[str, Any]:
    to_dict = getattr(result, "to_dict", None)
    if not callable(to_dict):
        raise PromotionError("experimental smoke 结果缺少 to_dict")
    value = to_dict()
    _require_json_native(value, "smoke")
    if not isinstance(value, dict):
        raise PromotionError("experimental smoke JSON 根节点必须是对象")
    return value


def _require_json_native(value: Any, path: str) -> None:
    if value is None or isinstance(value, bool | str):
        return
    if isinstance(value, int) and not isinstance(value, bool):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise PromotionError(f"{path} 包含非有限 JSON 数值")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _require_json_native(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise PromotionError(f"{path} 包含非字符串 JSON 键")
            _require_json_native(item, f"{path}.{key}")
        return
    raise PromotionError(f"{path} 包含非 JSON 原生类型：{type(value).__name__}")


def _require_distinct_paths(paths: Mapping[str, Path]) -> None:
    for (left_name, left), (right_name, right) in combinations(paths.items(), 2):
        if os.path.normcase(str(left)) == os.path.normcase(str(right)):
            raise PromotionError(f"路径冲突：{left_name} 与 {right_name} 指向同一路径 {left}")


def _reject_target_alias(target: Path, paths: Mapping[str, Path]) -> None:
    if not target.exists():
        return
    for name, path in paths.items():
        if name == "target" or not path.exists():
            continue
        try:
            aliased = os.path.samefile(target, path)
        except OSError as exc:
            raise PromotionError(f"无法检查 target hardlink 别名：{path}") from exc
        if aliased:
            raise PromotionError(f"target hardlink 别名冲突：{name}")


def _validate_metrics(value: Any, partition: str) -> dict[str, float]:
    required = {
        "class_order",
        "confusion_matrix",
        "per_class",
        "macro_f1",
        "balanced_accuracy",
        "auroc",
        "pr_auc",
        "uncalibrated_probability_quality",
        "calibrated_probability_quality",
        "bootstrap_95",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise PromotionError(f"{partition} report metrics 字段不完整或包含未知字段")
    if value.get("class_order") != ["authentic", "counterfeit"]:
        raise PromotionError(f"{partition} report class_order 不兼容")
    _validate_confusion_matrix(value.get("confusion_matrix"), partition)
    per_class = value.get("per_class")
    if not isinstance(per_class, Mapping) or list(per_class) != ["authentic", "counterfeit"]:
        raise PromotionError(f"{partition} report per_class 类别或顺序不兼容")
    parsed_classes = {
        name: _validate_class_metrics(per_class[name], f"{partition}.per_class.{name}")
        for name in ("authentic", "counterfeit")
    }
    result = {
        "macro_f1": _finite_probability(value.get("macro_f1"), f"{partition}.macro_f1"),
        "balanced_accuracy": _finite_probability(
            value.get("balanced_accuracy"), f"{partition}.balanced_accuracy"
        ),
        "auroc": _finite_probability(value.get("auroc"), f"{partition}.auroc"),
        "pr_auc": _finite_probability(value.get("pr_auc"), f"{partition}.pr_auc"),
        "counterfeit_recall": parsed_classes["counterfeit"]["recall"],
    }
    _validate_quality(
        value.get("uncalibrated_probability_quality"),
        f"{partition}.uncalibrated_probability_quality",
    )
    _validate_quality(
        value.get("calibrated_probability_quality"),
        f"{partition}.calibrated_probability_quality",
    )
    _validate_bootstrap(value.get("bootstrap_95"), partition)
    return result


def _validate_confusion_matrix(value: Any, partition: str) -> None:
    valid = (
        isinstance(value, list)
        and len(value) == 2
        and all(isinstance(row, list) and len(row) == 2 for row in value)
        and all(type(item) is int and item >= 0 for row in value for item in row)
    )
    if not valid:
        raise PromotionError(f"{partition} report confusion_matrix 类型错误")


def _validate_class_metrics(value: Any, label: str) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(value) != {"precision", "recall", "f1", "support"}:
        raise PromotionError(f"{label} 字段不完整或类型错误")
    support = value.get("support")
    if type(support) is not int or support < 0:
        raise PromotionError(f"{label}.support 必须是非负整数")
    return {
        name: _finite_probability(value.get(name), f"{label}.{name}")
        for name in ("precision", "recall", "f1")
    }


def _validate_quality(value: Any, label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != {"brier", "ece", "nll"}:
        raise PromotionError(f"{label} 字段不完整或类型错误")
    for name in ("brier", "ece", "nll"):
        _finite_nonnegative(value.get(name), f"{label}.{name}")


def _validate_bootstrap(value: Any, partition: str) -> None:
    required = {
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
    if not isinstance(value, Mapping) or set(value) != required:
        raise PromotionError(f"{partition} report bootstrap_95 字段不完整")
    for name, interval in value.items():
        label = f"{partition}.bootstrap_95.{name}"
        if not isinstance(interval, Mapping) or set(interval) != {
            "lower",
            "median",
            "upper",
            "samples",
            "seed",
        }:
            raise PromotionError(f"{label} 字段或类型错误")
        lower = _finite_nonnegative(interval.get("lower"), f"{label}.lower")
        median = _finite_nonnegative(interval.get("median"), f"{label}.median")
        upper = _finite_nonnegative(interval.get("upper"), f"{label}.upper")
        if not lower <= median <= upper:
            raise PromotionError(f"{label} 区间顺序错误")
        if type(interval.get("samples")) is not int or interval["samples"] <= 0:
            raise PromotionError(f"{label}.samples 必须是正整数")
        if type(interval.get("seed")) is not int:
            raise PromotionError(f"{label}.seed 必须是整数")


def _validate_gate_thresholds(value: Any, expected: Mapping[str, float], partition: str) -> None:
    if not isinstance(value, Mapping) or set(value) != set(_METRIC_KEYS):
        raise PromotionError(f"{partition} report 门槛字段不完整")
    parsed = {
        name: _finite_probability(value.get(name), f"{partition}.thresholds.{name}")
        for name in _METRIC_KEYS
    }
    if parsed != expected:
        raise PromotionError(f"{partition} report 门槛不是固定 0.80")


def _finite_nonnegative(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or value < 0
    ):
        raise PromotionError(f"{label} 必须是有限非负数")
    return float(value)


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _HASH.fullmatch(value) is None:
        raise PromotionError(f"{label} 必须是 64 位 SHA-256")
    return value


def _artifact_paths(preflight: _Preflight) -> dict[str, Path]:
    return {
        "validation_report_sha256": preflight.request.validation_report,
        "locked_report_sha256": preflight.request.locked_test_report,
        "reservation_sha256": preflight.request.run_dir / "locked_test" / "reservation.json",
    }


def _verify_artifact_hashes(preflight: _Preflight) -> None:
    for name, path in _artifact_paths(preflight).items():
        try:
            actual = sha256_file(path)
        except OSError as exc:
            raise PromotionError(f"{name} 并发改变或不可读：{path}") from exc
        if actual != preflight.artifact_sha256[name]:
            raise PromotionError(f"{name} 并发改变：{path}")
    actual_candidate = sha256_file(preflight.request.candidate)
    if actual_candidate != preflight.candidate_sha256:
        raise PromotionError("candidate 并发改变")


def _verify_target_unchanged(preflight: _Preflight) -> None:
    target = preflight.request.target
    if preflight.old_sha256 is None:
        if target.exists():
            raise PromotionError("并发改变 target：预检后新建了目标")
        return
    if not target.is_file() or sha256_file(target) != preflight.old_sha256:
        raise PromotionError("并发改变 target：旧 SHA-256 与预检不一致")


def _require_target_hash(target: Path, expected: str) -> None:
    if not target.is_file() or sha256_file(target) != expected:
        raise PromotionError("target SHA-256 与候选哈希不一致")


def _write_promotion_record_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(
            dict(payload),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
