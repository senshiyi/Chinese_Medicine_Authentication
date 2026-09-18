"""评估汇总、来源门禁与 UTF-8 工件写入。"""

from __future__ import annotations

import csv
import io
import json
import math
import os
import re
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from herb_authentication.training.metrics import (
    AggregatedPredictions,
    ImageLevelPredictions,
    aggregate_by_sample,
    compute_entity_metrics,
)


@dataclass(frozen=True, slots=True)
class EvaluationArtifacts:
    json_path: Path
    csv_path: Path


@dataclass(frozen=True, slots=True)
class ClassMetrics:
    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True, slots=True)
class AuthenticityMetrics:
    confusion_matrix: tuple[tuple[int, int], tuple[int, int]]
    macro_f1: float
    balanced_accuracy: float
    per_class: Mapping[str, ClassMetrics]
    auroc: float
    pr_auc: float
    brier: float
    ece: float
    nll: float


@dataclass(frozen=True, slots=True)
class BootstrapInterval:
    lower: float
    median: float
    upper: float
    samples: int
    seed: int


@dataclass(frozen=True, slots=True)
class UnknownPrediction:
    unknown_type: str
    sample_id: str
    group_id: str
    rejected: bool


def compute_authenticity_metrics(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    class_names: tuple[str, str] = ("authentic", "counterfeit"),
    bins: int = 10,
) -> AuthenticityMetrics:
    """按固定类别顺序显式计算闭集真伪二分类指标。"""

    clean_probabilities, clean_targets = _validate_authenticity_inputs(
        probabilities, targets, class_names, bins
    )
    predictions = clean_probabilities.argmax(dim=1)
    confusion = tuple(
        tuple(
            int(((clean_targets == actual) & (predictions == predicted)).sum().item())
            for predicted in range(2)
        )
        for actual in range(2)
    )
    class_metrics: dict[str, ClassMetrics] = {}
    for index, name in enumerate(class_names):
        true_positive = confusion[index][index]
        false_positive = confusion[1 - index][index]
        false_negative = confusion[index][1 - index]
        support = sum(confusion[index])
        precision = _safe_ratio(true_positive, true_positive + false_positive)
        recall = _safe_ratio(true_positive, true_positive + false_negative)
        f1 = _safe_ratio(2.0 * precision * recall, precision + recall)
        class_metrics[name] = ClassMetrics(precision, recall, f1, support)

    from sklearn.metrics import average_precision_score, roc_auc_score

    positive_scores = clean_probabilities[:, 1].numpy()
    target_values = clean_targets.numpy()
    expected = torch.nn.functional.one_hot(clean_targets, num_classes=2).to(torch.float64)
    brier = float(((clean_probabilities - expected) ** 2).sum(dim=1).mean().item())
    selected = clean_probabilities.gather(1, clean_targets[:, None]).squeeze(1)
    nll = float((-selected.log()).mean().item())
    confidence, predicted = clean_probabilities.max(dim=1)
    correct = predicted.eq(clean_targets).to(torch.float64)
    ece = 0.0
    boundaries = torch.linspace(0.0, 1.0, bins + 1, dtype=torch.float64)
    for index in range(bins):
        lower, upper = boundaries[index], boundaries[index + 1]
        included = (
            (confidence >= lower) & (confidence <= upper)
            if index == bins - 1
            else (confidence >= lower) & (confidence < upper)
        )
        if bool(included.any()):
            weight = float(included.to(torch.float64).mean().item())
            ece += weight * abs(
                float(correct[included].mean().item() - confidence[included].mean().item())
            )
    return AuthenticityMetrics(
        confusion_matrix=(confusion[0], confusion[1]),
        macro_f1=sum(item.f1 for item in class_metrics.values()) / 2.0,
        balanced_accuracy=sum(item.recall for item in class_metrics.values()) / 2.0,
        per_class=class_metrics,
        auroc=float(roc_auc_score(target_values, positive_scores)),
        pr_auc=float(average_precision_score(target_values, positive_scores)),
        brier=brier,
        ece=ece,
        nll=nll,
    )


def bootstrap_authenticity_metrics(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    *,
    samples: int,
    seed: int,
    class_names: tuple[str, str] = ("authentic", "counterfeit"),
    bins: int = 10,
) -> Mapping[str, BootstrapInterval]:
    """以分层 sample bootstrap 计算二分类标量指标的 95% 区间。"""

    clean_probabilities, clean_targets = _validate_authenticity_inputs(
        probabilities, targets, class_names, bins
    )
    if isinstance(samples, bool) or not isinstance(samples, int) or samples <= 0:
        raise ValueError("bootstrap samples 必须是正整数")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("bootstrap seed 必须是整数")
    generator = torch.Generator().manual_seed(seed)
    indexes_by_class = tuple(torch.where(clean_targets == index)[0] for index in range(2))
    values: dict[str, list[float]] = {
        name: []
        for name in (
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
        )
    }
    for _ in range(samples):
        drawn = torch.cat(
            tuple(
                indexes[torch.randint(len(indexes), (len(indexes),), generator=generator)]
                for indexes in indexes_by_class
            )
        )
        result = compute_authenticity_metrics(
            clean_probabilities.index_select(0, drawn),
            clean_targets.index_select(0, drawn),
            class_names,
            bins,
        )
        row = {
            "macro_f1": result.macro_f1,
            "balanced_accuracy": result.balanced_accuracy,
            "authentic_recall": result.per_class[class_names[0]].recall,
            "authentic_f1": result.per_class[class_names[0]].f1,
            "counterfeit_recall": result.per_class[class_names[1]].recall,
            "counterfeit_f1": result.per_class[class_names[1]].f1,
            "auroc": result.auroc,
            "pr_auc": result.pr_auc,
            "brier": result.brier,
            "ece": result.ece,
            "nll": result.nll,
        }
        for name, value in row.items():
            values[name].append(value)
    quantile_points = torch.tensor([0.025, 0.5, 0.975], dtype=torch.float64)
    return {
        name: BootstrapInterval(
            *(
                float(value)
                for value in torch.quantile(
                    torch.tensor(rows, dtype=torch.float64), quantile_points
                )
            ),
            samples=samples,
            seed=seed,
        )
        for name, rows in values.items()
    }


def build_closed_set_evaluation_report(
    *,
    run_id: str,
    partition: str,
    uncalibrated: AuthenticityMetrics,
    calibrated: AuthenticityMetrics,
    bootstrap: Mapping[str, BootstrapInterval],
    fingerprints: Mapping[str, str],
    thresholds: Mapping[str, float],
) -> dict[str, Any]:
    """构造 schema v2 实验闭集报告；只有 validation 产生晋升门禁。"""

    _require_safe_run_id(run_id)
    if partition not in {"validation", "locked-test"}:
        raise ValueError("partition 只允许 validation 或 locked-test")
    required_fingerprints = {
        "manifest",
        "split",
        "config",
        "source_candidate",
        "calibration",
    }
    if set(fingerprints) != required_fingerprints or any(
        not isinstance(value, str) or len(value) != 64 for value in fingerprints.values()
    ):
        raise ValueError("评估报告指纹字段不完整")
    required_thresholds = {"macro_f1", "balanced_accuracy", "counterfeit_recall"}
    if set(thresholds) != required_thresholds or any(
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or not 0 <= value <= 1
        for value in thresholds.values()
    ):
        raise ValueError("晋升门槛字段不完整或不是有限概率")
    class_order = ("authentic", "counterfeit")
    if tuple(calibrated.per_class) != class_order or tuple(uncalibrated.per_class) != class_order:
        raise ValueError("真伪指标类别顺序不兼容")
    checks = {
        "macro_f1": calibrated.macro_f1 >= thresholds["macro_f1"],
        "balanced_accuracy": (calibrated.balanced_accuracy >= thresholds["balanced_accuracy"]),
        "counterfeit_recall": (
            calibrated.per_class["counterfeit"].recall >= thresholds["counterfeit_recall"]
        ),
    }
    gate = {
        "thresholds": dict(thresholds),
        "checks": checks,
        "passed": all(checks.values()),
    }
    report: dict[str, Any] = {
        "schema_version": 2,
        "run_id": run_id,
        "evaluation_scope": "closed_set_authenticity",
        "partition": partition,
        "experimental_closed_set": True,
        "open_set_calibrated": False,
        "limitations": [
            "仅适用于本数据集定义的骨碎补正品/伪品范围",
            "缺少批次、来源、设备和拍摄会话元数据，无法排除采集捷径",
            "伪品实际物种未检验",
            "没有 unknown calibration/test 数据",
        ],
        "fingerprints": dict(fingerprints),
        "metrics": {
            "class_order": list(class_order),
            "confusion_matrix": [list(row) for row in calibrated.confusion_matrix],
            "per_class": {name: asdict(calibrated.per_class[name]) for name in class_order},
            "macro_f1": calibrated.macro_f1,
            "balanced_accuracy": calibrated.balanced_accuracy,
            "auroc": calibrated.auroc,
            "pr_auc": calibrated.pr_auc,
            "uncalibrated_probability_quality": {
                "brier": uncalibrated.brier,
                "ece": uncalibrated.ece,
                "nll": uncalibrated.nll,
            },
            "calibrated_probability_quality": {
                "brier": calibrated.brier,
                "ece": calibrated.ece,
                "nll": calibrated.nll,
            },
            "bootstrap_95": {name: asdict(interval) for name, interval in bootstrap.items()},
        },
    }
    report["promotion_gate" if partition == "validation" else "thresholds_met"] = gate
    return report


def _require_safe_run_id(run_id: object) -> None:
    if (
        not isinstance(run_id, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id) is None
        or run_id.endswith((".", " "))
    ):
        raise ValueError(f"不安全的 run ID：{run_id!r}")


def reserve_locked_test(run_dir: str | Path, fingerprint_bundle: Mapping[str, str]) -> Path:
    """在任何 locked-test inference 前独占创建一次性 reservation。"""

    required = {"manifest", "split", "config", "source_candidate", "calibration"}
    if set(fingerprint_bundle) != required or any(
        not isinstance(value, str) or len(value) != 64 for value in fingerprint_bundle.values()
    ):
        raise ValueError("locked test reservation 指纹字段不完整")
    resolved_run = Path(run_dir).expanduser().resolve()
    _require_safe_run_id(resolved_run.name)
    destination = resolved_run / "locked_test" / "reservation.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            {
                "schema_version": 1,
                "run_id": resolved_run.name,
                "fingerprints": dict(fingerprint_bundle),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    try:
        descriptor = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
            0o600,
        )
    except FileExistsError as exc:
        raise ValueError("该 run 已执行 locked test") from exc
    try:
        remaining = memoryview(payload)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError("写入 locked test reservation 时未取得进展")
            remaining = remaining[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return destination


def _validate_authenticity_inputs(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    class_names: tuple[str, str],
    bins: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if (
        not isinstance(probabilities, torch.Tensor)
        or probabilities.ndim != 2
        or probabilities.shape[1] != 2
        or probabilities.shape[0] == 0
    ):
        raise ValueError("真伪 probabilities 必须是非空 [N, 2]")
    if not isinstance(targets, torch.Tensor) or targets.shape != (probabilities.shape[0],):
        raise ValueError("真伪 targets 必须是 [N]")
    if not bool(torch.isfinite(probabilities).all()) or not bool(torch.isfinite(targets).all()):
        raise ValueError("真伪评估输入必须全部是有限数值")
    if bool((probabilities < 0).any()) or bool((probabilities > 1).any()):
        raise ValueError("真伪 probabilities 必须位于 [0, 1]")
    clean_probabilities = probabilities.detach().to(dtype=torch.float64, device="cpu")
    row_sums = clean_probabilities.sum(dim=1)
    if not bool(torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6, rtol=0)):
        raise ValueError("真伪 probabilities 每行之和必须为 1")
    if not bool(((targets == 0) | (targets == 1)).all()):
        raise ValueError("真伪 targets 只能是 0 或 1")
    clean_targets = targets.detach().to(dtype=torch.long, device="cpu")
    if not bool(((clean_targets == 0) | (clean_targets == 1)).all()):
        raise ValueError("真伪 targets 只能是 0 或 1")
    if torch.unique(clean_targets).numel() != 2:
        raise ValueError("真伪 targets 必须包含两个类别")
    if class_names != ("authentic", "counterfeit"):
        raise ValueError("class_names 必须严格依次为 authentic、counterfeit")
    if isinstance(bins, bool) or not isinstance(bins, int) or bins <= 0:
        raise ValueError("ECE bins 必须是正整数")
    return clean_probabilities.clamp_min(torch.finfo(torch.float64).tiny), clean_targets


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def collect_known_predictions(
    model: torch.nn.Module,
    loader: Iterable[Mapping[str, Any]],
    *,
    device: str | torch.device,
    temperatures: Mapping[str, float],
) -> ImageLevelPredictions:
    """一次遍历 known loader，收集校准概率、标签与实体元数据。"""

    expected_heads = {"claimed_herb", "actual_species", "authenticity"}
    if set(temperatures) != expected_heads or any(
        isinstance(value, bool) or not isinstance(value, int | float) or value <= 0
        for value in temperatures.values()
    ):
        raise ValueError("temperatures 必须包含三个任务头的有限正数")
    resolved_device = torch.device(device)
    model = model.to(resolved_device).eval()
    claimed_probabilities: list[torch.Tensor] = []
    species_probabilities: list[torch.Tensor] = []
    authenticity_probabilities: list[torch.Tensor] = []
    claimed_targets: list[torch.Tensor] = []
    species_targets: list[torch.Tensor] = []
    authenticity_targets: list[torch.Tensor] = []
    species_masks: list[torch.Tensor] = []
    sample_ids: list[str] = []
    group_ids: list[str] = []

    with torch.inference_mode():
        for batch in loader:
            try:
                images = batch["image"].to(resolved_device)
                metadata = tuple(batch["metadata"])
                output = model(images)
                claimed_logits = output.claimed_herb
                species_logits = output.actual_species
                authenticity_logits = output.authenticity
            except (AttributeError, KeyError, TypeError) as exc:
                raise ValueError("known 评估 batch 字段不完整") from exc
            batch_size = int(images.shape[0])
            logits = (claimed_logits, species_logits, authenticity_logits)
            if any(
                not isinstance(value, torch.Tensor)
                or value.ndim != 2
                or value.shape[0] != batch_size
                or not bool(torch.isfinite(value).all())
                for value in logits
            ):
                raise ValueError("known 评估模型输出必须是有限的二维 Tensor")
            if len(metadata) != batch_size:
                raise ValueError("known 评估 metadata 数量与 batch 不一致")
            claimed_probabilities.append(
                torch.softmax(claimed_logits / temperatures["claimed_herb"], dim=1).cpu()
            )
            species_probabilities.append(
                torch.sigmoid(species_logits / temperatures["actual_species"]).cpu()
            )
            authenticity_probabilities.append(
                torch.softmax(authenticity_logits / temperatures["authenticity"], dim=1).cpu()
            )
            claimed_targets.append(batch["claimed_target"].detach().cpu())
            species_targets.append(batch["actual_species_target"].detach().cpu())
            authenticity_targets.append(batch["authenticity_target"].detach().cpu())
            species_masks.append(batch["actual_species_mask"].detach().cpu())
            for item in metadata:
                if not isinstance(item, Mapping):
                    raise ValueError("known 评估 metadata 必须是映射")
                sample_id, group_id = item.get("sample_id"), item.get("group_id")
                if not isinstance(sample_id, str) or not isinstance(group_id, str):
                    raise ValueError("known 评估 metadata 缺少 sample_id 或 group_id")
                sample_ids.append(sample_id)
                group_ids.append(group_id)
    if not claimed_probabilities:
        raise ValueError("known test DataLoader 不能为空")
    return ImageLevelPredictions(
        claimed_probabilities=torch.cat(claimed_probabilities),
        species_probabilities=torch.cat(species_probabilities),
        authenticity_probabilities=torch.cat(authenticity_probabilities),
        claimed_targets=torch.cat(claimed_targets),
        species_targets=torch.cat(species_targets),
        authenticity_targets=torch.cat(authenticity_targets),
        species_mask=torch.cat(species_masks).bool(),
        sample_ids=tuple(sample_ids),
        group_ids=tuple(group_ids),
    )


def summarize_known_predictions(
    predictions: ImageLevelPredictions,
    *,
    min_species_supervision_coverage: float,
    species_threshold: float,
    bootstrap_samples: int,
    claimed_threshold: float = 0.0,
    authenticity_threshold: float = 0.0,
    seed: int,
) -> dict[str, Any]:
    """分别报告图片、实体和 group bootstrap 指标。"""

    samples = aggregate_by_sample(predictions)
    image_metrics = compute_entity_metrics(
        _as_aggregated(predictions),
        min_species_supervision_coverage=min_species_supervision_coverage,
        species_threshold=species_threshold,
    )
    sample_metrics = compute_entity_metrics(
        samples,
        min_species_supervision_coverage=min_species_supervision_coverage,
        species_threshold=species_threshold,
    )
    interval = _group_bootstrap(
        samples,
        min_species_supervision_coverage=min_species_supervision_coverage,
        species_threshold=species_threshold,
        samples=bootstrap_samples,
        seed=seed,
    )
    return {
        "status": "complete",
        "counts": {
            "images": len(predictions.sample_ids),
            "samples": len(samples.sample_ids),
            "groups": len(set(samples.group_ids)),
        },
        "image_level": asdict(image_metrics),
        "sample_level": asdict(sample_metrics),
        "group_bootstrap": interval,
        "selective": _selective_metrics(
            predictions, claimed_threshold, species_threshold, authenticity_threshold
        ),
    }


def summarize_unknown_predictions(
    predictions: Sequence[UnknownPrediction],
) -> dict[str, Any]:
    """未知数据以“任一图片被接受即实体未拒识”的保守规则聚合。"""

    rows = tuple(predictions)
    if not rows:
        raise ValueError("unknown test 预测不能为空")
    if any(
        not item.unknown_type.strip()
        or not item.sample_id.strip()
        or not item.group_id.strip()
        or not isinstance(item.rejected, bool)
        for item in rows
    ):
        raise ValueError("unknown test 预测字段不完整")
    sample_rows: dict[str, list[UnknownPrediction]] = defaultdict(list)
    group_rows: dict[str, list[UnknownPrediction]] = defaultdict(list)
    type_rows: dict[str, list[bool]] = defaultdict(list)
    for item in rows:
        sample_rows[item.sample_id].append(item)
        group_rows[item.group_id].append(item)
        type_rows[item.unknown_type].append(item.rejected)
    return {
        "status": "complete",
        "counts": {
            "images": len(rows),
            "samples": len(sample_rows),
            "groups": len(group_rows),
        },
        "image_level": {
            "reject_rate": _rate(item.rejected for item in rows),
            "by_unknown_type": {name: _rate(values) for name, values in sorted(type_rows.items())},
        },
        "sample_level": {
            "reject_rate": _rate(
                all(item.rejected for item in values) for values in sample_rows.values()
            )
        },
        "group_level": {
            "reject_rate": _rate(
                all(item.rejected for item in values) for values in group_rows.values()
            )
        },
    }


def build_evaluation_report(
    *,
    data_provenance: str,
    checkpoint_provenance: str | None = None,
    data_fingerprint: str | None,
    checkpoint_fingerprint: str | None,
    locked_split_fingerprint: str | None,
    checkpoint_split_fingerprint: str | None,
    is_calibrated: bool,
    known_metrics: Mapping[str, Any] | None = None,
    unknown_metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """只有真实数据且锁定切分匹配时才标记为正式指标。"""

    if data_provenance not in {"real", "synthetic", "unknown"}:
        raise ValueError("data_provenance 只允许 real、synthetic 或 unknown")
    compatible = bool(
        locked_split_fingerprint
        and checkpoint_split_fingerprint
        and locked_split_fingerprint == checkpoint_split_fingerprint
    )
    selective = known_metrics.get("selective") if isinstance(known_metrics, Mapping) else None
    selective_values = (
        tuple(selective.get(name) for name in ("coverage", "selective_accuracy", "reject_rate"))
        if isinstance(selective, Mapping)
        else ()
    )
    valid_selective_values = len(selective_values) == 3 and all(
        not isinstance(value, bool)
        and isinstance(value, int | float)
        and math.isfinite(value)
        and 0 <= value <= 1
        for value in selective_values
    )
    known_complete = (
        isinstance(known_metrics, Mapping)
        and known_metrics.get("status") == "complete"
        and isinstance(selective, Mapping)
        and valid_selective_values
    )
    unknown_sample = (
        unknown_metrics.get("sample_level") if isinstance(unknown_metrics, Mapping) else None
    )
    unknown_complete = (
        isinstance(unknown_metrics, Mapping)
        and unknown_metrics.get("status") == "complete"
        and isinstance(unknown_sample, Mapping)
        and "reject_rate" in unknown_sample
    )
    official = (
        data_provenance == checkpoint_provenance == "real"
        and compatible
        and is_calibrated
        and known_complete
        and unknown_complete
    )
    return {
        "schema_version": 1,
        "data_provenance": data_provenance,
        "official_metrics": bool(official),
        "locked_split_compatible": compatible,
        "is_calibrated": bool(is_calibrated),
        "checkpoint_provenance": checkpoint_provenance,
        "fingerprints": {
            "data": data_fingerprint,
            "checkpoint": checkpoint_fingerprint,
            "locked_split": locked_split_fingerprint,
            "checkpoint_split": checkpoint_split_fingerprint,
        },
        "metric_scopes": {
            "image_level": "单张图片",
            "sample_level": "同一 sample_id 聚合",
            "group_level": "按 group_id 聚合或 bootstrap",
        },
        "metrics": {
            "known": dict(known_metrics or {"status": "not_evaluated"}),
            "unknown": dict(unknown_metrics or {"status": "not_evaluated"}),
        },
    }


def write_evaluation_artifacts(
    report: Mapping[str, Any], output_dir: str | Path
) -> EvaluationArtifacts:
    """完整生成并校验同目录临时文件，再成对发布且失败时回滚。"""

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    json_path, csv_path = destination / "evaluation.json", destination / "evaluation.csv"
    json_bytes = (
        json.dumps(dict(report), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=("scope", "level", "metric", "value"))
    writer.writeheader()
    metrics = report.get("metrics", {})
    if isinstance(metrics, Mapping):
        for scope, values in metrics.items():
            _write_metric_rows(writer, str(scope), values)
    csv_bytes = b"\xef\xbb\xbf" + csv_buffer.getvalue().encode("utf-8")
    json.loads(json_bytes.decode("utf-8"))
    tuple(csv.DictReader(io.StringIO(csv_bytes.decode("utf-8-sig"))))

    previous = {
        json_path: json_path.read_bytes() if json_path.exists() else None,
        csv_path: csv_path.read_bytes() if csv_path.exists() else None,
    }
    json_temp: Path | None = None
    csv_temp: Path | None = None
    try:
        json_temp = _write_fsynced_temp(destination, json_path.name, json_bytes)
        csv_temp = _write_fsynced_temp(destination, csv_path.name, csv_bytes)
        try:
            os.replace(json_temp, json_path)
            os.replace(csv_temp, csv_path)
        except OSError as publish_error:
            rollback_failures: list[Exception] = []
            for path, old_bytes in previous.items():
                restore: Path | None = None
                try:
                    if old_bytes is None:
                        path.unlink(missing_ok=True)
                    else:
                        restore = _write_fsynced_temp(destination, path.name, old_bytes)
                        os.replace(restore, path)
                except Exception as rollback_error:
                    rollback_failures.append(rollback_error)
                finally:
                    if restore is not None:
                        restore.unlink(missing_ok=True)
            if rollback_failures:
                publish_error.add_note(
                    f"报告回滚出现 {len(rollback_failures)} 项失败；详见 chained ExceptionGroup"
                )
                raise publish_error from ExceptionGroup("报告回滚失败", rollback_failures)
            raise
    finally:
        if json_temp is not None:
            json_temp.unlink(missing_ok=True)
        if csv_temp is not None:
            csv_temp.unlink(missing_ok=True)
    return EvaluationArtifacts(json_path, csv_path)


def _write_fsynced_temp(directory: Path, name: str, payload: bytes) -> Path:
    descriptor, raw_path = tempfile.mkstemp(prefix=f".{name}.", suffix=".tmp", dir=directory)
    path = Path(raw_path)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if path.read_bytes() != payload:
            raise OSError(f"临时工件校验失败：{path}")
        return path
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _as_aggregated(predictions: ImageLevelPredictions) -> AggregatedPredictions:
    return AggregatedPredictions(**asdict(predictions))


def _group_bootstrap(
    predictions: AggregatedPredictions,
    *,
    min_species_supervision_coverage: float,
    species_threshold: float,
    samples: int,
    seed: int,
) -> dict[str, int | float]:
    if isinstance(samples, bool) or not isinstance(samples, int) or samples <= 0:
        raise ValueError("bootstrap samples 必须是正整数")
    groups = tuple(dict.fromkeys(predictions.group_ids))
    group_indexes = {
        group: [index for index, value in enumerate(predictions.group_ids) if value == group]
        for group in groups
    }
    generator = torch.Generator().manual_seed(seed)
    scores: list[float] = []
    for _ in range(samples):
        draws = torch.randint(len(groups), (len(groups),), generator=generator).tolist()
        indexes = [index for draw in draws for index in group_indexes[groups[draw]]]
        subset = _select_rows(predictions, indexes)
        scores.append(
            compute_entity_metrics(
                subset,
                min_species_supervision_coverage=min_species_supervision_coverage,
                species_threshold=species_threshold,
            ).selection_score
        )
    quantiles = torch.quantile(
        torch.tensor(scores, dtype=torch.float64),
        torch.tensor([0.025, 0.5, 0.975], dtype=torch.float64),
    )
    return {
        "lower": float(quantiles[0]),
        "median": float(quantiles[1]),
        "upper": float(quantiles[2]),
        "samples": samples,
        "seed": seed,
    }


def _selective_metrics(
    predictions: ImageLevelPredictions,
    claimed_threshold: float,
    species_threshold: float,
    authenticity_threshold: float,
) -> dict[str, float | int | None]:
    claimed_confidence, claimed_prediction = predictions.claimed_probabilities.max(dim=1)
    authenticity_confidence, authenticity_prediction = predictions.authenticity_probabilities.max(
        dim=1
    )
    species_prediction = predictions.species_probabilities >= species_threshold
    species_available = species_prediction.any(dim=1)
    eligible = predictions.species_mask.bool()
    accepted = (
        eligible
        & (claimed_confidence >= claimed_threshold)
        & species_available
        & (authenticity_confidence >= authenticity_threshold)
    )
    correct = (
        (claimed_prediction == predictions.claimed_targets)
        & (species_prediction == predictions.species_targets.bool()).all(dim=1)
        & (authenticity_prediction == predictions.authenticity_targets)
    )
    eligible_count = int(eligible.sum().item())
    accepted_count = int(accepted.sum().item())
    correct_count = int((accepted & correct).sum().item())
    return {
        "eligible": eligible_count,
        "accepted": accepted_count,
        "coverage": accepted_count / eligible_count if eligible_count else 0.0,
        "selective_accuracy": correct_count / accepted_count if accepted_count else None,
        "reject_rate": 1.0 - accepted_count / eligible_count if eligible_count else 1.0,
    }


def _select_rows(predictions: AggregatedPredictions, indexes: list[int]) -> AggregatedPredictions:
    index = torch.tensor(indexes, dtype=torch.long)
    return AggregatedPredictions(
        claimed_probabilities=predictions.claimed_probabilities.index_select(0, index),
        species_probabilities=predictions.species_probabilities.index_select(0, index),
        authenticity_probabilities=predictions.authenticity_probabilities.index_select(0, index),
        claimed_targets=predictions.claimed_targets.index_select(0, index),
        species_targets=predictions.species_targets.index_select(0, index),
        authenticity_targets=predictions.authenticity_targets.index_select(0, index),
        species_mask=predictions.species_mask.index_select(0, index),
        sample_ids=tuple(predictions.sample_ids[i] for i in indexes),
        group_ids=tuple(predictions.group_ids[i] for i in indexes),
    )


def _rate(values: Any) -> float:
    rows = tuple(bool(value) for value in values)
    return sum(rows) / len(rows)


def _write_metric_rows(writer: csv.DictWriter, scope: str, values: Any) -> None:
    if not isinstance(values, Mapping):
        writer.writerow({"scope": scope, "level": "", "metric": "value", "value": values})
        return
    for name, value in values.items():
        if isinstance(value, Mapping):
            for metric, metric_value in value.items():
                writer.writerow(
                    {"scope": scope, "level": name, "metric": metric, "value": metric_value}
                )
        else:
            writer.writerow({"scope": scope, "level": "", "metric": name, "value": value})
