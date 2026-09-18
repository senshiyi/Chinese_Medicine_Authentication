"""范围分类训练 / 校准 / 评估管线（frozen encoder 特征 + sklearn）。

阶段划分（严格按照范围门控合同）：
1. 只在 train 拟合 ``StandardScaler`` + ``LogisticRegression``；
2. 只在 calibration 选择阈值，以同时满足操作者给定的 max-false-accept / max-false-reject；
3. 只在 test 做一次评估；held-out 通过才 ``approved=true``，否则只是候选（candidate）。

本模块导入 numpy / sklearn / torch，只应在带 sklearn + torch 的训练 / 评估进程（通常是
原项目 ``.venv/python.exe``）中按需导入。``run_scope_training`` 通过注入的
``feature_extract`` 可调用对象解耦特征来源，便于用扁平合成特征做不针对真实性能的
小型自检（见测试说明：合成数据不代表真实 OOD 表现）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from .artifact import build_artifact
from .manifest import (
    SCOPE_IN_SCOPE,
    SPLITS,
    ScopeManifestRow,
    load_manifest,
    validate_group_separation,
)

FeatureExtract = Callable[[bytes | Path], list[float]]


class ScopeTrainingError(RuntimeError):
    """训练 / 校准 / 评估失败。错误信息不带真伪结论。"""


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float  # 最终使用的阈值（找不到满足条件时为 0.5 占位）
    satisfied: bool
    false_accept: float
    false_reject: float
    counts: dict[str, int]


@dataclass(frozen=True)
class EvalResult:
    false_accept: float
    false_reject: float
    false_accept_ci: tuple[float, float]
    false_reject_ci: tuple[float, float]
    counts: dict[str, int]
    satisfied: bool


def _import_sklearn():
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:  # pragma: no cover
        raise ScopeTrainingError("缺少 sklearn，无法训练范围分类器") from exc
    return StandardScaler, LogisticRegression


def split_rows(rows: list[ScopeManifestRow]) -> dict[str, list[ScopeManifestRow]]:
    return {split: [r for r in rows if r.split == split] for split in SPLITS}


def feature_matrix(
    rows: list[ScopeManifestRow], extract: FeatureExtract
) -> tuple[list[list[float]], list[int]]:
    """为一批行提取特征并返回 (X, y)；y=1 表示 in_scope。"""
    X: list[list[float]] = []
    y: list[int] = []
    for row in rows:
        X.append(extract(row.absolute_path))
        y.append(1 if row.scope_label == SCOPE_IN_SCOPE else 0)
    return X, y


def fit_scaler_classifier(
    X_train: list[list[float]], y_train: list[int], *, feature_dim: int
) -> dict[str, Any]:
    if len(X_train) < 2 or len({y for y in y_train}) < 2:
        raise ScopeTrainingError("train 集必须包含 in_scope 与 out_of_scope 两类才能拟合")
    StandardScaler, LogisticRegression = _import_sklearn()
    scaler = StandardScaler()
    scaler.fit(X_train)
    classifier = LogisticRegression(max_iter=2000, solver="lbfgs")
    classifier.fit(scaler.transform(X_train), y_train)
    coef = classifier.coef_[0]
    if len(coef) != feature_dim:
        raise ScopeTrainingError(
            f"分类器 coef 维度 {len(coef)} 与 feature_dim={feature_dim} 不一致"
        )
    # classes_ = [0, 1]（out_of_scope, in_scope）
    return {
        "mean": [float(v) for v in scaler.mean_],
        "scale": [float(v) for v in scaler.scale_],
        "coef": [float(v) for v in coef],
        "intercept": float(classifier.intercept_[0]),
    }


def _predict_scores(
    X: list[list[float]],
    *,
    mean: Sequence[float],
    scale: Sequence[float],
    coef: Sequence[float],
    intercept: float,
) -> list[float]:
    """用纯 Python 复算 LogisticRegression 的 predict_proba[:, 1]（in_scope 概率）。"""
    scores: list[float] = []
    for features in X:
        z = intercept
        for i, x in enumerate(features):
            scaled = (x - mean[i]) / scale[i] if scale[i] != 0 else 0.0
            z += coef[i] * scaled
        scores.append(1.0 / (1.0 + math.exp(-max(-500.0, min(500.0, z)))))
    return scores


def _rates_for_threshold(
    scores: Sequence[float], labels: Sequence[int], threshold: float
) -> dict[str, float]:
    accepted = [s >= threshold for s in scores]
    in_scope = [int(l) for l in labels]
    neg = sum(1 for i, l in enumerate(in_scope) if l == 0)
    pos = sum(1 for i, l in enumerate(in_scope) if l == 1)
    false_accept = sum(1 for i, l in enumerate(in_scope) if l == 0 and accepted[i])
    false_reject = sum(1 for i, l in enumerate(in_scope) if l == 1 and not accepted[i])
    return {
        "neg_total": neg,
        "pos_total": pos,
        "false_accept": false_accept,
        "false_reject": false_reject,
        "false_accept_rate": (false_accept / neg) if neg else 0.0,
        "false_reject_rate": (false_reject / pos) if pos else 0.0,
    }


def choose_threshold(
    scores: Sequence[float],
    labels: Sequence[int],
    *,
    max_false_accept: float,
    max_false_reject: float,
) -> ThresholdResult:
    """在 calibration 集上选择同时满足两个上限的阈值。

    找不到时 ``satisfied=False``、threshold=0.5（占位），调用方据此标记候选。
    """
    if len(scores) != len(labels) or not scores:
        raise ScopeTrainingError("calibration 集为空或与标签长度不一致")
    if not (0 <= max_false_accept <= 1 and 0 <= max_false_reject <= 1):
        raise ScopeTrainingError("max-false-accept / max-false-reject 必须在 [0,1]")

    candidates = sorted({float(s) for s in scores})
    best: tuple[float, float] | None = None  # (far, frr, threshold)
    best_threshold = 0.5
    for threshold in candidates:
        rates = _rates_for_threshold(scores, labels, threshold)
        far = rates["false_accept_rate"]
        frr = rates["false_reject_rate"]
        if far <= max_false_accept and frr <= max_false_reject:
            cost = far + frr
            if best is None or cost < best[0]:
                best = (cost, far, frr, threshold)
                best_threshold = threshold

    if best is None:
        far = _rates_for_threshold(scores, labels, 0.5)["false_accept_rate"]
        frr = _rates_for_threshold(scores, labels, 0.5)["false_reject_rate"]
        return ThresholdResult(
            threshold=0.5,
            satisfied=False,
            false_accept=far,
            false_reject=frr,
            counts={},
        )
    return ThresholdResult(
        threshold=best_threshold,
        satisfied=True,
        false_accept=best[1],
        false_reject=best[2],
        counts={},
    )


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval：用于比例（false-accept / false-reject）置信区间。"""
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = (z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def evaluate(
    scores: Sequence[float],
    labels: Sequence[int],
    threshold: float,
    *,
    max_false_accept: float,
    max_false_reject: float,
) -> EvalResult:
    if len(scores) != len(labels) or not scores:
        raise ScopeTrainingError("test 集为空或与标签长度不一致")
    neg = sum(1 for l in labels if l == 0)
    pos = sum(1 for l in labels if l == 1)
    if neg == 0 or pos == 0:
        raise ScopeTrainingError("test 集必须同时包含 in_scope 与 out_of_scope 两类")
    rates = _rates_for_threshold(scores, labels, threshold)
    far = rates["false_accept_rate"]
    frr = rates["false_reject_rate"]
    return EvalResult(
        false_accept=far,
        false_reject=frr,
        false_accept_ci=wilson_interval(rates["false_accept"], neg),
        false_reject_ci=wilson_interval(rates["false_reject"], pos),
        counts={
            "neg_total": neg,
            "pos_total": pos,
            "false_accept": rates["false_accept"],
            "false_reject": rates["false_reject"],
        },
        satisfied=(
            wilson_interval(rates["false_accept"], neg)[1] <= max_false_accept
            and wilson_interval(rates["false_reject"], pos)[1] <= max_false_reject
        ),
    )


def run_scope_training(
    manifest_csv: str | Path,
    image_root: str | Path,
    *,
    feature_extract: FeatureExtract,
    artifact_version: str,
    checkpoint_sha256: str,
    feature_dim: int,
    max_false_accept: float,
    max_false_reject: float,
    model_metadata: dict[str, Any] | None = None,
    require_min_samples: int = 8,
) -> dict[str, Any]:
    """执行完整训练 / 校准 / 评估管线，返回工件字典与指标报告。

    任何前置条件（清单校验、训练集类别、calibration/test 类别、样本量、上限满足度）
    不满足都不会被静默当作科学验证；只有 held-out 通过才 ``approved=true``。
    """
    rows = _validate_rows(manifest_csv, image_root, require_min_samples=require_min_samples)
    by_split = split_rows(rows)

    X_train, y_train = feature_matrix(by_split["train"], feature_extract)
    fitted = fit_scaler_classifier(X_train, y_train, feature_dim=feature_dim)

    X_cal, y_cal = feature_matrix(by_split["calibration"], feature_extract)
    cal_scores = _predict_scores(X_cal, **fitted)
    threshold_result = choose_threshold(
        cal_scores, y_cal, max_false_accept=max_false_accept, max_false_reject=max_false_reject
    )

    X_test, y_test = feature_matrix(by_split["test"], feature_extract)
    test_scores = _predict_scores(X_test, **fitted)
    eval_result = evaluate(
        test_scores,
        y_test,
        threshold_result.threshold,
        max_false_accept=max_false_accept,
        max_false_reject=max_false_reject,
    )

    approved = bool(threshold_result.satisfied and eval_result.satisfied)
    evaluation = {
        "threshold": float(threshold_result.threshold),
        "calibration": {
            "satisfied": threshold_result.satisfied,
            "false_accept_rate": threshold_result.false_accept,
            "false_reject_rate": threshold_result.false_reject,
            "max_false_accept": max_false_accept,
            "max_false_reject": max_false_reject,
        },
        "test": {
            "false_accept_rate": eval_result.false_accept,
            "false_reject_rate": eval_result.false_reject,
            "false_accept_ci95": [eval_result.false_accept_ci[0], eval_result.false_accept_ci[1]],
            "false_reject_ci95": [eval_result.false_reject_ci[0], eval_result.false_reject_ci[1]],
            "counts": eval_result.counts,
            "satisfied": eval_result.satisfied,
        },
        "approved": approved,
        "note": (
            "Held-out 评估通过，仅表示在此标注清单上满足操作者给定的错误率上限；"
            "不构成通用 OOD（开放集）防护，不替代真实世界抽样验证。"
            if approved
            else "Held-out 评估未通过或校准不满足上限：仅作为候选工件，不可部署；"
            "不得据此声称通用 OOD 防护。"
        ),
    }

    artifact = build_artifact(
        artifact_version=artifact_version,
        checkpoint_sha256=checkpoint_sha256,
        feature_dim=feature_dim,
        scaler_mean=fitted["mean"],
        scaler_scale=fitted["scale"],
        classifier_coef=fitted["coef"],
        classifier_intercept=fitted["intercept"],
        threshold=threshold_result.threshold,
        evaluation=evaluation,
        approved=approved,
        model_metadata=model_metadata or {},
    )
    report = {
        "artifact_version": artifact_version,
        "checkpoint_sha256": checkpoint_sha256,
        "feature_dim": feature_dim,
        "approved": approved,
        "threshold": threshold_result.threshold,
        "evaluation": evaluation,
        "train_counts": _row_counts(by_split["train"]),
        "calibration_counts": _row_counts(by_split["calibration"]),
        "test_counts": _row_counts(by_split["test"]),
    }
    return {"artifact": artifact, "report": report}


def _validate_rows(manifest_csv: str | Path, image_root: str | Path, *, require_min_samples: int):
    """校验清单并强制每个 split 都同时含两类且样本量足够。"""
    rows = load_manifest(manifest_csv, image_root)
    if not all(row.consent for row in rows):
        raise ScopeTrainingError("存在未获授权样本，拒绝训练")
    validate_group_separation(rows)
    by_split = split_rows(rows)
    for split in SPLITS:
        group = by_split[split]
        if len(group) < require_min_samples:
            raise ScopeTrainingError(
                f"split {split!r} 样本数 {len(group)} 少于要求的最小值 {require_min_samples}"
            )
        labels = {row.scope_label for row in group}
        if len(labels) < 2:
            raise ScopeTrainingError(
                f"split {split!r} 必须同时包含 in_scope 与 out_of_scope 两类"
            )
    return rows


def _row_counts(rows: list[ScopeManifestRow]) -> dict[str, int]:
    return {
        "total": len(rows),
        "in_scope": sum(1 for r in rows if r.scope_label == SCOPE_IN_SCOPE),
        "out_of_scope": sum(1 for r in rows if r.scope_label != SCOPE_IN_SCOPE),
    }
