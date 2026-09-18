"""范围门控工件（artifact）的 JSON 序列化、读取与绑定校验。

工件是范围分类器（frozen encoder 特征 → StandardScaler → LogisticRegression）的数值参数
与元数据的唯一载体。约定：

- 数值参数 / 元数据全部序列化为 JSON（不用 pickle），不嵌入任何训练样本或作者代码。
- 工件绑定 checkpoint SHA-256 与特征维度（feature_dim），与运行时特征长度校验。
- 只有 ``approved=true``（验证集后满足操作者给定的 false-accept / false-reject 上限）
  的工件允许部署；否则只能作为候选（candidate），门控会拒用。

本模块只依赖标准库，可被 API / Worker / 测试 / 训练进程共享导入。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCOPE_ARTIFACT_SCHEMA_VERSION = 1

POSITIVE_CLASS = "in_scope"  # 被接受的代表“在范围内”
NEGATIVE_CLASS = "out_of_scope"

# 运行时至少需要的顶层字段
_REQUIRED_KEYS = {
    "schema_version",
    "artifact_version",
    "checkpoint_sha256",
    "feature_dim",
    "scaler",
    "classifier",
    "threshold",
    "evaluation",
    "approved",
}


class ArtifactError(RuntimeError):
    """工件损坏 / 不匹配 / 未批准 / 无法反序列化。错误信息不带真伪结论。"""


@dataclass(frozen=True)
class ScopeArtifact:
    schema_version: int
    artifact_version: str
    checkpoint_sha256: str
    feature_dim: int
    scaler_mean: tuple[float, ...]
    scaler_scale: tuple[float, ...]
    classifier_coef: tuple[float, ...]
    classifier_intercept: float
    positive_class: str
    threshold: float
    evaluation: dict[str, Any]
    approved: bool
    model_metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str | None = None

    @property
    def n_features(self) -> int:
        return self.feature_dim


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(ch in "0123456789abcdef" for ch in value)
    )


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ArtifactError(f"{name} 必须是有限数值")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ArtifactError(f"{name} 必须是有限数值")
    return parsed


def _finite_vector(value: Any, name: str, length: int) -> tuple[float, ...]:
    if not isinstance(value, list):
        raise ArtifactError(f"{name} 必须是数组")
    if len(value) != length:
        raise ArtifactError(f"{name} 长度 {len(value)} 与 feature_dim={length} 不一致")
    return tuple(_finite_number(item, f"{name}[{i}]") for i, item in enumerate(value))


def parse_artifact(data: Any) -> ScopeArtifact:
    """从已反序列化的 JSON 结构构造并严格校验 ``ScopeArtifact``。"""
    if not isinstance(data, dict):
        raise ArtifactError("工件必须是 JSON 对象")
    missing = _REQUIRED_KEYS - set(data)
    if missing:
        raise ArtifactError(f"工件缺少必需字段：{sorted(missing)}")

    if data.get("schema_version") != SCOPE_ARTIFACT_SCHEMA_VERSION:
        raise ArtifactError(
            f"工件 schema_version 不是受支持的 {SCOPE_ARTIFACT_SCHEMA_VERSION}"
        )
    artifact_version = data.get("artifact_version")
    if not isinstance(artifact_version, str) or not artifact_version.strip():
        raise ArtifactError("artifact_version 必须是非空字符串")
    checkpoint_sha256 = data.get("checkpoint_sha256")
    if not _is_sha256(checkpoint_sha256):
        raise ArtifactError("checkpoint_sha256 必须是 64 位小写 SHA-256")

    feature_dim = data.get("feature_dim")
    if isinstance(feature_dim, bool) or not isinstance(feature_dim, int) or feature_dim <= 0:
        raise ArtifactError("feature_dim 必须是正整数")

    scaler = data.get("scaler")
    classifier = data.get("classifier")
    if not isinstance(scaler, dict) or not isinstance(classifier, dict):
        raise ArtifactError("scaler / classifier 必须是映射")

    mean = _finite_vector(scaler.get("mean"), "scaler.mean", feature_dim)
    scale = _finite_vector(scaler.get("scale"), "scaler.scale", feature_dim)
    coef = _finite_vector(classifier.get("coef"), "classifier.coef", feature_dim)
    intercept = _finite_number(classifier.get("intercept"), "classifier.intercept")

    positive_class = classifier.get("classes")
    if positive_class not in (
        [NEGATIVE_CLASS, POSITIVE_CLASS],
        {"0": NEGATIVE_CLASS, "1": POSITIVE_CLASS},
    ):
        raise ArtifactError("classifier.classes 必须依次为 out_of_scope 和 in_scope")
    positive = POSITIVE_CLASS

    threshold = _finite_number(data.get("threshold"), "threshold")
    if not 0.0 <= threshold <= 1.0:
        raise ArtifactError("threshold 必须在 [0, 1] 范围内")

    evaluation = data.get("evaluation")
    if not isinstance(evaluation, dict):
        raise ArtifactError("evaluation 必须是映射")
    approved = data.get("approved")
    if not isinstance(approved, bool):
        raise ArtifactError("approved 必须是布尔值")

    model_metadata = data.get("model_metadata") or {}
    if not isinstance(model_metadata, dict):
        raise ArtifactError("model_metadata 必须是映射")
    created_at = data.get("created_at")

    return ScopeArtifact(
        schema_version=SCOPE_ARTIFACT_SCHEMA_VERSION,
        artifact_version=artifact_version,
        checkpoint_sha256=checkpoint_sha256,
        feature_dim=feature_dim,
        scaler_mean=mean,
        scaler_scale=scale,
        classifier_coef=coef,
        classifier_intercept=intercept,
        positive_class=positive,
        threshold=threshold,
        evaluation=evaluation,
        approved=approved,
        model_metadata=model_metadata,
        created_at=created_at,
    )


def to_dict(artifact: ScopeArtifact) -> dict[str, Any]:
    """把工件转回可 JSON 序列化的字典（数值全部为普通 float / list）。"""
    return {
        "schema_version": artifact.schema_version,
        "artifact_version": artifact.artifact_version,
        "checkpoint_sha256": artifact.checkpoint_sha256,
        "feature_dim": artifact.feature_dim,
        "scaler": {
            "mean": [float(v) for v in artifact.scaler_mean],
            "scale": [float(v) for v in artifact.scaler_scale],
        },
        "classifier": {
            "coef": [float(v) for v in artifact.classifier_coef],
            "intercept": float(artifact.classifier_intercept),
            "classes": [NEGATIVE_CLASS, artifact.positive_class],
        },
        "threshold": float(artifact.threshold),
        "evaluation": artifact.evaluation,
        "approved": artifact.approved,
        "model_metadata": artifact.model_metadata,
        "created_at": artifact.created_at,
    }


def build_artifact(
    *,
    artifact_version: str,
    checkpoint_sha256: str,
    feature_dim: int,
    scaler_mean: list[float] | tuple[float, ...],
    scaler_scale: list[float] | tuple[float, ...],
    classifier_coef: list[float] | tuple[float, ...],
    classifier_intercept: float,
    threshold: float,
    evaluation: dict[str, Any],
    approved: bool,
    model_metadata: dict[str, Any] | None = None,
    created_at: str | None = None,
) -> ScopeArtifact:
    """构造并经 :func:`parse_artifact` 严格校验后的工件。"""
    payload = {
        "schema_version": SCOPE_ARTIFACT_SCHEMA_VERSION,
        "artifact_version": artifact_version,
        "checkpoint_sha256": checkpoint_sha256,
        "feature_dim": feature_dim,
        "scaler": {
            "mean": [float(v) for v in scaler_mean],
            "scale": [float(v) for v in scaler_scale],
        },
        "classifier": {
            "coef": [float(v) for v in classifier_coef],
            "intercept": float(classifier_intercept),
            "classes": [NEGATIVE_CLASS, POSITIVE_CLASS],
        },
        "threshold": float(threshold),
        "evaluation": evaluation,
        "approved": approved,
        "model_metadata": model_metadata or {},
        "created_at": created_at,
    }
    return parse_artifact(payload)


def save_artifact(artifact: ScopeArtifact, path: str | Path) -> None:
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(to_dict(artifact), handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def load_artifact(path: str | Path) -> ScopeArtifact:
    source = Path(path).expanduser()
    if not source.is_file():
        raise ArtifactError(f"工件文件不存在：{source}")
    try:
        with source.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (ValueError, OSError) as exc:
        raise ArtifactError(f"工件无法解析为 JSON：{source}") from exc
    return parse_artifact(data)


def verify_artifact(
    artifact: ScopeArtifact,
    *,
    checkpoint_sha256: str | None = None,
    feature_dim: int | None = None,
) -> ScopeArtifact:
    """校验工件绑定并返回；不满足则抛 :class:`ArtifactError`。

    - ``checkpoint_sha256`` 提供时须与工件绑定一致；
    - ``feature_dim`` 提供时须与工件一致；
    - ``approved`` 必须为真，否则视为候选（candidate）不可部署。
    """
    if checkpoint_sha256 is not None:
        if not _is_sha256(checkpoint_sha256):
            raise ArtifactError("提供的 checkpoint_sha256 不是 64 位 SHA-256")
        if artifact.checkpoint_sha256 != checkpoint_sha256:
            raise ArtifactError(
                "工件绑定的 checkpoint 与当前模型不一致，拒绝加载"
            )
    if feature_dim is not None:
        if isinstance(feature_dim, bool) or not isinstance(feature_dim, int):
            raise ArtifactError("feature_dim 必须是整数")
        if artifact.feature_dim != feature_dim:
            raise ArtifactError(
                f"工件 feature_dim={artifact.feature_dim} 与当前模型 "
                f"feature_dim={feature_dim} 不一致"
            )
    if artifact.approved is not True:
        raise ArtifactError(
            f"工件 {artifact.artifact_version} 未通过 held-out 验证（仅候选），"
            "不可部署；不要据此声称通用 OOD 防护。"
        )
    return artifact
