"""运行时范围门控：根据冻结 encoder 特征 + JSON 工件输出 accepted/rejected/unavailable。

逻辑（纯 Python，不依赖 numpy/torch）：
- 只有 ``scope_enforcement=true`` 才参与门控；false 表示旧行为（不门控）。
- 工件缺失 / 损坏 / checkpoint 不匹配 / 未批准 / 特征维度不一致 → ``unavailable``，
  绝不把一张图判成“在范围外”证明，也不给出肯定性结论。
- 工件可用时：``score >= threshold`` → ``accepted``；否则 ``rejected``。

``ScopeDecision.scope`` 只在 {accepted, rejected, unavailable} 中取值。
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .artifact import ArtifactError, ScopeArtifact, load_artifact, verify_artifact

SCOPE_ACCEPTED = "accepted"
SCOPE_REJECTED = "rejected"
SCOPE_UNAVAILABLE = "unavailable"

ACCEPTED_REASONS = {
    SCOPE_ACCEPTED: "scope_ok",
    SCOPE_REJECTED: "out_of_scope",
    SCOPE_UNAVAILABLE: "scope_evidence_absent",
}


@dataclass(frozen=True)
class ScopeDecision:
    scope: str
    reason: str
    enforced: bool
    version: str | None = None
    score: float | None = None
    threshold: float | None = None
    artifact_version: str | None = None
    feature_dim: int | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.scope not in {SCOPE_ACCEPTED, SCOPE_REJECTED, SCOPE_UNAVAILABLE}:
            raise ValueError(f"scope 只允许 accepted/rejected/unavailable，得到 {self.scope!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "reason": self.reason,
            "enforced": self.enforced,
            "version": self.version,
            "score": self.score,
            "threshold": self.threshold,
            "artifact_version": self.artifact_version,
            "feature_dim": self.feature_dim,
            "detail": self.detail,
        }


class ScopeGate:
    """持有已校验工件并在推理时给出范围决策。"""

    def __init__(
        self,
        *,
        enforced: bool,
        artifact: ScopeArtifact | None,
        artifact_path: str | None,
        unavailable_reason: str | None = None,
        checkpoint_sha256: str | None = None,
    ) -> None:
        self.enforced = enforced
        self.artifact = artifact
        self.artifact_path = artifact_path
        self._unavailable_reason = unavailable_reason
        self.checkpoint_sha256 = checkpoint_sha256

    @property
    def available(self) -> bool:
        return self.enforced and self.artifact is not None

    @property
    def version(self) -> str | None:
        return self.artifact.artifact_version if self.artifact else None

    def unavailable_decision(self, reason: str | None = None) -> ScopeDecision:
        return ScopeDecision(
            scope=SCOPE_UNAVAILABLE,
            reason=reason or self._unavailable_reason or "未提供可验证的范围工件",
            enforced=self.enforced,
            artifact_version=self.version,
        )

    def decide(self, features: Sequence[float]) -> ScopeDecision:
        """对给定特征向量给出范围决策。特征必须与工件 feature_dim 一致。"""
        if not self.enforced:
            return ScopeDecision(
                scope=SCOPE_UNAVAILABLE,
                reason="scope 门控未启用",
                enforced=False,
            )
        if self.artifact is None:
            return self.unavailable_decision()

        artifact = self.artifact
        if len(features) != artifact.feature_dim:
            return ScopeDecision(
                scope=SCOPE_UNAVAILABLE,
                reason=(
                    f"特征维度 {len(features)} 与工件 feature_dim="
                    f"{artifact.feature_dim} 不一致"
                ),
                enforced=True,
                artifact_version=artifact.artifact_version,
                feature_dim=artifact.feature_dim,
            )

        score = _logreg_score(
            features,
            mean=artifact.scaler_mean,
            scale=artifact.scaler_scale,
            coef=artifact.classifier_coef,
            intercept=artifact.classifier_intercept,
        )
        accepted = score >= artifact.threshold
        return ScopeDecision(
            scope=SCOPE_ACCEPTED if accepted else SCOPE_REJECTED,
            reason=ACCEPTED_REASONS[SCOPE_ACCEPTED if accepted else SCOPE_REJECTED],
            enforced=True,
            version=artifact.artifact_version,
            score=score,
            threshold=artifact.threshold,
            artifact_version=artifact.artifact_version,
            feature_dim=artifact.feature_dim,
        )


def build_scope_gate(settings: Any) -> ScopeGate:
    """根据配置构造范围门控；绝不抛出，一律退化为 unavailable 并给出原因。"""
    if not getattr(settings, "scope_enforcement", False):
        return ScopeGate(enforced=False, artifact=None, artifact_path=None)

    artifact_path = str(getattr(settings, "scope_artifact_path", "") or "").strip()
    if not artifact_path:
        return ScopeGate(
            enforced=True,
            artifact=None,
            artifact_path=None,
            unavailable_reason="scope_enforcement 已启用，但未配置可验证的范围工件",
        )

    path = Path(artifact_path).expanduser()
    if not path.is_file():
        return ScopeGate(
            enforced=True,
            artifact=None,
            artifact_path=artifact_path,
            unavailable_reason=f"范围工件不存在：{path}",
        )

    try:
        artifact = load_artifact(path)
    except ArtifactError as exc:
        return ScopeGate(
            enforced=True,
            artifact=None,
            artifact_path=artifact_path,
            unavailable_reason=f"范围工件不可用：{exc}",
        )

    checkpoint = getattr(settings, "model_checkpoint_path", None)
    checkpoint_sha = _checkpoint_sha256(checkpoint)
    if checkpoint_sha is None:
        # 无法读取 checkpoint 校验绑定：宁可 unavailable，也不会在未核对绑定时部署范围判定。
        return ScopeGate(
            enforced=True,
            artifact=None,
            artifact_path=artifact_path,
            unavailable_reason=f"无法读取 checkpoint 以校验工件绑定：{checkpoint}",
        )
    try:
        verify_artifact(artifact, checkpoint_sha256=checkpoint_sha)
    except ArtifactError as exc:
        return ScopeGate(
            enforced=True,
            artifact=None,
            artifact_path=artifact_path,
            unavailable_reason=f"范围工件校验失败：{exc}",
            checkpoint_sha256=checkpoint_sha,
        )
    return ScopeGate(
        enforced=True,
        artifact=artifact,
        artifact_path=artifact_path,
        checkpoint_sha256=checkpoint_sha,
    )


def scope_fingerprint(settings: Any) -> str:
    """分析幂等键中的范围策略指纹。

    只依赖配置与工件文件的字节哈希，不读取 checkpoint 大文件，可安全在 API 进程调用。
    """
    if not getattr(settings, "scope_enforcement", False):
        return "scope:disabled"
    artifact_path = str(getattr(settings, "scope_artifact_path", "") or "").strip()
    if not artifact_path:
        return "scope:unavailable"
    path = Path(artifact_path).expanduser()
    if not path.is_file():
        return "scope:unavailable"
    return "scope:" + _sha256_file(path)


def _logreg_score(
    features: Sequence[float],
    *,
    mean: Sequence[float],
    scale: Sequence[float],
    coef: Sequence[float],
    intercept: float,
) -> float:
    """StandardScaler + LogisticRegression 二分类的 in_scope 概率。

    等价于 sklearn ``LogisticRegression``（binary）的 ``predict_proba[:, 1]``：
    ``z = coef @ ((x - mean) / scale) + intercept``，``p = sigmoid(z)``。
    """
    z = intercept
    for i, x in enumerate(features):
        scaled = (x - mean[i]) / scale[i] if scale[i] != 0 else 0.0
        z += coef[i] * scaled
    if z >= 0:
        exp_neg = math.exp(-z)
        return 1.0 / (1.0 + exp_neg)
    exp_pos = math.exp(z)
    return exp_pos / (1.0 + exp_pos)


def _checkpoint_sha256(path: str | Path | None) -> str | None:
    if not path:
        return None
    resolved = Path(path).expanduser()
    if not resolved.is_file():
        return None
    return _sha256_file(resolved)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
