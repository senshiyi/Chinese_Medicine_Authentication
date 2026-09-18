"""范围（scope）门控：独立于闭集真伪的“是否本药材/本数据集范围”决策。

这一子包实现三件事，全部与原始闭集模型解耦，且不覆盖任何原模型权重：

1. 范围分类 Baseline：冻结现有 encoder 特征 + sklearn StandardScaler/LogisticRegression。
2. 可复现的 JSON 工件：数值参数（scaler、classifier、threshold）与元数据全部序列化为
   JSON，绑定 checkpoint SHA-256 与特征维度，拒绝损坏/不匹配/未批准工件。
3. 运行时门控：只有 ``scope=accepted`` 的图片才允许给出公开真伪结论；rejected /
   unavailable 一律 inconclusive，**不伪造**真伪预测。

隔离约定：
- 本包不加载 torch / sklearn（除 ``features``、``pipeline`` 两个模块，它们只在
  Worker / 训练 CLI 进程中被按需导入）。API 进程与常规测试进程永远不导入它们。
- ``manifest`` / ``artifact`` / ``gate`` 是纯 Python（仅标准库），可在任意环境运行。
"""

from __future__ import annotations

from .artifact import (
    SCOPE_ARTIFACT_SCHEMA_VERSION,
    ArtifactError,
    ScopeArtifact,
    build_artifact,
    load_artifact,
    save_artifact,
    verify_artifact,
)
from .gate import (
    ScopeDecision,
    build_scope_gate,
    scope_fingerprint,
)
from .manifest import (
    SCOPE_IN_SCOPE,
    SCOPE_OUT_OF_SCOPE,
    ManifestError,
    ScopeManifestRow,
    load_manifest,
    validate_manifest,
)

__all__ = [
    "SCOPE_ARTIFACT_SCHEMA_VERSION",
    "SCOPE_IN_SCOPE",
    "SCOPE_OUT_OF_SCOPE",
    "ArtifactError",
    "ManifestError",
    "ScopeArtifact",
    "ScopeDecision",
    "ScopeManifestRow",
    "build_artifact",
    "build_scope_gate",
    "load_artifact",
    "load_manifest",
    "save_artifact",
    "scope_fingerprint",
    "validate_manifest",
    "verify_artifact",
]
