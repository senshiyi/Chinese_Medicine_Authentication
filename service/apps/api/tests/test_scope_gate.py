"""范围门控（gate）测试：accepted/rejected/unavailable 决策与构建。"""

from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest

from drynaria_api.scope.artifact import build_artifact, save_artifact
from drynaria_api.scope.gate import (
    SCOPE_ACCEPTED,
    SCOPE_REJECTED,
    SCOPE_UNAVAILABLE,
    ScopeDecision,
    ScopeGate,
    build_scope_gate,
    scope_fingerprint,
)

CKPT = "a" * 64
FEATURE_DIM = 3


def _artifact(threshold: float = 0.5, approved: bool = True, checkpoint_sha256: str = CKPT):
    # 特征 [0,0,0] -> z=0 -> p=0.5；[-2,0,0] -> z=-2 -> p≈0.119
    return build_artifact(
        artifact_version="scope-logreg-v1",
        checkpoint_sha256=checkpoint_sha256,
        feature_dim=FEATURE_DIM,
        scaler_mean=[0.0, 0.0, 0.0],
        scaler_scale=[1.0, 1.0, 1.0],
        classifier_coef=[1.0, 0.0, 0.0],
        classifier_intercept=0.0,
        threshold=threshold,
        evaluation={"test": {"n": 1}},
        approved=approved,
    )


def _gate(**kwargs):
    return ScopeGate(enforced=True, artifact=_artifact(), artifact_path="in-memory", **kwargs)


def test_decide_accepts_in_scope():
    decision = _gate().decide([0.0, 0.0, 0.0])
    assert decision.scope == SCOPE_ACCEPTED
    assert decision.artifact_version == "scope-logreg-v1"
    assert decision.threshold == 0.5


def test_decide_rejects_out_of_scope():
    decision = _gate().decide([-2.0, 0.0, 0.0])
    assert decision.scope == SCOPE_REJECTED


def test_decide_unavailable_when_feature_dim_mismatch():
    decision = _gate().decide([0.0, 0.0])
    assert decision.scope == SCOPE_UNAVAILABLE
    assert "维度" in decision.reason


def test_unavailable_when_not_enforced():
    gate = ScopeGate(enforced=False, artifact=None, artifact_path=None)
    decision = gate.decide([0.0, 0.0, 0.0])
    assert decision.scope == SCOPE_UNAVAILABLE
    assert decision.enforced is False


def test_scope_decision_rejects_bad_scope_value():
    with pytest.raises(ValueError):
        ScopeDecision(scope="weird", reason="x", enforced=True)


def test_build_scope_gate_disabled():
    settings = SimpleNamespace(scope_enforcement=False, scope_artifact_path="")
    gate = build_scope_gate(settings)
    assert gate.enforced is False
    assert gate.available is False


def test_build_scope_gate_unavailable_without_artifact():
    settings = SimpleNamespace(scope_enforcement=True, scope_artifact_path="")
    gate = build_scope_gate(settings)
    assert gate.enforced is True
    assert gate.available is False
    assert gate.unavailable_decision().scope == SCOPE_UNAVAILABLE


def test_build_scope_gate_unavailable_when_path_missing():
    settings = SimpleNamespace(scope_enforcement=True, scope_artifact_path="C:/nope.json")
    gate = build_scope_gate(settings)
    assert gate.enforced is True
    assert gate.available is False


def test_build_scope_gate_available_with_approved_artifact(tmp_path):
    ckpt = tmp_path / "model.pth"
    ckpt.write_bytes(b"checkpoint-bytes")
    ckpt_sha = hashlib.sha256(b"checkpoint-bytes").hexdigest()
    path = tmp_path / "artifact.json"
    save_artifact(_artifact(checkpoint_sha256=ckpt_sha), path)
    settings = SimpleNamespace(
        scope_enforcement=True,
        scope_artifact_path=str(path),
        model_checkpoint_path=str(ckpt),
    )
    gate = build_scope_gate(settings)
    assert gate.enforced is True
    assert gate.available is True
    assert gate.decide([0.0, 0.0, 0.0]).scope == SCOPE_ACCEPTED


def test_build_scope_gate_unavailable_without_readable_checkpoint(tmp_path):
    # 工件存在但 checkpoint 不可读（无法校验绑定）=> unavailable，绝不草率部署
    path = tmp_path / "artifact.json"
    save_artifact(_artifact(), path)
    settings = SimpleNamespace(
        scope_enforcement=True,
        scope_artifact_path=str(path),
        model_checkpoint_path=str(tmp_path / "nonexistent.pth"),
    )
    gate = build_scope_gate(settings)
    assert gate.enforced is True
    assert gate.available is False
    assert "checkpoint" in gate.unavailable_decision().reason


def test_build_scope_gate_rejects_unapproved_artifact(tmp_path):
    ckpt = tmp_path / "model.pth"
    ckpt.write_bytes(b"checkpoint-bytes")
    ckpt_sha = hashlib.sha256(b"checkpoint-bytes").hexdigest()
    path = tmp_path / "artifact.json"
    save_artifact(_artifact(approved=False, checkpoint_sha256=ckpt_sha), path)
    settings = SimpleNamespace(
        scope_enforcement=True,
        scope_artifact_path=str(path),
        model_checkpoint_path=str(ckpt),
    )
    gate = build_scope_gate(settings)
    assert gate.enforced is True
    assert gate.available is False
    assert "未通过" in gate.unavailable_decision().reason


def test_build_scope_gate_rejects_checkpoint_mismatch(tmp_path):
    # checkpoint 文件存在，其 sha 与工件绑定不一致
    ckpt = tmp_path / "model.pth"
    ckpt.write_bytes(b"model-bytes")
    ckpt_sha = hashlib.sha256(b"model-bytes").hexdigest()
    path = tmp_path / "artifact.json"
    save_artifact(_artifact(), path)
    settings = SimpleNamespace(
        scope_enforcement=True,
        scope_artifact_path=str(path),
        model_checkpoint_path=str(ckpt),
    )
    assert ckpt_sha != CKPT  # 绑定必然不匹配
    gate = build_scope_gate(settings)
    assert gate.enforced is True
    assert gate.available is False


def test_scope_fingerprint_disabled():
    settings = SimpleNamespace(scope_enforcement=False, scope_artifact_path="")
    assert scope_fingerprint(settings) == "scope:disabled"


def test_scope_fingerprint_unavailable():
    settings = SimpleNamespace(scope_enforcement=True, scope_artifact_path="")
    assert scope_fingerprint(settings) == "scope:unavailable"


def test_scope_fingerprint_from_artifact(tmp_path):
    path = tmp_path / "artifact.json"
    save_artifact(_artifact(), path)
    expected = "scope:" + hashlib.sha256(path.read_bytes()).hexdigest()
    settings = SimpleNamespace(scope_enforcement=True, scope_artifact_path=str(path))
    assert scope_fingerprint(settings) == expected
