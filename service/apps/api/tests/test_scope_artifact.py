"""范围工件（artifact）测试：JSON 序列化、绑定校验、损坏/不匹配/未批准拒绝。"""

from __future__ import annotations

from pathlib import Path

import pytest

from drynaria_api.scope.artifact import (
    ArtifactError,
    ScopeArtifact,
    build_artifact,
    load_artifact,
    parse_artifact,
    save_artifact,
    to_dict,
    verify_artifact,
)

CKPT = "a" * 64
FEATURE_DIM = 3


def _artifact(**overrides) -> ScopeArtifact:
    kwargs = dict(
        artifact_version="scope-logreg-v1",
        checkpoint_sha256=CKPT,
        feature_dim=FEATURE_DIM,
        scaler_mean=[0.0, 0.0, 0.0],
        scaler_scale=[1.0, 1.0, 1.0],
        classifier_coef=[1.0, 0.0, 0.0],
        classifier_intercept=0.0,
        threshold=0.5,
        evaluation={"test": {"n": 1}},
        approved=True,
        model_metadata={"backbone": "efficientnet_b0"},
        created_at="2026-09-10T00:00:00Z",
    )
    kwargs.update(overrides)
    return build_artifact(**kwargs)


@pytest.mark.parametrize("classes", [
    ["in_scope", "in_scope"],
    [None, "in_scope"],
    {"0": "invalid", "1": "in_scope"},
])
def test_invalid_negative_class_is_rejected(classes):
    data = to_dict(_artifact())
    data["classifier"]["classes"] = classes
    with pytest.raises(ArtifactError):
        parse_artifact(data)


def test_round_trip_json(tmp_path):
    artifact = _artifact()
    path = tmp_path / "artifact.json"
    save_artifact(artifact, path)
    assert path.is_file()
    reloaded = load_artifact(path)
    assert reloaded == artifact
    assert reloaded.checkpoint_sha256 == CKPT
    assert reloaded.feature_dim == FEATURE_DIM
    assert reloaded.approved is True


def test_to_dict_is_json_serializable():
    import json

    artifact = _artifact()
    payload = to_dict(artifact)
    # 数值必须是纯 Python 类型，可 JSON 序列化（无 numpy）
    json.dumps(payload)
    assert payload["scaler"]["mean"] == [0.0, 0.0, 0.0]
    assert payload["classifier"]["classes"] == ["out_of_scope", "in_scope"]


def test_verify_artifact_binds_checkpoint():
    artifact = _artifact()
    verify_artifact(artifact, checkpoint_sha256=CKPT, feature_dim=FEATURE_DIM)


def test_reject_mismatched_checkpoint():
    artifact = _artifact()
    with pytest.raises(ArtifactError, match="checkpoint 与当前模型不一致"):
        verify_artifact(artifact, checkpoint_sha256="b" * 64)


def test_reject_mismatched_feature_dim():
    artifact = _artifact()
    with pytest.raises(ArtifactError, match="feature_dim"):
        verify_artifact(artifact, feature_dim=999)


def test_reject_unapproved_candidate():
    artifact = _artifact(approved=False)
    with pytest.raises(ArtifactError, match="未通过"):
        verify_artifact(artifact, checkpoint_sha256=CKPT)


def test_reject_wrong_schema_version(tmp_path):
    artifact = _artifact()
    payload = to_dict(artifact)
    payload["schema_version"] = 99
    with pytest.raises(ArtifactError, match="schema_version"):
        parse_artifact(payload)


def test_reject_missing_field(tmp_path):
    with pytest.raises(ArtifactError, match="缺少必需字段"):
        parse_artifact({"schema_version": 1})


def test_reject_corrupt_json(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(ArtifactError, match="无法解析"):
        load_artifact(path)


def test_reject_invalid_sha256():
    with pytest.raises(ArtifactError, match="checkpoint_sha256"):
        _artifact(checkpoint_sha256="not-a-sha256")


def test_reject_invalid_feature_dim():
    with pytest.raises(ArtifactError, match="feature_dim"):
        _artifact(feature_dim=0)


def test_reject_scale_length_mismatch():
    with pytest.raises(ArtifactError, match="scale"):
        _artifact(scaler_scale=[1.0, 1.0])


def test_reject_coef_length_mismatch():
    with pytest.raises(ArtifactError, match="coef"):
        _artifact(classifier_coef=[1.0, 0.0, 0.0, 0.0])


def test_reject_nonfinite_threshold():
    with pytest.raises(ArtifactError, match="threshold"):
        _artifact(threshold=float("nan"))
