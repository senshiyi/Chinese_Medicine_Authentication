"""范围训练管线自检（合成数据）。

**重要**：本测试只用扁平合成特征验证管线机械正确性（train 拟合 / calibration 选阈值 /
test 单次评估 / held-out 通过才 approved）。**合成数据不代表真实 OOD 性能，不构成任何
真实世界准确率的证据。** 需要 sklearn+numpy；缺依赖时整模块跳过。
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

sklearn = pytest.importorskip("sklearn")

from drynaria_api.scope.manifest import ManifestError
from drynaria_api.scope.pipeline import (
    ScopeTrainingError,
    choose_threshold,
    evaluate,
    run_scope_training,
    wilson_interval,
)

CKPT = "a" * 64
FEATURE_DIM = 2
IN_FEATURE = [1.0, 1.0]
OUT_FEATURE = [-1.0, -1.0]


def _write_dataset(tmp_path: Path, *, overlap: bool = False, consent_ok: bool = True):
    root = tmp_path / "images"
    root.mkdir()
    rows = []
    features = {}
    in_feature = [0.0, 0.0] if overlap else IN_FEATURE
    out_feature = [0.0, 0.0] if overlap else OUT_FEATURE
    for split, base_group in (("train", "gA"), ("calibration", "gB"), ("test", "gC")):
        class_rows = (("in_scope", in_feature), ("out_of_scope", out_feature))
        for i, (label, feature) in enumerate(class_rows):
            name = f"{split}_{i}.img"
            (root / name).write_bytes(f"{split}-{label}-{i}".encode("utf-8"))
            rows.append((name, label, split, base_group, "src", consent_ok))
            features[str((root / name).resolve())] = feature
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_path", "scope_label", "split", "group_id", "source", "consent"])
        writer.writerows(rows)
    return manifest, root, features


def _extractor(features):
    def extract(path: Path) -> list[float]:
        return features[str(path.resolve())]

    return extract


def test_zero_errors_on_tiny_test_set_remains_candidate(tmp_path):
    manifest, root, features = _write_dataset(tmp_path)
    result = run_scope_training(
        manifest,
        root,
        feature_extract=_extractor(features),
        artifact_version="scope-logreg-v1",
        checkpoint_sha256=CKPT,
        feature_dim=FEATURE_DIM,
        max_false_accept=0.05,
        max_false_reject=0.05,
        require_min_samples=2,
    )
    artifact = result["artifact"]
    report = result["report"]
    assert artifact.approved is False
    assert artifact.checkpoint_sha256 == CKPT
    assert artifact.feature_dim == FEATURE_DIM
    assert report["approved"] is False
    assert 0 <= artifact.threshold <= 1
    # 分阶段指标被记录
    assert "calibration" in report["evaluation"]
    assert "test" in report["evaluation"]
    test_eval = report["evaluation"]["test"]
    assert test_eval["satisfied"] is False


def test_heldout_passes_when_confidence_bounds_meet_limits():
    result = evaluate(
        [0.9] * 100 + [0.1] * 100,
        [1] * 100 + [0] * 100,
        0.5,
        max_false_accept=0.05,
        max_false_reject=0.05,
    )
    assert result.satisfied is True
    assert result.false_accept_ci[1] < 0.05
    assert result.false_reject_ci[1] < 0.05


def test_run_scope_training_candidate_when_limits_too_strict(tmp_path):
    # 重叠（不可分）特征：无论怎样阈值都无法同时满足 FAR=0 与 FRR=0 => 候选
    manifest, root, features = _write_dataset(tmp_path, overlap=True)
    result = run_scope_training(
        manifest,
        root,
        feature_extract=_extractor(features),
        artifact_version="scope-logreg-v1",
        checkpoint_sha256=CKPT,
        feature_dim=FEATURE_DIM,
        max_false_accept=0.0,  # 不可能满足 => 候选
        max_false_reject=0.0,
        require_min_samples=2,
    )
    assert result["artifact"].approved is False
    # 仍写出工件（候选），但绝不标记为可部署
    assert "仅作为候选" in result["report"]["evaluation"]["note"]


def test_run_scope_training_rejects_unconsented(tmp_path):
    manifest, root, features = _write_dataset(tmp_path, consent_ok=False)
    with pytest.raises(ScopeTrainingError, match="未获授权"):
        run_scope_training(
            manifest,
            root,
            feature_extract=_extractor(features),
            artifact_version="scope-logreg-v1",
            checkpoint_sha256=CKPT,
            feature_dim=FEATURE_DIM,
            max_false_accept=0.05,
            max_false_reject=0.05,
            require_min_samples=2,
        )


def test_run_scope_training_rejects_sample_too_small(tmp_path):
    manifest, root, features = _write_dataset(tmp_path)
    with pytest.raises(ScopeTrainingError, match="样本数"):
        run_scope_training(
            manifest,
            root,
            feature_extract=_extractor(features),
            artifact_version="scope-logreg-v1",
            checkpoint_sha256=CKPT,
            feature_dim=FEATURE_DIM,
            max_false_accept=0.05,
            max_false_reject=0.05,
            require_min_samples=50,
        )


def test_choose_threshold_picks_satisfying_threshold():
    # in_scope 分数高（>=0.8），out_of_scope 分数低（<=0.3）
    scores = [0.9, 0.85, 0.3, 0.2]
    labels = [1, 1, 0, 0]
    result = choose_threshold(scores, labels, max_false_accept=0.0, max_false_reject=0.0)
    assert result.satisfied is True
    assert result.threshold >= 0.8


def test_choose_threshold_reports_failure_when_impossible():
    # 分数完全不可分（两类的分数相同）：无法同时满足两个上限
    scores = [0.5, 0.5, 0.5, 0.5]
    labels = [1, 0, 1, 0]
    result = choose_threshold(scores, labels, max_false_accept=0.0, max_false_reject=0.0)
    assert result.satisfied is False


def test_wilson_interval_sane():
    lo, hi = wilson_interval(1, 10)
    assert 0 <= lo <= hi <= 1
    lo2, hi2 = wilson_interval(0, 0)
    assert (lo2, hi2) == (0.0, 0.0)


def test_feature_dim_mismatch_raises(tmp_path):
    manifest, root, features = _write_dataset(tmp_path)
    with pytest.raises(ScopeTrainingError, match="coef 维度"):
        run_scope_training(
            manifest,
            root,
            feature_extract=_extractor(features),
            artifact_version="scope-logreg-v1",
            checkpoint_sha256=CKPT,
            feature_dim=5,
            max_false_accept=0.05,
            max_false_reject=0.05,
            require_min_samples=2,
        )
