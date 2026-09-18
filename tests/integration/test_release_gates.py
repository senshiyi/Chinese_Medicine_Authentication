from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from herb_authentication.evaluation import build_evaluation_report

ROOT = Path(__file__).resolve().parents[2]


def _load_script(name: str):
    path = ROOT / "项目" / "Files" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_release_gate", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_known_validation_accepts_only_locked_partition_column() -> None:
    calibrate = _load_script("calibrate")

    calibrate._require_locked_known_validation([{"partition": "val"}])

    with pytest.raises(ValueError, match="test 数据禁止用于校准"):
        calibrate._require_locked_known_validation([{"partition": "test"}])


def test_calibration_requires_both_recorded_manifest_fingerprints(tmp_path: Path) -> None:
    calibrate = _load_script("calibrate")
    known = tmp_path / "val.csv"
    unknown = tmp_path / "unknown.csv"
    known.write_text("partition\nval\n", encoding="utf-8")
    unknown.write_text("dataset_role\ncalibration\n", encoding="utf-8")

    with pytest.raises(ValueError, match="缺少.*known_validation"):
        calibrate._validate_manifest_fingerprints({"metadata": {}}, known, unknown)


@pytest.mark.parametrize(
    ("calibrated", "known", "unknown", "checkpoint_provenance"),
    [
        (False, {"sample_level": {}}, {"sample_level": {}}, "real"),
        (True, None, {"sample_level": {}}, "real"),
        (True, {"sample_level": {}}, {"status": "not_available"}, "real"),
        (True, {"sample_level": {}}, {"sample_level": {}}, "synthetic"),
    ],
)
def test_official_metrics_fail_closed_without_complete_calibrated_real_evaluation(
    calibrated: bool,
    known: dict[str, object] | None,
    unknown: dict[str, object] | None,
    checkpoint_provenance: str,
) -> None:
    report = build_evaluation_report(
        data_provenance="real",
        checkpoint_provenance=checkpoint_provenance,
        data_fingerprint="data",
        checkpoint_fingerprint="checkpoint",
        locked_split_fingerprint="same",
        checkpoint_split_fingerprint="same",
        is_calibrated=calibrated,
        known_metrics=known,
        unknown_metrics=unknown,
    )

    assert report["official_metrics"] is False


def test_complete_calibrated_real_evaluation_can_be_official() -> None:
    report = build_evaluation_report(
        data_provenance="real",
        checkpoint_provenance="real",
        data_fingerprint="data",
        checkpoint_fingerprint="checkpoint",
        locked_split_fingerprint="same",
        checkpoint_split_fingerprint="same",
        is_calibrated=True,
        known_metrics={
            "status": "complete",
            "selective": {"coverage": 0.8, "selective_accuracy": 0.9, "reject_rate": 0.2},
        },
        unknown_metrics={"status": "complete", "sample_level": {"reject_rate": 0.9}},
    )

    assert report["official_metrics"] is True


def test_full_split_file_fingerprint_changes_when_group_assignment_changes(tmp_path: Path) -> None:
    evaluate = _load_script("evaluate")
    split = tmp_path / "splits.json"
    value = {
        "schema_version": 1,
        "algorithm_version": "group-stratified-v1",
        "seed": 42,
        "ratios": {"train": 0.7, "val": 0.2, "test": 0.1},
        "input_fingerprint": "input",
        "by_group": {"g1": "train", "g2": "test"},
    }
    split.write_text(json.dumps(value), encoding="utf-8")
    before = evaluate._locked_split_fingerprint(split)
    value["by_group"] = {"g1": "test", "g2": "train"}
    split.write_text(json.dumps(value), encoding="utf-8")

    assert evaluate._locked_split_fingerprint(split) != before


def test_zero_known_acceptance_cannot_be_official() -> None:
    report = build_evaluation_report(
        data_provenance="real",
        checkpoint_provenance="real",
        data_fingerprint="data",
        checkpoint_fingerprint="checkpoint",
        locked_split_fingerprint="same",
        checkpoint_split_fingerprint="same",
        is_calibrated=True,
        known_metrics={
            "status": "complete",
            "selective": {
                "coverage": 0.0,
                "selective_accuracy": None,
                "reject_rate": 1.0,
            },
        },
        unknown_metrics={"status": "complete", "sample_level": {"reject_rate": 1.0}},
    )

    assert report["official_metrics"] is False


def test_promote_cli_orchestrates_fixed_config_and_prints_native_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    promote = _load_script("promote")
    from herb_authentication import config as config_module
    from herb_authentication.training.promotion import PromotionRecord

    target = tmp_path / "official.pth"
    monkeypatch.setattr(
        config_module,
        "load_config",
        lambda _path: SimpleNamespace(
            promotion=SimpleNamespace(
                min_macro_f1=0.8,
                min_balanced_accuracy=0.8,
                min_counterfeit_recall=0.8,
                target_checkpoint_path=target,
            )
        ),
    )
    captured: dict[str, object] = {}

    def fake_promote(request, thresholds, smoke_image):
        captured.update(request=request, thresholds=thresholds, smoke_image=smoke_image)
        return PromotionRecord(None, "a" * 64, target.resolve(), "run-1", "2026-08-23T00:00:00Z")

    monkeypatch.setattr(promote, "promote_candidate", fake_promote)
    run_dir = tmp_path / "run-1"
    result = promote.main(
        [
            "--config",
            str(tmp_path / "config.yaml"),
            "--run-dir",
            str(run_dir),
            "--candidate",
            str(tmp_path / "candidate.pth"),
            "--smoke-image",
            str(tmp_path / "smoke.jpg"),
        ]
    )

    assert result == 0
    request = captured["request"]
    assert request.target == target.resolve()
    assert request.validation_report == run_dir.resolve() / "validation" / "evaluation.json"
    assert request.locked_test_report == run_dir.resolve() / "locked_test" / "evaluation.json"
    assert captured["thresholds"].to_dict() == {
        "macro_f1": 0.8,
        "balanced_accuracy": 0.8,
        "counterfeit_recall": 0.8,
    }
    assert json.loads(capsys.readouterr().out)["new_sha256"] == "a" * 64


def test_promote_cli_rejects_lowered_typed_thresholds_before_service(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    promote = _load_script("promote")
    from herb_authentication import config as config_module

    monkeypatch.setattr(
        config_module,
        "load_config",
        lambda _path: SimpleNamespace(
            promotion=SimpleNamespace(
                min_macro_f1=0.79,
                min_balanced_accuracy=0.8,
                min_counterfeit_recall=0.8,
                target_checkpoint_path=tmp_path / "official.pth",
            )
        ),
    )
    monkeypatch.setattr(
        promote,
        "promote_candidate",
        lambda *_args: pytest.fail("lowered config must not call promotion service"),
    )

    result = promote.main(
        [
            "--config",
            str(tmp_path / "config.yaml"),
            "--run-dir",
            str(tmp_path / "run-1"),
            "--candidate",
            str(tmp_path / "candidate.pth"),
            "--smoke-image",
            str(tmp_path / "smoke.jpg"),
        ]
    )

    assert result == 2
    assert "固定" in capsys.readouterr().err
