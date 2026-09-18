from __future__ import annotations

import json
import os
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from herb_authentication.evaluation import (
    AuthenticityMetrics,
    BootstrapInterval,
    ClassMetrics,
    build_closed_set_evaluation_report,
)
from herb_authentication.inference import service as service_module
from herb_authentication.inference.calibration import CalibrationBundle
from herb_authentication.models.multitask import MultiTaskHerbModel
from herb_authentication.training import promotion
from herb_authentication.training.promotion import (
    DEFAULT_PROMOTION_THRESHOLDS,
    PromotionError,
    PromotionRequest,
    PromotionThresholds,
    promote_candidate,
)
from PIL import Image
from torch import nn


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=True), encoding="utf-8")


def _checkpoint(run_id: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "metadata": {
            "run_id": run_id,
            "data_fingerprint": "a" * 64,
            "split_input_fingerprint": "a" * 64,
            "split_fingerprint": "b" * 64,
            "experimental_closed_set": True,
            "open_set_calibrated": False,
            "fingerprints": {
                "manifest": "a" * 64,
                "known_validation": "f" * 64,
                "locked_split": "b" * 64,
            },
            "split_artifact_fingerprint": "b" * 64,
            "training_config_fingerprint": "c" * 64,
            "closed_set_calibration": {
                "temperature": 1.0,
                "algorithm_version": "closed-set-temperature-v1",
                "validation_fingerprint": "f" * 64,
                "checkpoint_fingerprint": "d" * 64,
                "nll_before": 0.5,
                "nll_after": 0.4,
                "brier_before": 0.3,
                "brier_after": 0.2,
                "ece_before": 0.1,
                "ece_after": 0.05,
            },
        },
        "training": {},
        "model_state": {},
        "optimizer_state": {},
    }


def _authenticity_metrics(
    macro_f1: float,
    balanced_accuracy: float,
    counterfeit_recall: float,
) -> AuthenticityMetrics:
    return AuthenticityMetrics(
        confusion_matrix=((5, 0), (0, 5)),
        macro_f1=macro_f1,
        balanced_accuracy=balanced_accuracy,
        per_class={
            "authentic": ClassMetrics(0.9, 0.9, 0.9, 5),
            "counterfeit": ClassMetrics(0.9, counterfeit_recall, 0.9, 5),
        },
        auroc=0.9,
        pr_auc=0.9,
        brier=0.1,
        ece=0.05,
        nll=0.2,
    )


def _bootstrap() -> dict[str, BootstrapInterval]:
    names = (
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
    return {name: BootstrapInterval(0.8, 0.9, 1.0, 10, 42) for name in names}


def _report(
    run_id: str,
    partition: str,
    candidate_hash: str,
    macro_f1: float,
    balanced_accuracy: float,
    counterfeit_recall: float,
) -> dict[str, Any]:
    calibrated = _authenticity_metrics(
        macro_f1,
        balanced_accuracy,
        counterfeit_recall,
    )
    return build_closed_set_evaluation_report(
        run_id=run_id,
        partition=partition,
        uncalibrated=_authenticity_metrics(0.7, 0.7, 0.7),
        calibrated=calibrated,
        bootstrap=_bootstrap(),
        fingerprints={
            "manifest": "a" * 64,
            "split": "b" * 64,
            "config": "c" * 64,
            "source_candidate": "d" * 64,
            "calibration": candidate_hash,
        },
        thresholds={
            "macro_f1": 0.8,
            "balanced_accuracy": 0.8,
            "counterfeit_recall": 0.8,
        },
    )


def make_request(
    root: Path,
    *,
    macro_f1: float,
    balanced_accuracy: float,
    counterfeit_recall: float,
) -> PromotionRequest:
    run_dir = root / "run-123"
    run_dir.mkdir()
    candidate = root / "candidate.calibrated.pth"
    torch.save(_checkpoint(run_dir.name), candidate)
    candidate_hash = _sha256(candidate)
    validation = run_dir / "validation" / "evaluation.json"
    locked = run_dir / "locked_test" / "evaluation.json"
    reports = (
        _report(
            run_dir.name,
            "validation",
            candidate_hash,
            macro_f1,
            balanced_accuracy,
            counterfeit_recall,
        ),
        _report(
            run_dir.name,
            "locked-test",
            candidate_hash,
            macro_f1,
            balanced_accuracy,
            counterfeit_recall,
        ),
    )
    _write_json(validation, reports[0])
    _write_json(locked, reports[1])
    _write_json(
        run_dir / "locked_test" / "reservation.json",
        {
            "schema_version": 1,
            "run_id": run_dir.name,
            "fingerprints": reports[1]["fingerprints"],
        },
    )
    target = root / "official.pth"
    target.write_bytes(b"old-model")
    Image.new("RGB", (4, 4), "green").save(root / "smoke.jpg", format="JPEG")
    return PromotionRequest(candidate, target, validation, locked, run_dir)


def passing_request(root: Path) -> PromotionRequest:
    return make_request(
        root,
        macro_f1=0.8,
        balanced_accuracy=0.8,
        counterfeit_recall=0.8,
    )


class _Result:
    def __init__(self, marker: str = "same") -> None:
        self.marker = marker

    def to_dict(self) -> dict[str, object]:
        return {"schema_version": 1, "marker": self.marker}


class _Service:
    def predict_experimental_closed_set(self, image: Path | bytes) -> _Result:
        assert isinstance(image, Path | bytes)
        return _Result()


@pytest.fixture(autouse=True)
def safe_service(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    if request.node.name == "test_promotion_uses_real_inference_loader_for_candidate_and_copy":
        return
    monkeypatch.setattr(
        promotion.InferenceService,
        "from_checkpoint",
        lambda *_args, **_kwargs: _Service(),
    )


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _rewrite_candidate(request: PromotionRequest, mutate: Any) -> None:
    state = torch.load(request.candidate, map_location="cpu", weights_only=True)
    mutate(state)
    torch.save(state, request.candidate)
    candidate_hash = _sha256(request.candidate)
    for path in (
        request.validation_report,
        request.locked_test_report,
        request.run_dir / "locked_test" / "reservation.json",
    ):
        value = _json(path)
        value["fingerprints"]["calibration"] = candidate_hash
        _write_json(path, value)


def test_promotion_rejects_below_threshold_without_touching_target(tmp_path: Path) -> None:
    request = make_request(
        tmp_path,
        macro_f1=0.79,
        balanced_accuracy=0.9,
        counterfeit_recall=0.9,
    )
    before = request.target.read_bytes()
    with pytest.raises(PromotionError, match="macro_f1"):
        promote_candidate(request, DEFAULT_PROMOTION_THRESHOLDS, tmp_path / "smoke.jpg")
    assert request.target.read_bytes() == before


def test_successful_promotion_records_old_and_new_hashes(tmp_path: Path) -> None:
    request = passing_request(tmp_path)
    old_hash, new_hash = _sha256(request.target), _sha256(request.candidate)
    record = promote_candidate(request, DEFAULT_PROMOTION_THRESHOLDS, tmp_path / "smoke.jpg")
    assert (record.old_sha256, record.new_sha256) == (old_hash, new_hash)
    assert _sha256(request.target) == new_hash
    assert record.run_id == request.run_dir.name
    assert record.promoted_at.endswith("Z")
    payload = _json(request.run_dir / "promotion.json")
    assert payload["schema_version"] == 1
    assert payload["old_sha256"] == old_hash
    assert payload["new_sha256"] == new_hash
    assert payload["thresholds"] == {
        "macro_f1": 0.8,
        "balanced_accuracy": 0.8,
        "counterfeit_recall": 0.8,
    }


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_promotion_rejects_nonfinite_metric(tmp_path: Path, value: float) -> None:
    request = make_request(
        tmp_path,
        macro_f1=value,
        balanced_accuracy=0.9,
        counterfeit_recall=0.9,
    )
    with pytest.raises(PromotionError, match="有限"):
        promote_candidate(request, DEFAULT_PROMOTION_THRESHOLDS, tmp_path / "smoke.jpg")
    assert request.target.read_bytes() == b"old-model"


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda report: report["fingerprints"].__setitem__("manifest", "9" * 64), "manifest"),
        (lambda report: report.__setitem__("run_id", "other-run"), "run_id"),
        (lambda report: report.__setitem__("schema_version", 99), "schema"),
        (lambda report: report.__setitem__("open_set_calibrated", True), "open_set"),
    ],
)
def test_promotion_rejects_bad_locked_report(tmp_path: Path, mutation: Any, match: str) -> None:
    request = passing_request(tmp_path)
    report = _json(request.locked_test_report)
    mutation(report)
    _write_json(request.locked_test_report, report)
    with pytest.raises(PromotionError, match=match):
        promote_candidate(request, DEFAULT_PROMOTION_THRESHOLDS, tmp_path / "smoke.jpg")
    assert request.target.read_bytes() == b"old-model"


@pytest.mark.parametrize("missing", ["reservation", "locked-report"])
def test_promotion_requires_locked_reservation_and_report(tmp_path: Path, missing: str) -> None:
    request = passing_request(tmp_path)
    path = (
        request.run_dir / "locked_test" / "reservation.json"
        if missing == "reservation"
        else request.locked_test_report
    )
    path.unlink()
    with pytest.raises(PromotionError, match="locked"):
        promote_candidate(request, DEFAULT_PROMOTION_THRESHOLDS, tmp_path / "smoke.jpg")
    assert request.target.read_bytes() == b"old-model"


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda state: state.__setitem__("schema_version", 99), "schema"),
        (
            lambda state: state["metadata"].__setitem__("experimental_closed_set", False),
            "experimental_closed_set",
        ),
        (
            lambda state: state["metadata"].__setitem__("open_set_calibrated", True),
            "open_set_calibrated",
        ),
        (
            lambda state: state["metadata"]["closed_set_calibration"].__setitem__(
                "checkpoint_fingerprint", "9" * 64
            ),
            "source_candidate",
        ),
    ],
)
def test_promotion_rejects_bad_candidate_contract(tmp_path: Path, mutate: Any, match: str) -> None:
    request = passing_request(tmp_path)
    _rewrite_candidate(request, mutate)
    with pytest.raises(PromotionError, match=match):
        promote_candidate(request, DEFAULT_PROMOTION_THRESHOLDS, tmp_path / "smoke.jpg")
    assert request.target.read_bytes() == b"old-model"


def test_promotion_rejects_changed_candidate_hash_and_same_path(tmp_path: Path) -> None:
    request = passing_request(tmp_path)
    request.candidate.write_bytes(request.candidate.read_bytes() + b"tampered")
    with pytest.raises(PromotionError, match="候选.*哈希"):
        promote_candidate(request, DEFAULT_PROMOTION_THRESHOLDS, tmp_path / "smoke.jpg")
    before = request.candidate.read_bytes()
    with pytest.raises(PromotionError, match="同一路径"):
        promote_candidate(
            replace(request, target=request.candidate),
            DEFAULT_PROMOTION_THRESHOLDS,
            tmp_path / "smoke.jpg",
        )
    assert request.candidate.read_bytes() == before


def test_promotion_rejects_nonstandard_report_path(tmp_path: Path) -> None:
    request = passing_request(tmp_path)
    moved = tmp_path / "validation.json"
    moved.write_bytes(request.validation_report.read_bytes())
    with pytest.raises(PromotionError, match="标准位置"):
        promote_candidate(
            replace(request, validation_report=moved),
            DEFAULT_PROMOTION_THRESHOLDS,
            tmp_path / "smoke.jpg",
        )


def test_promotion_rejects_load_or_smoke_failure_without_target_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = passing_request(tmp_path)
    monkeypatch.setattr(
        promotion.InferenceService,
        "from_checkpoint",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("load failed")),
    )
    with pytest.raises(PromotionError, match="安全加载"):
        promote_candidate(request, DEFAULT_PROMOTION_THRESHOLDS, tmp_path / "smoke.jpg")
    assert request.target.read_bytes() == b"old-model"


def test_promotion_rejects_path_bytes_smoke_disagreement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = passing_request(tmp_path)

    class DifferentService:
        def predict_experimental_closed_set(self, image: Path | bytes) -> _Result:
            return _Result("path" if isinstance(image, Path) else "bytes")

    monkeypatch.setattr(
        promotion.InferenceService,
        "from_checkpoint",
        lambda *_args, **_kwargs: DifferentService(),
    )
    with pytest.raises(PromotionError, match="Path.*bytes"):
        promote_candidate(request, DEFAULT_PROMOTION_THRESHOLDS, tmp_path / "smoke.jpg")
    assert request.target.read_bytes() == b"old-model"


def test_replace_failure_preserves_previous_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = passing_request(tmp_path)
    before = request.target.read_bytes()
    monkeypatch.setattr(
        promotion.os,
        "replace",
        lambda *_args: (_ for _ in ()).throw(OSError("simulated replace failure")),
    )
    with pytest.raises(PromotionError, match="replace"):
        promote_candidate(request, DEFAULT_PROMOTION_THRESHOLDS, tmp_path / "smoke.jpg")
    assert request.target.read_bytes() == before


def test_record_failure_after_replace_reports_recovery_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = passing_request(tmp_path)
    old_hash, new_hash = _sha256(request.target), _sha256(request.candidate)
    monkeypatch.setattr(
        promotion,
        "_write_promotion_record_atomic",
        lambda *_args: (_ for _ in ()).throw(OSError("disk full")),
    )
    with pytest.raises(PromotionError) as captured:
        promote_candidate(request, DEFAULT_PROMOTION_THRESHOLDS, tmp_path / "smoke.jpg")
    assert "恢复" in str(captured.value)
    assert old_hash in str(captured.value)
    assert new_hash in str(captured.value)
    assert _sha256(request.target) == new_hash


def test_promotion_rejects_nonfixed_thresholds(tmp_path: Path) -> None:
    request = passing_request(tmp_path)
    with pytest.raises(PromotionError, match="固定.*0.80"):
        promote_candidate(
            request,
            PromotionThresholds(0.79, 0.8, 0.8),
            tmp_path / "smoke.jpg",
        )
    assert request.target.read_bytes() == b"old-model"


def test_training_package_exports_promotion_contract() -> None:
    import herb_authentication.training as training

    assert training.DEFAULT_PROMOTION_THRESHOLDS == DEFAULT_PROMOTION_THRESHOLDS
    assert training.PromotionError is PromotionError
    assert training.PromotionRequest is PromotionRequest
    assert training.PromotionThresholds is PromotionThresholds
    assert training.promote_candidate is promote_candidate


def test_success_record_hashes_all_consumed_reports(tmp_path: Path) -> None:
    request = passing_request(tmp_path)
    expected = {
        "validation_report_sha256": _sha256(request.validation_report),
        "locked_report_sha256": _sha256(request.locked_test_report),
        "reservation_sha256": _sha256(request.run_dir / "locked_test" / "reservation.json"),
    }

    promote_candidate(request, DEFAULT_PROMOTION_THRESHOLDS, tmp_path / "smoke.jpg")

    payload = _json(request.run_dir / "promotion.json")
    assert {name: payload[name] for name in expected} == expected


@pytest.mark.parametrize("artifact", ["validation", "locked", "reservation", "promotion"])
def test_promotion_rejects_target_colliding_with_control_artifact(
    tmp_path: Path, artifact: str
) -> None:
    request = passing_request(tmp_path)
    paths = {
        "validation": request.validation_report,
        "locked": request.locked_test_report,
        "reservation": request.run_dir / "locked_test" / "reservation.json",
        "promotion": request.run_dir / "promotion.json",
    }
    target = paths[artifact]
    before = target.read_bytes() if target.exists() else None

    with pytest.raises(PromotionError, match="路径冲突"):
        promote_candidate(
            replace(request, target=target),
            DEFAULT_PROMOTION_THRESHOLDS,
            tmp_path / "smoke.jpg",
        )

    assert (target.read_bytes() if target.exists() else None) == before


def test_promotion_rejects_target_hardlink_alias_of_candidate(tmp_path: Path) -> None:
    request = passing_request(tmp_path)
    alias = tmp_path / "official-hardlink.pth"
    try:
        os.link(request.candidate, alias)
    except OSError as exc:
        pytest.skip(f"hardlinks unavailable: {exc}")
    before = alias.read_bytes()

    with pytest.raises(PromotionError, match="hardlink|别名"):
        promote_candidate(
            replace(request, target=alias),
            DEFAULT_PROMOTION_THRESHOLDS,
            tmp_path / "smoke.jpg",
        )

    assert alias.read_bytes() == before


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda report: report.pop("limitations"), "字段"),
        (
            lambda report: report["metrics"].__setitem__(
                "class_order", ["counterfeit", "authentic"]
            ),
            "class_order",
        ),
        (
            lambda report: report["metrics"]["uncalibrated_probability_quality"].pop("nll"),
            "uncalibrated",
        ),
        (lambda report: report["metrics"]["per_class"].pop("authentic"), "per_class"),
        (lambda report: report["metrics"]["bootstrap_95"].pop("nll"), "bootstrap"),
        (
            lambda report: report["metrics"]["bootstrap_95"]["macro_f1"].__setitem__(
                "upper", float("nan")
            ),
            "有限",
        ),
        (
            lambda report: report["thresholds_met"]["checks"].__setitem__("macro_f1", "true"),
            "门禁",
        ),
    ],
)
def test_promotion_strictly_rejects_malformed_task5_schema_v2(
    tmp_path: Path, mutation: Any, match: str
) -> None:
    request = passing_request(tmp_path)
    report = _json(request.locked_test_report)
    mutation(report)
    _write_json(request.locked_test_report, report)

    with pytest.raises(PromotionError, match=match):
        promote_candidate(request, DEFAULT_PROMOTION_THRESHOLDS, tmp_path / "smoke.jpg")

    assert request.target.read_bytes() == b"old-model"


def test_promotion_accepts_distinct_manifest_and_split_input_fingerprints(
    tmp_path: Path,
) -> None:
    request = passing_request(tmp_path)

    def align_split_input_fingerprint(state: dict[str, Any]) -> None:
        state["metadata"]["data_fingerprint"] = "e" * 64
        state["metadata"]["split_input_fingerprint"] = "e" * 64

    _rewrite_candidate(request, align_split_input_fingerprint)
    record = promote_candidate(
        request,
        DEFAULT_PROMOTION_THRESHOLDS,
        tmp_path / "smoke.jpg",
    )

    assert record.run_id == request.run_dir.name


@pytest.mark.parametrize("field", ["data_fingerprint", "split_fingerprint"])
def test_promotion_requires_top_level_candidate_provenance(tmp_path: Path, field: str) -> None:
    request = passing_request(tmp_path)
    _rewrite_candidate(
        request,
        lambda state: state["metadata"].__setitem__(field, "9" * 64),
    )

    with pytest.raises(PromotionError, match=field):
        promote_candidate(request, DEFAULT_PROMOTION_THRESHOLDS, tmp_path / "smoke.jpg")

    assert request.target.read_bytes() == b"old-model"


def test_promotion_rejects_target_changed_after_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = passing_request(tmp_path)
    original_copy = promotion.shutil.copyfile

    def drift_then_copy(source: Path, destination: Path) -> Path:
        request.target.write_bytes(b"concurrent-target")
        return original_copy(source, destination)

    monkeypatch.setattr(promotion.shutil, "copyfile", drift_then_copy)

    with pytest.raises(PromotionError, match="并发.*target"):
        promote_candidate(request, DEFAULT_PROMOTION_THRESHOLDS, tmp_path / "smoke.jpg")

    assert request.target.read_bytes() == b"concurrent-target"


def test_promotion_detects_target_corruption_during_audit_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = passing_request(tmp_path)
    original_writer = promotion._write_promotion_record_atomic

    def corrupt_after_write(path: Path, payload: Any) -> None:
        original_writer(path, payload)
        request.target.write_bytes(b"audit-corruption")

    monkeypatch.setattr(promotion, "_write_promotion_record_atomic", corrupt_after_write)

    with pytest.raises(PromotionError, match="target.*哈希"):
        promote_candidate(request, DEFAULT_PROMOTION_THRESHOLDS, tmp_path / "smoke.jpg")


def test_promotion_detects_report_drift_during_audit_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = passing_request(tmp_path)
    original_writer = promotion._write_promotion_record_atomic

    def drift_after_write(path: Path, payload: Any) -> None:
        original_writer(path, payload)
        request.validation_report.write_bytes(b"concurrent-report")

    monkeypatch.setattr(promotion, "_write_promotion_record_atomic", drift_after_write)

    with pytest.raises(PromotionError, match="validation.*并发"):
        promote_candidate(request, DEFAULT_PROMOTION_THRESHOLDS, tmp_path / "smoke.jpg")


class _TinyEncoder(nn.Module):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return images.mean(dim=(2, 3))


def test_promotion_uses_real_inference_loader_for_candidate_and_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = passing_request(tmp_path)
    model = MultiTaskHerbModel(_TinyEncoder(), 3, 1, 1, dropout=0.0)
    state = _checkpoint(request.run_dir.name)
    metadata = state["metadata"]
    metadata.update(
        {
            "backbone": "efficientnet_b0",
            "claimed_herb_classes": ("Drynaria_fortunei",),
            "actual_species_classes": ("Drynaria_fortunei",),
            "authenticity_classes": ("authentic", "counterfeit"),
            "image_size": 8,
            "preprocessing_version": "test-v1",
            "model_version": "tiny-promotion-v1",
            "dropout": 0.0,
            "calibration": CalibrationBundle.uncalibrated().to_dict(),
            "counterfeit_species_verified": False,
        }
    )
    state["model_state"] = model.state_dict()
    torch.save(state, request.candidate)
    candidate_hash = _sha256(request.candidate)
    for path in (
        request.validation_report,
        request.locked_test_report,
        request.run_dir / "locked_test" / "reservation.json",
    ):
        value = _json(path)
        value["fingerprints"]["calibration"] = candidate_hash
        _write_json(path, value)

    calls = 0

    def tiny_backbone(_backbone: str, *, pretrained: bool):
        nonlocal calls
        calls += 1
        assert pretrained is False
        return SimpleNamespace(encoder=_TinyEncoder(), feature_dim=3)

    monkeypatch.setattr(service_module, "build_torchvision_encoder", tiny_backbone)

    record = promote_candidate(
        request,
        DEFAULT_PROMOTION_THRESHOLDS,
        tmp_path / "smoke.jpg",
    )

    assert calls == 2
    assert _sha256(request.target) == record.new_sha256
