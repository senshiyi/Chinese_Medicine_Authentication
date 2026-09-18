"""Worker 范围门控集成测试：默认 unavailable 门控 → inconclusive；accepted → conclusion；
rejected → inconclusive；特征失败 → unavailable。"""

from __future__ import annotations

from uuid import UUID

from fastapi.testclient import TestClient

from drynaria_api.database import make_session_factory
from drynaria_api.main import create_app
from drynaria_api.models import (
    Case,
    CaseStatus,
    Prediction,
    ResultType,
)
from drynaria_api.scope.artifact import build_artifact
from drynaria_api.scope.gate import (
    SCOPE_ACCEPTED,
    SCOPE_REJECTED,
    SCOPE_UNAVAILABLE,
    ScopeGate,
)
from drynaria_api.settings import Settings
from drynaria_api.storage import LocalStorageBackend
from drynaria_api.worker import compute_case_outcome, run_analysis_job
from helpers import analyze_case, create_case, register, upload_image

CKPT = "a" * 64
FEATURE_DIM = 3


def _payload(label: str = "authentic", confidence: float = 0.99) -> dict:
    return {
        "schema_version": 1,
        "model_version": "closed-set-v1",
        "experimental_closed_set": True,
        "open_set_calibrated": False,
        "scope_warning": "仅适用于本数据集定义的骨碎补正品/伪品范围",
        "message": "此药品为正品，有99.0%的可能该药材为正品。",
        "authenticity": {"label": label, "confidence": confidence},
        "probabilities": {"authentic": confidence, "counterfeit": round(1 - confidence, 6)},
        "actual_species_verified": False,
        "model_fingerprint": "a" * 64,
        "data_fingerprint": "b" * 64,
        "split_fingerprint": "c" * 64,
    }


def _artifact(threshold: float = 0.5):
    # 特征 [0,0,0] -> z=0 -> p=0.5（accept）；[-2,0,0] -> z=-2 -> p≈0.119（reject）
    return build_artifact(
        artifact_version="scope-logreg-v1",
        checkpoint_sha256=CKPT,
        feature_dim=FEATURE_DIM,
        scaler_mean=[0.0, 0.0, 0.0],
        scaler_scale=[1.0, 1.0, 1.0],
        classifier_coef=[1.0, 0.0, 0.0],
        classifier_intercept=0.0,
        threshold=threshold,
        evaluation={"test": {"n": 1}},
        approved=True,
    )


def _in_memory_gate() -> ScopeGate:
    return ScopeGate(enforced=True, artifact=_artifact(), artifact_path="in-memory")


def _enforced_app(tmp_path) -> tuple[TestClient, Settings]:
    settings = Settings(
        environment="test",
        database_url=f"sqlite:///{(tmp_path / 'w.db').as_posix()}",
        storage_backend="local",
        storage_local_dir=tmp_path / "storage",
        celery_broker_url="",
        celery_result_backend="",
        auto_create_tables=True,
        allowed_origins=["http://testserver"],
        cookie_secure=False,
        model_checkpoint_path=str(tmp_path / "nonexistent.pth"),
        scope_enforcement=True,
        scope_artifact_path="",
    )
    return TestClient(create_app(settings)), settings


def _build_case_with_images(client: TestClient, account: dict) -> dict:
    case = create_case(client, account)
    upload_image(client, case["id"], "detail", account)
    upload_image(client, case["id"], "cross_section", account)
    upload_image(client, case["id"], "whole", account)
    result = analyze_case(client, case["id"], account)
    return {"case": case, "job_id": result["analysis_job_id"]}


def _run(job_id: str, settings: Settings, predict, *, scope_gate=None, feature_extract=None):
    storage = LocalStorageBackend(settings.storage_local_dir)
    return run_analysis_job(
        job_id,
        session_factory=make_session_factory(settings.database_url),
        storage=storage,
        predict=predict,
        settings=settings,
        scope_gate=scope_gate,
        feature_extract=feature_extract,
    )


def test_worker_unavailable_gate_yields_inconclusive(tmp_path):
    client, settings = _enforced_app(tmp_path)
    account = register(client)
    built = _build_case_with_images(client, account)
    outcome = _run(built["job_id"], settings, lambda _b: _payload("authentic", 0.99))
    assert outcome["status"] == "succeeded"
    assert outcome["case_status"] == "inconclusive"
    with make_session_factory(settings.database_url)() as db:
        case = db.get(Case, UUID(built["case"]["id"]))
        assert case.status == CaseStatus.Inconclusive
        assert case.result_type == ResultType.Inconclusive
        prediction = db.get(Prediction, case.current_prediction_id)
        assert prediction is not None
        # 原始闭集预测仍被保留（不可变），但范围证据缺失
        assert prediction.authenticity_label == "authentic"
        assert prediction.scope_decision["scope"] == SCOPE_UNAVAILABLE
    detail = client.get(f"/api/v1/cases/{built['case']['id']}").json()
    assert detail["result_display"] == "无法判断"
    assert detail["result_verdict_valid"] is False
    assert detail["scope"]["scope"] == SCOPE_UNAVAILABLE
    assert detail["scope"]["enforced"] is True
    assert detail["prediction"]["scope_decision"]["scope"] == SCOPE_UNAVAILABLE


def test_worker_scope_accepted_yields_completed(tmp_path):
    client, settings = _enforced_app(tmp_path)
    account = register(client)
    built = _build_case_with_images(client, account)
    outcome = _run(
        built["job_id"],
        settings,
        lambda _b: _payload("authentic", 0.99),
        scope_gate=_in_memory_gate(),
        feature_extract=lambda _b: [0.0, 0.0, 0.0],
    )
    assert outcome["status"] == "succeeded"
    assert outcome["case_status"] == "completed"
    with make_session_factory(settings.database_url)() as db:
        case = db.get(Case, UUID(built["case"]["id"]))
        assert case.status == CaseStatus.Completed
        assert case.result_type == ResultType.AuthenticLikely
        prediction = db.get(Prediction, case.current_prediction_id)
        assert prediction.scope_decision["scope"] == SCOPE_ACCEPTED
    detail = client.get(f"/api/v1/cases/{built['case']['id']}").json()
    assert detail["result_display"] == "正品特征吻合度高"
    assert detail["result_verdict_valid"] is True
    assert detail["scope"]["scope"] == SCOPE_ACCEPTED


def test_worker_scope_rejected_yields_inconclusive(tmp_path):
    client, settings = _enforced_app(tmp_path)
    account = register(client)
    built = _build_case_with_images(client, account)
    outcome = _run(
        built["job_id"],
        settings,
        lambda _b: _payload("authentic", 0.99),
        scope_gate=_in_memory_gate(),
        feature_extract=lambda _b: [-2.0, 0.0, 0.0],
    )
    assert outcome["status"] == "succeeded"
    assert outcome["case_status"] == "inconclusive"
    with make_session_factory(settings.database_url)() as db:
        case = db.get(Case, UUID(built["case"]["id"]))
        assert case.result_type == ResultType.Inconclusive
        prediction = db.get(Prediction, case.current_prediction_id)
        assert prediction.scope_decision["scope"] == SCOPE_REJECTED
    detail = client.get(f"/api/v1/cases/{built['case']['id']}").json()
    assert detail["result_display"] == "无法判断"
    assert detail["result_verdict_valid"] is False
    assert detail["scope"]["scope"] == SCOPE_REJECTED


def test_worker_scope_feature_error_falls_back_to_unavailable(tmp_path):
    client, settings = _enforced_app(tmp_path)
    account = register(client)
    built = _build_case_with_images(client, account)
    outcome = _run(
        built["job_id"],
        settings,
        lambda _b: _payload("authentic", 0.99),
        scope_gate=_in_memory_gate(),
        feature_extract=lambda _b: (_ for _ in ()).throw(RuntimeError("encoder down")),
    )
    assert outcome["status"] == "succeeded"
    assert outcome["case_status"] == "inconclusive"


def test_compute_case_outcome_without_scope_preserves_old_behavior():
    assert compute_case_outcome(_payload("authentic"), 0.0) == (
        CaseStatus.Completed,
        ResultType.AuthenticLikely,
    )


def test_compute_case_outcome_rejected_is_inconclusive():
    gate = _in_memory_gate()
    rejected = gate.decide([-2.0, 0.0, 0.0])
    assert rejected.scope == SCOPE_REJECTED
    assert compute_case_outcome(_payload("authentic"), 0.0, rejected) == (
        CaseStatus.Inconclusive,
        ResultType.Inconclusive,
    )
