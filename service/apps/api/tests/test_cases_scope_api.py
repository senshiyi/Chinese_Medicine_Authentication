"""API 范围契约测试：范围门控开启时历史/实验闭集预测不作为有效结论呈现。"""

from __future__ import annotations

from uuid import UUID

from fastapi.testclient import TestClient

from drynaria_api.cases import _idempotency_key
from drynaria_api.database import make_session_factory
from drynaria_api.main import create_app
from drynaria_api.models import (
    AnalysisJob,
    AnalysisStatus,
    Case,
    CaseImage,
    CaseStatus,
    Prediction,
    ResultType,
    ViewType,
)
from drynaria_api.settings import Settings
from helpers import make_image_bytes, register, upload_image, create_case, csrf_headers


def _make_app(tmp_path, *, scope_enforcement: bool) -> tuple[TestClient, Settings]:
    settings = Settings(
        environment="test",
        database_url=f"sqlite:///{(tmp_path / 'case.db').as_posix()}",
        storage_backend="local",
        storage_local_dir=tmp_path / "storage",
        celery_broker_url="",
        celery_result_backend="",
        auto_create_tables=True,
        allowed_origins=["http://testserver"],
        cookie_secure=False,
        model_checkpoint_path=str(tmp_path / "nonexistent.pth"),
        scope_enforcement=scope_enforcement,
        scope_artifact_path="",
    )
    app = create_app(settings)
    return TestClient(app), settings


def _seed_prediction(
    settings: Settings,
    case_id,
    whole_image_id,
    *,
    scope_decision: dict | None,
) -> UUID:
    """直接落一条不可变预测（模拟历史闭集预测），并把案件指向它。"""
    factory = make_session_factory(settings.database_url)
    whole_uuid = UUID(str(whole_image_id))
    with factory() as db:
        case = db.get(Case, UUID(str(case_id)))
        job = AnalysisJob(
            case_id=case.id,
            idempotency_key="scope-api-test-" + str(case.id)[:8],
            whole_image_id=whole_uuid,
            model_version_hint="closed-set-v1",
            status=AnalysisStatus.Succeeded,
        )
        db.add(job)
        db.flush()
        prediction = Prediction(
            case_id=case.id,
            analysis_job_id=job.id,
            image_id=whole_uuid,
            view_type=ViewType.Whole,
            model_version="closed-set-v1",
            strategy_version="stage-a-strategy-v1",
            experimental_closed_set=True,
            open_set_calibrated=False,
            authenticity_label="authentic",
            confidence=0.99,
            probabilities={"authentic": 0.99, "counterfeit": 0.01},
            actual_species_verified=False,
            message="此药品为正品，有99.0%的可能该药材为正品。",
            scope_warning="仅适用于本数据集定义的骨碎补正品/伪品范围",
            scope_decision=scope_decision,
            model_fingerprint="a" * 64,
            data_fingerprint="b" * 64,
            split_fingerprint="c" * 64,
            payload={"experimental_closed_set": True, "open_set_calibrated": False},
        )
        db.add(prediction)
        db.flush()
        case.current_prediction_id = prediction.id
        case.status = CaseStatus.Completed
        case.result_type = ResultType.AuthenticLikely
        db.commit()
        return prediction.id


def test_old_high_confidence_prediction_not_presented_as_valid_when_scope_absent(tmp_path):
    """历史（未记录范围决策）高置信度预测：范围门控开启时不得作为有效结论呈现。"""
    client, settings = _make_app(tmp_path, scope_enforcement=True)
    account = register(client)
    case = create_case(client, account)
    whole = upload_image(client, case["id"], "whole", account)["image"]
    _seed_prediction(settings, case["id"], whole["id"], scope_decision=None)

    detail = client.get(f"/api/v1/cases/{case['id']}").json()
    assert detail["scope_enforced"] is True
    assert detail["result_verdict_valid"] is False
    assert detail["result_display"] == "无法判断"
    assert detail["scope"]["scope"] == "unavailable"
    # 原始预测仍保留（不可变），但不作为有效结论呈现
    assert detail["prediction"]["authenticity_label"] == "authentic"
    assert detail["prediction"]["scope_decision"] is None
    assert detail["prediction"]["scope_enforced"] is True
    assert "此药品为正品" not in detail["prediction_message"]
    assert "此药品为正品" not in detail["prediction"]["message"]


def test_old_high_confidence_prediction_valid_when_scope_disabled(tmp_path):
    """范围门控关闭时保留旧闭集呈现（实验性、无范围断言）。"""
    client, settings = _make_app(tmp_path, scope_enforcement=False)
    account = register(client)
    case = create_case(client, account)
    whole = upload_image(client, case["id"], "whole", account)["image"]
    _seed_prediction(settings, case["id"], whole["id"], scope_decision=None)

    detail = client.get(f"/api/v1/cases/{case['id']}").json()
    assert detail["scope_enforced"] is False
    assert detail["result_verdict_valid"] is True
    assert detail["result_display"] == "正品特征吻合度高"
    assert detail["scope"] is None


def test_accepted_scope_prediction_presented_as_valid(tmp_path):
    client, settings = _make_app(tmp_path, scope_enforcement=True)
    account = register(client)
    case = create_case(client, account)
    whole = upload_image(client, case["id"], "whole", account)["image"]
    _seed_prediction(
        settings,
        case["id"],
        whole["id"],
        scope_decision={
            "scope": "accepted",
            "reason": "scope_ok",
            "enforced": True,
            "version": "scope-logreg-v1",
            "score": 0.9,
            "threshold": 0.5,
            "artifact_version": "scope-logreg-v1",
            "feature_dim": 3,
        },
    )
    detail = client.get(f"/api/v1/cases/{case['id']}").json()
    assert detail["result_verdict_valid"] is True
    assert detail["result_display"] == "正品特征吻合度高"
    assert detail["scope"]["scope"] == "accepted"


def test_summary_includes_scope_information(tmp_path):
    client, settings = _make_app(tmp_path, scope_enforcement=True)
    account = register(client)
    case = create_case(client, account)
    whole = upload_image(client, case["id"], "whole", account)["image"]
    _seed_prediction(settings, case["id"], whole["id"], scope_decision=None)

    rows = client.get("/api/v1/cases").json()
    assert len(rows) == 1
    assert rows[0]["scope_enforced"] is True
    assert rows[0]["result_verdict_valid"] is False
    assert rows[0]["result_type"] == "inconclusive"
    assert rows[0]["scope_decision"]["scope"] == "unavailable"


def test_idempotency_key_differs_by_scope_policy():
    """幂等键纳入范围策略：避免把 pre-gate 结果当作可重放的新结果。"""
    from types import SimpleNamespace

    from drynaria_api.scope.gate import scope_fingerprint

    case_id = UUID("11111111-1111-1111-1111-111111111111")
    whole_sha = "a" * 64
    version = "closed-set-v1"
    disabled = scope_fingerprint(SimpleNamespace(scope_enforcement=False, scope_artifact_path=""))
    unavailable = scope_fingerprint(SimpleNamespace(scope_enforcement=True, scope_artifact_path=""))
    assert disabled != unavailable
    key_disabled = _idempotency_key(case_id, whole_sha, version, disabled)
    key_unavailable = _idempotency_key(case_id, whole_sha, version, unavailable)
    assert key_disabled != key_unavailable
