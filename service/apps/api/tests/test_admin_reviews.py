"""专家复核 API：角色门禁、队列、追加版本决定、预测不可变。"""

from __future__ import annotations

import uuid
from uuid import UUID

from fastapi.testclient import TestClient
from sqlalchemy import select

from drynaria_api.database import make_session_factory
from drynaria_api.models import Prediction, UserRole
from drynaria_api.security import new_user
from drynaria_api.settings import Settings
from drynaria_api.worker import run_analysis_job
from helpers import analyze_case, create_case, csrf_headers, login, register, upload_image


def _payload_authentic() -> dict:
    return {
        "schema_version": 1,
        "model_version": "closed-set-v1",
        "experimental_closed_set": True,
        "open_set_calibrated": False,
        "scope_warning": "仅适用于本数据集定义的骨碎补正品/伪品范围",
        "message": "此药品为正品，有91.0%的可能该药材为正品。",
        "authenticity": {"label": "authentic", "confidence": 0.91},
        "probabilities": {"authentic": 0.91, "counterfeit": 0.09},
        "actual_species_verified": False,
        "model_fingerprint": "a" * 64,
        "data_fingerprint": "b" * 64,
        "split_fingerprint": "c" * 64,
    }


def _complete_case(client: TestClient, app_settings: Settings, account: dict) -> str:
    case = create_case(client, account)
    upload_image(client, case["id"], "detail", account)
    upload_image(client, case["id"], "whole", account)
    result = analyze_case(client, case["id"], account)
    from drynaria_api.storage import LocalStorageBackend

    run_analysis_job(
        result["analysis_job_id"],
        session_factory=make_session_factory(app_settings.database_url),
        storage=LocalStorageBackend(app_settings.storage_local_dir),
        predict=lambda _b: _payload_authentic(),
        settings=app_settings,
    )
    return case["id"]


def _create_expert_client(app, app_settings) -> TestClient:
    with make_session_factory(app_settings.database_url)() as db:
        expert = new_user(
            db,
            email="expert@example.com",
            password="expert-pass-123",
            role=UserRole.Expert,
            display_name="专家",
        )
        db.commit()
    client = TestClient(app)
    expert_account = login(client, "expert@example.com", "expert-pass-123")
    return client, expert_account


def _decision_body(**overrides) -> dict:
    body = {
        "actual_species": "槲蕨",
        "authenticity_class": "authentic",
        "processing_status": "compliant",
        "verification_method": "专家",
        "basis": "鳞片与叶痕形态符合",
    }
    body.update(overrides)
    return body


def test_review_workflow_appends_versions_without_touching_prediction(
    app, app_settings: Settings, client: TestClient
):
    account = register(client)
    case_id = _complete_case(client, app_settings, account)

    request_response = client.post(
        f"/api/v1/cases/{case_id}/review-requests",
        json={"reason": "请专家确认"},
        headers=csrf_headers(account["csrf_token"]),
    )
    assert request_response.status_code == 201
    review_id = request_response.json()["id"]

    expert_client, expert_account = _create_expert_client(app, app_settings)

    queue = expert_client.get("/api/v1/admin/reviews?status=requested")
    assert queue.status_code == 200
    assert any(item["id"] == str(review_id) for item in queue.json())

    detail = expert_client.get(f"/api/v1/admin/reviews/{review_id}")
    assert detail.status_code == 200
    body = detail.json()
    assert body["model_prediction"] is not None
    assert len(body["images"]) == 2
    assert body["case"]["id"] == case_id

    # 记录复核前的不可变预测
    with make_session_factory(app_settings.database_url)() as db:
        case_uuid = UUID(case_id)
        prediction_before = db.scalar(
            select(Prediction).where(Prediction.case_id == case_uuid)
        )
        payload_before = dict(prediction_before.payload)

    first = expert_client.post(
        f"/api/v1/admin/reviews/{review_id}/decisions",
        json=_decision_body(keep_open=True),
        headers=csrf_headers(expert_account["csrf_token"]),
    )
    assert first.status_code == 201
    assert first.json()["version"] == 1

    second = expert_client.post(
        f"/api/v1/admin/reviews/{review_id}/decisions",
        json=_decision_body(authenticity_class="unknown", processing_status="unknown"),
        headers=csrf_headers(expert_account["csrf_token"]),
    )
    assert second.status_code == 201
    assert second.json()["version"] == 2

    # 追加版本没有修改任何历史模型预测
    with make_session_factory(app_settings.database_url)() as db:
        prediction_after = db.scalar(
            select(Prediction).where(Prediction.case_id == UUID(case_id))
        )
        assert prediction_after.payload == payload_before

    detail_after = expert_client.get(f"/api/v1/admin/reviews/{review_id}").json()
    assert len(detail_after["expert_reviews"]) == 2
    assert [r["version"] for r in detail_after["expert_reviews"]] == [1, 2]
    assert detail_after["review_request"]["status"] == "completed"

    completed_queue = expert_client.get("/api/v1/admin/reviews?status=completed")
    assert any(item["id"] == str(review_id) for item in completed_queue.json())


def test_expert_decision_requires_fields_and_reviewer_role(
    app, app_settings: Settings, client: TestClient
):
    account = register(client)
    case_id = _complete_case(client, app_settings, account)
    request_response = client.post(
        f"/api/v1/cases/{case_id}/review-requests",
        json={"reason": "复核"},
        headers=csrf_headers(account["csrf_token"]),
    )
    review_id = request_response.json()["id"]

    # 普通用户不能提交专家决定
    denied = client.post(
        f"/api/v1/admin/reviews/{review_id}/decisions",
        json=_decision_body(),
        headers=csrf_headers(account["csrf_token"]),
    )
    assert denied.status_code == 403

    expert_client, expert_account = _create_expert_client(app, app_settings)

    missing_method = expert_client.post(
        f"/api/v1/admin/reviews/{review_id}/decisions",
        json=_decision_body(verification_method=""),
        headers=csrf_headers(expert_account["csrf_token"]),
    )
    assert missing_method.status_code == 422

    missing_class = expert_client.post(
        f"/api/v1/admin/reviews/{review_id}/decisions",
        json={k: v for k, v in _decision_body().items() if k != "authenticity_class"},
        headers=csrf_headers(expert_account["csrf_token"]),
    )
    assert missing_class.status_code == 422


def test_model_versions_registered_after_worker_run(
    app, app_settings: Settings, client: TestClient
):
    account = register(client)
    _complete_case(client, app_settings, account)
    expert_client, expert_account = _create_expert_client(app, app_settings)
    versions = expert_client.get("/api/v1/admin/model-versions")
    assert versions.status_code == 200
    assert any(item["model_version"] == "closed-set-v1" for item in versions.json())
    entry = next(item for item in versions.json() if item["model_version"] == "closed-set-v1")
    assert entry["experimental_closed_set"] is True
    assert entry["open_set_calibrated"] is False
