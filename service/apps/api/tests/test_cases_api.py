"""案件 API 集成：上传视角/状态流转、analyze 幂等、CSRF/所有权门禁。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from fastapi.testclient import TestClient

from drynaria_api.cases import _idempotency_key
from drynaria_api.main import create_app
from drynaria_api.models import AnalysisJob, CaseStatus
from helpers import analyze_case, create_case, csrf_headers, login, make_image_bytes, register, upload_image


def test_create_case_defaults_to_draft_and_requires_consent(client: TestClient):
    account = register(client)
    # 拒绝显式取消分析授权
    response = client.post(
        "/api/v1/cases",
        json={"claimed_herb": "骨碎补", "analysis_consent": False},
        headers=csrf_headers(account["csrf_token"]),
    )
    assert response.status_code == 400

    case = create_case(client, account)
    assert case["status"] == "draft"
    assert case["claimed_herb"] == "骨碎补"
    assert case["id"]


def test_uploads_transition_whole_and_keep_optional_views(client: TestClient):
    account = register(client)
    case = create_case(client, account)
    case_id = case["id"]

    detail = upload_image(client, case_id, "detail", account)
    assert detail["image"]["view_type"] == "detail"
    assert detail["image"]["participates_in_inference"] is False
    # 只有 detail 时案件仍是 draft
    assert detail["case"]["status"] == "draft"

    cross = upload_image(client, case_id, "cross_section", account)
    assert cross["image"]["view_type"] == "cross_section"

    whole = upload_image(client, case_id, "whole", account)
    assert whole["image"]["participates_in_inference"] is True
    assert whole["case"]["status"] == "ready"

    fetched = client.get(f"/api/v1/cases/{case_id}")
    assert fetched.status_code == 200
    body = fetched.json()
    assert body["case"]["whole_image_count"] == 1
    assert body["case"]["detail_image_count"] == 1
    assert body["case"]["cross_section_image_count"] == 1
    views = {item["view_type"] for item in body["images"]}
    assert views == {"whole", "detail", "cross_section"}


def test_low_quality_whole_leads_to_needs_retake(client: TestClient):
    account = register(client)
    case = create_case(client, account)
    # 纯色图被质量策略标记 blur -> whole 不允许参与分析，进入 needs_retake
    flat = make_image_bytes(size=(64, 64), texture=False, color=(120, 120, 120))
    response = client.post(
        f"/api/v1/cases/{case['id']}/images",
        data={"view_type": "whole"},
        files={"file": ("flat.jpg", flat, "application/octet-stream")},
        headers=csrf_headers(account["csrf_token"]),
    )
    assert response.status_code == 201
    body = response.json()
    assert "blur" in body["image"]["quality_flags"]
    assert body["case"]["status"] == "needs_retake"

    # needs_retake 不能直接分析
    analyze = client.post(
        f"/api/v1/cases/{case['id']}/analyze", headers=csrf_headers(account["csrf_token"])
    )
    assert analyze.status_code == 409
    assert analyze.json()["detail"]["code"] == "case_not_ready"

    # 换上合格 whole 后 -> ready
    whole = upload_image(client, case["id"], "whole", account)
    assert whole["case"]["status"] == "ready"


def test_analyze_is_idempotent_and_returns_same_job(client: TestClient):
    account = register(client)
    case = create_case(client, account)
    upload_image(client, case["id"], "whole", account)

    first = analyze_case(client, case["id"], account)
    second = analyze_case(client, case["id"], account)

    assert first["analysis_job_id"] == second["analysis_job_id"]
    assert second["replayed"] is True
    assert first["job_status"] == "pending"
    assert first["case_status"] == "queued"

    detail = client.get(f"/api/v1/cases/{case['id']}").json()
    assert detail["analysis_job_id"] == first["analysis_job_id"]


def test_analyze_without_whole_is_rejected(client: TestClient):
    account = register(client)
    case = create_case(client, account)
    response = client.post(
        f"/api/v1/cases/{case['id']}/analyze", headers=csrf_headers(account["csrf_token"])
    )
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "whole_image_required"


def test_state_changing_routes_require_csrf(client: TestClient):
    account = register(client)
    response = client.post(
        "/api/v1/cases",
        json={"analysis_consent": True},
    )
    assert response.status_code == 403
    assert response.json()["detail"]["code"] == "csrf_invalid"

    case = create_case(client, account)
    upload = client.post(
        f"/api/v1/cases/{case['id']}/images",
        data={"view_type": "whole"},
        files={"file": ("p.jpg", make_image_bytes(), "application/octet-stream")},
    )
    assert upload.status_code == 403


def test_user_cannot_access_other_users_case(app: TestClient):
    alice_client = TestClient(app)
    bob_client = TestClient(app)
    alice = register(alice_client)
    bob = register(bob_client)

    case = create_case(alice_client, alice)
    case_id = case["id"]

    # bob 读取/上传/分析 alice 的案件都应 404（不泄露存在性）
    assert bob_client.get(f"/api/v1/cases/{case_id}").status_code == 404
    upload_response = bob_client.post(
        f"/api/v1/cases/{case_id}/images",
        data={"view_type": "whole"},
        files={"file": ("p.jpg", make_image_bytes(), "application/octet-stream")},
        headers=csrf_headers(bob["csrf_token"]),
    )
    assert upload_response.status_code == 404
    analyze_response = bob_client.post(
        f"/api/v1/cases/{case_id}/analyze", headers=csrf_headers(bob["csrf_token"])
    )
    assert analyze_response.status_code == 404


def test_whole_replacement_keeps_single_participant(client: TestClient):
    account = register(client)
    case = create_case(client, account)
    first = upload_image(client, case["id"], "whole", account)
    second = upload_image(client, case["id"], "whole", account)
    assert second["image"]["id"] != first["image"]["id"]
    detail = client.get(f"/api/v1/cases/{case['id']}").json()
    whole_images = [i for i in detail["images"] if i["view_type"] == "whole"]
    assert len(whole_images) == 1


def test_delete_whole_from_ready_reverts_to_draft(client: TestClient):
    account = register(client)
    case = create_case(client, account)
    whole = upload_image(client, case["id"], "whole", account)
    assert whole["case"]["status"] == "ready"

    response = client.delete(
        f"/api/v1/cases/{case['id']}/images/{whole['image']['id']}",
        headers=csrf_headers(account["csrf_token"]),
    )
    assert response.status_code == 200
    assert response.json()["case_status"] == "draft"


def test_concurrent_analyze_replays_same_job_without_error(app):
    """并发 analyze：select→insert 竞态必须恢复为幂等重放，绝不 500。"""
    alice = TestClient(app)
    account = register(alice)

    def race_round(_: int) -> tuple[list[int], set[str]]:
        case = create_case(alice, account)
        upload_image(alice, case["id"], "whole", account)
        # 第二个独立会话（同一账号），模拟不同请求并发打同一个案件
        other = TestClient(app)
        other_account = login(other, account["email"], account["password"])
        case_id = case["id"]

        def hit(which: int):
            if which == 0:
                response = alice.post(
                    f"/api/v1/cases/{case_id}/analyze",
                    headers=csrf_headers(account["csrf_token"]),
                )
            else:
                response = other.post(
                    f"/api/v1/cases/{case_id}/analyze",
                    headers=csrf_headers(other_account["csrf_token"]),
                )
            body = response.json() if response.status_code == 200 else None
            return response.status_code, body

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(hit, (0, 1)))
        codes = [code for code, _ in results]
        job_ids = {body["analysis_job_id"] for _, body in results if body is not None}
        return codes, job_ids

    with ThreadPoolExecutor(max_workers=2) as pool:
        rounds = list(pool.map(race_round, range(5)))

    for codes, job_ids in rounds:
        assert codes == [200, 200], f"并发 analyze 不应 500: {codes}"
        assert len(job_ids) == 1, f"同一案件并发 analyze 必须复用同一 job: {job_ids}"


def _tight_upload_client(app_settings):
    """构造上传上限收窄的独立 app（默认 fixture 的 20MB/4000 万像素上限不触发）。"""
    settings = app_settings.model_copy(
        update={"upload_max_bytes": 4096, "upload_max_pixels": 4096}
    )
    application = create_app(settings)
    client = TestClient(application)
    return application, client


def test_upload_rejects_over_configured_byte_limit(app_settings):
    application, client = _tight_upload_client(app_settings)
    try:
        account = register(client)
        case = create_case(client, account)
        payload = make_image_bytes(size=(320, 240))  # 纹理 JPEG 远超 4096 字节
        assert len(payload) > 4096
        response = client.post(
            f"/api/v1/cases/{case['id']}/images",
            data={"view_type": "whole"},
            files={"file": ("photo.jpg", payload, "application/octet-stream")},
            headers=csrf_headers(account["csrf_token"]),
        )
        assert response.status_code == 400
        assert response.json()["detail"]["code"] == "file_too_large"
    finally:
        application.state.engine.dispose()


def test_upload_rejects_over_configured_pixel_limit(app_settings):
    application, client = _tight_upload_client(app_settings)
    try:
        account = register(client)
        case = create_case(client, account)
        # 90x90=8100 像素 < 20MB，但 > 配置的 4096 像素上限
        payload = make_image_bytes(size=(90, 90), image_format="PNG", texture=False)
        response = client.post(
            f"/api/v1/cases/{case['id']}/images",
            data={"view_type": "whole"},
            files={"file": ("photo.png", payload, "application/octet-stream")},
            headers=csrf_headers(account["csrf_token"]),
        )
        assert response.status_code == 400
        assert response.json()["detail"]["code"] == "too_many_pixels"
    finally:
        application.state.engine.dispose()
