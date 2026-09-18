"""鉴权与权限：Argon2id、公开注册角色、CSRF、管理员建号、会话撤销。"""

from __future__ import annotations

from fastapi.testclient import TestClient

from drynaria_api.database import make_session_factory
from drynaria_api.models import UserRole
from drynaria_api.security import hash_password, lookup_user_by_email, new_user, verify_password
from helpers import create_case, csrf_headers, login, register


def test_argon2id_hash_roundtrip():
    password = "s3cret-password!"
    digest = hash_password(password)
    assert digest != password
    assert verify_password(password, digest) is True
    assert verify_password("wrong-password", digest) is False
    assert verify_password(password, "not-a-hash") is False


def test_register_creates_only_user_role_and_duplicate_rejected(client: TestClient):
    account = register(client)
    assert account["user"]["role"] == "user"

    duplicate = client.post(
        "/api/v1/auth/register",
        json={"email": account["email"], "password": "password123"},
    )
    assert duplicate.status_code == 409


def test_public_register_cannot_request_expert_or_admin(client: TestClient, app_settings):
    # 公开注册 schema 根本没有 role 字段；直接构造请求体忽略该字段并断言仍是 user
    response = client.post(
        "/api/v1/auth/register",
        json={"email": "hacker@example.com", "password": "password123", "role": "admin"},
    )
    assert response.status_code == 201
    assert response.json()["user"]["role"] == "user"
    with make_session_factory(app_settings.database_url)() as db:
        user = lookup_user_by_email(db, "hacker@example.com")
        assert user.role == UserRole.User


def test_admin_creates_expert_who_can_log_in(app, app_settings):
    # 用数据库直接创建管理员（等价于 drynaria-cli create-admin）
    with make_session_factory(app_settings.database_url)() as db:
        admin = new_user(
            db, email="root@example.com", password="admin-pass-123", role=UserRole.Admin, display_name="管理员"
        )
        db.commit()
        admin_id = admin.id
    client = TestClient(app)
    admin_account = login(client, "root@example.com", "admin-pass-123")
    assert admin_account["user"]["role"] == "admin"

    response = client.post(
        "/api/v1/admin/users",
        json={"email": "expert@example.com", "password": "password123", "role": "expert", "display_name": "专家"},
        headers=csrf_headers(admin_account["csrf_token"]),
    )
    assert response.status_code == 201
    assert response.json()["role"] == "expert"

    expert_login = client.post("/api/v1/auth/login", json={"email": "expert@example.com", "password": "password123"})
    assert expert_login.status_code == 200


def test_plain_user_cannot_access_admin_endpoints(app):
    client = TestClient(app)
    account = register(client)
    response = client.get("/api/v1/admin/reviews")
    assert response.status_code == 403
    response2 = client.post(
        "/api/v1/admin/users",
        json={"email": "x@example.com", "password": "password123", "role": "expert"},
        headers=csrf_headers(account["csrf_token"]),
    )
    assert response2.status_code == 403


def test_logout_revokes_session(client: TestClient):
    account = register(client)
    me = client.get("/api/v1/auth/me")
    assert me.status_code == 200
    # /me 只读回读当前会话 CSRF（不轮换），登出直接用 /me 返回的 token
    current_csrf = me.json()["csrf_token"]
    logout = client.post("/api/v1/auth/logout", headers=csrf_headers(current_csrf))
    assert logout.status_code == 200
    me_after = client.get("/api/v1/auth/me")
    assert me_after.status_code == 401


def test_me_is_read_only_and_repeated_calls_return_stable_csrf(client: TestClient):
    account = register(client)
    me1 = client.get("/api/v1/auth/me").json()
    me2 = client.get("/api/v1/auth/me").json()
    # /me 不轮换：路由守卫与 App 挂载各调一次 /me 不会让客户端 token 失配
    assert me1["csrf_token"] == me2["csrf_token"] == account["csrf_token"]


def test_refreshed_me_csrf_is_used_on_next_mutation(client: TestClient):
    account = register(client)
    me = client.get("/api/v1/auth/me")
    assert me.status_code == 200
    refreshed = me.json()["csrf_token"]
    # “刷新”后的 token 用于下一次状态变更必须成功
    response = client.post(
        "/api/v1/cases",
        json={"analysis_consent": True},
        headers=csrf_headers(refreshed),
    )
    assert response.status_code == 201
