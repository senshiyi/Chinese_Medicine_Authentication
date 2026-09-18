"""Settings 环境变量解码：ALLOWED_ORIGINS 必须可靠支持 Compose 的逗号分隔格式。"""

from __future__ import annotations

from fastapi.testclient import TestClient

from drynaria_api.main import create_app
from drynaria_api.settings import Settings


def test_allowed_origins_decodes_comma_separated_env(monkeypatch):
    # Compose 实际使用的格式：ALLOWED_ORIGINS=http://localhost:8080,http://127.0.0.1:8080
    monkeypatch.setenv(
        "ALLOWED_ORIGINS", "http://localhost:8080,http://127.0.0.1:8080"
    )
    settings = Settings(_env_file=None)
    assert settings.allowed_origins == [
        "http://localhost:8080",
        "http://127.0.0.1:8080",
    ]


def test_allowed_origins_json_list_env_still_supported(monkeypatch):
    # 保留 JSON 数组字符串写法（旧格式兼容）
    monkeypatch.setenv(
        "ALLOWED_ORIGINS", '["http://a.example.com", "http://b.example.com"]'
    )
    settings = Settings(_env_file=None)
    assert settings.allowed_origins == ["http://a.example.com", "http://b.example.com"]


def test_allowed_origins_default_and_init_kwarg(monkeypatch):
    monkeypatch.delenv("ALLOWED_ORIGINS", raising=False)
    assert Settings(_env_file=None).allowed_origins == ["http://localhost:5173"]
    injected = Settings(_env_file=None, allowed_origins=["http://testserver"])
    assert injected.allowed_origins == ["http://testserver"]


def test_compose_origins_env_builds_app_and_health_ok(monkeypatch, tmp_path):
    # 复现原始崩溃点：Compose 同款 env 曾经在 Settings 构造阶段抛 SettingsError
    monkeypatch.setenv(
        "ALLOWED_ORIGINS", "http://localhost:8080,http://127.0.0.1:8080"
    )
    settings = Settings(
        _env_file=None,
        environment="test",
        database_url=f"sqlite:///{(tmp_path / 's.db').as_posix()}",
        storage_backend="local",
        storage_local_dir=tmp_path / "storage",
        celery_broker_url="",
        celery_result_backend="",
        auto_create_tables=True,
        cookie_secure=False,
    )
    app = create_app(settings)
    try:
        with TestClient(app) as client:
            assert settings.allowed_origins == [
                "http://localhost:8080",
                "http://127.0.0.1:8080",
            ]
            assert client.get("/api/v1/health").status_code == 200
    finally:
        app.state.engine.dispose()
