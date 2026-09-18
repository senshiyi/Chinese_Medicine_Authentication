"""pytest 共享夹具：为每个测试建独立 SQLite 文件库 + 本地磁盘对象存储。"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from drynaria_api.database import make_session_factory
from drynaria_api.main import create_app
from drynaria_api.settings import Settings


@pytest.fixture()
def app_settings(tmp_path) -> Settings:
    return Settings(
        environment="test",
        database_url=f"sqlite:///{(tmp_path / 'test.db').as_posix()}",
        storage_backend="local",
        storage_local_dir=tmp_path / "storage",
        celery_broker_url="",
        celery_result_backend="",
        auto_create_tables=True,
        allowed_origins=["http://testserver"],
        cookie_secure=False,
        model_checkpoint_path=str(tmp_path / "nonexistent.pth"),
        # 默认关闭范围门控，保留既有闭集映射语义；范围门控测试显式开启。
        scope_enforcement=False,
    )


@pytest.fixture()
def app(app_settings):
    application = create_app(app_settings)
    yield application
    application.state.engine.dispose()


@pytest.fixture()
def client(app) -> TestClient:
    return TestClient(app)


@pytest.fixture()
def session_factory(app_settings):
    return make_session_factory(app_settings.database_url)
