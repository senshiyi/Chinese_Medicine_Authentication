"""FastAPI 应用入口：CORS、请求 ID、健康检查、路由与共享资源装配。

约定：API 进程不加载模型、不在路由中拼接领域结论、不重算概率。
"""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import sessionmaker
from starlette.middleware.base import BaseHTTPMiddleware

from .admin_reviews import router as admin_reviews_router
from .auth import admin_router as admin_users_router
from .auth import router as auth_router
from .cases import router as cases_router
from .database import make_engine
from .models import Base
from .settings import Settings, get_settings
from .storage import build_storage


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):  # noqa: ANN001
        request_id = request.headers.get("X-Request-Id") or uuid.uuid4().hex
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-Id"] = request_id
        return response


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        docs_url="/api/v1/docs",
        openapi_url="/api/v1/openapi.json",
        redoc_url=None,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type", "X-CSRF-Token", "X-Request-Id"],
        expose_headers=["X-Request-Id"],
    )
    app.add_middleware(RequestContextMiddleware)

    # 共享资源：engine / session / storage 挂在 app.state，避免模块级全局状态
    engine = make_engine(settings.database_url)
    if settings.auto_create_tables:
        Base.metadata.create_all(engine)
    app.state.engine = engine
    app.state.session_factory = sessionmaker(bind=engine, expire_on_commit=False)
    app.state.storage = build_storage(settings)
    app.state.settings = settings

    @app.get("/api/v1/health")
    def health() -> dict:
        return {
            "service": settings.app_name,
            "stage": "A",
            "experimental_closed_set": True,
            "open_set_calibrated": False,
        }

    @app.get("/api/v1/status")
    def service_status() -> dict:
        """供前端/运维查看实验范围；API 不加载模型，只做路径存在性检查。"""
        checkpoint = Path(settings.model_checkpoint_path)
        project = Path(settings.inference_project_path)
        artifact_path = Path(settings.scope_artifact_path) if settings.scope_artifact_path else None
        return {
            "stage": "A",
            "experimental_closed_set": True,
            "open_set_calibrated": False,
            "model_checkpoint_available": checkpoint.is_file(),
            "inference_project_available": project.is_dir(),
            "model_version_hint": settings.model_version_hint,
            "strategy_version": settings.strategy_version,
            "scope_enforcement": bool(settings.scope_enforcement),
            "scope_artifact_available": bool(
                artifact_path and artifact_path.is_file()
            ),
        }

    app.include_router(auth_router)
    app.include_router(admin_users_router)
    app.include_router(cases_router)
    app.include_router(admin_reviews_router)
    return app


app = create_app()
