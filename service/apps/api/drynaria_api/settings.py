"""环境变量配置（pydantic-settings）。

密钥只通过环境变量/秘密系统注入；仓库内不保存真实值。模型路径默认指向
只读的原闭集实验项目（`Chinese_Medicine_Authentication`），禁止把模型权重
复制进本仓库。
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import Request
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INFERENCE_PROJECT = PROJECT_ROOT.parent / "项目" / "Files"
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT.parent / "项目" / "Models" / "best_multitask_model.calibrated.pth"
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    app_name: str = "drynaria-online-authentication-v2"
    environment: str = "dev"  # dev | test | prod

    # ---- 数据库 / 队列 / 存储 ----
    database_url: str = (
        "postgresql+psycopg://drynaria:drynaria@localhost:5432/drynaria_stage_a"
    )
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/0"

    storage_backend: str = "auto"  # auto | local | s3
    storage_local_dir: Path = PROJECT_ROOT / "var" / "local-storage"
    # 默认留空 -> auto 走本地磁盘；部署（Compose/.env）显式配置 S3 端点
    s3_endpoint_url: str = ""
    s3_bucket: str = "drynaria-stage-a"
    s3_access_key: str = "drynaria-minio"
    s3_secret_key: str = ""

    # ---- 推理（仅 Worker 读取；API 不加载模型）----
    inference_project_path: Path = DEFAULT_INFERENCE_PROJECT
    model_checkpoint_path: Path = DEFAULT_CHECKPOINT
    model_version_hint: str = "closed-set-v1"
    strategy_version: str = "stage-a-strategy-v1"
    closed_set_min_confidence: float = 0.0
    worker_device: str = "auto"
    # Worker 领取 job 后若超过该秒数仍未完成，视为进程崩溃，其它 Worker 可
    # 原子重领并继续（started_at 是领取时写入的）。
    worker_claim_timeout_seconds: float = 3600.0

    # ---- 范围（scope）门控 ----
    # 是否启用范围门控。默认启用：只有 scope=accepted 的图片才允许给出公开真伪结论；
    # 否则一律落为 inconclusive（rejected / unavailable 都不伪造真伪预测）。
    scope_enforcement: bool = True
    # 范围分类工件路径（JSON artifact）。为空或路径无效时，默认门控为 unavailable：
    # 公开接口不把（历史）实验闭集预测当作有效真伪结论呈现。
    scope_artifact_path: str = ""

    # ---- 上传与内容策略 ----
    upload_max_bytes: int = 20 * 1024 * 1024
    upload_max_pixels: int = 40_000_000
    image_retention_days: int = 30

    # ---- 会话 / 安全 ----
    session_ttl_days: int = 14
    cookie_secure: bool = False
    csrf_header: str = "X-CSRF-Token"
    session_cookie: str = "drynaria_session"
    # NoDecode：禁止 pydantic-settings 对 list 复杂字段先按 JSON 解码
    # （`ALLOWED_ORIGINS=http://a,http://b` 不是合法 JSON，会在字段校验前抛
    # SettingsError）；原始字符串交给下方 before 校验器按逗号拆分。
    allowed_origins: Annotated[list[str], NoDecode] = ["http://localhost:5173"]

    # ---- 其它 ----
    auto_create_tables: bool = False

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def _parse_origins(cls, value: object) -> object:
        """兼容三种输入：真实 list、JSON 数组字符串、逗号分隔字符串。

        Compose/.env 使用 `ALLOWED_ORIGINS=http://localhost:8080,http://127.0.0.1:8080`
        这类逗号分隔格式；同时保留 JSON 数组字符串与直接传 list 的用法。
        """
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return []
            if text.startswith("[") and text.endswith("]"):
                try:
                    parsed = json.loads(text)
                except (ValueError, TypeError):
                    parsed = None
                if isinstance(parsed, list):
                    return [str(item).strip() for item in parsed if str(item).strip()]
            return [
                item.strip().strip("\"'")
                for item in text.split(",")
                if item.strip()
            ]
        return value

    @property
    def cookie_secure_effective(self) -> bool:
        if self.environment == "prod":
            return True
        return self.cookie_secure

    @property
    def effective_storage_backend(self) -> str:
        """auto：配置了 S3 端点则用 S3，否则本地磁盘（本地开发默认）。"""
        if self.storage_backend != "auto":
            return self.storage_backend
        return "s3" if self.s3_endpoint_url else "local"


@lru_cache
def get_settings() -> Settings:
    return Settings()


def get_app_settings(request: Request) -> Settings:
    """请求级配置：优先使用应用实例的 settings（测试注入），否则全局缓存。"""
    app_settings = getattr(request.app.state, "settings", None)
    return app_settings if isinstance(app_settings, Settings) else get_settings()
