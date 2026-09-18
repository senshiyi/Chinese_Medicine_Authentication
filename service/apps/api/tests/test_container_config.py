"""容器/部署配置静态检查：模型只挂 Worker、命令分离、无真实密钥。"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]  # Chinese_Medicine_Authentication_v2
COMPOSE = ROOT / "infra" / "docker-compose.yml"
ENV_EXAMPLE = ROOT / ".env.example"
API_DOCKERFILE = ROOT / "apps" / "api" / "Dockerfile"


def _services_block(text: str, name: str) -> str:
    # (?s) 让 .*? 跨行；(?m) 提供行首锚点；先归一化 CRLF 防止平台差异
    text = text.replace("\r\n", "\n")
    match = re.search(rf"(?sm)^  {name}:.*?(?=^  [a-z0-9_-]+:|\Z)", text)
    assert match is not None, f"compose 缺少 service {name}"
    return match.group(0)


def test_env_example_lists_required_variables():
    content = ENV_EXAMPLE.read_text(encoding="utf-8")
    for key in (
        "DATABASE_URL",
        "REDIS_URL",
        "CELERY_BROKER_URL",
        "STORAGE_BACKEND",
        "S3_ENDPOINT_URL",
        "INFERENCE_PROJECT_PATH",
        "MODEL_CHECKPOINT_PATH",
        "MODEL_VERSION_HINT",
        "ALLOWED_ORIGINS",
    ):
        assert key in content, f".env.example 缺少 {key}"
    # 示例环境变量不含真实密钥/本地 compose 口令
    assert "drynaria-stage-a-local-only" not in content
    assert "minio-stage-a-local-only" not in content


def test_model_mount_only_on_worker_and_commands_are_separate():
    compose = COMPOSE.read_text(encoding="utf-8")
    api = _services_block(compose, "api")
    worker = _services_block(compose, "worker")

    # Worker 单独挂载原推理项目与模型目录（只读），命令是 celery
    assert "/opt/inference:ro" in worker
    assert "/models:ro" in worker
    assert "celery" in worker and "worker" in worker
    # API 不出现模型挂载，命令是 uvicorn（先跑 alembic 迁移）
    assert "项目/Models" not in api and "项目/Files" not in api
    assert "uvicorn" in api and "alembic" in api
    # API 构建不装 torch：同一 Dockerfile 用 ARG INSTALL_TORCH 控制，
    # api service 不传该参数（默认 0），只有 worker 传 INSTALL_TORCH=1
    api_dockerfile = API_DOCKERFILE.read_text(encoding="utf-8")
    assert "ARG INSTALL_TORCH" in api_dockerfile
    assert 'INSTALL_TORCH" = "1"' in api_dockerfile  # torch 安装受开关保护
    assert 'INSTALL_TORCH: "1"' in worker
    assert "INSTALL_TORCH" not in api  # api service 不启用 torch


def test_required_services_defined():
    compose = COMPOSE.read_text(encoding="utf-8")
    for service in ("postgres", "redis", "minio", "api", "worker", "web", "nginx"):
        _services_block(compose, service)
    for service in ("api", "postgres", "redis", "minio"):
        assert "healthcheck" in _services_block(compose, service)


@pytest.mark.skipif(shutil.which("docker") is None, reason="本机无 docker CLI")
def test_docker_compose_config_renders():
    result = subprocess.run(
        ["docker", "compose", "-f", str(COMPOSE), "config", "--quiet"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
