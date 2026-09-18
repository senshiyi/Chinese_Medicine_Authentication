"""只读推理适配层：仅在 Worker 进程内 sys.path 挂载原项目并按需加载模型。

API 进程绝不 import 本模块中的 torch 依赖路径；模型在 Worker 内加载一次复用。
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Any

from .settings import Settings, get_settings


class InferenceError(RuntimeError):
    """Worker 侧推理错误：category 只允许工程错误类别，不含真伪结论。"""

    CATEGORIES = {"image_decode_failed", "model_load_failed", "prediction_failed"}

    def __init__(self, category: str, message: str) -> None:
        if category not in self.CATEGORIES:
            raise ValueError(f"unknown inference error category: {category}")
        self.category = category
        super().__init__(message)


_service_lock = threading.Lock()
_service_cache: dict[str, Any] = {}  # checkpoint 路径 -> InferenceService（进程级单例）


def _ensure_project_on_path(settings: Settings) -> Path:
    project = Path(settings.inference_project_path).expanduser().resolve()
    if not project.is_dir():
        raise InferenceError(
            "model_load_failed",
            f"推理项目路径不存在：{project}（不要复制原项目到本仓库，只读引用）。",
        )
    if str(project) not in sys.path:
        sys.path.insert(0, str(project))
    return project


def _checkpoint_path(settings: Settings) -> Path:
    checkpoint = Path(settings.model_checkpoint_path).expanduser().resolve()
    if not checkpoint.is_file():
        raise InferenceError(
            "model_load_failed", f"checkpoint 不存在：{checkpoint}（不随仓库复制模型权重）。"
        )
    return checkpoint


def get_inference_service(settings: Settings | None = None) -> Any:
    """加载并缓存 InferenceService；只允许 Worker 调用。"""
    settings = settings or get_settings()
    key = str(Path(settings.model_checkpoint_path).resolve())
    with _service_lock:
        service = _service_cache.get(key)
        if service is not None:
            return service
        _ensure_project_on_path(settings)
        checkpoint = _checkpoint_path(settings)
        try:
            from herb_authentication.inference.service import InferenceService

            service = InferenceService.from_checkpoint(
                checkpoint,
                device=settings.worker_device,
                max_image_bytes=settings.upload_max_bytes,
                max_pixels=settings.upload_max_pixels,
                max_decoded_bytes=settings.upload_max_pixels * 3,
            )
        except InferenceError:
            raise
        except Exception as exc:  # noqa: BLE001 —— 统一归类为工程失败
            raise InferenceError("model_load_failed", f"模型装配失败：{type(exc).__name__}") from exc
        if (
            getattr(service, "experimental_closed_set", None) is not True
            or getattr(service, "open_set_calibrated", None) is not False
        ):
            raise InferenceError(
                "model_load_failed",
                "checkpoint 不是已批准的实验闭集模型（需 experimental_closed_set=true、"
                "open_set_calibrated=false）。",
            )
        _service_cache[key] = service
        return service


def predict_experimental_closed_set_bytes(
    payload: bytes, settings: Settings | None = None
) -> dict[str, Any]:
    """Worker 调用 InferenceService.predict_experimental_closed_set 的唯一入口。"""
    settings = settings or get_settings()
    service = get_inference_service(settings)
    try:
        result = service.predict_experimental_closed_set(payload)
    except Exception as exc:  # noqa: BLE001 —— 见下方分类
        from herb_authentication.errors import InputImageError  # 已挂载 sys.path

        if isinstance(exc, InputImageError):
            raise InferenceError("image_decode_failed", "存储图片无法解码。") from exc
        raise InferenceError("prediction_failed", f"推理执行失败：{type(exc).__name__}") from exc
    data = result.to_dict()
    if data.get("experimental_closed_set") is not True or data.get("open_set_calibrated") is not False:
        raise InferenceError(
            "prediction_failed", "推理结果缺少批准的实验闭集状态标志。"
        )
    return data
