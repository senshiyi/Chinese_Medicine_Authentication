"""冻结 encoder 特征提取：复用原闭集模型（只读）的共享骨干。

这是范围分类 Baseline 的特征侧：对图片走与闭集推理完全相同的解码与预处理，再取
``MultiTaskHerbModel.encoder`` 的输出向量（不是三头 logits）。原项目只读，不复制权重、
不覆盖任何参数。

隔离约定：本模块导入 torch 与 ``herb_authentication``，因此只允许在 Worker / 训练 CLI
进程（带 torch 的环境）按需导入；API 进程与常规测试进程不得导入本模块。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch


def extract_scope_feature_vector(image: bytes | Path, service: Any) -> list[float]:
    """对单张图片提取冻结 encoder 特征向量（``[feature_dim]`` 的 Python float 列表）。"""
    decoded = service._decode_image(image)  # noqa: SLF001 —— 复用服务的安全解码/预处理
    tensor = service.preprocess(decoded)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 4 or tensor.shape[0] != 1:
        raise ValueError("preprocess 输出必须是 [C,H,W] 或 [1,C,H,W]")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError("preprocess 输出包含非有限数值")

    with torch.inference_mode():
        features = service.model.encoder(tensor.to(service.device))
    if not isinstance(features, torch.Tensor) or features.ndim != 2:
        raise RuntimeError("encoder 输出必须是 [1, feature_dim]")
    feature_dim = int(getattr(service.model, "feature_dim", features.shape[-1]))
    if features.shape[-1] != feature_dim:
        raise RuntimeError(
            f"encoder 输出维度 {features.shape[-1]} 与 feature_dim={feature_dim} 不一致"
        )
    flattened = features.squeeze(0).detach().to(dtype=torch.float64, device="cpu").tolist()
    return [float(v) for v in flattened]


def build_feature_extractor(settings: Any):
    """返回一个 ``(image: bytes|Path) -> list[float]`` 的可调用对象。

    复用 ``inference_adapter.get_inference_service`` 的进程级缓存：范围特征与闭集推理
    共享同一份冻结模型，特征与 checkpoint 绑定一致。
    """
    from ..inference_adapter import get_inference_service

    service = get_inference_service(settings)
    feature_dim = int(getattr(service.model, "feature_dim", 0))

    def extract(image: bytes | Path) -> list[float]:
        return extract_scope_feature_vector(image, service)

    extract.feature_dim = feature_dim  # type: ignore[attr-defined]
    return extract
