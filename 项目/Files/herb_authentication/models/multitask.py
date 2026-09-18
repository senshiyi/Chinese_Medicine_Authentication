"""共享视觉骨干的声称药材、实际物种与真伪三头模型。"""

from __future__ import annotations

import math
import pickle
from collections.abc import Iterable
from dataclasses import dataclass
from hashlib import sha256
from hmac import compare_digest
from itertools import chain
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import torch
from torch import nn

from herb_authentication.errors import ConfigurationError


@dataclass(frozen=True, slots=True)
class MultiTaskLogits:
    claimed_herb: torch.Tensor
    actual_species: torch.Tensor
    authenticity: torch.Tensor


@dataclass(frozen=True, slots=True)
class EncoderSpec:
    encoder: nn.Module
    feature_dim: int
    backbone: str
    weights_source: Path | None


class MultiTaskHerbModel(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        feature_dim: int,
        num_claimed: int,
        num_species: int,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if not isinstance(encoder, nn.Module):
            raise TypeError("encoder 必须是 torch.nn.Module")
        self._validate_positive_integer("feature_dim", feature_dim)
        self._validate_positive_integer("num_claimed", num_claimed)
        self._validate_positive_integer("num_species", num_species)
        if isinstance(dropout, bool) or not isinstance(dropout, int | float):
            raise TypeError("dropout 必须是有限数值")
        if not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError("dropout 必须满足 0 <= dropout < 1")

        self.encoder = encoder
        self.feature_dim = feature_dim
        self.dropout = nn.Dropout(float(dropout))
        self.claimed_head = nn.Linear(feature_dim, num_claimed)
        self.species_head = nn.Linear(feature_dim, num_species)
        self.authenticity_head = nn.Linear(feature_dim, 2)
        self._encoder_trainable = True

    @staticmethod
    def _validate_positive_integer(field: str, value: int) -> None:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{field} 必须是正整数")
        if value <= 0:
            raise ValueError(f"{field} 必须是正整数")

    @property
    def encoder_trainable(self) -> bool:
        return self._encoder_trainable

    def head_parameters(self) -> Iterable[nn.Parameter]:
        """返回三个任务头的参数，供分阶段优化器显式使用。"""

        return chain(
            self.claimed_head.parameters(),
            self.species_head.parameters(),
            self.authenticity_head.parameters(),
        )

    def add_encoder_to_optimizer(
        self,
        optimizer: torch.optim.Optimizer,
        **group_options: Any,
    ) -> int:
        """把解冻后的骨干参数补入既有优化器，返回新增参数对象数量。"""

        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError("optimizer 必须是 torch.optim.Optimizer")
        if "params" in group_options:
            raise ValueError("group_options 不得覆盖 params")
        if not self._encoder_trainable:
            raise ConfigurationError("encoder 尚未解冻，不能加入优化器")

        existing_parameter_ids = {
            id(parameter) for group in optimizer.param_groups for parameter in group["params"]
        }
        missing = [
            parameter
            for parameter in self.encoder.parameters()
            if id(parameter) not in existing_parameter_ids
        ]
        if not missing:
            return 0
        optimizer.add_param_group({"params": missing, **group_options})
        return len(missing)

    def set_encoder_trainable(self, trainable: bool) -> None:
        """切换骨干训练阶段，并同步梯度与 BatchNorm/Dropout 模式。"""

        if not isinstance(trainable, bool):
            raise TypeError("trainable 必须是 bool")
        self._encoder_trainable = trainable
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(trainable)
        if trainable:
            self.encoder.train(self.training)
        else:
            self.encoder.eval()

    def train(self, mode: bool = True) -> MultiTaskHerbModel:
        if not isinstance(mode, bool):
            raise TypeError("mode 必须是 bool")
        super().train(mode)
        if not self._encoder_trainable:
            self.encoder.eval()
        return self

    def get_extra_state(self) -> dict[str, int | bool]:
        return {"schema_version": 1, "encoder_trainable": self._encoder_trainable}

    def set_extra_state(self, state: Any) -> None:
        if not isinstance(state, dict) or state.get("schema_version") != 1:
            raise ConfigurationError("模型训练阶段状态格式不受支持")
        trainable = state.get("encoder_trainable")
        if not isinstance(trainable, bool):
            raise ConfigurationError("模型训练阶段缺少 encoder_trainable 布尔值")
        self.set_encoder_trainable(trainable)

    def forward(self, images: torch.Tensor) -> MultiTaskLogits:
        features = self.encoder(images)
        if not isinstance(features, torch.Tensor):
            raise RuntimeError("encoder 必须返回 torch.Tensor")
        if features.ndim != 2 or features.shape[-1] != self.feature_dim:
            shape = tuple(features.shape)
            raise RuntimeError(
                f"encoder 输出形状应为 [batch, feature_dim={self.feature_dim}]，实际为 {shape}"
            )
        shared_features = self.dropout(features)
        return MultiTaskLogits(
            claimed_herb=self.claimed_head(shared_features),
            actual_species=self.species_head(shared_features),
            authenticity=self.authenticity_head(shared_features),
        )


@dataclass(frozen=True, slots=True)
class _TorchvisionBackbone:
    constructor: Any
    weights: Any


def _torchvision_backbones() -> dict[str, _TorchvisionBackbone]:
    from torchvision.models import (
        EfficientNet_B0_Weights,
        EfficientNet_B3_Weights,
        efficientnet_b0,
        efficientnet_b3,
    )

    return {
        "efficientnet_b0": _TorchvisionBackbone(efficientnet_b0, EfficientNet_B0_Weights),
        "efficientnet_b3": _TorchvisionBackbone(efficientnet_b3, EfficientNet_B3_Weights),
    }


def _local_weights_path(weights: Any, weights_cache_dir: str | Path | None) -> Path:
    filename = Path(urlparse(weights.DEFAULT.url).path).name
    if not filename:
        raise ConfigurationError("torchvision 预训练权重地址无有效文件名")
    cache_dir = (
        Path(weights_cache_dir)
        if weights_cache_dir is not None
        else Path(torch.hub.get_dir()) / "checkpoints"
    )
    path = cache_dir.expanduser().resolve() / filename
    if not path.is_file():
        raise ConfigurationError(
            f"预训练权重不在本地缓存中：{path}。请先显式下载并校验权重，系统不会自动联网。"
        )
    return path


def _verify_local_weights_hash(path: Path) -> None:
    expected_prefix = path.stem.rpartition("-")[2].lower()
    if not 8 <= len(expected_prefix) <= 64 or any(
        character not in "0123456789abcdef" for character in expected_prefix
    ):
        raise ConfigurationError(f"预训练权重文件名缺少有效 SHA-256 前缀：{path.name}")

    digest = sha256()
    try:
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
    except OSError as exc:
        raise ConfigurationError(f"无法读取本地预训练权重：{path}") from exc

    actual = digest.hexdigest()
    if not compare_digest(actual[: len(expected_prefix)], expected_prefix):
        raise ConfigurationError(f"本地预训练权重 SHA-256 校验失败：{path}")


def build_torchvision_encoder(
    backbone: str,
    *,
    pretrained: bool,
    weights_cache_dir: str | Path | None = None,
) -> EncoderSpec:
    """构建 torchvision 骨干；预训练模式只读取已有本地缓存。"""

    if not isinstance(backbone, str) or not backbone.strip():
        raise ConfigurationError("backbone 必须是非空字符串")
    if not isinstance(pretrained, bool):
        raise ConfigurationError("pretrained 必须是 bool")

    registry = _torchvision_backbones()
    normalized = backbone.strip().lower()
    definition = registry.get(normalized)
    if definition is None:
        supported = "、".join(sorted(registry))
        raise ConfigurationError(f"不支持的 backbone：{backbone}；可选值：{supported}")

    # 始终以 weights=None 构造，避免 torchvision 在此路径中触发隐式下载。
    model = definition.constructor(weights=None)
    weights_source: Path | None = None
    if pretrained:
        weights_source = _local_weights_path(definition.weights, weights_cache_dir)
        try:
            _verify_local_weights_hash(weights_source)
            state_dict = torch.load(weights_source, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict, strict=True)
        except (
            EOFError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
            pickle.UnpicklingError,
        ) as exc:
            raise ConfigurationError(f"无法加载本地预训练权重：{weights_source}") from exc

    feature_dim = model.classifier[-1].in_features
    encoder = nn.Sequential(model.features, model.avgpool, nn.Flatten(start_dim=1))
    return EncoderSpec(
        encoder=encoder,
        feature_dim=feature_dim,
        backbone=normalized,
        weights_source=weights_source,
    )
