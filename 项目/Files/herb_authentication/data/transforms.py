"""相互独立的训练增强与评估预处理工厂。"""

from __future__ import annotations

import math
from collections.abc import Sequence

from torchvision import transforms

DEFAULT_MEAN = (0.485, 0.456, 0.406)
DEFAULT_STD = (0.229, 0.224, 0.225)


def build_train_transform(
    image_size: int,
    mean: Sequence[float] = DEFAULT_MEAN,
    std: Sequence[float] = DEFAULT_STD,
) -> transforms.Compose:
    """每次调用都创建独立的训练增强流水线。"""

    normalized_mean, normalized_std = _validate_transform_config(image_size, mean, std)
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.75, 1.0),
                ratio=(0.85, 1.15),
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(normalized_mean, normalized_std),
        ]
    )


def build_eval_transform(
    image_size: int,
    mean: Sequence[float] = DEFAULT_MEAN,
    std: Sequence[float] = DEFAULT_STD,
) -> transforms.Compose:
    """每次调用都创建不含随机操作的验证/测试流水线。"""

    normalized_mean, normalized_std = _validate_transform_config(image_size, mean, std)
    resize_size = max(image_size, round(image_size / 0.875))
    return transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(normalized_mean, normalized_std),
        ]
    )


def _validate_transform_config(
    image_size: int,
    mean: Sequence[float],
    std: Sequence[float],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    if isinstance(image_size, bool) or not isinstance(image_size, int) or image_size <= 0:
        raise ValueError("image_size 必须是正整数。")
    if len(mean) != 3 or len(std) != 3:
        raise ValueError("mean 和 std 必须各包含 3 个通道值。")
    try:
        normalized_mean = tuple(float(value) for value in mean)
        normalized_std = tuple(float(value) for value in std)
    except (TypeError, ValueError) as exc:
        raise ValueError("mean 和 std 必须只包含有限数。") from exc
    if not all(math.isfinite(value) for value in (*normalized_mean, *normalized_std)):
        raise ValueError("mean 和 std 必须只包含有限数。")
    if any(value <= 0 for value in normalized_std):
        raise ValueError("std 的每个通道值必须大于 0。")
    return normalized_mean, normalized_std
