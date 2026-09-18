"""三任务图片 Dataset 与 DataLoader 随机性辅助。"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset, default_collate

from ..errors import InputImageError, ManifestValidationError
from .labels import LabelSpace
from .splitting import PartitionedImageRecord


class HerbImageDataset(Dataset):
    def __init__(
        self,
        rows: Sequence[PartitionedImageRecord],
        label_space: LabelSpace,
        transform: Callable[[Any], Any] | None,
        image_loader: Callable[[Path], Any] | None = None,
    ) -> None:
        self.rows = tuple(rows)
        self.verify_manifest_content = image_loader is None
        self.label_space = label_space
        self.transform = transform
        self.image_loader = image_loader or load_rgb_image

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.rows[index]
        row = item.row
        if self.verify_manifest_content:
            _verify_validated_image(item)
        try:
            image = self.image_loader(row.image_path)
        except InputImageError:
            raise
        except Image.DecompressionBombError as exc:
            raise InputImageError(f"图片触发解压炸弹保护：{row.image_path}（{exc}）") from exc
        except (OSError, SyntaxError, ValueError, UnidentifiedImageError) as exc:
            raise InputImageError(f"无法读取图片：{row.image_path}（{exc}）") from exc
        if self.transform is not None:
            image = self.transform(image)

        claimed_index = self.label_space.claimed_to_index.get(row.claimed_herb_name)
        if claimed_index is None:
            raise ManifestValidationError(
                f"声称药材 {row.claimed_herb_name!r} 在训练标签空间中不存在。"
            )
        try:
            authenticity_index = self.label_space.authenticity_classes.index(row.authenticity)
        except ValueError as exc:
            raise ManifestValidationError(
                f"真伪标签 {row.authenticity!r} 在标签空间中不存在。"
            ) from exc

        species_target = torch.zeros(
            len(self.label_space.actual_species_classes), dtype=torch.float32
        )
        unknown_species: tuple[str, ...] = ()
        species_mask = False
        if row.species_annotation_status == "verified":
            unknown_species = tuple(
                sorted(
                    species
                    for species in row.actual_species
                    if species not in self.label_space.species_to_index
                )
            )
            for species in row.actual_species:
                species_index = self.label_space.species_to_index.get(species)
                if species_index is not None:
                    species_target[species_index] = 1.0
            species_mask = not unknown_species

        return {
            "image": image,
            "claimed_target": torch.tensor(claimed_index, dtype=torch.long),
            "authenticity_target": torch.tensor(authenticity_index, dtype=torch.long),
            "actual_species_target": species_target,
            "actual_species_mask": torch.tensor(species_mask, dtype=torch.bool),
            "unknown_species_labels": unknown_species,
            "metadata": {
                "image_path": row.image_path,
                "group_id": row.group_id,
                "sample_id": row.sample_id,
                "partition": item.partition,
                "claimed_herb_name": row.claimed_herb_name,
            },
        }


def _verify_validated_image(item: PartitionedImageRecord) -> None:
    path = item.row.image_path
    try:
        stat = path.stat()
        if stat.st_size != item.record.file_size:
            raise InputImageError(f"图片在 manifest 验证后发生变化：{path}")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != item.record.sha256:
            raise InputImageError(f"图片在 manifest 验证后发生变化：{path}")
        with Image.open(path) as image:
            if image.size != (item.record.width, item.record.height):
                raise InputImageError(f"图片在 manifest 验证后发生变化：{path}")
    except InputImageError:
        raise
    except Image.DecompressionBombError as exc:
        raise InputImageError(f"图片触发解压炸弹保护：{path}（{exc}）") from exc
    except (OSError, SyntaxError, ValueError, UnidentifiedImageError) as exc:
        raise InputImageError(f"无法重新验证图片：{path}（{exc}）") from exc


def load_rgb_image(path: Path) -> Image.Image:
    """读取并复制 RGB 图片，确保返回值不依赖已关闭的文件句柄。"""

    with Image.open(path) as image:
        return image.convert("RGB")


def collate_herb_batch(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """堆叠固定形状张量，并保留逐样本开放集标签与实体元数据。"""

    if not batch:
        raise ValueError("不能整理空 batch。")
    tensor_keys = (
        "image",
        "claimed_target",
        "authenticity_target",
        "actual_species_target",
        "actual_species_mask",
    )
    return {key: default_collate([item[key] for item in batch]) for key in tensor_keys} | {
        "unknown_species_labels": tuple(item["unknown_species_labels"] for item in batch),
        "metadata": tuple(item["metadata"] for item in batch),
    }


def create_herb_dataloader(
    dataset: HerbImageDataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int = 0,
    drop_last: bool = False,
    pin_memory: bool = False,
) -> DataLoader:
    """创建强制绑定 collate、generator 与 worker seed 的唯一 DataLoader。"""

    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size 必须是正整数。")
    if isinstance(num_workers, bool) or not isinstance(num_workers, int) or num_workers < 0:
        raise ValueError("num_workers 必须是非负整数。")
    if dataset.transform is None:
        raise ValueError("批量加载必须提供返回 Tensor 的 transform。")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_herb_batch,
        generator=create_dataloader_generator(seed),
        worker_init_fn=seed_worker,
        drop_last=drop_last,
        pin_memory=pin_memory,
    )


def create_dataloader_generator(seed: int) -> torch.Generator:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("DataLoader seed 必须是整数。")
    return torch.Generator().manual_seed(seed)


def seed_worker(_worker_id: int) -> None:
    """从 PyTorch worker seed 派生 Python 与 NumPy 的 32 位 seed。"""

    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
