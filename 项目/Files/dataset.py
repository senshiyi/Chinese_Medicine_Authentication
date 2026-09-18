"""数据集加载与 DataLoader 工厂。"""

import json
from collections import Counter
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.datasets import ImageFolder

from transforms import get_train_transforms, get_val_transforms


class HerbDataset:
    """中药数据集管理器：封装 ImageFolder + 数据划分 + DataLoader 创建。"""

    def __init__(
        self,
        data_root: str,
        image_size: int = 224,
        batch_size: int = 32,
        num_workers: int = 4,
        train_ratio: float = 0.7,
        val_ratio: float = 0.2,
        use_weighted_sampler: bool = True,
    ):
        self.data_root = Path(data_root)
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.use_weighted_sampler = use_weighted_sampler

        if not self.data_root.exists():
            raise FileNotFoundError(f"数据目录不存在: {self.data_root}")

        # 用 ImageFolder 自动从文件夹名解析类别标签
        self.full_dataset = ImageFolder(root=str(self.data_root))
        self.classes = self.full_dataset.classes
        self.num_classes = len(self.classes)
        self.class_to_idx = self.full_dataset.class_to_idx

        print(f"发现 {self.num_classes} 个类别: {self.classes}")
        print(f"总样本数: {len(self.full_dataset)}")
        for class_idx, count in Counter(self.full_dataset.targets).items():
            print(f"  {self.classes[class_idx]}: {count} 张")

    def split(self, seed: int = 42):
        """按比例划分训练集/验证集/测试集。"""
        total = len(self.full_dataset)
        indices = list(range(total))
        labels = self.full_dataset.targets

        # 分层抽样：每个类别按比例划分
        from sklearn.model_selection import train_test_split

        train_idx, temp_idx = train_test_split(
            indices, test_size=1 - self.train_ratio, stratify=labels, random_state=seed
        )
        temp_labels = [labels[i] for i in temp_idx]
        val_ratio_in_temp = self.val_ratio / (1 - self.train_ratio)
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=1 - val_ratio_in_temp,
            stratify=temp_labels,
            random_state=seed,
        )

        self.train_indices = train_idx
        self.val_indices = val_idx
        self.test_indices = test_idx

        print(f"\n数据划分完成:")
        print(f"  训练集: {len(train_idx)} 张")
        print(f"  验证集: {len(val_idx)} 张")
        print(f"  测试集: {len(test_idx)} 张")

        return self

    def _create_sampler(self, indices: list[int]) -> Optional[WeightedRandomSampler]:
        """为训练集创建加权采样器，处理类别不平衡。"""
        if not self.use_weighted_sampler:
            return None

        labels = [self.full_dataset.targets[i] for i in indices]
        class_counts = Counter(labels)
        num_samples = len(indices)
        class_weights = {cls: num_samples / count for cls, count in class_counts.items()}
        sample_weights = [class_weights[label] for label in labels]
        sampler = WeightedRandomSampler(sample_weights, num_samples=num_samples, replacement=True)
        return sampler

    def get_train_loader(self) -> DataLoader:
        """获取训练集 DataLoader（带数据增强）。"""
        subset = torch.utils.data.Subset(self.full_dataset, self.train_indices)
        subset.dataset.transform = get_train_transforms(self.image_size)
        sampler = self._create_sampler(self.train_indices)
        return DataLoader(
            subset,
            batch_size=self.batch_size,
            sampler=sampler,
            shuffle=(sampler is None),
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def get_val_loader(self) -> DataLoader:
        """获取验证集 DataLoader。"""
        subset = torch.utils.data.Subset(self.full_dataset, self.val_indices)
        subset.dataset.transform = get_val_transforms(self.image_size)
        return DataLoader(
            subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def get_test_loader(self) -> DataLoader:
        """获取测试集 DataLoader。"""
        subset = torch.utils.data.Subset(self.full_dataset, self.test_indices)
        subset.dataset.transform = get_val_transforms(self.image_size)
        return DataLoader(
            subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
