"""配置管理：dataclass + YAML 驱动。"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class DataConfig:
    root: str = "../数据/raw"
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    train_ratio: float = 0.7
    val_ratio: float = 0.2
    use_weighted_sampler: bool = True


@dataclass
class ModelConfig:
    backbone: str = "efficientnet_b3"
    pretrained: bool = True
    freeze_backbone: bool = True
    unfreeze_epochs: int = 5
    dropout: float = 0.3
    num_classes: Optional[int] = None


@dataclass
class TrainConfig:
    epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    early_stop_patience: int = 10
    mixed_precision: bool = True
    gradient_clip: float = 1.0


@dataclass
class LoggingConfig:
    log_dir: str = "../模型/Results/logs"
    checkpoint_dir: str = "../模型/Models"
    save_best_only: bool = True
    log_interval: int = 10


@dataclass
class InferenceConfig:
    checkpoint: str = "../模型/Models/best_model.pth"
    top_k: int = 3


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "Config":
        """从 YAML 文件加载配置。"""
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {yaml_path}")

        with open(yaml_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        return cls(
            data=DataConfig(**raw.get("data", {})),
            model=ModelConfig(**raw.get("model", {})),
            train=TrainConfig(**raw.get("train", {})),
            logging=LoggingConfig(**raw.get("logging", {})),
            inference=InferenceConfig(**raw.get("inference", {})),
        )

    def resolve_paths(self, project_root: Path) -> None:
        """将配置中的相对路径转为基于 project_root 的绝对路径。"""
        self.data.root = str((project_root / self.data.root).resolve())
        self.logging.log_dir = str((project_root / self.logging.log_dir).resolve())
        self.logging.checkpoint_dir = str((project_root / self.logging.checkpoint_dir).resolve())
        self.inference.checkpoint = str((project_root / self.inference.checkpoint).resolve())
