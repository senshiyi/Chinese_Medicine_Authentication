"""模型定义：视觉分类器 + 多模态融合预留骨架。

架构设计：
  - 单视觉模式：Image → CNN Backbone → Pooling → Classifier Head → 输出
  - 多模态模式：Image → CNN + Sensor → MLP → concat → Fusion Head → 输出

当前阶段仅训练视觉模型，多模态接口已预留，后续扩展无需重写视觉部分。
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models


# ── 骨干网络工厂 ──────────────────────────────────────────────

BACKBONE_CONFIGS = {
    "resnet50": {"fn": tv_models.resnet50, "dim": 2048},
    "resnet101": {"fn": tv_models.resnet101, "dim": 2048},
    "efficientnet_b0": {"fn": tv_models.efficientnet_b0, "dim": 1280},
    "efficientnet_b3": {"fn": tv_models.efficientnet_b3, "dim": 1536},
    "efficientnet_b4": {"fn": tv_models.efficientnet_b4, "dim": 1792},
    "convnext_tiny": {"fn": tv_models.convnext_tiny, "dim": 768},
    "convnext_small": {"fn": tv_models.convnext_small, "dim": 768},
}


def _build_backbone(name: str, pretrained: bool = True) -> tuple[nn.Module, int]:
    """创建骨干网络，返回 (backbone_module, feature_dim)。"""
    if name not in BACKBONE_CONFIGS:
        raise ValueError(f"不支持的骨干网络: {name}。可选: {list(BACKBONE_CONFIGS.keys())}")

    cfg = BACKBONE_CONFIGS[name]
    model = cfg["fn"](weights="IMAGENET1K_V1" if pretrained else None)

    # 去掉原始分类头
    if name.startswith("resnet"):
        backbone = nn.Sequential(*list(model.children())[:-2])  # 保留到 avgpool 之前
    elif name.startswith("efficientnet"):
        backbone = nn.Sequential(
            model.features,
            model.avgpool,
        )
    elif name.startswith("convnext"):
        backbone = nn.Sequential(
            model.features,
            nn.AdaptiveAvgPool2d(1),
        )
    else:
        raise NotImplementedError(f"骨干网络 {name} 的去头逻辑未实现")

    return backbone, cfg["dim"]


# ── 分类头 ────────────────────────────────────────────────────

class ClassifierHead(nn.Module):
    """MLP 分类头。"""

    def __init__(self, in_features: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        hidden = in_features // 3
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── 视觉分类模型 ──────────────────────────────────────────────

class VisionClassifier(nn.Module):
    """纯视觉中药真伪分类器。

    Args:
        backbone_name: 骨干网络名称
        num_classes: 类别数
        pretrained: 是否加载预训练权重
        freeze_backbone: 是否冻结骨干网络
        dropout: 分类头 dropout 比例
    """

    def __init__(
        self,
        backbone_name: str = "efficientnet_b3",
        num_classes: int = 2,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.backbone, self.feature_dim = _build_backbone(backbone_name, pretrained)

        if freeze_backbone:
            self.freeze_backbone()

        self.head = ClassifierHead(self.feature_dim, num_classes, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """提取特征向量（用于多模态融合或特征分析）。"""
        features = self.backbone(x)
        return features.flatten(1)

    def freeze_backbone(self):
        """冻结骨干网络参数。"""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """解冻骨干网络参数（用于微调阶段）。"""
        for param in self.backbone.parameters():
            param.requires_grad = True


# ── 多模态分类模型（预留，后续扩展）─────────────────────────────

class MultiModalClassifier(nn.Module):
    """多模态中药真伪分类器：视觉 + 嗅觉传感器 + 味觉传感器。

    当前仅实现视觉分支，嗅觉/味觉分支在传感器数据到位后激活。

    Usage:
        # 视觉模式
        model = MultiModalClassifier(num_classes=5)
        output = model(image=x)

        # 多模态模式（未来）
        output = model(image=x, smell=s, taste=t)
    """

    def __init__(
        self,
        num_classes: int = 2,
        backbone_name: str = "efficientnet_b3",
        pretrained: bool = True,
        freeze_backbone: bool = True,
        dropout: float = 0.3,
        smell_dim: Optional[int] = None,   # 电子鼻传感器特征维度
        taste_dim: Optional[int] = None,   # 电子舌传感器特征维度
    ):
        super().__init__()
        # 视觉分支
        self.vision_encoder, vision_dim = _build_backbone(backbone_name, pretrained)
        self.vision_pool = nn.AdaptiveAvgPool2d(1)

        if freeze_backbone:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False

        # 嗅觉分支（预留）
        self.has_smell = smell_dim is not None
        if self.has_smell:
            self.smell_encoder = nn.Sequential(
                nn.Linear(smell_dim, 128), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(128, 64), nn.ReLU(),
            )
        else:
            self.smell_encoder = None

        # 味觉分支（预留）
        self.has_taste = taste_dim is not None
        if self.has_taste:
            self.taste_encoder = nn.Sequential(
                nn.Linear(taste_dim, 128), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(128, 64), nn.ReLU(),
            )
        else:
            self.taste_encoder = None

        # 融合头：拼接所有模态的特征
        fused_dim = vision_dim
        if self.has_smell:
            fused_dim += 64
        if self.has_taste:
            fused_dim += 64

        self.fusion_head = ClassifierHead(fused_dim, num_classes, dropout)

    def forward(
        self,
        image: torch.Tensor,
        smell: Optional[torch.Tensor] = None,
        taste: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 视觉特征
        vis_feat = self.vision_encoder(image)
        vis_feat = self.vision_pool(vis_feat)

        features = [vis_feat.flatten(1)]

        # 嗅觉特征
        if self.has_smell:
            if smell is None:
                raise ValueError("smell 参数不能为 None，当前模型配置了嗅觉分支")
            features.append(self.smell_encoder(smell))

        # 味觉特征
        if self.has_taste:
            if taste is None:
                raise ValueError("taste 参数不能为 None，当前模型配置了味觉分支")
            features.append(self.taste_encoder(taste))

        fused = torch.cat(features, dim=1)
        return self.fusion_head(fused)

    def freeze_backbone(self):
        for param in self.vision_encoder.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.vision_encoder.parameters():
            param.requires_grad = True


# ── 便捷工厂函数 ──────────────────────────────────────────────

def create_model(
    backbone: str = "efficientnet_b3",
    num_classes: int = 2,
    pretrained: bool = True,
    freeze_backbone: bool = True,
    dropout: float = 0.3,
) -> VisionClassifier:
    """创建视觉分类模型。"""
    return VisionClassifier(
        backbone_name=backbone,
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        dropout=dropout,
    )
