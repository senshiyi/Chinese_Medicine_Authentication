"""数据增强 pipeline。"""

from torchvision import transforms


def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    """训练集增强：随机裁剪 + 翻转 + 色彩抖动 + 旋转 + 归一化。"""
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def get_val_transforms(image_size: int = 224) -> transforms.Compose:
    """验证/测试集增强：缩放 + 中心裁剪 + 归一化。"""
    return transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def get_inference_transforms(image_size: int = 224) -> transforms.Compose:
    """推理时使用，与验证集增强一致。"""
    return get_val_transforms(image_size)
