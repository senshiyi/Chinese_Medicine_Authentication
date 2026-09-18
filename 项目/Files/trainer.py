"""训练循环：epoch/step/batch 逻辑，与具体模型解耦。"""

import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# AMP：PyTorch 2.6 新旧 API 均可用，用旧版 API 避免兼容问题
from torch.cuda.amp import GradScaler, autocast

try:
    from torch.utils.tensorboard import SummaryWriter

    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False


class Trainer:
    """通用训练器。

    支持：
      - 混合精度训练 (AMP)
      - 梯度裁剪
      - 标签平滑
      - 早停
      - 骨干网络分阶段解冻
      - TensorBoard 日志
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config,  # TrainConfig
        logging_config,  # LoggingConfig
        device: torch.device,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.logging_config = logging_config
        self.device = device

        # 损失函数（标签平滑）
        self.criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

        # 优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.epochs
        )

        # 混合精度
        self.use_amp = config.mixed_precision and device.type == "cuda"
        self.scaler = GradScaler() if self.use_amp else None

        # 日志
        if HAS_TENSORBOARD:
            self.writer = SummaryWriter(log_dir=logging_config.log_dir)
        else:
            self.writer = None
        self.checkpoint_dir = Path(logging_config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # 状态
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0

    def _log_scalar(self, tag: str, value: float, step: int):
        """记录标量到 TensorBoard（如果可用）。"""
        if self.writer is not None:
            self.writer.add_scalar(tag, value, step)

    def train_epoch(self) -> dict[str, float]:
        """训练一个 epoch。"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                self.scaler.scale(loss).backward()
                if self.config.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.gradient_clip
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                if self.config.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.gradient_clip
                    )
                self.optimizer.step()

            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += images.size(0)

            if batch_idx % self.logging_config.log_interval == 0:
                self._log_scalar(
                    "train/step_loss",
                    loss.item(),
                    self.current_epoch * len(self.train_loader) + batch_idx,
                )

        avg_loss = total_loss / total
        accuracy = correct / total
        self.scheduler.step()

        return {"loss": avg_loss, "accuracy": accuracy}

    @torch.no_grad()
    def validate_epoch(self) -> dict[str, float]:
        """验证一个 epoch。"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in self.val_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += images.size(0)

        avg_loss = total_loss / total
        accuracy = correct / total

        return {"loss": avg_loss, "accuracy": accuracy}

    def save_checkpoint(self, filename: str, is_best: bool = False) -> Path:
        """保存模型权重。"""
        path = self.checkpoint_dir / filename
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_acc": self.best_val_acc,
            "best_val_loss": self.best_val_loss,
        }
        if self.scaler:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        torch.save(checkpoint, path)
        print(f"  📦 模型已保存: {path}")

        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"  ⭐ 最佳模型已保存: {best_path}")

        return path

    def fit(self, unfreeze_epoch: Optional[int] = None) -> dict:
        """完整训练流程。

        Args:
            unfreeze_epoch: 在第 N 轮后解冻骨干网络（None = 不解冻）
        """
        print(f"\n{'='*50}")
        print(f"开始训练 | 设备: {self.device} | AMP: {self.use_amp}")
        print(f"训练集: {len(self.train_loader.dataset)} 张")
        print(f"验证集: {len(self.val_loader.dataset)} 张")
        print(f"{'='*50}\n")

        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        start_time = time.time()

        for epoch in range(1, self.config.epochs + 1):
            self.current_epoch = epoch
            epoch_start = time.time()

            # 分阶段解冻骨干网络
            if unfreeze_epoch and epoch == unfreeze_epoch:
                print(f"\n🔓 第 {epoch} 轮：解冻骨干网络，开始微调...")
                if hasattr(self.model, "unfreeze_backbone"):
                    self.model.unfreeze_backbone()
                # 解冻后用更小的学习率
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.config.learning_rate * 0.1

            # 训练 + 验证
            train_metrics = self.train_epoch()
            val_metrics = self.validate_epoch()

            elapsed = time.time() - epoch_start
            lr = self.optimizer.param_groups[0]["lr"]

            # 记录
            for k, v in train_metrics.items():
                history[f"train_{'acc' if k == 'accuracy' else 'loss'}"].append(v)
                self._log_scalar(f"train/{k}", v, epoch)
            for k, v in val_metrics.items():
                history[f"val_{'acc' if k == 'accuracy' else 'loss'}"].append(v)
                self._log_scalar(f"val/{k}", v, epoch)
            self._log_scalar("lr", lr, epoch)

            print(
                f"Epoch {epoch:3d}/{self.config.epochs} | "
                f"lr: {lr:.2e} | "
                f"train loss: {train_metrics['loss']:.4f} acc: {train_metrics['accuracy']:.4f} | "
                f"val loss: {val_metrics['loss']:.4f} acc: {val_metrics['accuracy']:.4f} | "
                f"⏱ {elapsed:.1f}s"
            )

            # 早停检查
            is_best = val_metrics["accuracy"] > self.best_val_acc
            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

            if is_best:
                self.best_val_acc = val_metrics["accuracy"]
                self.save_checkpoint(f"epoch_{epoch:03d}.pth", is_best=True)
            elif epoch % 10 == 0:
                self.save_checkpoint(f"epoch_{epoch:03d}.pth")

            if self.epochs_without_improvement >= self.config.early_stop_patience:
                print(f"\n⏹ 早停：连续 {self.config.early_stop_patience} 轮验证损失未改善")
                break

        total_time = time.time() - start_time
        print(f"\n{'='*50}")
        print(f"训练完成 | 总耗时: {total_time/60:.1f} 分钟")
        print(f"最佳验证准确率: {self.best_val_acc:.4f} | 最佳验证损失: {self.best_val_loss:.4f}")
        print(f"{'='*50}")

        if self.writer is not None:
            self.writer.close()
        return history
