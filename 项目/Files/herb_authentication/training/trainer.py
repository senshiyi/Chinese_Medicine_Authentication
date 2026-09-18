"""可组装的多任务训练循环与骨干冻结调度。"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import torch

from herb_authentication.errors import CheckpointCompatibilityError
from herb_authentication.models import MultiTaskHerbModel
from herb_authentication.training.checkpoints import (
    SCHEMA_VERSION,
    CheckpointContract,
    capture_rng_state,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint_atomic,
)
from herb_authentication.training.losses import (
    LossWeights,
    MultiTaskMasks,
    MultiTaskTargets,
    compute_multitask_loss,
)
from herb_authentication.training.metrics import (
    EntityMetrics,
    ImageLevelPredictions,
    aggregate_by_sample,
    compute_entity_metrics,
)

FreezeStage = Literal["heads_only", "full_model"]


@dataclass(frozen=True, slots=True)
class EpochResult:
    epoch: int
    training: bool
    total_loss: float
    claimed_herb_loss: float
    actual_species_loss: float
    authenticity_loss: float
    species_supervised_rows: int
    batches: int
    samples: int
    freeze_stage: FreezeStage
    learning_rates: tuple[float, ...]
    duration_seconds: float
    entity_metrics: EntityMetrics | None = None


@dataclass(frozen=True, slots=True)
class TrainingSummary:
    train_epochs: tuple[EpochResult, ...]
    validation_epochs: tuple[EpochResult, ...]
    amp_enabled: bool
    device: str
    best_epoch: int | None = None
    best_selection_score: float | None = None
    stopped_early: bool = False


class Trainer:
    """训练三任务模型，并保证冻结状态与优化器参数组始终一致。"""

    def __init__(
        self,
        model: MultiTaskHerbModel,
        *,
        epochs: int,
        learning_rate: float,
        encoder_learning_rate: float | None = None,
        weight_decay: float = 0.0,
        freeze_backbone: bool = True,
        unfreeze_epoch: int | None = None,
        loss_weights: LossWeights | None = None,
        label_smoothing: float = 0.0,
        gradient_clip: float | None = None,
        mixed_precision: bool = False,
        device: str | torch.device = "cpu",
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
        checkpoint_path: str | Path | None = None,
        checkpoint_contract: CheckpointContract | None = None,
        checkpoint_metadata: Mapping[str, Any] | None = None,
        protected_checkpoint_paths: tuple[str | Path, ...] = (),
        early_stop_patience: int | None = None,
        authenticity_class_weight: torch.Tensor | None = None,
        min_species_supervision_coverage: float = 0.2,
    ) -> None:
        if not isinstance(model, MultiTaskHerbModel):
            raise TypeError("model 必须是 MultiTaskHerbModel")
        _require_positive_integer(epochs, "epochs")
        _require_positive_number(learning_rate, "learning_rate")
        _require_nonnegative_number(weight_decay, "weight_decay")
        if not isinstance(freeze_backbone, bool):
            raise TypeError("freeze_backbone 必须是 bool")
        if not isinstance(mixed_precision, bool):
            raise TypeError("mixed_precision 必须是 bool")
        if freeze_backbone:
            if unfreeze_epoch is None:
                raise ValueError("冻结骨干时必须配置 unfreeze_epoch")
            _require_nonnegative_integer(unfreeze_epoch, "unfreeze_epoch")
        elif unfreeze_epoch is not None:
            _require_nonnegative_integer(unfreeze_epoch, "unfreeze_epoch")
        if encoder_learning_rate is None:
            encoder_learning_rate = learning_rate * 0.1
        _require_positive_number(encoder_learning_rate, "encoder_learning_rate")
        if gradient_clip is not None:
            _require_positive_number(gradient_clip, "gradient_clip")
        if not isinstance(label_smoothing, int | float) or isinstance(label_smoothing, bool):
            raise TypeError("label_smoothing 必须是位于 [0, 1) 的有限数值")
        if not math.isfinite(label_smoothing) or not 0 <= label_smoothing < 1:
            raise ValueError("label_smoothing 必须位于 [0, 1)")
        if early_stop_patience is not None:
            _require_positive_integer(early_stop_patience, "early_stop_patience")
        _require_probability(min_species_supervision_coverage, "min_species_supervision_coverage")

        self.model = model
        self.epochs = epochs
        self.learning_rate = float(learning_rate)
        self.encoder_learning_rate = float(encoder_learning_rate)
        self.weight_decay = float(weight_decay)
        self.freeze_backbone = freeze_backbone
        self.unfreeze_epoch = unfreeze_epoch
        self.loss_weights = loss_weights or LossWeights()
        self.label_smoothing = float(label_smoothing)
        self.gradient_clip = None if gradient_clip is None else float(gradient_clip)
        self.early_stop_patience = early_stop_patience
        self.min_species_supervision_coverage = float(min_species_supervision_coverage)
        self.device = torch.device(device)
        self.model.to(self.device)
        if authenticity_class_weight is not None:
            if not isinstance(authenticity_class_weight, torch.Tensor):
                raise TypeError("authenticity_class_weight 必须是 Tensor 或 None")
            if authenticity_class_weight.ndim != 1 or authenticity_class_weight.numel() != 2:
                raise ValueError("authenticity_class_weight 必须包含两个类别权重")
            if not bool(torch.isfinite(authenticity_class_weight).all().item()) or not bool(
                (authenticity_class_weight > 0).all().item()
            ):
                raise ValueError("authenticity_class_weight 必须是有限正数")
        self.authenticity_class_weight = (
            None
            if authenticity_class_weight is None
            else authenticity_class_weight.detach().to(self.device)
        )
        self.scheduler = scheduler
        self.checkpoint_path = (
            None if checkpoint_path is None else Path(checkpoint_path).expanduser().resolve()
        )
        self.checkpoint_contract = checkpoint_contract
        self.checkpoint_metadata = dict(checkpoint_metadata or {})
        self.protected_checkpoint_paths = protected_checkpoint_paths
        if self.checkpoint_path is not None and self.checkpoint_contract is None:
            raise ValueError("保存 checkpoint 必须提供 checkpoint_contract")
        if self.checkpoint_contract is not None and (
            self.checkpoint_contract.schema_version != SCHEMA_VERSION
        ):
            raise ValueError("checkpoint_contract schema_version 与当前实现不一致")
        self.global_step = 0
        self.best_epoch: int | None = None
        self.best_selection_score: float | None = None
        self._best_model_state: dict[str, Any] | None = None
        self._epochs_without_improvement = 0
        self._train_history: tuple[EpochResult, ...] = ()
        self._validation_history: tuple[EpochResult, ...] = ()

        if freeze_backbone:
            self.model.set_encoder_trainable(False)
            parameters = tuple(self.model.head_parameters())
            self.freeze_stage: FreezeStage = "heads_only"
        else:
            self.model.set_encoder_trainable(True)
            parameters = tuple(self.model.parameters())
            self.freeze_stage = "full_model"
        self.optimizer = optimizer or torch.optim.AdamW(
            parameters,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        self._validate_optimizer_parameters(parameters)

        self.amp_enabled = bool(
            mixed_precision and self.device.type == "cuda" and torch.cuda.is_available()
        )
        self._scaler = torch.amp.GradScaler("cuda", enabled=True) if self.amp_enabled else None

    def run_epoch(
        self,
        epoch: int,
        loader: Iterable[Mapping[str, Any]],
        *,
        training: bool,
    ) -> EpochResult:
        _require_nonnegative_integer(epoch, "epoch")
        self._update_freeze_stage(epoch)
        self.model.train(training)
        started = perf_counter()
        totals = {"total": 0.0, "claimed": 0.0, "species": 0.0, "authenticity": 0.0}
        batches = 0
        samples = 0
        supervised_rows = 0
        claimed_probabilities: list[torch.Tensor] = []
        species_probabilities: list[torch.Tensor] = []
        authenticity_probabilities: list[torch.Tensor] = []
        claimed_targets: list[torch.Tensor] = []
        species_targets: list[torch.Tensor] = []
        authenticity_targets: list[torch.Tensor] = []
        species_masks: list[torch.Tensor] = []
        sample_ids: list[str] = []
        group_ids: list[str] = []
        metadata_mode: bool | None = None

        gradient_context = torch.enable_grad() if training else torch.no_grad()
        with gradient_context:
            for batch in loader:
                images, targets, masks = self._prepare_batch(batch)
                batch_size = int(images.shape[0])
                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                amp_context = (
                    torch.autocast(device_type="cuda", dtype=torch.float16)
                    if self.amp_enabled
                    else nullcontext()
                )
                with amp_context:
                    logits = self.model(images)
                    losses = compute_multitask_loss(
                        logits,
                        targets,
                        masks,
                        self.loss_weights,
                        label_smoothing=self.label_smoothing,
                        authenticity_class_weight=self.authenticity_class_weight,
                    )
                if not bool(torch.isfinite(losses.total).item()):
                    raise RuntimeError(f"epoch={epoch} 出现非有限 total loss")
                if training:
                    self._backward_and_step(losses.total)
                    self.global_step += 1

                totals["total"] += float(losses.total.detach()) * batch_size
                totals["claimed"] += float(losses.claimed_herb.detach()) * batch_size
                totals["species"] += float(losses.actual_species.detach()) * batch_size
                totals["authenticity"] += float(losses.authenticity.detach()) * batch_size
                supervised_rows += losses.species_supervised_rows
                claimed_probabilities.append(
                    torch.softmax(logits.claimed_herb.detach(), dim=1).cpu()
                )
                species_probabilities.append(torch.sigmoid(logits.actual_species.detach()).cpu())
                authenticity_probabilities.append(
                    torch.softmax(logits.authenticity.detach(), dim=1).cpu()
                )
                claimed_targets.append(targets.claimed_herb.detach().cpu())
                species_targets.append(targets.actual_species.detach().cpu())
                authenticity_targets.append(targets.authenticity.detach().cpu())
                species_masks.append(masks.actual_species.detach().cpu())
                raw_metadata = batch.get("metadata")
                current_metadata_mode = raw_metadata is not None
                if metadata_mode is None:
                    metadata_mode = current_metadata_mode
                elif metadata_mode != current_metadata_mode:
                    raise ValueError("同一 epoch 的 batch 不得混用有/无 metadata")
                if raw_metadata is not None:
                    metadata = tuple(raw_metadata)
                    if len(metadata) != batch_size:
                        raise ValueError("batch metadata 数量与图片数量不一致")
                    for item in metadata:
                        if not isinstance(item, Mapping):
                            raise ValueError("batch metadata 必须是映射")
                        sample_id, group_id = item.get("sample_id"), item.get("group_id")
                        if not isinstance(sample_id, str) or not isinstance(group_id, str):
                            raise ValueError("batch metadata 缺少 sample_id 或 group_id")
                        sample_ids.append(sample_id)
                        group_ids.append(group_id)
                batches += 1
                samples += batch_size

        if batches == 0 or samples == 0:
            raise ValueError("DataLoader 不能为空")
        if not training:
            observed_authenticity = set(torch.cat(authenticity_targets).tolist())
            if observed_authenticity != {0, 1}:
                raise ValueError("验证集必须同时包含两个 authenticity 类别，才能选择 checkpoint")
        entity_metrics: EntityMetrics | None = None
        if metadata_mode:
            predictions = ImageLevelPredictions(
                claimed_probabilities=torch.cat(claimed_probabilities),
                species_probabilities=torch.cat(species_probabilities),
                authenticity_probabilities=torch.cat(authenticity_probabilities),
                claimed_targets=torch.cat(claimed_targets),
                species_targets=torch.cat(species_targets),
                authenticity_targets=torch.cat(authenticity_targets),
                species_mask=torch.cat(species_masks).bool(),
                sample_ids=tuple(sample_ids),
                group_ids=tuple(group_ids),
            )
            entity_metrics = compute_entity_metrics(
                aggregate_by_sample(predictions),
                min_species_supervision_coverage=self.min_species_supervision_coverage,
            )
        return EpochResult(
            epoch=epoch,
            training=training,
            total_loss=totals["total"] / samples,
            claimed_herb_loss=totals["claimed"] / samples,
            actual_species_loss=totals["species"] / samples,
            authenticity_loss=totals["authenticity"] / samples,
            species_supervised_rows=supervised_rows,
            batches=batches,
            samples=samples,
            freeze_stage=self.freeze_stage,
            learning_rates=tuple(float(group["lr"]) for group in self.optimizer.param_groups),
            duration_seconds=perf_counter() - started,
            entity_metrics=entity_metrics,
        )

    def fit(
        self,
        train_loader: Iterable[Mapping[str, Any]],
        val_loader: Iterable[Mapping[str, Any]],
        resume_from: str | Path | None = None,
    ) -> TrainingSummary:
        start_epoch = 0
        if resume_from is not None:
            start_epoch = self._restore_checkpoint(resume_from, train_loader)
            if start_epoch >= self.epochs:
                raise ValueError(
                    f"resume start_epoch={start_epoch} 必须小于 configured epochs={self.epochs}"
                )
            if self.checkpoint_path is None:
                self.checkpoint_path = Path(resume_from).expanduser().resolve()
        train_results = list(self._train_history)
        validation_results = list(self._validation_history)
        stopped_early = False
        for epoch in range(start_epoch, self.epochs):
            train_result = self.run_epoch(epoch, train_loader, training=True)
            validation_result = self.run_epoch(epoch, val_loader, training=False)
            train_results.append(train_result)
            validation_results.append(validation_result)
            self._train_history = tuple(train_results)
            self._validation_history = tuple(validation_results)
            selection_score = (
                validation_result.entity_metrics.selection_score
                if validation_result.entity_metrics is not None
                else -validation_result.total_loss
            )
            if self.best_selection_score is None or selection_score > self.best_selection_score:
                self.best_epoch = epoch
                self.best_selection_score = selection_score
                self._best_model_state = _clone_state_dict(self.model.state_dict())
                self._epochs_without_improvement = 0
            else:
                self._epochs_without_improvement += 1
            if self.scheduler is not None:
                self.scheduler.step()
            self._save_checkpoint(epoch, train_loader)
            if (
                self.early_stop_patience is not None
                and self._epochs_without_improvement >= self.early_stop_patience
            ):
                stopped_early = True
                break
        return TrainingSummary(
            train_epochs=tuple(train_results),
            validation_epochs=tuple(validation_results),
            amp_enabled=self.amp_enabled,
            device=str(self.device),
            best_epoch=self.best_epoch,
            best_selection_score=self.best_selection_score,
            stopped_early=stopped_early,
        )

    def _update_freeze_stage(self, epoch: int) -> None:
        if (
            self.freeze_stage == "heads_only"
            and self.unfreeze_epoch is not None
            and epoch >= self.unfreeze_epoch
        ):
            self.model.set_encoder_trainable(True)
            added = self.model.add_encoder_to_optimizer(
                self.optimizer,
                lr=self.encoder_learning_rate,
                weight_decay=self.weight_decay,
            )
            if added <= 0:
                raise RuntimeError("骨干解冻后没有参数加入优化器")
            self.freeze_stage = "full_model"
            self._sync_scheduler_parameter_groups()

    def _prepare_batch(
        self, batch: Mapping[str, Any]
    ) -> tuple[torch.Tensor, MultiTaskTargets, MultiTaskMasks]:
        try:
            images = batch["image"].to(self.device, non_blocking=True)
            targets = MultiTaskTargets(
                claimed_herb=batch["claimed_target"].to(self.device, non_blocking=True),
                actual_species=batch["actual_species_target"].to(self.device, non_blocking=True),
                authenticity=batch["authenticity_target"].to(self.device, non_blocking=True),
            )
            masks = MultiTaskMasks(
                actual_species=batch["actual_species_mask"].to(self.device, non_blocking=True)
            )
        except (AttributeError, KeyError, TypeError) as exc:
            raise ValueError("batch 缺少三任务训练所需的 Tensor 字段") from exc
        return images, targets, masks

    def _backward_and_step(self, loss: torch.Tensor) -> None:
        if self._scaler is not None:
            self._scaler.scale(loss).backward()
            if self.gradient_clip is not None:
                self._scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            self._scaler.step(self.optimizer)
            self._scaler.update()
            return
        loss.backward()
        if self.gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
        self.optimizer.step()

    def _validate_optimizer_parameters(self, expected: tuple[torch.nn.Parameter, ...]) -> None:
        actual_ids = {
            id(parameter) for group in self.optimizer.param_groups for parameter in group["params"]
        }
        expected_ids = {id(parameter) for parameter in expected}
        if not expected_ids <= actual_ids:
            raise ValueError("optimizer 缺少当前训练阶段所需参数")
        if self.freeze_stage == "heads_only":
            encoder_ids = {id(parameter) for parameter in self.model.encoder.parameters()}
            if actual_ids & encoder_ids:
                raise ValueError("heads_only 阶段 optimizer 不得包含 encoder 参数")

    def _sync_scheduler_parameter_groups(self) -> None:
        if self.scheduler is None:
            return
        base_lrs = getattr(self.scheduler, "base_lrs", None)
        if isinstance(base_lrs, list) and len(base_lrs) < len(self.optimizer.param_groups):
            base_lrs.append(self.encoder_learning_rate)
        last_lrs = getattr(self.scheduler, "_last_lr", None)
        if isinstance(last_lrs, list) and len(last_lrs) < len(self.optimizer.param_groups):
            last_lrs.append(self.encoder_learning_rate)

    def _save_checkpoint(self, epoch: int, train_loader: Iterable[Mapping[str, Any]]) -> None:
        if self.checkpoint_path is None:
            return
        if self.checkpoint_contract is None:  # 构造器已保证，仅用于类型收窄。
            raise RuntimeError("缺少 checkpoint_contract")
        generator = getattr(train_loader, "generator", None)
        generator_state = generator.get_state() if isinstance(generator, torch.Generator) else None
        state = {
            "schema_version": SCHEMA_VERSION,
            "metadata": self.checkpoint_metadata | self.checkpoint_contract.to_dict(),
            "training": {
                "epoch": epoch,
                "global_step": self.global_step,
                "freeze_stage": self.freeze_stage,
                "best_epoch": self.best_epoch,
                "best_selection_score": self.best_selection_score,
                "epochs_without_improvement": self._epochs_without_improvement,
                "history": {
                    "train_epochs": [asdict(result) for result in self._train_history],
                    "validation_epochs": [asdict(result) for result in self._validation_history],
                },
            },
            "model_state": self._best_model_state or _clone_state_dict(self.model.state_dict()),
            "resume_model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": (None if self.scheduler is None else self.scheduler.state_dict()),
            "scaler_state": None if self._scaler is None else self._scaler.state_dict(),
            "rng_state": capture_rng_state(),
            "dataloader_generator_state": generator_state,
        }
        save_checkpoint_atomic(
            self.checkpoint_path,
            state,
            protected_paths=self.protected_checkpoint_paths,
        )

    def _restore_checkpoint(
        self,
        path: str | Path,
        train_loader: Iterable[Mapping[str, Any]],
    ) -> int:
        if self.checkpoint_contract is None:
            raise ValueError("恢复 checkpoint 必须提供 checkpoint_contract")
        required_fingerprints = (
            self.checkpoint_contract.data_fingerprint,
            self.checkpoint_contract.split_fingerprint,
            self.checkpoint_contract.training_config_fingerprint,
        )
        if any(not value for value in required_fingerprints):
            raise CheckpointCompatibilityError("恢复训练必须提供数据、完整切分和训练配置指纹")
        state = load_checkpoint(path, self.checkpoint_contract)
        _validate_resume_state(
            state,
            has_scheduler=self.scheduler is not None,
            has_scaler=self._scaler is not None,
            has_loader_generator=isinstance(
                getattr(train_loader, "generator", None), torch.Generator
            ),
        )
        training = state["training"]
        if not isinstance(training, Mapping):
            raise ValueError("checkpoint training 必须是映射")
        stage = training.get("freeze_stage")
        if stage not in ("heads_only", "full_model"):
            raise ValueError("checkpoint freeze_stage 不受支持")
        if stage == "full_model" and self.freeze_stage == "heads_only":
            self.model.set_encoder_trainable(True)
            self.model.add_encoder_to_optimizer(
                self.optimizer,
                lr=self.encoder_learning_rate,
                weight_decay=self.weight_decay,
            )
            self.freeze_stage = "full_model"
            self._sync_scheduler_parameter_groups()
        elif stage == "heads_only" and self.freeze_stage != "heads_only":
            raise ValueError("checkpoint 为 heads_only，但当前配置从首轮训练全模型")

        self.model.load_state_dict(state["resume_model_state"])
        best_model_state = state.get("model_state")
        if isinstance(best_model_state, Mapping):
            self._best_model_state = _clone_state_dict(best_model_state)
        self.optimizer.load_state_dict(state["optimizer_state"])
        scheduler_state = state.get("scheduler_state")
        if scheduler_state is not None:
            if self.scheduler is None:
                raise ValueError("checkpoint 包含 scheduler 状态，但当前未配置 scheduler")
            self.scheduler.load_state_dict(scheduler_state)
        scaler_state = state.get("scaler_state")
        if scaler_state is not None:
            if self._scaler is None:
                raise ValueError("checkpoint 包含 AMP scaler 状态，但当前设备未启用 AMP")
            self._scaler.load_state_dict(scaler_state)

        self.global_step = int(training.get("global_step", 0))
        best_epoch = training.get("best_epoch")
        if best_epoch is not None and (
            isinstance(best_epoch, bool) or not isinstance(best_epoch, int) or best_epoch < 0
        ):
            raise ValueError("checkpoint best_epoch 必须是非负整数或 null")
        self.best_epoch = best_epoch
        best_score = training.get("best_selection_score")
        if best_score is not None and (
            isinstance(best_score, bool)
            or not isinstance(best_score, int | float)
            or not math.isfinite(best_score)
        ):
            raise ValueError("checkpoint best_selection_score 必须是有限数或 null")
        self.best_selection_score = None if best_score is None else float(best_score)
        epochs_without_improvement = training.get("epochs_without_improvement", 0)
        _require_nonnegative_integer(epochs_without_improvement, "epochs_without_improvement")
        self._epochs_without_improvement = epochs_without_improvement
        generator_state = state.get("dataloader_generator_state")
        if generator_state is not None:
            generator = getattr(train_loader, "generator", None)
            generator.set_state(generator_state)
        restore_rng_state(state["rng_state"])
        epoch = training.get("epoch")
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("checkpoint epoch 必须是非负整数")
        train_history, validation_history = _restore_training_history(training, epoch)
        if self.best_epoch is not None and self.best_epoch not in {
            result.epoch for result in validation_history
        }:
            raise CheckpointCompatibilityError("checkpoint best_epoch 无法在完整验证历史中追溯")
        self._train_history = train_history
        self._validation_history = validation_history
        return epoch + 1


def _restore_training_history(
    training: Mapping[str, Any],
    current_epoch: int,
) -> tuple[tuple[EpochResult, ...], tuple[EpochResult, ...]]:
    history = training.get("history")
    if not isinstance(history, Mapping):
        raise CheckpointCompatibilityError("resume checkpoint 缺少完整 training history")

    def finite_number(value: object, name: str, *, nonnegative: bool = False) -> float:
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
            or (nonnegative and value < 0)
        ):
            raise CheckpointCompatibilityError(f"history {name} 必须是有限数值")
        return float(value)

    def integer(value: object, name: str, *, positive: bool = False) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < (1 if positive else 0):
            raise CheckpointCompatibilityError(f"history {name} 必须是有效整数")
        return value

    def metrics_from(value: object) -> EntityMetrics | None:
        if value is None:
            return None
        if not isinstance(value, Mapping):
            raise CheckpointCompatibilityError("history entity_metrics 必须是映射或 null")
        species = value.get("species_macro_f1")
        tasks = value.get("selection_tasks")
        if not isinstance(tasks, list | tuple) or not all(isinstance(task, str) for task in tasks):
            raise CheckpointCompatibilityError("history selection_tasks 必须是字符串序列")
        return EntityMetrics(
            claimed_macro_f1=finite_number(value.get("claimed_macro_f1"), "claimed_macro_f1"),
            authenticity_macro_f1=finite_number(
                value.get("authenticity_macro_f1"), "authenticity_macro_f1"
            ),
            authenticity_balanced_accuracy=finite_number(
                value.get("authenticity_balanced_accuracy"),
                "authenticity_balanced_accuracy",
            ),
            species_macro_f1=(
                None if species is None else finite_number(species, "species_macro_f1")
            ),
            species_supervision_coverage=finite_number(
                value.get("species_supervision_coverage"),
                "species_supervision_coverage",
            ),
            selection_score=finite_number(value.get("selection_score"), "selection_score"),
            selection_tasks=tuple(tasks),
        )

    def epoch_from(value: object, expected_training: bool) -> EpochResult:
        if not isinstance(value, Mapping):
            raise CheckpointCompatibilityError("history epoch 必须是映射")
        if value.get("training") is not expected_training:
            raise CheckpointCompatibilityError("history training 标记不一致")
        stage = value.get("freeze_stage")
        if stage not in ("heads_only", "full_model"):
            raise CheckpointCompatibilityError("history freeze_stage 不受支持")
        raw_learning_rates = value.get("learning_rates")
        if not isinstance(raw_learning_rates, list | tuple) or not raw_learning_rates:
            raise CheckpointCompatibilityError("history learning_rates 必须是非空序列")
        return EpochResult(
            epoch=integer(value.get("epoch"), "epoch"),
            training=expected_training,
            total_loss=finite_number(value.get("total_loss"), "total_loss"),
            claimed_herb_loss=finite_number(value.get("claimed_herb_loss"), "claimed_herb_loss"),
            actual_species_loss=finite_number(
                value.get("actual_species_loss"), "actual_species_loss"
            ),
            authenticity_loss=finite_number(value.get("authenticity_loss"), "authenticity_loss"),
            species_supervised_rows=integer(
                value.get("species_supervised_rows"), "species_supervised_rows"
            ),
            batches=integer(value.get("batches"), "batches", positive=True),
            samples=integer(value.get("samples"), "samples", positive=True),
            freeze_stage=stage,
            learning_rates=tuple(
                finite_number(rate, "learning_rate", nonnegative=True)
                for rate in raw_learning_rates
            ),
            duration_seconds=finite_number(
                value.get("duration_seconds"), "duration_seconds", nonnegative=True
            ),
            entity_metrics=metrics_from(value.get("entity_metrics")),
        )

    raw_train = history.get("train_epochs")
    raw_validation = history.get("validation_epochs")
    if not isinstance(raw_train, list | tuple) or not isinstance(raw_validation, list | tuple):
        raise CheckpointCompatibilityError("training history 必须包含 train/validation 序列")
    train_epochs = tuple(epoch_from(value, True) for value in raw_train)
    validation_epochs = tuple(epoch_from(value, False) for value in raw_validation)
    expected = tuple(range(current_epoch + 1))
    if (
        tuple(result.epoch for result in train_epochs) != expected
        or tuple(result.epoch for result in validation_epochs) != expected
    ):
        raise CheckpointCompatibilityError("training history 必须从 epoch 0 连续到当前 epoch")
    return train_epochs, validation_epochs


def compute_authenticity_class_weights(
    targets: Sequence[int],
    classes: int = 2,
) -> torch.Tensor:
    if isinstance(classes, bool) or not isinstance(classes, int) or classes <= 0:
        raise ValueError("classes 必须是正整数")
    values = tuple(targets)
    if not values:
        raise ValueError("authenticity targets 不能为空")
    if any(
        isinstance(target, bool) or not isinstance(target, int) or target < 0 or target >= classes
        for target in values
    ):
        raise ValueError("authenticity target 必须是有效类别索引")
    counts = torch.bincount(torch.tensor(values, dtype=torch.long), minlength=classes)
    if bool((counts == 0).any().item()):
        raise ValueError("训练集必须同时包含每个 authenticity 类别")
    weights = len(values) / (classes * counts.to(dtype=torch.float32))
    if not bool(torch.isfinite(weights).all().item()):
        raise ValueError("authenticity class weights 必须为有限数")
    return weights


def _validate_resume_state(
    state: Mapping[str, Any],
    *,
    has_scheduler: bool,
    has_scaler: bool,
    has_loader_generator: bool,
) -> None:
    required_fields = (
        "resume_model_state",
        "rng_state",
        "dataloader_generator_state",
        "scheduler_state",
        "scaler_state",
    )
    for field in required_fields:
        if field not in state:
            raise CheckpointCompatibilityError(f"resume checkpoint 缺少关键状态：{field}")
    if not isinstance(state["resume_model_state"], Mapping):
        raise CheckpointCompatibilityError("resume_model_state 必须是映射")
    if not isinstance(state["rng_state"], Mapping):
        raise CheckpointCompatibilityError("rng_state 必须是映射")
    training = state.get("training")
    if not isinstance(training, Mapping):
        raise CheckpointCompatibilityError("checkpoint training 必须是映射")
    for field in (
        "epoch",
        "global_step",
        "freeze_stage",
        "best_epoch",
        "best_selection_score",
        "epochs_without_improvement",
        "history",
    ):
        if field not in training:
            raise CheckpointCompatibilityError(f"resume checkpoint 缺少训练状态：{field}")
    if has_scheduler and state["scheduler_state"] is None:
        raise CheckpointCompatibilityError("当前训练配置需要 scheduler_state")
    if has_scaler and state["scaler_state"] is None:
        raise CheckpointCompatibilityError("当前训练配置需要 scaler_state")
    generator_state = state["dataloader_generator_state"]
    if has_loader_generator and generator_state is None:
        raise CheckpointCompatibilityError("当前 DataLoader 需要 generator 状态")
    if not has_loader_generator and generator_state is not None:
        raise CheckpointCompatibilityError(
            "checkpoint 包含 DataLoader generator 状态，但当前 loader 不支持"
        )


def _clone_state_dict(state: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(name): value.detach().cpu().clone() if isinstance(value, torch.Tensor) else value
        for name, value in state.items()
    }


def _require_probability(value: float, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise ValueError(f"{name} 必须是 [0, 1] 内的有限概率")


def _require_positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} 必须是正整数")


def _require_nonnegative_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} 必须是非负整数")


def _require_positive_number(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} 必须是有限正数")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} 必须是有限正数")


def _require_nonnegative_number(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} 必须是有限非负数")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} 必须是有限非负数")
