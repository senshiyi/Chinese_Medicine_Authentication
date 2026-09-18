"""三任务损失及实际物种监督掩码。"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from herb_authentication.models import MultiTaskLogits


@dataclass(frozen=True, slots=True)
class MultiTaskTargets:
    claimed_herb: torch.Tensor
    actual_species: torch.Tensor
    authenticity: torch.Tensor


@dataclass(frozen=True, slots=True)
class MultiTaskMasks:
    actual_species: torch.Tensor


@dataclass(frozen=True, slots=True)
class LossWeights:
    claimed_herb: float = 1.0
    actual_species: float = 1.0
    authenticity: float = 1.0

    def __post_init__(self) -> None:
        for name in ("claimed_herb", "actual_species", "authenticity"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise TypeError(f"{name} loss 权重必须是有限非负数")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} loss 权重必须是有限非负数")
        if self.claimed_herb == self.actual_species == self.authenticity == 0:
            raise ValueError("至少一个任务的 loss 权重必须大于零")


@dataclass(frozen=True, slots=True)
class LossBreakdown:
    total: torch.Tensor
    claimed_herb: torch.Tensor
    actual_species: torch.Tensor
    authenticity: torch.Tensor
    species_supervised_rows: int


def masked_species_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    *,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    if logits.ndim != 2 or targets.shape != logits.shape:
        raise ValueError("实际物种 logits 与 targets 必须是相同的二维形状")
    if mask.dtype != torch.bool:
        raise ValueError("实际物种 mask 必须是 bool Tensor")
    if mask.ndim != 1 or mask.shape[0] != logits.shape[0]:
        raise ValueError("实际物种 mask 必须是一维且与 batch 大小一致")
    if not bool(mask.any()):
        return logits.sum() * 0.0
    element_loss = F.binary_cross_entropy_with_logits(
        logits[mask],
        targets[mask],
        pos_weight=pos_weight,
        reduction="none",
    )
    return element_loss.mean()


def compute_multitask_loss(
    logits: MultiTaskLogits,
    targets: MultiTaskTargets,
    masks: MultiTaskMasks,
    weights: LossWeights,
    *,
    label_smoothing: float = 0.0,
    claimed_class_weight: torch.Tensor | None = None,
    authenticity_class_weight: torch.Tensor | None = None,
    species_pos_weight: torch.Tensor | None = None,
) -> LossBreakdown:
    claimed = F.cross_entropy(
        logits.claimed_herb,
        targets.claimed_herb,
        weight=claimed_class_weight,
        label_smoothing=label_smoothing,
    )
    species = masked_species_loss(
        logits.actual_species,
        targets.actual_species,
        masks.actual_species,
        pos_weight=species_pos_weight,
    )
    authenticity = F.cross_entropy(
        logits.authenticity,
        targets.authenticity,
        weight=authenticity_class_weight,
        label_smoothing=label_smoothing,
    )
    total = (
        weights.claimed_herb * claimed
        + weights.actual_species * species
        + weights.authenticity * authenticity
    )
    return LossBreakdown(
        total=total,
        claimed_herb=claimed,
        actual_species=species,
        authenticity=authenticity,
        species_supervised_rows=int(masks.actual_species.sum().item()),
    )
