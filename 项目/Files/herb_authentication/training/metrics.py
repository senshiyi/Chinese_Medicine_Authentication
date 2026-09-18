"""从图片级概率聚合到实体级指标，并按相关组计算 bootstrap 区间。"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class ImageLevelPredictions:
    claimed_probabilities: torch.Tensor
    species_probabilities: torch.Tensor
    authenticity_probabilities: torch.Tensor
    claimed_targets: torch.Tensor
    species_targets: torch.Tensor
    authenticity_targets: torch.Tensor
    species_mask: torch.Tensor
    sample_ids: tuple[str, ...]
    group_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AggregatedPredictions(ImageLevelPredictions):
    """按 sample_id 聚合后的实体级预测。"""


@dataclass(frozen=True, slots=True)
class EntityMetrics:
    claimed_macro_f1: float
    authenticity_macro_f1: float
    authenticity_balanced_accuracy: float
    species_macro_f1: float | None
    species_supervision_coverage: float
    selection_score: float
    selection_tasks: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BootstrapInterval:
    lower: float
    median: float
    upper: float
    samples: int
    seed: int


def aggregate_by_sample(predictions: ImageLevelPredictions) -> AggregatedPredictions:
    _validate_predictions(predictions)
    ordered_ids = tuple(dict.fromkeys(predictions.sample_ids))
    rows: list[dict[str, object]] = []
    for sample_id in ordered_ids:
        indexes = [i for i, value in enumerate(predictions.sample_ids) if value == sample_id]
        first = indexes[0]
        _require_consistent_sample(predictions, sample_id, indexes)
        index = torch.tensor(
            indexes, dtype=torch.long, device=predictions.claimed_probabilities.device
        )
        rows.append(
            {
                "claimed": predictions.claimed_probabilities.index_select(0, index).mean(0),
                "species": predictions.species_probabilities.index_select(0, index).mean(0),
                "authenticity": predictions.authenticity_probabilities.index_select(0, index).mean(
                    0
                ),
                "claimed_target": predictions.claimed_targets[first],
                "species_target": predictions.species_targets[first],
                "authenticity_target": predictions.authenticity_targets[first],
                "species_mask": predictions.species_mask[first],
                "group_id": predictions.group_ids[first],
            }
        )
    return AggregatedPredictions(
        claimed_probabilities=torch.stack([row["claimed"] for row in rows]),
        species_probabilities=torch.stack([row["species"] for row in rows]),
        authenticity_probabilities=torch.stack([row["authenticity"] for row in rows]),
        claimed_targets=torch.stack([row["claimed_target"] for row in rows]),
        species_targets=torch.stack([row["species_target"] for row in rows]),
        authenticity_targets=torch.stack([row["authenticity_target"] for row in rows]),
        species_mask=torch.stack([row["species_mask"] for row in rows]),
        sample_ids=ordered_ids,
        group_ids=tuple(str(row["group_id"]) for row in rows),
    )


def compute_entity_metrics(
    predictions: AggregatedPredictions,
    *,
    min_species_supervision_coverage: float,
    species_threshold: float = 0.5,
) -> EntityMetrics:
    _validate_predictions(predictions)
    _validate_probability(min_species_supervision_coverage, "min_species_supervision_coverage")
    _validate_probability(species_threshold, "species_threshold")
    claimed = _multiclass_macro_f1(predictions.claimed_probabilities, predictions.claimed_targets)
    authenticity = _multiclass_macro_f1(
        predictions.authenticity_probabilities, predictions.authenticity_targets
    )
    authenticity_balanced_accuracy = _balanced_accuracy(
        predictions.authenticity_probabilities.argmax(dim=1),
        predictions.authenticity_targets,
        predictions.authenticity_probabilities.shape[1],
    )
    coverage = float(predictions.species_mask.float().mean().item())
    species: float | None = None
    if bool(predictions.species_mask.any()):
        species = _multilabel_macro_f1(
            predictions.species_probabilities[predictions.species_mask],
            predictions.species_targets[predictions.species_mask],
            threshold=species_threshold,
        )
    return EntityMetrics(
        claimed_macro_f1=claimed,
        authenticity_macro_f1=authenticity,
        authenticity_balanced_accuracy=authenticity_balanced_accuracy,
        species_macro_f1=species,
        species_supervision_coverage=coverage,
        selection_score=(authenticity + authenticity_balanced_accuracy) / 2.0,
        selection_tasks=("authenticity_macro_f1", "authenticity_balanced_accuracy"),
    )


def harmonic_mean(values: tuple[float, ...] | list[float]) -> float:
    if not values:
        raise ValueError("调和平均至少一个值")
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("调和平均只接受有限非负值")
    if any(value == 0 for value in values):
        return 0.0
    return len(values) / sum(1.0 / value for value in values)


def group_bootstrap_selection_score(
    predictions: AggregatedPredictions,
    *,
    min_species_supervision_coverage: float,
    samples: int,
    seed: int,
) -> BootstrapInterval:
    _validate_predictions(predictions)
    if isinstance(samples, bool) or not isinstance(samples, int) or samples <= 0:
        raise ValueError("bootstrap samples 必须是正整数")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("bootstrap seed 必须是整数")
    groups = tuple(dict.fromkeys(predictions.group_ids))
    group_indexes = {
        group: [i for i, value in enumerate(predictions.group_ids) if value == group]
        for group in groups
    }
    generator = torch.Generator().manual_seed(seed)
    scores: list[float] = []
    for _ in range(samples):
        draws = torch.randint(len(groups), (len(groups),), generator=generator).tolist()
        indexes = [index for draw in draws for index in group_indexes[groups[draw]]]
        subset = _select_rows(predictions, indexes)
        scores.append(
            compute_entity_metrics(
                subset,
                min_species_supervision_coverage=min_species_supervision_coverage,
            ).selection_score
        )
    quantiles = torch.quantile(
        torch.tensor(scores, dtype=torch.float64),
        torch.tensor([0.025, 0.5, 0.975], dtype=torch.float64),
    )
    return BootstrapInterval(
        lower=float(quantiles[0]),
        median=float(quantiles[1]),
        upper=float(quantiles[2]),
        samples=samples,
        seed=seed,
    )


def _select_rows(predictions: AggregatedPredictions, indexes: list[int]) -> AggregatedPredictions:
    index = torch.tensor(indexes, dtype=torch.long, device=predictions.claimed_probabilities.device)
    return AggregatedPredictions(
        claimed_probabilities=predictions.claimed_probabilities.index_select(0, index),
        species_probabilities=predictions.species_probabilities.index_select(0, index),
        authenticity_probabilities=predictions.authenticity_probabilities.index_select(0, index),
        claimed_targets=predictions.claimed_targets.index_select(0, index),
        species_targets=predictions.species_targets.index_select(0, index),
        authenticity_targets=predictions.authenticity_targets.index_select(0, index),
        species_mask=predictions.species_mask.index_select(0, index),
        sample_ids=tuple(predictions.sample_ids[i] for i in indexes),
        group_ids=tuple(predictions.group_ids[i] for i in indexes),
    )


def _multiclass_macro_f1(probabilities: torch.Tensor, targets: torch.Tensor) -> float:
    predicted = probabilities.argmax(dim=1)
    return _macro_f1(predicted, targets, probabilities.shape[1])


def _multilabel_macro_f1(
    probabilities: torch.Tensor, targets: torch.Tensor, *, threshold: float
) -> float:
    predicted = probabilities >= threshold
    truth = targets >= 0.5
    scores = [_binary_f1(predicted[:, i], truth[:, i]) for i in range(probabilities.shape[1])]
    return sum(scores) / len(scores)


def _macro_f1(predicted: torch.Tensor, targets: torch.Tensor, classes: int) -> float:
    scores = [_binary_f1(predicted == i, targets == i) for i in range(classes)]
    return sum(scores) / len(scores)


def _balanced_accuracy(predicted: torch.Tensor, targets: torch.Tensor, classes: int) -> float:
    recalls: list[float] = []
    for index in range(classes):
        class_rows = targets == index
        denominator = int(class_rows.sum().item())
        recalls.append(
            0.0
            if denominator == 0
            else int(((predicted == index) & class_rows).sum().item()) / denominator
        )
    return sum(recalls) / len(recalls)


def _binary_f1(predicted: torch.Tensor, targets: torch.Tensor) -> float:
    true_positive = int((predicted & targets).sum().item())
    false_positive = int((predicted & ~targets).sum().item())
    false_negative = int((~predicted & targets).sum().item())
    denominator = 2 * true_positive + false_positive + false_negative
    return 0.0 if denominator == 0 else 2 * true_positive / denominator


def _require_consistent_sample(
    predictions: ImageLevelPredictions, sample_id: str, indexes: list[int]
) -> None:
    first = indexes[0]
    for index in indexes[1:]:
        consistent = (
            predictions.group_ids[index] == predictions.group_ids[first]
            and torch.equal(predictions.claimed_targets[index], predictions.claimed_targets[first])
            and torch.equal(predictions.species_targets[index], predictions.species_targets[first])
            and torch.equal(
                predictions.authenticity_targets[index], predictions.authenticity_targets[first]
            )
            and torch.equal(predictions.species_mask[index], predictions.species_mask[first])
        )
        if not consistent:
            raise ValueError(f"sample_id={sample_id!r} 的 group 或标签不一致")


def _validate_predictions(predictions: ImageLevelPredictions) -> None:
    size = len(predictions.sample_ids)
    if size == 0 or len(predictions.group_ids) != size:
        raise ValueError("预测必须包含等长且非空的 sample_ids 与 group_ids")
    tensors = (
        predictions.claimed_probabilities,
        predictions.species_probabilities,
        predictions.authenticity_probabilities,
        predictions.claimed_targets,
        predictions.species_targets,
        predictions.authenticity_targets,
        predictions.species_mask,
    )
    if any(tensor.shape[0] != size for tensor in tensors):
        raise ValueError("所有预测、标签和掩码的首维必须一致")
    if predictions.species_mask.dtype != torch.bool:
        raise ValueError("species_mask 必须是 bool Tensor")


def _validate_probability(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float) or not 0 <= value <= 1:
        raise ValueError(f"{name} 必须位于 [0, 1]")
