"""只由训练分区冻结的多任务标签空间。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from ..encoding import write_json_utf8
from ..errors import ManifestValidationError
from .splitting import PARTITIONS, PartitionedImageRecord


@dataclass(frozen=True, slots=True)
class LabelSpace:
    """训练 taxonomy 及验证/测试中的开放集物种标签。"""

    claimed_classes: tuple[str, ...]
    authenticity_classes: tuple[str, str]
    actual_species_classes: tuple[str, ...]
    unknown_species_labels: tuple[str, ...]
    claimed_to_index: Mapping[str, int]
    species_to_index: Mapping[str, int]

    def __reduce__(self):
        return (
            _restore_label_space,
            (
                self.claimed_classes,
                self.authenticity_classes,
                self.actual_species_classes,
                self.unknown_species_labels,
            ),
        )


def _restore_label_space(
    claimed_classes: tuple[str, ...],
    authenticity_classes: tuple[str, str],
    actual_species_classes: tuple[str, ...],
    unknown_species_labels: tuple[str, ...],
) -> LabelSpace:
    return LabelSpace(
        claimed_classes=claimed_classes,
        authenticity_classes=authenticity_classes,
        actual_species_classes=actual_species_classes,
        unknown_species_labels=unknown_species_labels,
        claimed_to_index=MappingProxyType(
            {name: index for index, name in enumerate(claimed_classes)}
        ),
        species_to_index=MappingProxyType(
            {name: index for index, name in enumerate(actual_species_classes)}
        ),
    )


def build_label_space(rows: Sequence[PartitionedImageRecord]) -> LabelSpace:
    """从 train 冻结类别；val/test 只能报告而不能扩张 taxonomy。"""

    invalid_partitions = sorted({record.partition for record in rows} - set(PARTITIONS))
    if invalid_partitions:
        raise ManifestValidationError(f"存在非法 partition：{invalid_partitions}")

    training = tuple(record for record in rows if record.partition == "train")
    if not training:
        raise ManifestValidationError("训练分区为空，无法冻结标签空间。")

    claimed_classes = tuple(sorted({record.row.claimed_herb_name for record in training}))
    unseen_claimed = sorted(
        {
            record.row.claimed_herb_name
            for record in rows
            if record.partition != "train" and record.row.claimed_herb_name not in claimed_classes
        }
    )
    if unseen_claimed:
        raise ManifestValidationError(f"验证/测试包含训练未见的声称药材：{unseen_claimed}")

    actual_species_classes = tuple(
        sorted(
            {
                species
                for record in training
                if record.row.species_annotation_status == "verified"
                for species in record.row.actual_species
            }
        )
    )
    unknown_species_labels = tuple(
        sorted(
            {
                species
                for record in rows
                if record.partition != "train"
                and record.row.species_annotation_status == "verified"
                for species in record.row.actual_species
                if species not in actual_species_classes
            }
        )
    )
    claimed_to_index = MappingProxyType({name: index for index, name in enumerate(claimed_classes)})
    species_to_index = MappingProxyType(
        {name: index for index, name in enumerate(actual_species_classes)}
    )
    return LabelSpace(
        claimed_classes=claimed_classes,
        authenticity_classes=("authentic", "counterfeit"),
        actual_species_classes=actual_species_classes,
        unknown_species_labels=unknown_species_labels,
        claimed_to_index=claimed_to_index,
        species_to_index=species_to_index,
    )


def write_label_space(label_space: LabelSpace, path: str | Path) -> Path:
    """使用 UTF-8 写出顺序稳定、可供 checkpoint 校验的标签空间。"""

    destination = Path(path).expanduser().resolve()
    write_json_utf8(
        destination,
        {
            "schema_version": 1,
            "claimed_classes": label_space.claimed_classes,
            "authenticity_classes": label_space.authenticity_classes,
            "actual_species_classes": label_space.actual_species_classes,
            "unknown_species_labels": label_space.unknown_species_labels,
            "claimed_to_index": dict(label_space.claimed_to_index),
            "species_to_index": dict(label_space.species_to_index),
        },
    )
    return destination
