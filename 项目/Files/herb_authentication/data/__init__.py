"""数据清单、切分与 Dataset 契约。"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .datasets import HerbImageDataset
    from .transforms import build_eval_transform, build_train_transform

from .labels import LabelSpace, build_label_space, write_label_space
from .manifest import (
    KnownManifestRow,
    ManifestBundle,
    UnknownManifestRow,
    ValidatedImageRecord,
    ValidatedKnownRecord,
    ValidatedUnknownRecord,
    load_known_manifest,
    load_unknown_manifest,
    validate_manifest_pair,
)
from .preparation import ManifestBuildResult, ManifestSource, build_closed_set_manifest
from .splitting import (
    V1_ALGORITHM_VERSION,
    V2_ALGORITHM_VERSION,
    PartitionedImageRecord,
    SplitArtifacts,
    SplitAssignment,
    SplitAudit,
    apply_split,
    create_group_splits,
    split_assignment_from_payload,
    split_assignment_to_payload,
    write_split_artifacts,
)

__all__ = [
    "KnownManifestRow",
    "LabelSpace",
    "ManifestBundle",
    "ManifestBuildResult",
    "ManifestSource",
    "HerbImageDataset",
    "PartitionedImageRecord",
    "SplitArtifacts",
    "SplitAssignment",
    "SplitAudit",
    "UnknownManifestRow",
    "V1_ALGORITHM_VERSION",
    "V2_ALGORITHM_VERSION",
    "ValidatedImageRecord",
    "ValidatedKnownRecord",
    "ValidatedUnknownRecord",
    "apply_split",
    "build_closed_set_manifest",
    "build_label_space",
    "build_eval_transform",
    "build_train_transform",
    "collate_herb_batch",
    "create_dataloader_generator",
    "create_herb_dataloader",
    "create_group_splits",
    "load_known_manifest",
    "load_unknown_manifest",
    "seed_worker",
    "split_assignment_from_payload",
    "split_assignment_to_payload",
    "validate_manifest_pair",
    "write_label_space",
    "write_split_artifacts",
]

_DATASET_EXPORTS = {
    "HerbImageDataset",
    "collate_herb_batch",
    "create_dataloader_generator",
    "create_herb_dataloader",
    "seed_worker",
}
_TRANSFORM_EXPORTS = {
    "build_eval_transform",
    "build_train_transform",
}


def __getattr__(name: str) -> Any:
    """延迟加载 Torch/torchvision，避免 manifest-only 命令提前导入重依赖。"""

    if name in _DATASET_EXPORTS:
        from . import datasets

        value = getattr(datasets, name)
        globals()[name] = value
        return value
    if name in _TRANSFORM_EXPORTS:
        from . import transforms

        value = getattr(transforms, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
