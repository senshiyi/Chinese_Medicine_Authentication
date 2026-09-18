"""安全、版本化、原子写入并可恢复随机状态的 checkpoint。"""

from __future__ import annotations

import os
import pickle
import random
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from herb_authentication.errors import CheckpointCompatibilityError

SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class CheckpointContract:
    schema_version: int
    backbone: str
    claimed_herb_classes: tuple[str, ...]
    actual_species_classes: tuple[str, ...]
    authenticity_classes: tuple[str, ...]
    image_size: int
    preprocessing_version: str
    data_fingerprint: str | None = None
    split_fingerprint: str | None = None
    training_config_fingerprint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result = {
            "schema_version": self.schema_version,
            "backbone": self.backbone,
            "claimed_herb_classes": self.claimed_herb_classes,
            "actual_species_classes": self.actual_species_classes,
            "authenticity_classes": self.authenticity_classes,
            "image_size": self.image_size,
            "preprocessing_version": self.preprocessing_version,
        }
        optional_fingerprints = {
            "data_fingerprint": self.data_fingerprint,
            "split_fingerprint": self.split_fingerprint,
            "training_config_fingerprint": self.training_config_fingerprint,
        }
        result.update(
            (name, value) for name, value in optional_fingerprints.items() if value is not None
        )
        return result


def save_checkpoint_atomic(
    path: str | Path,
    state: Mapping[str, Any],
    protected_paths: Sequence[str | Path],
) -> Path:
    """校验后写同目录临时文件，并用原子替换发布。"""

    target = Path(path).expanduser().resolve()
    protected = {
        os.path.normcase(str(Path(item).expanduser().resolve())) for item in protected_paths
    }
    if os.path.normcase(str(target)) in protected:
        raise CheckpointCompatibilityError(f"不允许覆盖旧模型：{target}")
    _validate_checkpoint_state(state)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(dict(state), temporary)
        verified = torch.load(temporary, map_location="cpu", weights_only=True)
        _validate_checkpoint_state(verified)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def load_checkpoint(
    path: str | Path,
    expected_contract: CheckpointContract,
) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise CheckpointCompatibilityError(f"checkpoint 不存在：{source}")
    try:
        state = torch.load(source, map_location="cpu", weights_only=True)
    except (EOFError, OSError, RuntimeError, TypeError, ValueError, pickle.UnpicklingError) as exc:
        raise CheckpointCompatibilityError(f"无法安全加载 checkpoint：{source}") from exc
    _validate_checkpoint_state(state)
    _validate_contract(state, expected_contract)
    return dict(state)


def capture_rng_state() -> dict[str, Any]:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "algorithm": numpy_state[0],
            "keys": torch.from_numpy(numpy_state[1].astype(np.int64)),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": tuple(torch.cuda.get_rng_state_all()) if torch.cuda.is_available() else (),
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    try:
        random.setstate(tuple(state["python"]))
        numpy_state = state["numpy"]
        np.random.set_state(
            (
                str(numpy_state["algorithm"]),
                numpy_state["keys"].cpu().numpy().astype(np.uint32),
                int(numpy_state["position"]),
                int(numpy_state["has_gauss"]),
                float(numpy_state["cached_gaussian"]),
            )
        )
        torch.set_rng_state(state["torch_cpu"].cpu())
        cuda_state = tuple(state.get("torch_cuda", ()))
        if cuda_state:
            if not torch.cuda.is_available():
                raise CheckpointCompatibilityError("checkpoint 包含 CUDA RNG，但当前无可用 CUDA")
            torch.cuda.set_rng_state_all(list(cuda_state))
    except CheckpointCompatibilityError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise CheckpointCompatibilityError("checkpoint RNG 状态不完整或不兼容") from exc


def _validate_checkpoint_state(state: Any) -> None:
    if not isinstance(state, Mapping):
        raise CheckpointCompatibilityError("checkpoint 根节点必须是映射")
    if "schema_version" not in state:
        raise CheckpointCompatibilityError("checkpoint 缺少 schema_version，可能是旧单头模型")
    if state["schema_version"] != SCHEMA_VERSION:
        raise CheckpointCompatibilityError(f"不支持的 schema_version：{state['schema_version']!r}")
    for required in ("metadata", "training", "model_state", "optimizer_state"):
        if required not in state:
            raise CheckpointCompatibilityError(f"checkpoint 缺少字段：{required}")
    _validate_safe_tree(state, "checkpoint")


def _validate_contract(state: Mapping[str, Any], expected: CheckpointContract) -> None:
    metadata = state.get("metadata")
    if not isinstance(metadata, Mapping):
        raise CheckpointCompatibilityError("checkpoint metadata 必须是映射")
    for field, expected_value in expected.to_dict().items():
        actual = metadata.get(field) if field != "schema_version" else state.get(field)
        if isinstance(expected_value, tuple) and isinstance(actual, list):
            actual = tuple(actual)
        if actual != expected_value:
            raise CheckpointCompatibilityError(
                f"checkpoint 契约不兼容：{field}，期望 {expected_value!r}，实际 {actual!r}"
            )


def _validate_safe_tree(value: Any, path: str) -> None:
    if value is None or isinstance(value, bool | int | float | str | torch.Tensor):
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str | int):
                raise CheckpointCompatibilityError(f"checkpoint 包含不安全键：{path}")
            _validate_safe_tree(item, f"{path}.{key}")
        return
    if isinstance(value, list | tuple):
        for index, item in enumerate(value):
            _validate_safe_tree(item, f"{path}[{index}]")
        return
    raise CheckpointCompatibilityError(
        f"checkpoint 包含不安全对象：{path} ({type(value).__name__})"
    )
