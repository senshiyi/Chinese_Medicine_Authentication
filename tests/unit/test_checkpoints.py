from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest
import torch
from herb_authentication.errors import CheckpointCompatibilityError
from herb_authentication.training import checkpoints
from herb_authentication.training.checkpoints import (
    CheckpointContract,
    load_checkpoint,
    save_checkpoint_atomic,
)


def contract() -> CheckpointContract:
    return CheckpointContract(
        schema_version=1,
        backbone="tiny",
        claimed_herb_classes=("当归",),
        actual_species_classes=("当归", "欧当归"),
        authenticity_classes=("authentic", "counterfeit"),
        image_size=8,
        preprocessing_version="test-v1",
        data_fingerprint="data-a",
        split_fingerprint="split-a",
        training_config_fingerprint="training-a",
    )


def valid_state() -> dict[str, object]:
    return {
        "schema_version": 1,
        "metadata": contract().to_dict()
        | {
            "provenance": "real",
            "source_encoding": "utf-8",
            "fingerprints": {"manifest": "abc", "split": "def", "images": "ghi"},
            "calibration": {"is_calibrated": False, "method": None, "parameters": {}},
        },
        "training": {"epoch": 2, "global_step": 7, "freeze_stage": "heads_only"},
        "model_state": {"weight": torch.tensor([1.0])},
        "optimizer_state": {"state": {}, "param_groups": []},
        "scheduler_state": None,
        "scaler_state": None,
        "rng_state": None,
        "dataloader_generator_state": None,
    }


def file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("legacy_name", ["best_model.pth", "epoch_001.pth"])
def test_save_rejects_both_legacy_paths(tmp_path: Path, legacy_name: str) -> None:
    legacy_path = tmp_path / legacy_name
    legacy_path.write_bytes(b"legacy-model")
    before = file_hash(legacy_path)

    with pytest.raises(CheckpointCompatibilityError, match="旧模型"):
        save_checkpoint_atomic(
            legacy_path,
            valid_state(),
            protected_paths=(tmp_path / "best_model.pth", tmp_path / "epoch_001.pth"),
        )

    assert file_hash(legacy_path) == before


def test_atomic_replace_failure_preserves_previous_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "new.pth"
    save_checkpoint_atomic(path, valid_state(), protected_paths=())
    before = path.read_bytes()

    def fail_replace(_source: object, _target: object) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(checkpoints.os, "replace", fail_replace)
    updated = valid_state()
    updated["training"] = {"epoch": 3}

    with pytest.raises(OSError, match="simulated"):
        save_checkpoint_atomic(path, updated, protected_paths=())

    assert path.read_bytes() == before


def test_old_single_head_checkpoint_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "old.pth"
    torch.save({"model_state_dict": {}}, path)

    with pytest.raises(CheckpointCompatibilityError, match="schema_version"):
        load_checkpoint(path, contract())


def test_corrupted_checkpoint_uses_domain_error(tmp_path: Path) -> None:
    path = tmp_path / "corrupted.pth"
    path.write_bytes(b"not-a-checkpoint")

    with pytest.raises(CheckpointCompatibilityError, match="安全加载"):
        load_checkpoint(path, contract())


def test_load_rejects_contract_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "new.pth"
    save_checkpoint_atomic(path, valid_state(), protected_paths=())
    wrong = CheckpointContract(**(contract().to_dict() | {"image_size": 224}))

    with pytest.raises(CheckpointCompatibilityError, match="image_size"):
        load_checkpoint(path, wrong)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("data_fingerprint", "data-b"),
        ("split_fingerprint", "split-b"),
        ("training_config_fingerprint", "training-b"),
    ],
)
def test_resume_contract_rejects_same_vocab_with_different_inputs(
    tmp_path: Path, field: str, replacement: str
) -> None:
    path = tmp_path / "resume.pth"
    save_checkpoint_atomic(path, valid_state(), protected_paths=())

    with pytest.raises(CheckpointCompatibilityError, match=field):
        load_checkpoint(path, replace(contract(), **{field: replacement}))


def test_save_rejects_numpy_objects(tmp_path: Path) -> None:
    state = valid_state()
    state["unsafe"] = np.array([1, 2, 3])

    with pytest.raises(CheckpointCompatibilityError, match="不安全"):
        save_checkpoint_atomic(tmp_path / "unsafe.pth", state, protected_paths=())
