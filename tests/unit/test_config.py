from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from herb_authentication.config import load_config
from herb_authentication.errors import ConfigurationError


def write_config(tmp_path: Path, **sections: object) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(sections, allow_unicode=True), encoding="utf-8")
    return path


def test_config_rejects_unknown_nested_key(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("training:\n  unknow_key: 1\n", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="training.unknow_key"):
        load_config(path)


def test_config_rejects_invalid_split_ratio_sum(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "data:\n  train_ratio: 0.8\n  val_ratio: 0.2\n  test_ratio: 0.2\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="切分比例之和"):
        load_config(path)


def test_config_resolves_relative_paths_from_project_root_idempotently(tmp_path) -> None:
    config_dir = tmp_path / "settings"
    config_dir.mkdir()
    path = config_dir / "config.yaml"
    path.write_text(
        "project_root: '..'\n"
        "data:\n"
        "  manifest_path: 'data/metadata.csv'\n"
        "inference:\n"
        "  checkpoint_path: 'models/model.pth'\n",
        encoding="utf-8",
    )

    first = load_config(path)
    second = load_config(path.resolve())

    assert first == second
    assert first.project_root == tmp_path.resolve()
    assert first.data.manifest_path == (tmp_path / "data" / "metadata.csv").resolve()
    assert first.inference.checkpoint_path == (tmp_path / "models" / "model.pth").resolve()


def test_empty_config_uses_safe_multitask_defaults(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("", encoding="utf-8")

    config = load_config(path)

    assert config.data.provenance == "unknown"
    assert (
        config.data.train_ratio + config.data.val_ratio + config.data.test_ratio == pytest.approx(1)
    )
    assert config.data.num_workers == 0
    assert config.inference.checkpoint_path.name == "best_multitask_model.pth"
    assert config.inference.allow_uncalibrated is False
    assert isinstance(config.project_root, Path)


def test_closed_set_config_rejects_double_class_balancing(tmp_path: Path) -> None:
    path = write_config(
        tmp_path,
        data={"use_weighted_sampler": True},
        training={"use_authenticity_class_weights": True},
        calibration={"mode": "closed_set_temperature"},
    )

    with pytest.raises(ConfigurationError, match="只能启用一种"):
        load_config(path)


def test_closed_set_config_rejects_unknown_manifest(tmp_path: Path) -> None:
    path = write_config(
        tmp_path,
        data={"unknown_manifest_path": "unknown.csv", "use_weighted_sampler": False},
        calibration={"mode": "closed_set_temperature"},
    )

    with pytest.raises(ConfigurationError, match="data.unknown_manifest_path"):
        load_config(path)


def test_promotion_thresholds_are_probabilities(tmp_path: Path) -> None:
    path = write_config(tmp_path, promotion={"min_macro_f1": 1.1})

    with pytest.raises(ConfigurationError, match="promotion.min_macro_f1"):
        load_config(path)


@pytest.mark.parametrize("invalid_weight", [-0.1, float("nan"), float("inf")])
def test_task_loss_weights_must_be_finite_nonnegative(
    tmp_path: Path, invalid_weight: float
) -> None:
    path = write_config(tmp_path, training={"claimed_herb_weight": invalid_weight})

    with pytest.raises(ConfigurationError, match="training.claimed_herb_weight"):
        load_config(path)


def test_task_loss_weights_must_not_all_be_zero(tmp_path: Path) -> None:
    path = write_config(
        tmp_path,
        training={
            "claimed_herb_weight": 0.0,
            "actual_species_weight": 0.0,
            "authenticity_weight": 0.0,
        },
    )

    with pytest.raises(ConfigurationError, match="不能同时为 0"):
        load_config(path)


def test_run_and_promotion_paths_are_resolved_from_project_root(tmp_path: Path) -> None:
    path = write_config(
        tmp_path,
        project_root=".",
        run={"root_dir": "runs"},
        promotion={"target_checkpoint_path": "models/official.pth"},
    )

    config = load_config(path)

    assert config.run.root_dir == (tmp_path / "runs").resolve()
    assert config.promotion.target_checkpoint_path == (tmp_path / "models/official.pth").resolve()


@pytest.mark.parametrize(
    ("section", "field"),
    [("run", "root_dir"), ("promotion", "target_checkpoint_path")],
)
def test_run_and_promotion_paths_reject_non_path_values(
    tmp_path: Path, section: str, field: str
) -> None:
    path = write_config(tmp_path, **{section: {field: 123}})

    with pytest.raises(ConfigurationError, match=rf"{section}\.{field}"):
        load_config(path)


def test_split_algorithm_defaults_to_replayable_v1(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path))

    assert config.data.split_algorithm_version == "group-stratified-v1"


def test_split_algorithm_accepts_explicit_component_v2(tmp_path: Path) -> None:
    config = load_config(
        write_config(
            tmp_path,
            data={"split_algorithm_version": ("group-stratified-near-duplicate-components-v2")},
        )
    )

    assert config.data.split_algorithm_version == "group-stratified-near-duplicate-components-v2"


def test_split_algorithm_rejects_unsupported_version(tmp_path: Path) -> None:
    path = write_config(tmp_path, data={"split_algorithm_version": "forged-v999"})

    with pytest.raises(ConfigurationError, match="data.split_algorithm_version"):
        load_config(path)
