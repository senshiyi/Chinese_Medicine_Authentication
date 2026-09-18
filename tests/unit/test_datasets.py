from __future__ import annotations

import hashlib
import pickle
import random
from dataclasses import replace

import numpy as np
import pytest
import torch
from herb_authentication.data import datasets as datasets_module
from herb_authentication.data.datasets import (
    HerbImageDataset,
    create_dataloader_generator,
    seed_worker,
)
from herb_authentication.data.labels import build_label_space
from herb_authentication.data.splitting import PartitionedImageRecord
from herb_authentication.data.transforms import build_eval_transform, build_train_transform
from herb_authentication.errors import InputImageError, ManifestValidationError
from PIL import Image
from test_splitting import make_record
from torch.utils.data import DataLoader


def partitioned(record, partition: str = "train") -> PartitionedImageRecord:
    return PartitionedImageRecord(record=record, partition=partition)


def training_label_space():
    authentic = partitioned(make_record("g1", 0), "train")
    counterfeit = partitioned(make_record("g2", 1, authenticity="counterfeit"), "train")
    return build_label_space((authentic, counterfeit))


def test_validation_transform_cannot_mutate_training_transform() -> None:
    row = partitioned(make_record("g1", 0))
    label_space = training_label_space()
    train = HerbImageDataset(
        (row,),
        label_space,
        transform=lambda image: f"train:{image}",
        image_loader=lambda _path: "raw",
    )
    validation = HerbImageDataset(
        (replace(row, partition="val"),),
        label_space,
        transform=lambda image: f"val:{image}",
        image_loader=lambda _path: "raw",
    )

    assert train[0]["image"] == "train:raw"
    assert validation[0]["image"] == "val:raw"
    assert train[0]["image"] == "train:raw"


def test_unverified_species_has_false_mask_and_zero_target() -> None:
    record = make_record("g1", 0)
    record = replace(
        record,
        row=replace(record.row, species_annotation_status="unverified"),
    )
    dataset = HerbImageDataset(
        (partitioned(record),),
        training_label_space(),
        transform=None,
        image_loader=lambda _path: "raw",
    )

    item = dataset[0]

    assert item["actual_species_mask"].item() is False
    assert item["actual_species_target"].sum().item() == 0


def test_verified_multilabel_species_target_is_fixed_length() -> None:
    record = make_record("g3", 2)
    record = replace(
        record,
        row=replace(record.row, actual_species=("当归", "欧当归")),
    )
    dataset = HerbImageDataset(
        (partitioned(record, "val"),),
        training_label_space(),
        transform=None,
        image_loader=lambda _path: "raw",
    )

    item = dataset[0]

    assert item["actual_species_target"].tolist() == [1.0, 1.0]
    assert item["actual_species_mask"].item() is True
    assert item["unknown_species_labels"] == ()


def test_unseen_verified_species_is_reported_and_excluded_from_closed_loss() -> None:
    record = make_record("g3", 2)
    record = replace(
        record,
        row=replace(record.row, actual_species=("川芎",)),
    )
    dataset = HerbImageDataset(
        (partitioned(record, "test"),),
        training_label_space(),
        transform=None,
        image_loader=lambda _path: "raw",
    )

    item = dataset[0]

    assert item["actual_species_target"].sum().item() == 0
    assert item["actual_species_mask"].item() is False
    assert item["unknown_species_labels"] == ("川芎",)


def test_dataset_returns_three_targets_and_entity_metadata() -> None:
    row = partitioned(make_record("g1", 0))
    dataset = HerbImageDataset(
        (row,),
        training_label_space(),
        transform=None,
        image_loader=lambda _path: "raw",
    )

    item = dataset[0]

    assert item["claimed_target"].dtype == torch.long
    assert item["authenticity_target"].dtype == torch.long
    assert item["actual_species_target"].dtype == torch.float32
    assert item["metadata"] == {
        "image_path": row.row.image_path,
        "group_id": "g1",
        "sample_id": "sample-g1",
        "partition": "train",
        "claimed_herb_name": "当归",
    }


def test_unknown_claimed_class_fails_with_domain_error() -> None:
    record = partitioned(make_record("g3", 2, claimed_herb_name="黄芪"), "val")
    dataset = HerbImageDataset(
        (record,),
        training_label_space(),
        transform=None,
        image_loader=lambda _path: "raw",
    )

    with pytest.raises(ManifestValidationError, match="标签空间中不存在"):
        dataset[0]


def test_default_image_loader_wraps_error_with_path() -> None:
    row = partitioned(make_record("missing", 0))
    dataset = HerbImageDataset((row,), training_label_space(), transform=None)

    with pytest.raises(InputImageError, match="missing-0.png"):
        dataset[0]


def test_train_and_eval_transform_factories_are_independent() -> None:
    train_transform = build_train_transform(24)
    eval_transform = build_eval_transform(24)
    image = Image.new("RGB", (40, 32), color=(128, 64, 32))

    assert train_transform is not eval_transform
    assert train_transform(image).shape == (3, 24, 24)
    assert eval_transform(image).shape == (3, 24, 24)
    assert torch.equal(eval_transform(image), eval_transform(image))


def test_dataloader_generator_is_seeded_deterministically() -> None:
    first = create_dataloader_generator(123)
    second = create_dataloader_generator(123)

    assert torch.equal(torch.rand(5, generator=first), torch.rand(5, generator=second))


def test_seed_worker_seeds_python_and_numpy_deterministically(monkeypatch) -> None:
    monkeypatch.setattr(torch, "initial_seed", lambda: 2**32 + 123)

    seed_worker(0)
    first = (random.random(), np.random.random())
    seed_worker(0)
    second = (random.random(), np.random.random())

    assert first == second


def test_default_image_loader_converts_to_rgb(tmp_path) -> None:
    image_path = tmp_path / "grayscale.png"
    Image.new("L", (8, 8), color=127).save(image_path)
    payload = image_path.read_bytes()
    record = replace(
        make_record("g1", 0),
        row=replace(make_record("g1", 0).row, image_path=image_path),
        sha256=hashlib.sha256(payload).hexdigest(),
        file_size=len(payload),
        width=8,
        height=8,
    )
    dataset = HerbImageDataset(
        (partitioned(record),),
        training_label_space(),
        transform=None,
    )

    item = dataset[0]

    assert item["image"].mode == "RGB"


def test_default_image_loader_wraps_decompression_bomb(tmp_path, monkeypatch) -> None:
    image_path = tmp_path / "bomb.png"
    Image.new("RGB", (16, 16), color=(127, 64, 32)).save(image_path)
    payload = image_path.read_bytes()
    record = replace(
        make_record("g1", 0),
        row=replace(make_record("g1", 0).row, image_path=image_path),
        sha256=hashlib.sha256(payload).hexdigest(),
        file_size=len(payload),
        width=16,
        height=16,
    )
    dataset = HerbImageDataset(
        (partitioned(record),),
        training_label_space(),
        transform=None,
    )
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 10)

    with pytest.raises(InputImageError, match="解压炸弹"):
        dataset[0]


def test_default_loader_rejects_image_replaced_after_manifest_validation(tmp_path) -> None:
    image_path = tmp_path / "validated.png"
    Image.new("RGB", (8, 8), color=(10, 20, 30)).save(image_path)
    original = image_path.read_bytes()
    record = make_record("g1", 0)
    record = replace(
        record,
        row=replace(record.row, image_path=image_path),
        sha256=hashlib.sha256(original).hexdigest(),
        file_size=len(original),
        width=8,
        height=8,
    )
    dataset = HerbImageDataset(
        (partitioned(record),),
        training_label_space(),
        transform=None,
    )
    Image.new("RGB", (8, 8), color=(200, 100, 50)).save(image_path)

    with pytest.raises(InputImageError, match="验证后发生变化"):
        dataset[0]


def test_default_image_loader_wraps_corrupted_file(tmp_path) -> None:
    image_path = tmp_path / "corrupted.png"
    image_path.write_bytes(b"not-an-image")
    record = make_record("g1", 0)
    record = replace(record, row=replace(record.row, image_path=image_path))
    dataset = HerbImageDataset(
        (partitioned(record),),
        training_label_space(),
        transform=None,
    )

    with pytest.raises(InputImageError, match="corrupted.png"):
        dataset[0]


def test_data_package_exports_task5_public_api() -> None:
    import herb_authentication.data as data

    assert data.HerbImageDataset is HerbImageDataset
    assert data.build_train_transform is build_train_transform
    assert data.build_eval_transform is build_eval_transform
    assert data.collate_herb_batch is datasets_module.collate_herb_batch
    assert data.create_dataloader_generator is create_dataloader_generator
    assert data.create_herb_dataloader is datasets_module.create_herb_dataloader
    assert data.seed_worker is seed_worker


@pytest.mark.parametrize(
    ("image_size", "message"),
    ((0, "正整数"), (-1, "正整数"), (True, "正整数")),
)
def test_transform_factory_rejects_invalid_image_size(image_size, message) -> None:
    with pytest.raises(ValueError, match=message):
        build_eval_transform(image_size)


@pytest.mark.parametrize(
    "kwargs",
    (
        {"mean": (float("nan"), 0.0, 0.0)},
        {"mean": (float("inf"), 0.0, 0.0)},
        {"std": (0.2, float("nan"), 0.2)},
        {"std": (0.2, float("inf"), 0.2)},
        {"std": (0.2, float("-inf"), 0.2)},
    ),
)
def test_transform_factory_rejects_nonfinite_normalization(kwargs) -> None:
    with pytest.raises(ValueError, match="有限数"):
        build_eval_transform(24, **kwargs)


def test_label_space_and_default_dataset_are_pickleable_for_windows_workers() -> None:
    label_space = training_label_space()
    row = partitioned(make_record("g1", 0))
    dataset = HerbImageDataset((row,), label_space, transform=None)

    restored_label_space = pickle.loads(pickle.dumps(label_space))
    restored_dataset = pickle.loads(pickle.dumps(dataset))

    assert dict(restored_label_space.claimed_to_index) == {"当归": 0}
    assert dict(restored_label_space.species_to_index) == {"当归": 0, "欧当归": 1}
    assert len(restored_dataset) == 1
    assert restored_dataset.rows[0] == row


def test_dataloader_factory_always_applies_required_collate() -> None:
    known = partitioned(make_record("g3", 2), "val")
    unknown_record = make_record("g4", 3)
    unknown_record = replace(
        unknown_record,
        row=replace(unknown_record.row, actual_species=("川芎",)),
    )
    dataset = HerbImageDataset(
        (known, partitioned(unknown_record, "test")),
        training_label_space(),
        transform=lambda _image: torch.zeros(3, 8, 8),
        image_loader=lambda _path: "raw",
    )

    loader = datasets_module.create_herb_dataloader(
        dataset,
        batch_size=2,
        shuffle=False,
        seed=123,
        num_workers=0,
    )
    batch = next(iter(loader))

    assert loader.collate_fn is datasets_module.collate_herb_batch
    assert batch["unknown_species_labels"] == ((), ("川芎",))
    assert batch["actual_species_mask"].tolist() == [True, False]


def test_dataloader_factory_rejects_missing_tensor_transform() -> None:
    dataset = HerbImageDataset(
        (partitioned(make_record("g1", 0)),),
        training_label_space(),
        transform=None,
    )

    with pytest.raises(ValueError, match="返回 Tensor"):
        datasets_module.create_herb_dataloader(
            dataset,
            batch_size=1,
            shuffle=False,
            seed=123,
        )


def test_dataloader_factory_shuffle_order_is_reproducible() -> None:
    rows = tuple(partitioned(make_record(f"g{index}", index)) for index in range(6))
    dataset = HerbImageDataset(
        rows,
        training_label_space(),
        transform=lambda _image: torch.zeros(3, 8, 8),
        image_loader=lambda _path: "raw",
    )
    kwargs = {"batch_size": 2, "shuffle": True, "seed": 123, "num_workers": 0}

    first = datasets_module.create_herb_dataloader(dataset, **kwargs)
    second = datasets_module.create_herb_dataloader(dataset, **kwargs)
    first_order = tuple(meta["sample_id"] for batch in first for meta in batch["metadata"])
    second_order = tuple(meta["sample_id"] for batch in second for meta in batch["metadata"])

    assert first_order == second_order


def test_custom_collate_supports_variable_unknown_species_labels() -> None:
    known = partitioned(make_record("g3", 2), "val")
    unknown_record = make_record("g4", 3)
    unknown_record = replace(
        unknown_record,
        row=replace(unknown_record.row, actual_species=("川芎",)),
    )
    unknown = partitioned(unknown_record, "test")
    dataset = HerbImageDataset(
        (known, unknown),
        training_label_space(),
        transform=lambda _image: torch.zeros(3, 8, 8),
        image_loader=lambda _path: "raw",
    )
    loader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=0,
        collate_fn=datasets_module.collate_herb_batch,
    )

    batch = next(iter(loader))

    assert batch["image"].shape == (2, 3, 8, 8)
    assert batch["claimed_target"].shape == (2,)
    assert batch["authenticity_target"].shape == (2,)
    assert batch["actual_species_target"].shape == (2, 2)
    assert batch["actual_species_mask"].tolist() == [True, False]
    assert batch["unknown_species_labels"] == ((), ("川芎",))
    assert tuple(item["sample_id"] for item in batch["metadata"]) == (
        "sample-g3",
        "sample-g4",
    )
