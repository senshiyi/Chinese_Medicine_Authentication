from __future__ import annotations

import json
from dataclasses import replace

import pytest
from herb_authentication.data import labels
from herb_authentication.data.labels import build_label_space
from herb_authentication.data.splitting import PartitionedImageRecord
from herb_authentication.errors import ManifestValidationError
from test_splitting import make_record


def partitioned(record, partition: str) -> PartitionedImageRecord:
    return PartitionedImageRecord(record=record, partition=partition)


def test_label_space_is_frozen_from_training_partition() -> None:
    authentic = make_record("g1", 0)
    counterfeit = make_record("g2", 1, authenticity="counterfeit")
    unseen_species = make_record("g3", 2)
    unseen_species = replace(
        unseen_species,
        row=replace(unseen_species.row, actual_species=("川芎",)),
    )

    label_space = build_label_space(
        (
            partitioned(authentic, "train"),
            partitioned(counterfeit, "train"),
            partitioned(unseen_species, "val"),
        )
    )

    assert label_space.claimed_classes == ("当归",)
    assert label_space.authenticity_classes == ("authentic", "counterfeit")
    assert label_space.actual_species_classes == ("当归", "欧当归")
    assert label_space.unknown_species_labels == ("川芎",)
    assert dict(label_space.claimed_to_index) == {"当归": 0}
    assert dict(label_space.species_to_index) == {"当归": 0, "欧当归": 1}


def test_validation_claimed_class_must_exist_in_train() -> None:
    train = make_record("g1", 0, claimed_herb_name="当归")
    validation = make_record("g2", 1, claimed_herb_name="黄芪")

    with pytest.raises(ManifestValidationError, match="训练未见.*黄芪"):
        build_label_space((partitioned(train, "train"), partitioned(validation, "val")))


def test_unverified_training_species_is_not_added_to_taxonomy() -> None:
    record = make_record("g1", 0)
    record = replace(
        record,
        row=replace(
            record.row,
            actual_species=("未核验物种",),
            species_annotation_status="unverified",
        ),
    )

    label_space = build_label_space((partitioned(record, "train"),))

    assert label_space.actual_species_classes == ()


def test_label_space_requires_training_rows() -> None:
    validation = partitioned(make_record("g1", 0), "val")

    with pytest.raises(ManifestValidationError, match="训练分区为空"):
        build_label_space((validation,))


def test_label_space_json_preserves_chinese_and_class_order(tmp_path) -> None:
    train = partitioned(make_record("g1", 0), "train")
    validation_record = make_record("g2", 1)
    validation_record = replace(
        validation_record,
        row=replace(validation_record.row, actual_species=("川芎",)),
    )
    label_space = build_label_space((train, partitioned(validation_record, "val")))
    output = tmp_path / "label_space.json"

    labels.write_label_space(label_space, output)

    raw = output.read_bytes()
    assert "当归".encode() in raw
    assert b"\\u5f53\\u5f52" not in raw
    payload = json.loads(raw.decode("utf-8"))
    assert payload["claimed_classes"] == ["当归"]
    assert payload["unknown_species_labels"] == ["川芎"]
