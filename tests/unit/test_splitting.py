from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import replace
from itertools import product
from pathlib import Path

import pytest
from herb_authentication.data import splitting
from herb_authentication.data.manifest import KnownManifestRow, ValidatedKnownRecord
from herb_authentication.data.splitting import apply_split, create_group_splits
from herb_authentication.errors import ManifestValidationError


def make_record(
    group_id: str,
    index: int,
    *,
    claimed_herb_name: str = "当归",
    authenticity: str = "authentic",
    sample_id: str | None = None,
    sha256: str | None = None,
    perceptual_hash: str | None = None,
) -> ValidatedKnownRecord:
    digest = hashlib.sha256(f"{group_id}-{index}".encode()).hexdigest()
    row = KnownManifestRow(
        image_path=Path(f"{group_id}-{index}.png"),
        group_id=group_id,
        sample_id=sample_id or f"sample-{group_id}",
        claimed_herb_name=claimed_herb_name,
        authenticity=authenticity,
        actual_species=("当归",) if authenticity == "authentic" else ("欧当归",),
        species_annotation_status="verified",
        batch_id=None,
        origin=None,
        capture_device=None,
        collected_at=None,
        annotator=None,
        notes=None,
        source_line=index + 2,
    )
    return ValidatedKnownRecord(
        row=row,
        sha256=sha256 or digest,
        perceptual_hash=perceptual_hash or digest[:16],
        image_format="PNG",
        width=16,
        height=16,
        file_size=100,
    )


def make_rows() -> tuple[ValidatedKnownRecord, ...]:
    rows: list[ValidatedKnownRecord] = []
    for authenticity in ("authentic", "counterfeit"):
        prefix = "a" if authenticity == "authentic" else "c"
        for group_index in range(6):
            group_id = f"{prefix}{group_index}"
            rows.extend(
                make_record(group_id, image_index, authenticity=authenticity)
                for image_index in range(2)
            )
    return tuple(rows)


def test_split_has_no_group_sample_or_hash_leakage() -> None:
    rows = make_rows()

    assignment = create_group_splits(rows, ratios=(0.5, 0.25, 0.25), seed=7)
    partitioned = apply_split(rows, assignment)

    for key in (
        lambda item: item.row.group_id,
        lambda item: item.row.sample_id,
        lambda item: item.sha256,
    ):
        owners: dict[str, str] = {}
        for item in partitioned:
            value = key(item)
            previous = owners.setdefault(value, item.partition)
            assert previous == item.partition


def test_split_is_independent_of_input_order() -> None:
    rows = make_rows()

    first = create_group_splits(rows, ratios=(0.5, 0.25, 0.25), seed=7)
    second = create_group_splits(tuple(reversed(rows)), ratios=(0.5, 0.25, 0.25), seed=7)

    assert dict(first.by_group) == dict(second.by_group)
    assert first.input_fingerprint == second.input_fingerprint


def test_each_stratum_appears_in_every_partition() -> None:
    rows = tuple(make_record(f"g{index}", index) for index in range(3))

    assignment = create_group_splits(rows, seed=11)

    assert set(assignment.by_group.values()) == {"train", "val", "test"}


def test_split_rejects_mixed_label_group() -> None:
    rows = (
        make_record("g1", 0, claimed_herb_name="当归"),
        make_record("g1", 1, claimed_herb_name="黄芪"),
        make_record("g2", 2),
        make_record("g3", 3),
    )

    with pytest.raises(ManifestValidationError, match="混合分层"):
        create_group_splits(rows)


def test_split_rejects_stratum_with_fewer_than_three_groups() -> None:
    rows = (make_record("g1", 0), make_record("g2", 1))

    with pytest.raises(ManifestValidationError, match="至少需要 3"):
        create_group_splits(rows)


def test_split_rejects_near_duplicate_images_across_groups() -> None:
    rows = tuple(
        make_record(f"g{index}", index, perceptual_hash="0000000000000000") for index in range(3)
    )

    with pytest.raises(ManifestValidationError, match="感知近重复"):
        create_group_splits(rows)


def test_apply_split_rejects_missing_group_assignment() -> None:
    rows = make_rows()
    assignment = create_group_splits(rows, ratios=(0.5, 0.25, 0.25), seed=7)
    missing_group_rows = rows + (make_record("not-assigned", 99),)

    with pytest.raises(ManifestValidationError, match="没有切分分配"):
        apply_split(missing_group_rows, assignment)


def test_split_artifacts_are_utf8_sig_and_auditable(tmp_path) -> None:
    rows = make_rows()
    assignment = create_group_splits(rows, ratios=(0.5, 0.25, 0.25), seed=7)

    artifacts = splitting.write_split_artifacts(rows, assignment, tmp_path)

    for path in (artifacts.train_csv, artifacts.val_csv, artifacts.test_csv):
        assert path.read_bytes().startswith(b"\xef\xbb\xbf")
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            records = tuple(csv.DictReader(handle))
        assert records
        expected_partition = path.stem
        assert {record["partition"] for record in records} == {expected_partition}

    payload = json.loads(artifacts.splits_json.read_text(encoding="utf-8"))
    assert payload["algorithm_version"] == assignment.algorithm_version
    assert payload["seed"] == 7
    assert payload["ratios"] == {"train": 0.5, "val": 0.25, "test": 0.25}
    assert payload["input_fingerprint"] == assignment.input_fingerprint
    assert payload["by_group"] == dict(sorted(assignment.by_group.items()))


def test_invalid_split_ratios_are_rejected() -> None:
    with pytest.raises(ManifestValidationError, match="比例之和必须为 1"):
        create_group_splits(make_rows(), ratios=(0.8, 0.2, 0.2))


def test_locked_split_rejects_changed_input_fingerprint() -> None:
    rows = make_rows()
    assignment = create_group_splits(rows, ratios=(0.5, 0.25, 0.25), seed=7)
    changed = replace(
        rows[0],
        sha256="f" * 64,
        perceptual_hash="f0f0f0f0f0f0f0f0",
    )

    with pytest.raises(ManifestValidationError, match="输入数据指纹"):
        apply_split((changed, *rows[1:]), assignment)


def test_locked_split_requires_every_stratum_in_all_partitions() -> None:
    rows = make_rows()
    assignment = create_group_splits(rows, ratios=(0.5, 0.25, 0.25), seed=7)
    all_training = replace(
        assignment,
        by_group={group_id: "train" for group_id in assignment.by_group},
    )

    with pytest.raises(ManifestValidationError, match="覆盖 train、val、test"):
        apply_split(rows, all_training)


def test_locked_split_fingerprint_tracks_manifest_audit_metadata() -> None:
    rows = make_rows()
    assignment = create_group_splits(rows, ratios=(0.5, 0.25, 0.25), seed=7)
    changed = replace(
        rows[0],
        row=replace(
            rows[0].row,
            batch_id="新批次",
            origin="新来源",
            capture_device="新设备",
            collected_at="2026-07-29",
            annotator="复核员",
            notes="审计元数据已改变",
        ),
    )

    with pytest.raises(ManifestValidationError, match="输入数据指纹"):
        apply_split((changed, *rows[1:]), assignment)


def test_locked_split_rejects_unsupported_algorithm_version() -> None:
    rows = make_rows()
    assignment = create_group_splits(rows, ratios=(0.5, 0.25, 0.25), seed=7)
    unsupported = replace(assignment, algorithm_version="unsupported-v999")

    with pytest.raises(ManifestValidationError, match="算法版本"):
        apply_split(rows, unsupported)


def test_locked_split_rejects_invalid_recorded_ratios() -> None:
    rows = make_rows()
    assignment = create_group_splits(rows, ratios=(0.5, 0.25, 0.25), seed=7)
    invalid = replace(assignment, ratios=(1.0, 0.0, 0.0))

    with pytest.raises(ManifestValidationError, match="比例"):
        apply_split(rows, invalid)


def test_locked_split_rejects_ratios_that_cannot_reproduce_mapping() -> None:
    rows = make_rows()
    assignment = create_group_splits(rows, ratios=(0.5, 0.25, 0.25), seed=7)
    forged = replace(assignment, ratios=(0.7, 0.2, 0.1))

    with pytest.raises(ManifestValidationError, match="无法重现 group 映射"):
        apply_split(rows, forged)


def test_locked_split_rejects_seed_that_cannot_reproduce_mapping() -> None:
    rows = make_rows()
    assignment = create_group_splits(rows, ratios=(0.5, 0.25, 0.25), seed=7)
    forged = replace(assignment, seed=999_999)

    with pytest.raises(ManifestValidationError, match="无法重现 group 映射"):
        apply_split(rows, forged)


@pytest.mark.parametrize(
    "invalid_hash",
    (
        "-000000000000001",
        "+000000000000001",
        "0x00000000000001",
        "00000000000000000",
        "ABCDEF0123456789",
        "not-a-hash-value",
    ),
)
def test_split_rejects_noncanonical_perceptual_hash(invalid_hash: str) -> None:
    rows = (
        make_record("g0", 0, perceptual_hash=invalid_hash),
        make_record("g1", 1, perceptual_hash="0f0f0f0f0f0f0f0f"),
        make_record("g2", 2, perceptual_hash="ffffffffffffffff"),
    )

    with pytest.raises(ManifestValidationError, match="16 位小写十六进制"):
        create_group_splits(rows)


def test_perceptual_hash_distance_four_is_rejected() -> None:
    rows = (
        make_record("g0", 0, perceptual_hash="0000000000000000"),
        make_record("g1", 1, perceptual_hash="000000000000000f"),
        make_record("g2", 2, perceptual_hash="ffffffffffffffff"),
    )

    with pytest.raises(ManifestValidationError, match="距离 4"):
        create_group_splits(rows)


def test_perceptual_hash_distance_five_is_not_rejected() -> None:
    rows = (
        make_record("g0", 0, perceptual_hash="0000000000000000"),
        make_record("g1", 1, perceptual_hash="000000000000001f"),
        make_record("g2", 2, perceptual_hash="ffffffffffffffff"),
    )

    assignment = create_group_splits(rows)

    assert set(assignment.by_group.values()) == {"train", "val", "test"}


def test_configured_perceptual_hash_distance_is_applied_and_audited(tmp_path) -> None:
    rows = (
        make_record("g0", 0, perceptual_hash="0000000000000000"),
        make_record("g1", 1, perceptual_hash="0000000000000001"),
        make_record("g2", 2, perceptual_hash="ffffffffffffffff"),
    )

    assignment = create_group_splits(rows, perceptual_hash_distance=0)
    artifacts = splitting.write_split_artifacts(rows, assignment, tmp_path)
    payload = json.loads(artifacts.splits_json.read_text(encoding="utf-8"))

    assert assignment.perceptual_hash_distance == 0
    assert payload["perceptual_hash_distance"] == 0
    assert set(assignment.by_group.values()) == {"train", "val", "test"}


@pytest.mark.parametrize("invalid_distance", (-1, 65, True))
def test_split_rejects_invalid_perceptual_hash_distance(invalid_distance) -> None:
    with pytest.raises(ManifestValidationError, match="感知哈希距离"):
        create_group_splits(make_rows(), perceptual_hash_distance=invalid_distance)


def test_species_label_order_does_not_change_input_fingerprint() -> None:
    rows = make_rows()
    first = replace(
        rows[0],
        row=replace(rows[0].row, actual_species=("当归", "欧当归")),
    )
    normalized_rows = (first, *rows[1:])
    assignment = create_group_splits(normalized_rows, ratios=(0.5, 0.25, 0.25), seed=7)
    reordered = replace(first, row=replace(first.row, actual_species=("欧当归", "当归")))

    apply_split((reordered, *rows[1:]), assignment)


def test_largest_remainder_uses_expected_group_counts() -> None:
    rows = tuple(make_record(f"g{index}", index) for index in range(10))

    assignment = create_group_splits(rows, ratios=(0.7, 0.2, 0.1), seed=7)
    counts = {
        partition: tuple(assignment.by_group.values()).count(partition)
        for partition in ("train", "val", "test")
    }

    assert counts == {"train": 7, "val": 2, "test": 1}


def test_data_package_exports_task4_public_api() -> None:
    import herb_authentication.data as data

    assert data.create_group_splits is create_group_splits
    assert data.apply_split is apply_split
    assert data.PartitionedImageRecord is splitting.PartitionedImageRecord
    assert data.build_label_space.__name__ == "build_label_space"
    assert data.write_split_artifacts is splitting.write_split_artifacts


def _v2_targets(
    train: tuple[int, int],
    val: tuple[int, int],
    test: tuple[int, int],
) -> dict[str, dict[str, int]]:
    return {
        "train": {"authentic": train[0], "counterfeit": train[1]},
        "val": {"authentic": val[0], "counterfeit": val[1]},
        "test": {"authentic": test[0], "counterfeit": test[1]},
    }


_V2_BASE_HASHES = (
    "0000000000000000",
    "ffff000000000000",
    "0000ffff00000000",
    "00000000ffff0000",
    "000000000000ffff",
    "aaaaaaaaaaaaaaaa",
    "5555555555555555",
    "cccccccccccccccc",
    "3333333333333333",
    "f0f0f0f0f0f0f0f0",
    "0f0f0f0f0f0f0f0f",
    "ff00ff00ff00ff00",
    "00ff00ff00ff00ff",
)


def _v2_rows(
    authentic: int,
    counterfeit: int,
    *,
    hashes: dict[str, str] | None = None,
) -> tuple[ValidatedKnownRecord, ...]:
    overrides = hashes or {}
    rows = []
    for label, count, prefix, offset in (
        ("authentic", authentic, "a", 0),
        ("counterfeit", counterfeit, "c", authentic),
    ):
        for index in range(count):
            group_id = f"{prefix}{index}"
            rows.append(
                make_record(
                    group_id,
                    offset + index,
                    authenticity=label,
                    sample_id=f"sample-{group_id}",
                    perceptual_hash=overrides.get(
                        group_id,
                        hashlib.sha256(f"v2-{group_id}".encode()).hexdigest()[:16],
                    ),
                )
            )
    return tuple(rows)


def _v2_assignment(
    rows: tuple[ValidatedKnownRecord, ...],
    targets: dict[str, dict[str, int]],
    *,
    distance: int = 4,
):
    return create_group_splits(
        rows,
        ratios=(0.5, 0.25, 0.25),
        seed=42,
        perceptual_hash_distance=distance,
        algorithm_version=splitting.V2_ALGORITHM_VERSION,
        target_counts=targets,
    )


def test_v2_threshold_is_inclusive_at_four_but_not_five() -> None:
    targets = _v2_targets((4, 4), (1, 1), (1, 1))
    at_four = _v2_rows(
        6,
        6,
        hashes={"a0": "0000000000000000", "a1": "000000000000000f"},
    )
    at_five = _v2_rows(
        6,
        6,
        hashes={"a0": "0000000000000000", "a1": "000000000000001f"},
    )

    four = _v2_assignment(at_four, targets)
    five = _v2_assignment(at_five, targets)

    assert four.audit is not None and five.audit is not None
    assert four.audit.component_by_group["a0"] == four.audit.component_by_group["a1"]
    assert five.audit.component_by_group["a0"] != five.audit.component_by_group["a1"]


def test_v2_transitive_near_duplicate_chain_stays_atomic() -> None:
    rows = _v2_rows(
        6,
        6,
        hashes={
            "a0": "0000000000000000",
            "a1": "000000000000000f",
            "a2": "000000000000001f",
        },
    )

    assignment = _v2_assignment(rows, _v2_targets((4, 4), (1, 1), (1, 1)))

    assert len({assignment.by_group[group_id] for group_id in ("a0", "a1", "a2")}) == 1
    assert assignment.audit is not None
    assert (
        len({assignment.audit.component_by_group[group_id] for group_id in ("a0", "a1", "a2")}) == 1
    )


def test_v2_mapping_is_independent_of_input_order() -> None:
    rows = _v2_rows(
        6,
        6,
        hashes={"a0": "0000000000000000", "a1": "000000000000000f"},
    )
    targets = _v2_targets((4, 4), (1, 1), (1, 1))

    first = _v2_assignment(rows, targets)
    second = _v2_assignment(tuple(reversed(rows)), targets)

    assert dict(first.by_group) == dict(second.by_group)
    assert first.input_fingerprint == second.input_fingerprint
    assert first.audit == second.audit


def test_v2_preserves_unique_sample_and_group_ids() -> None:
    rows = _v2_rows(6, 6)
    assignment = _v2_assignment(rows, _v2_targets((4, 4), (1, 1), (1, 1)))

    partitioned = apply_split(rows, assignment)

    assert {item.row.sample_id for item in partitioned} == {row.row.sample_id for row in rows}
    assert {item.row.group_id for item in partitioned} == {row.row.group_id for row in rows}
    assert len(partitioned) == len(rows) == len(assignment.by_group)


def test_v2_exactly_solves_same_and_mixed_label_components() -> None:
    rows = _v2_rows(
        6,
        6,
        hashes={
            "a0": _V2_BASE_HASHES[0],
            "a1": _V2_BASE_HASHES[0],
            "c0": _V2_BASE_HASHES[1],
            "c1": _V2_BASE_HASHES[1],
            "a2": _V2_BASE_HASHES[2],
            "c2": _V2_BASE_HASHES[2],
        },
    )
    targets = _v2_targets((2, 2), (2, 2), (2, 2))

    assignment = _v2_assignment(rows, targets)

    assert assignment.audit is not None
    assert assignment.audit.actual_counts == targets
    assert assignment.by_group["a0"] == assignment.by_group["a1"]
    assert assignment.by_group["c0"] == assignment.by_group["c1"]
    assert assignment.by_group["a2"] == assignment.by_group["c2"]


def test_v2_rejects_unsatisfiable_exact_targets() -> None:
    rows = _v2_rows(
        6,
        6,
        hashes={f"a{index}": "0000000000000000" for index in range(6)},
    )

    with pytest.raises(ManifestValidationError, match="精确.*解"):
        _v2_assignment(rows, _v2_targets((4, 4), (1, 1), (1, 1)))


def test_v2_secondary_objective_distributes_multi_member_components() -> None:
    hashes: dict[str, str] = {}
    for label in ("a", "c"):
        for pair_index, base in enumerate(_V2_BASE_HASHES[:3]):
            hashes[f"{label}{pair_index * 2}"] = base
            hashes[f"{label}{pair_index * 2 + 1}"] = base
    rows = _v2_rows(12, 12, hashes=hashes)

    assignment = _v2_assignment(rows, _v2_targets((6, 6), (3, 3), (3, 3)))

    assert assignment.audit is not None
    multi_partitions = {
        assignment.by_group[group_id]
        for group_id, component_id in assignment.audit.component_by_group.items()
        if assignment.audit.component_sizes[component_id] > 1
    }
    assert multi_partitions == {"train", "val", "test"}
    assert assignment.audit.objective_value >= 0


def _brute_force_component_optimum(
    components,
    targets: dict[str, dict[str, int]],
    *,
    seed: int,
    component_fingerprint: str,
):
    best = None
    for choices in product(splitting.PARTITIONS, repeat=len(components)):
        counts = {
            partition: {label: 0 for label in splitting.LABELS}
            for partition in splitting.PARTITIONS
        }
        multi_counts = {
            partition: {label: 0 for label in splitting.LABELS}
            for partition in splitting.PARTITIONS
        }
        mapping = {}
        hash_score = 0
        for component, partition in zip(components, choices, strict=True):
            counts[partition]["authentic"] += component.authentic
            counts[partition]["counterfeit"] += component.counterfeit
            for group_id in component.groups:
                mapping[group_id] = partition
            if len(component.groups) > 1:
                multi_counts[partition]["authentic"] += component.authentic
                multi_counts[partition]["counterfeit"] += component.counterfeit
            digest = hashlib.sha256(
                (f"{seed}|{component_fingerprint}|{component.component_id}|{partition}").encode()
            ).digest()
            hash_score += int.from_bytes(digest, "big")
        if counts != targets:
            continue
        total_multi = {
            label: sum(values[label] for values in multi_counts.values())
            for label in splitting.LABELS
        }
        objective = sum(
            abs(
                multi_counts[partition][label]
                * sum(targets[item][label] for item in splitting.PARTITIONS)
                - targets[partition][label] * total_multi[label]
            )
            for partition in splitting.PARTITIONS
            for label in splitting.LABELS
        )
        key = (objective, hash_score, tuple(sorted(mapping.items())))
        if best is None or key < best[0]:
            best = key, mapping
    assert best is not None
    return best


def test_v2_solver_matches_global_brute_force_optimum_for_reviewer_counterexample() -> None:
    components = (
        splitting._Component("M", ("m-a", "m-c"), 2, 1),
        splitting._Component("S", ("s",), 2, 1),
        splitting._Component("A0", ("a0",), 1, 0),
        splitting._Component("A1", ("a1",), 1, 0),
        splitting._Component("C0", ("c0",), 0, 1),
    )
    targets = _v2_targets((3, 1), (2, 1), (1, 1))
    expected_key, expected_mapping = _brute_force_component_optimum(
        components,
        targets,
        seed=42,
        component_fingerprint="fp0",
    )

    mapping, objective = splitting._solve_components(components, targets, 42, "fp0")
    reversed_mapping, reversed_objective = splitting._solve_components(
        tuple(reversed(components)), targets, 42, "fp0"
    )

    assert expected_key[0] == 16
    assert (objective, mapping) == (16, expected_mapping)
    assert (reversed_objective, reversed_mapping) == (16, expected_mapping)
    assert mapping["m-a"] == mapping["m-c"]


def test_v2_solver_uses_complete_mapping_lex_when_hash_scores_collide(monkeypatch) -> None:
    components = tuple(
        splitting._Component(group_id, (group_id,), 1, 0) for group_id in ("g2", "g1", "g0")
    )
    targets = _v2_targets((1, 0), (1, 0), (1, 0))
    monkeypatch.setattr(splitting, "_tie_value", lambda *_args: 0)

    mapping, objective = splitting._solve_components(components, targets, 42, "collision")

    assert objective == 0
    assert tuple(sorted(mapping.items())) == (
        ("g0", "test"),
        ("g1", "train"),
        ("g2", "val"),
    )


@pytest.mark.parametrize(
    ("components", "targets"),
    (
        (
            (
                splitting._Component("x0", ("x0-a", "x0-c"), 1, 1),
                splitting._Component("x1", ("x1-a", "x1-c"), 1, 1),
                splitting._Component("a", ("a-a", "a-b"), 2, 0),
                splitting._Component("s", ("s",), 1, 1),
                splitting._Component("a0", ("a0",), 1, 0),
                splitting._Component("c0", ("c0",), 0, 1),
            ),
            _v2_targets((2, 2), (2, 1), (2, 1)),
        ),
        (
            (
                splitting._Component("m0", ("m0-a", "m0-b"), 2, 0),
                splitting._Component("m1", ("m1-a", "m1-b"), 2, 0),
                splitting._Component("m2", ("m2-a", "m2-b"), 2, 0),
                splitting._Component("c0", ("c0",), 0, 1),
                splitting._Component("c1", ("c1",), 0, 1),
                splitting._Component("c2", ("c2",), 0, 1),
            ),
            _v2_targets((2, 1), (2, 1), (2, 1)),
        ),
    ),
)
def test_v2_decomposed_solver_matches_full_exhaustive_ordering(components, targets) -> None:
    seed, fingerprint = 42, "property-fixture"
    expected_key, expected_mapping = _brute_force_component_optimum(
        components, targets, seed=seed, component_fingerprint=fingerprint
    )

    for ordered in (components, tuple(reversed(components))):
        mapping, objective = splitting._solve_components(ordered, targets, seed, fingerprint)
        hash_score = sum(
            splitting._tie_value(
                seed, fingerprint, component.component_id, mapping[component.groups[0]]
            )
            for component in components
        )
        actual_key = (objective, hash_score, tuple(sorted(mapping.items())))
        assert actual_key == expected_key
        assert mapping == expected_mapping


def test_identical_vector_options_cover_all_count_states_exactly() -> None:
    components = tuple(
        splitting._Component(f"m{index}", (f"m{index}-a", f"m{index}-b"), 1, 1)
        for index in range(4)
    )

    options = splitting._vector_assignment_options(
        components, seed=42, component_fingerprint="vector-options"
    )

    assert set(options) == {
        (val_count, test_count) for val_count in range(5) for test_count in range(5 - val_count)
    }
    for state, actual in options.items():
        candidates = []
        for choices in product(splitting.PARTITIONS, repeat=len(components)):
            if (choices.count("val"), choices.count("test")) != state:
                continue
            mapping = {
                group_id: partition
                for component, partition in zip(components, choices, strict=True)
                for group_id in component.groups
            }
            candidates.append(
                (
                    sum(
                        splitting._tie_value(
                            42, "vector-options", component.component_id, partition
                        )
                        for component, partition in zip(components, choices, strict=True)
                    ),
                    tuple(sorted(mapping.items())),
                )
            )
        assert actual == min(candidates)


def test_v2_independent_all_pairs_oracle_reports_zero_crossings() -> None:
    rows = _v2_rows(
        6,
        6,
        hashes={
            "a0": "0000000000000000",
            "a1": "000000000000000f",
            "a2": "000000000000001f",
        },
    )
    assignment = _v2_assignment(rows, _v2_targets((4, 4), (1, 1), (1, 1)))

    for left_index, left in enumerate(rows):
        for right in rows[left_index + 1 :]:
            distance = (int(left.perceptual_hash, 16) ^ int(right.perceptual_hash, 16)).bit_count()
            if distance <= 4:
                assert (
                    assignment.by_group[left.row.group_id]
                    == assignment.by_group[right.row.group_id]
                )
    assert assignment.audit is not None
    assert assignment.audit.final_cross_partition_near_duplicate_count == 0


def test_v1_payload_replays_without_v2_audit(tmp_path: Path) -> None:
    rows = make_rows()
    assignment = create_group_splits(rows, ratios=(0.5, 0.25, 0.25), seed=7)

    payload = splitting.split_assignment_to_payload(assignment)
    replayed = splitting.split_assignment_from_payload(payload)

    assert "component_algorithm_version" not in payload
    assert replayed.audit is None
    assert dict(replayed.by_group) == dict(assignment.by_group)
    apply_split(rows, replayed)


def test_v2_payload_round_trips_and_replays() -> None:
    rows = _v2_rows(
        6,
        6,
        hashes={"a0": "0000000000000000", "a1": "000000000000000f"},
    )
    assignment = _v2_assignment(rows, _v2_targets((4, 4), (1, 1), (1, 1)))

    payload = splitting.split_assignment_to_payload(assignment)
    replayed = splitting.split_assignment_from_payload(payload)

    assert replayed == assignment
    apply_split(rows, replayed)


def test_v2_audit_includes_portable_component_diagnostics() -> None:
    rows = _v2_rows(
        6,
        6,
        hashes={
            "a0": _V2_BASE_HASHES[0],
            "a1": _V2_BASE_HASHES[0],
            "a2": _V2_BASE_HASHES[1],
            "c2": _V2_BASE_HASHES[1],
        },
    )
    assignment = _v2_assignment(rows, _v2_targets((2, 2), (2, 2), (2, 2)))
    payload = splitting.split_assignment_to_payload(assignment)

    assert payload["edge_count"] == 2
    assert payload["singleton_component_count"] == 8
    assert payload["mixed_label_component_count"] == 1
    vectors = payload["component_label_counts"]
    assert len(vectors) == payload["component_count"] == 10
    assert sum(vector["authentic"] for vector in vectors.values()) == 6
    assert sum(vector["counterfeit"] for vector in vectors.values()) == 6
    encoded = json.dumps(
        vectors,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    assert payload["component_label_counts_fingerprint"] == hashlib.sha256(encoded).hexdigest()


def _tamper_v2_payload(payload: dict[str, object], field: str) -> None:
    if field == "schema_version":
        payload[field] = 2
    elif field == "by_group":
        mapping = dict(payload[field])
        mapping["a0"] = "test" if mapping["a0"] != "test" else "train"
        payload[field] = mapping
    elif field == "component_by_group":
        mapping = dict(payload[field])
        mapping["a0"] = "component-forged"
        payload[field] = mapping
    elif field == "component_sizes":
        mapping = dict(payload[field])
        key = next(iter(mapping))
        mapping[key] += 1
        payload[field] = mapping
    elif field == "component_label_counts":
        counts = {key: dict(values) for key, values in payload[field].items()}
        counts[next(iter(counts))]["authentic"] += 1
        payload[field] = counts
    elif field in ("target_counts", "actual_counts"):
        counts = {partition: dict(values) for partition, values in payload[field].items()}
        counts["train"]["authentic"] += 1
        payload[field] = counts
    elif field in (
        "component_count",
        "edge_count",
        "singleton_component_count",
        "mixed_label_component_count",
        "multi_member_component_count",
        "largest_component_size",
        "objective_value",
        "oracle_pair_count",
        "final_cross_partition_near_duplicate_count",
        "seed",
        "perceptual_hash_distance",
    ):
        payload[field] = int(payload[field]) + 1
    elif field == "ratios":
        payload[field] = {"train": 0.6, "val": 0.2, "test": 0.2}
    else:
        payload[field] = "forged"


@pytest.mark.parametrize(
    "field",
    (
        "schema_version",
        "algorithm_version",
        "seed",
        "ratios",
        "input_fingerprint",
        "perceptual_hash_distance",
        "by_group",
        "perceptual_hash_algorithm",
        "threshold_comparison",
        "component_algorithm_version",
        "component_count",
        "edge_count",
        "singleton_component_count",
        "mixed_label_component_count",
        "multi_member_component_count",
        "largest_component_size",
        "component_fingerprint",
        "component_by_group",
        "component_sizes",
        "component_label_counts",
        "component_label_counts_fingerprint",
        "solver_algorithm_version",
        "solver_status",
        "objective_name",
        "objective_value",
        "tie_break",
        "target_counts",
        "actual_counts",
        "mapping_fingerprint",
        "oracle_algorithm_version",
        "oracle_pair_count",
        "final_cross_partition_near_duplicate_count",
    ),
)
def test_v2_replay_rejects_every_critical_tampered_field(field: str) -> None:
    rows = _v2_rows(
        6,
        6,
        hashes={"a0": "0000000000000000", "a1": "000000000000000f"},
    )
    assignment = _v2_assignment(rows, _v2_targets((4, 4), (1, 1), (1, 1)))
    payload = splitting.split_assignment_to_payload(assignment)
    _tamper_v2_payload(payload, field)

    with pytest.raises(ManifestValidationError):
        replayed = splitting.split_assignment_from_payload(payload)
        apply_split(rows, replayed)
