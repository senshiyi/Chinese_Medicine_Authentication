"""确定性的 group-level 分层切分与泄漏校验。"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal

from ..encoding import write_csv_utf8_sig, write_json_utf8
from ..errors import ManifestValidationError
from .manifest import KnownManifestRow, ValidatedImageRecord, ValidatedKnownRecord

Partition = Literal["train", "val", "test"]
PARTITIONS: tuple[Partition, Partition, Partition] = ("train", "val", "test")
V1_ALGORITHM_VERSION = "group-stratified-v1"
V2_ALGORITHM_VERSION = "group-stratified-near-duplicate-components-v2"
ALGORITHM_VERSION = V1_ALGORITHM_VERSION
SUPPORTED_ALGORITHM_VERSIONS = frozenset({V1_ALGORITHM_VERSION, V2_ALGORITHM_VERSION})
PERCEPTUAL_HASH_ALGORITHM = "average-hash-8x8-lanczos-v1"
THRESHOLD_COMPARISON = "inclusive-lte"
COMPONENT_ALGORITHM_VERSION = "all-pairs-union-find-v1"
SOLVER_ALGORITHM_VERSION = "exact-component-dp-v1"
SOLVER_OBJECTIVE_NAME = "multi-label-target-proportion-absolute-deviation-v1"
SOLVER_TIE_BREAK = (
    "min-objective-then-sum-sha256(seed,component_fingerprint,component_id,partition)"
    "-then-sorted-group-mapping-lex-v1"
)
ORACLE_ALGORITHM_VERSION = "independent-all-pairs-v1"
MAX_NEAR_DUPLICATE_ERRORS = 100
MAX_CANDIDATES_PER_IMAGE = 100_000
LABELS = ("authentic", "counterfeit")


@dataclass(frozen=True, slots=True)
class SplitAudit:
    """V2 component、solver 与独立泄漏 oracle 审计。"""

    perceptual_hash_algorithm: str
    threshold_comparison: str
    component_algorithm_version: str
    component_count: int
    edge_count: int
    singleton_component_count: int
    mixed_label_component_count: int
    multi_member_component_count: int
    largest_component_size: int
    component_fingerprint: str
    component_by_group: Mapping[str, str]
    component_sizes: Mapping[str, int]
    component_label_counts: Mapping[str, Mapping[str, int]]
    component_label_counts_fingerprint: str
    solver_algorithm_version: str
    solver_status: str
    objective_name: str
    objective_value: int
    tie_break: str
    target_counts: Mapping[str, Mapping[str, int]]
    actual_counts: Mapping[str, Mapping[str, int]]
    mapping_fingerprint: str
    oracle_algorithm_version: str
    oracle_pair_count: int
    final_cross_partition_near_duplicate_count: int


@dataclass(frozen=True, slots=True)
class SplitAssignment:
    """不可变的 group 到分区映射及可审计参数。"""

    by_group: Mapping[str, Partition]
    ratios: tuple[float, float, float]
    seed: int
    input_fingerprint: str
    perceptual_hash_distance: int = 4
    algorithm_version: str = ALGORITHM_VERSION
    audit: SplitAudit | None = None


@dataclass(frozen=True, slots=True)
class SplitArtifacts:
    train_csv: Path
    val_csv: Path
    test_csv: Path
    splits_json: Path


@dataclass(frozen=True, slots=True)
class PartitionedImageRecord:
    """切分后唯一允许携带 partition 的图片记录类型。"""

    record: ValidatedKnownRecord
    partition: Partition

    @property
    def row(self) -> KnownManifestRow:
        return self.record.row

    @property
    def sha256(self) -> str:
        return self.record.sha256

    @property
    def perceptual_hash(self) -> str:
        return self.record.perceptual_hash

    @property
    def manifest_encoding(self) -> str:
        return self.record.manifest_encoding


def create_group_splits(
    rows: Sequence[ValidatedImageRecord],
    ratios: tuple[float, float, float] = (0.7, 0.2, 0.1),
    seed: int = 42,
    perceptual_hash_distance: int = 4,
    *,
    algorithm_version: str = V1_ALGORITHM_VERSION,
    target_counts: Mapping[str, Mapping[str, int]] | None = None,
) -> SplitAssignment:
    """按显式算法版本创建确定性、可重放的 group 切分。"""

    normalized_ratios = _validate_ratios(ratios)
    _validate_seed(seed)
    _validate_perceptual_hash_distance(perceptual_hash_distance)
    known_rows = _validate_known_rows(rows)
    if algorithm_version == V1_ALGORITHM_VERSION:
        if target_counts is not None:
            raise ManifestValidationError("V1 切分不接受 target_counts。")
        group_strata = _collect_group_strata(known_rows)
        by_group = _build_group_mapping(group_strata, normalized_ratios, seed)
        assignment = SplitAssignment(
            by_group=MappingProxyType(dict(sorted(by_group.items()))),
            ratios=normalized_ratios,
            seed=seed,
            input_fingerprint=_input_fingerprint(known_rows),
            perceptual_hash_distance=perceptual_hash_distance,
        )
        apply_split(known_rows, assignment)
        return assignment
    if algorithm_version == V2_ALGORITHM_VERSION:
        if target_counts is None:
            raise ManifestValidationError("V2 切分必须提供精确 target_counts。")
        assignment = _create_v2_assignment(
            known_rows,
            normalized_ratios,
            seed,
            perceptual_hash_distance,
            target_counts,
        )
        apply_split(known_rows, assignment)
        return assignment
    raise ManifestValidationError(f"不支持的切分算法版本：{algorithm_version!r}。")


def apply_split(
    rows: Sequence[ValidatedImageRecord],
    assignment: SplitAssignment,
) -> tuple[PartitionedImageRecord, ...]:
    """应用新建或锁定切分，并重新执行全部跨分区泄漏校验。"""

    if assignment.algorithm_version not in SUPPORTED_ALGORITHM_VERSIONS:
        raise ManifestValidationError(f"不支持的切分算法版本：{assignment.algorithm_version!r}。")
    normalized_ratios = _validate_ratios(assignment.ratios)
    _validate_seed(assignment.seed)
    _validate_perceptual_hash_distance(assignment.perceptual_hash_distance)
    known_rows = _validate_known_rows(rows)
    group_strata = _collect_group_strata(known_rows)
    row_groups = {record.row.group_id for record in known_rows}
    assigned_groups = set(assignment.by_group)
    missing = sorted(row_groups - assigned_groups)
    if missing:
        raise ManifestValidationError(f"以下 group 没有切分分配：{missing}")
    stale = sorted(assigned_groups - row_groups)
    if stale:
        raise ManifestValidationError(f"切分包含当前数据不存在的 group：{stale}")

    invalid_partitions = sorted(set(assignment.by_group.values()) - set(PARTITIONS))
    if invalid_partitions:
        raise ManifestValidationError(f"切分包含非法 partition：{invalid_partitions}")

    coverage: dict[tuple[str, str], set[Partition]] = defaultdict(set)
    for group_id, stratum in group_strata.items():
        coverage[stratum].add(assignment.by_group[group_id])
    incomplete = {
        stratum: sorted(set(PARTITIONS) - partitions)
        for stratum, partitions in sorted(coverage.items())
        if partitions != set(PARTITIONS)
    }
    if incomplete:
        raise ManifestValidationError(
            f"每个分层必须覆盖 train、val、test；当前缺失情况：{incomplete}"
        )

    if assignment.algorithm_version == V1_ALGORITHM_VERSION:
        expected_mapping = _build_group_mapping(group_strata, normalized_ratios, assignment.seed)
        if dict(assignment.by_group) != expected_mapping:
            raise ManifestValidationError("记录的算法版本、seed 与 ratios 无法重现 group 映射。")
        if assignment.audit is not None:
            raise ManifestValidationError("V1 切分不得携带 V2 audit。")
    else:
        if assignment.audit is None:
            raise ManifestValidationError("V2 切分缺少 component/solver/oracle audit。")
        expected = _create_v2_assignment(
            known_rows,
            normalized_ratios,
            assignment.seed,
            assignment.perceptual_hash_distance,
            assignment.audit.target_counts,
        )
        if dict(assignment.by_group) != dict(expected.by_group):
            raise ManifestValidationError("V2 by_group 无法由当前输入与审计参数重现。")
        if assignment.audit != expected.audit:
            raise ManifestValidationError("V2 component/solver/oracle audit 与重建结果不一致。")

    current_fingerprint = _input_fingerprint(known_rows)
    if current_fingerprint != assignment.input_fingerprint:
        raise ManifestValidationError("锁定切分的输入数据指纹与当前 manifest/图片内容不一致。")

    partitioned = tuple(
        PartitionedImageRecord(
            record=record,
            partition=assignment.by_group[record.row.group_id],
        )
        for record in known_rows
    )
    _validate_partition_leakage(partitioned, assignment.perceptual_hash_distance)
    if assignment.algorithm_version == V2_ALGORITHM_VERSION:
        cross_count = _independent_cross_partition_count(
            partitioned, assignment.perceptual_hash_distance
        )
        if cross_count != 0:
            raise ManifestValidationError(
                f"V2 独立全量 oracle 检测到 {cross_count} 个跨分区近重复图对。"
            )
    return partitioned


def write_split_artifacts(
    rows: Sequence[ValidatedImageRecord],
    assignment: SplitAssignment,
    output_dir: str | Path,
) -> SplitArtifacts:
    """写出稳定排序的 UTF-8-SIG 分区 CSV 和 UTF-8 切分审计 JSON。"""

    partitioned = apply_split(rows, assignment)
    destination = Path(output_dir).expanduser().resolve()
    paths: dict[Partition, Path] = {
        partition: destination / f"{partition}.csv" for partition in PARTITIONS
    }
    fieldnames = (
        "image_path",
        "group_id",
        "sample_id",
        "claimed_herb_name",
        "authenticity",
        "actual_species",
        "species_annotation_status",
        "batch_id",
        "origin",
        "capture_device",
        "collected_at",
        "annotator",
        "notes",
        "sha256",
        "perceptual_hash",
        "manifest_encoding",
        "partition",
    )
    ordered = sorted(
        partitioned,
        key=lambda item: (
            item.partition,
            item.row.group_id,
            item.row.sample_id,
            item.row.image_path.as_posix(),
            item.sha256,
        ),
    )
    for partition, path in paths.items():
        write_csv_utf8_sig(
            path,
            fieldnames,
            (_artifact_row(item) for item in ordered if item.partition == partition),
        )

    splits_json = destination / "splits.json"
    write_json_utf8(
        splits_json,
        split_assignment_to_payload(assignment),
    )
    return SplitArtifacts(
        train_csv=paths["train"],
        val_csv=paths["val"],
        test_csv=paths["test"],
        splits_json=splits_json,
    )


def _artifact_row(item: PartitionedImageRecord) -> dict[str, object]:
    row = item.row
    return {
        "image_path": row.image_path.as_posix(),
        "group_id": row.group_id,
        "sample_id": row.sample_id,
        "claimed_herb_name": row.claimed_herb_name,
        "authenticity": row.authenticity,
        "actual_species": "|".join(sorted(row.actual_species)),
        "species_annotation_status": row.species_annotation_status,
        "batch_id": row.batch_id,
        "origin": row.origin,
        "capture_device": row.capture_device,
        "collected_at": row.collected_at,
        "annotator": row.annotator,
        "notes": row.notes,
        "sha256": item.sha256,
        "perceptual_hash": item.perceptual_hash,
        "manifest_encoding": item.manifest_encoding,
        "partition": item.partition,
    }


@dataclass(frozen=True, slots=True)
class _Component:
    component_id: str
    groups: tuple[str, ...]
    authentic: int
    counterfeit: int

    @property
    def row_count(self) -> int:
        return self.authentic + self.counterfeit


def split_assignment_to_payload(assignment: SplitAssignment) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "algorithm_version": assignment.algorithm_version,
        "seed": assignment.seed,
        "ratios": dict(zip(PARTITIONS, assignment.ratios, strict=True)),
        "input_fingerprint": assignment.input_fingerprint,
        "perceptual_hash_distance": assignment.perceptual_hash_distance,
        "by_group": dict(sorted(assignment.by_group.items())),
    }
    if assignment.audit is not None:
        audit = assignment.audit
        payload.update(
            {
                "perceptual_hash_algorithm": audit.perceptual_hash_algorithm,
                "threshold_comparison": audit.threshold_comparison,
                "component_algorithm_version": audit.component_algorithm_version,
                "component_count": audit.component_count,
                "edge_count": audit.edge_count,
                "singleton_component_count": audit.singleton_component_count,
                "mixed_label_component_count": audit.mixed_label_component_count,
                "multi_member_component_count": audit.multi_member_component_count,
                "largest_component_size": audit.largest_component_size,
                "component_fingerprint": audit.component_fingerprint,
                "component_by_group": dict(sorted(audit.component_by_group.items())),
                "component_sizes": dict(sorted(audit.component_sizes.items())),
                "component_label_counts": _plain_component_counts(audit.component_label_counts),
                "component_label_counts_fingerprint": (audit.component_label_counts_fingerprint),
                "solver_algorithm_version": audit.solver_algorithm_version,
                "solver_status": audit.solver_status,
                "objective_name": audit.objective_name,
                "objective_value": audit.objective_value,
                "tie_break": audit.tie_break,
                "target_counts": _plain_counts(audit.target_counts),
                "actual_counts": _plain_counts(audit.actual_counts),
                "mapping_fingerprint": audit.mapping_fingerprint,
                "oracle_algorithm_version": audit.oracle_algorithm_version,
                "oracle_pair_count": audit.oracle_pair_count,
                "final_cross_partition_near_duplicate_count": (
                    audit.final_cross_partition_near_duplicate_count
                ),
            }
        )
    return payload


def split_assignment_from_payload(value: Mapping[str, object]) -> SplitAssignment:
    if value.get("schema_version") != 1:
        raise ManifestValidationError("锁定 split schema_version 必须为 1。")
    try:
        ratios = value["ratios"]
        by_group = value["by_group"]
        if not isinstance(ratios, Mapping) or not isinstance(by_group, Mapping):
            raise TypeError("ratios/by_group 必须是映射")
        algorithm_version = str(value["algorithm_version"])
        audit = None
        if algorithm_version == V2_ALGORITHM_VERSION:
            component_by_group = value["component_by_group"]
            component_sizes = value["component_sizes"]
            component_label_counts = value["component_label_counts"]
            if not isinstance(component_by_group, Mapping) or not isinstance(
                component_sizes, Mapping
            ):
                raise TypeError("component mapping 必须是映射")
            audit = SplitAudit(
                perceptual_hash_algorithm=str(value["perceptual_hash_algorithm"]),
                threshold_comparison=str(value["threshold_comparison"]),
                component_algorithm_version=str(value["component_algorithm_version"]),
                component_count=int(value["component_count"]),
                edge_count=int(value["edge_count"]),
                singleton_component_count=int(value["singleton_component_count"]),
                mixed_label_component_count=int(value["mixed_label_component_count"]),
                multi_member_component_count=int(value["multi_member_component_count"]),
                largest_component_size=int(value["largest_component_size"]),
                component_fingerprint=str(value["component_fingerprint"]),
                component_by_group=_string_mapping(component_by_group),
                component_sizes=MappingProxyType(
                    {str(key): int(item) for key, item in component_sizes.items()}
                ),
                component_label_counts=_component_counts_mapping(component_label_counts),
                component_label_counts_fingerprint=str(value["component_label_counts_fingerprint"]),
                solver_algorithm_version=str(value["solver_algorithm_version"]),
                solver_status=str(value["solver_status"]),
                objective_name=str(value["objective_name"]),
                objective_value=int(value["objective_value"]),
                tie_break=str(value["tie_break"]),
                target_counts=_counts_mapping(value["target_counts"]),
                actual_counts=_counts_mapping(value["actual_counts"]),
                mapping_fingerprint=str(value["mapping_fingerprint"]),
                oracle_algorithm_version=str(value["oracle_algorithm_version"]),
                oracle_pair_count=int(value["oracle_pair_count"]),
                final_cross_partition_near_duplicate_count=int(
                    value["final_cross_partition_near_duplicate_count"]
                ),
            )
        return SplitAssignment(
            by_group=MappingProxyType({str(key): str(item) for key, item in by_group.items()}),
            ratios=tuple(float(ratios[name]) for name in PARTITIONS),
            seed=int(value["seed"]),
            input_fingerprint=str(value["input_fingerprint"]),
            perceptual_hash_distance=int(value.get("perceptual_hash_distance", 4)),
            algorithm_version=algorithm_version,
            audit=audit,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ManifestValidationError(f"无法解析锁定 split：{exc}") from exc


def _create_v2_assignment(
    rows: Sequence[ValidatedKnownRecord],
    ratios: tuple[float, float, float],
    seed: int,
    maximum_distance: int,
    target_counts: Mapping[str, Mapping[str, int]],
) -> SplitAssignment:
    _collect_group_strata(rows)
    targets = _normalize_target_counts(rows, target_counts)
    components, component_by_group, component_sizes, edge_count = _build_components(
        rows, maximum_distance
    )
    component_label_counts = {
        component.component_id: {
            "authentic": component.authentic,
            "counterfeit": component.counterfeit,
        }
        for component in components
    }
    component_label_counts_fingerprint = _json_fingerprint(component_label_counts)
    component_fingerprint = _json_fingerprint(
        {
            "algorithm": COMPONENT_ALGORITHM_VERSION,
            "distance": maximum_distance,
            "ratios": dict(zip(PARTITIONS, ratios, strict=True)),
            "edge_count": edge_count,
            "component_by_group": component_by_group,
            "component_sizes": component_sizes,
            "component_label_counts": component_label_counts,
        }
    )
    by_group, objective_value = _solve_components(components, targets, seed, component_fingerprint)
    actual = _partition_label_counts(rows, by_group)
    if actual != targets:
        raise ManifestValidationError(
            f"V2 solver 实际计数与精确 target_counts 不一致：{actual} != {targets}"
        )
    partitioned = tuple(
        PartitionedImageRecord(record=row, partition=by_group[row.row.group_id]) for row in rows
    )
    cross_count = _independent_cross_partition_count(partitioned, maximum_distance)
    if cross_count:
        raise ManifestValidationError(
            f"V2 独立全量 oracle 检测到 {cross_count} 个跨分区近重复图对。"
        )
    mapping_fingerprint = _json_fingerprint(dict(sorted(by_group.items())))
    audit = SplitAudit(
        perceptual_hash_algorithm=PERCEPTUAL_HASH_ALGORITHM,
        threshold_comparison=THRESHOLD_COMPARISON,
        component_algorithm_version=COMPONENT_ALGORITHM_VERSION,
        component_count=len(components),
        edge_count=edge_count,
        singleton_component_count=sum(len(item.groups) == 1 for item in components),
        mixed_label_component_count=sum(
            item.authentic > 0 and item.counterfeit > 0 for item in components
        ),
        multi_member_component_count=sum(len(item.groups) > 1 for item in components),
        largest_component_size=max(len(item.groups) for item in components),
        component_fingerprint=component_fingerprint,
        component_by_group=_string_mapping(component_by_group),
        component_sizes=MappingProxyType(dict(sorted(component_sizes.items()))),
        component_label_counts=_freeze_component_counts(component_label_counts),
        component_label_counts_fingerprint=component_label_counts_fingerprint,
        solver_algorithm_version=SOLVER_ALGORITHM_VERSION,
        solver_status="optimal",
        objective_name=SOLVER_OBJECTIVE_NAME,
        objective_value=objective_value,
        tie_break=SOLVER_TIE_BREAK,
        target_counts=_freeze_counts(targets),
        actual_counts=_freeze_counts(actual),
        mapping_fingerprint=mapping_fingerprint,
        oracle_algorithm_version=ORACLE_ALGORITHM_VERSION,
        oracle_pair_count=len(rows) * (len(rows) - 1) // 2,
        final_cross_partition_near_duplicate_count=cross_count,
    )
    return SplitAssignment(
        by_group=MappingProxyType(dict(sorted(by_group.items()))),
        ratios=ratios,
        seed=seed,
        input_fingerprint=_input_fingerprint(rows),
        perceptual_hash_distance=maximum_distance,
        algorithm_version=V2_ALGORITHM_VERSION,
        audit=audit,
    )


def _build_components(
    rows: Sequence[ValidatedKnownRecord], maximum_distance: int
) -> tuple[tuple[_Component, ...], dict[str, str], dict[str, int], int]:
    groups = sorted({row.row.group_id for row in rows})
    parent = {group_id: group_id for group_id in groups}

    def find(group_id: str) -> str:
        while parent[group_id] != group_id:
            parent[group_id] = parent[parent[group_id]]
            group_id = parent[group_id]
        return group_id

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return
        lower, higher = sorted((left_root, right_root))
        parent[higher] = lower

    ordered_rows = sorted(
        rows,
        key=lambda row: (
            row.row.group_id,
            row.row.sample_id,
            row.row.image_path.as_posix(),
            row.sha256,
        ),
    )
    edges: set[tuple[str, str]] = set()
    for left_index, left in enumerate(ordered_rows):
        left_value = _perceptual_hash_value(left.perceptual_hash)
        for right in ordered_rows[left_index + 1 :]:
            distance = (left_value ^ _perceptual_hash_value(right.perceptual_hash)).bit_count()
            if distance <= maximum_distance:
                left_group, right_group = sorted((left.row.group_id, right.row.group_id))
                if left_group != right_group:
                    edges.add((left_group, right_group))
                    union(left_group, right_group)
    grouped: dict[str, list[str]] = defaultdict(list)
    for group_id in groups:
        grouped[find(group_id)].append(group_id)
    rows_by_group: dict[str, list[ValidatedKnownRecord]] = defaultdict(list)
    for row in rows:
        rows_by_group[row.row.group_id].append(row)
    components: list[_Component] = []
    component_by_group: dict[str, str] = {}
    component_sizes: dict[str, int] = {}
    for member_groups in sorted(tuple(tuple(sorted(items)) for items in grouped.values())):
        component_id = "component-" + _json_fingerprint(member_groups)[:16]
        component_rows = [row for group in member_groups for row in rows_by_group[group]]
        counts = defaultdict(int)
        for row in component_rows:
            counts[row.row.authenticity] += 1
        component = _Component(
            component_id,
            member_groups,
            counts["authentic"],
            counts["counterfeit"],
        )
        components.append(component)
        component_sizes[component_id] = len(member_groups)
        for group_id in member_groups:
            component_by_group[group_id] = component_id
    return tuple(components), component_by_group, component_sizes, len(edges)


def _solve_components(
    components: Sequence[_Component],
    targets: Mapping[str, Mapping[str, int]],
    seed: int,
    component_fingerprint: str,
) -> tuple[dict[str, Partition], int]:
    ordered = tuple(sorted(components, key=lambda item: (item.component_id, item.groups)))
    coupled = tuple(
        component for component in ordered if component.authentic > 0 and component.counterfeit > 0
    )
    exclusive_multi = {
        label: tuple(
            component
            for component in ordered
            if len(component.groups) > 1
            and getattr(component, label) > 0
            and getattr(component, LABELS[1 - LABELS.index(label)]) == 0
        )
        for label in LABELS
    }
    exclusive_nonmulti = {
        label: tuple(
            component
            for component in ordered
            if len(component.groups) == 1
            and getattr(component, label) > 0
            and getattr(component, LABELS[1 - LABELS.index(label)]) == 0
        )
        for label in LABELS
    }
    coupled_states = _coupled_assignment_states(coupled, targets, seed, component_fingerprint)
    multi_states = {
        label: _label_assignment_states(
            exclusive_multi[label], targets, label, seed, component_fingerprint
        )
        for label in LABELS
    }
    nonmulti_states = {
        label: _label_assignment_states(
            exclusive_nonmulti[label], targets, label, seed, component_fingerprint
        )
        for label in LABELS
    }
    coupled_totals = {
        label: sum(getattr(component, label) for component in coupled) for label in LABELS
    }
    coupled_multi_totals = {
        label: sum(getattr(component, label) for component in coupled if len(component.groups) > 1)
        for label in LABELS
    }
    exclusive_multi_totals = {
        label: sum(getattr(component, label) for component in exclusive_multi[label])
        for label in LABELS
    }
    exclusive_nonmulti_totals = {
        label: sum(getattr(component, label) for component in exclusive_nonmulti[label])
        for label in LABELS
    }
    total_multi = {
        label: coupled_multi_totals[label] + exclusive_multi_totals[label] for label in LABELS
    }
    # 固定 coupled 分配后，两标签的配额、目标值与 hash 和彼此可加。
    # 两标签 group 不相交，各自 sorted mapping lex 最小也使合并 mapping lex 最小。
    best: (
        tuple[
            tuple[int, int, tuple[tuple[str, Partition], ...]],
            dict[str, Partition],
        ]
        | None
    ) = None
    for state, (coupled_score, coupled_mapping) in coupled_states.items():
        (
            coupled_val_a,
            coupled_val_c,
            coupled_test_a,
            coupled_test_c,
            coupled_multi_val_a,
            coupled_multi_val_c,
            coupled_multi_test_a,
            coupled_multi_test_c,
        ) = state
        coupled_partition_counts = {
            "authentic": (
                coupled_totals["authentic"] - coupled_val_a - coupled_test_a,
                coupled_val_a,
                coupled_test_a,
            ),
            "counterfeit": (
                coupled_totals["counterfeit"] - coupled_val_c - coupled_test_c,
                coupled_val_c,
                coupled_test_c,
            ),
        }
        coupled_multi_counts = {
            "authentic": (
                coupled_multi_totals["authentic"] - coupled_multi_val_a - coupled_multi_test_a,
                coupled_multi_val_a,
                coupled_multi_test_a,
            ),
            "counterfeit": (
                coupled_multi_totals["counterfeit"] - coupled_multi_val_c - coupled_multi_test_c,
                coupled_multi_val_c,
                coupled_multi_test_c,
            ),
        }
        label_solutions: dict[
            str,
            tuple[int, int, tuple[tuple[str, Partition], ...]],
        ] = {}
        for label in LABELS:
            label_best: tuple[int, int, tuple[tuple[str, Partition], ...]] | None = None
            for (multi_val, multi_test), (multi_score, multi_mapping) in multi_states[
                label
            ].items():
                multi_train = exclusive_multi_totals[label] - multi_val - multi_test
                fixed_counts = (
                    coupled_partition_counts[label][0] + multi_train,
                    coupled_partition_counts[label][1] + multi_val,
                    coupled_partition_counts[label][2] + multi_test,
                )
                residual = tuple(
                    targets[partition][label] - fixed_counts[index]
                    for index, partition in enumerate(PARTITIONS)
                )
                if any(value < 0 for value in residual):
                    continue
                if sum(residual) != exclusive_nonmulti_totals[label]:
                    continue
                nonmulti_option = nonmulti_states[label].get((residual[1], residual[2]))
                if nonmulti_option is None:
                    continue
                nonmulti_score, nonmulti_mapping = nonmulti_option
                multi_partition_counts = (
                    coupled_multi_counts[label][0] + multi_train,
                    coupled_multi_counts[label][1] + multi_val,
                    coupled_multi_counts[label][2] + multi_test,
                )
                label_total = sum(targets[partition][label] for partition in PARTITIONS)
                objective = sum(
                    abs(
                        multi_partition_counts[index] * label_total
                        - targets[partition][label] * total_multi[label]
                    )
                    for index, partition in enumerate(PARTITIONS)
                )
                mapping_lex = tuple(sorted((*multi_mapping, *nonmulti_mapping)))
                candidate = (objective, multi_score + nonmulti_score, mapping_lex)
                if label_best is None or candidate < label_best:
                    label_best = candidate
            if label_best is None:
                break
            label_solutions[label] = label_best
        if len(label_solutions) != len(LABELS):
            continue
        auth_solution = label_solutions["authentic"]
        counterfeit_solution = label_solutions["counterfeit"]
        full_mapping_lex = tuple(
            sorted((*coupled_mapping, *auth_solution[2], *counterfeit_solution[2]))
        )
        key = (
            auth_solution[0] + counterfeit_solution[0],
            coupled_score + auth_solution[1] + counterfeit_solution[1],
            full_mapping_lex,
        )
        if best is None or key < best[0]:
            best = key, dict(full_mapping_lex)
    if best is None:
        raise ManifestValidationError("V2 component target_counts 不存在精确可行解。")
    key, by_group = best
    return by_group, key[0]


def _coupled_assignment_states(
    components: Sequence[_Component],
    targets: Mapping[str, Mapping[str, int]],
    seed: int,
    component_fingerprint: str,
) -> dict[
    tuple[int, int, int, int, int, int, int, int],
    tuple[int, tuple[tuple[str, Partition], ...]],
]:
    grouped: dict[tuple[int, int, bool], list[_Component]] = defaultdict(list)
    for component in components:
        grouped[(component.authentic, component.counterfeit, len(component.groups) > 1)].append(
            component
        )
    states = {(0, 0, 0, 0, 0, 0, 0, 0): (0, ())}
    processed_a = processed_c = 0
    for authentic, counterfeit, is_multi in sorted(
        grouped, key=lambda vector: (-(vector[0] + vector[1]), -vector[1], -vector[0], vector[2])
    ):
        vector_components = tuple(
            sorted(
                grouped[(authentic, counterfeit, is_multi)],
                key=lambda item: item.component_id,
            )
        )
        val_limits = [
            targets["val"][label] // count
            for label, count in zip(LABELS, (authentic, counterfeit), strict=True)
            if count > 0
        ]
        test_limits = [
            targets["test"][label] // count
            for label, count in zip(LABELS, (authentic, counterfeit), strict=True)
            if count > 0
        ]
        options = _vector_assignment_options(
            vector_components,
            seed=seed,
            component_fingerprint=component_fingerprint,
            max_val_items=min(val_limits),
            max_test_items=min(test_limits),
        )
        processed_a += authentic * len(vector_components)
        processed_c += counterfeit * len(vector_components)
        next_states = {}
        for state, (tie_score, mapping_lex) in states.items():
            for (val_items, test_items), (option_score, option_mapping) in options.items():
                values = list(state)
                values[0] += val_items * authentic
                values[1] += val_items * counterfeit
                values[2] += test_items * authentic
                values[3] += test_items * counterfeit
                if is_multi:
                    values[4] += val_items * authentic
                    values[5] += val_items * counterfeit
                    values[6] += test_items * authentic
                    values[7] += test_items * counterfeit
                candidate_state = tuple(values)
                train_a = processed_a - values[0] - values[2]
                train_c = processed_c - values[1] - values[3]
                if any(
                    value > limit
                    for value, limit in (
                        (train_a, targets["train"]["authentic"]),
                        (train_c, targets["train"]["counterfeit"]),
                        (values[0], targets["val"]["authentic"]),
                        (values[1], targets["val"]["counterfeit"]),
                        (values[2], targets["test"]["authentic"]),
                        (values[3], targets["test"]["counterfeit"]),
                    )
                ):
                    continue
                candidate = (
                    tie_score + option_score,
                    tuple(sorted((*mapping_lex, *option_mapping))),
                )
                previous = next_states.get(candidate_state)
                if previous is None or candidate < previous:
                    next_states[candidate_state] = candidate
        states = next_states
        if not states:
            raise ManifestValidationError("V2 component target_counts 不存在精确可行解。")
    return states


def _label_assignment_states(
    components: Sequence[_Component],
    targets: Mapping[str, Mapping[str, int]],
    label: str,
    seed: int,
    component_fingerprint: str,
) -> dict[tuple[int, int], tuple[int, tuple[tuple[str, Partition], ...]]]:
    grouped: dict[int, list[_Component]] = defaultdict(list)
    for component in components:
        grouped[getattr(component, label)].append(component)
    states: dict[tuple[int, int], tuple[int, tuple[tuple[str, Partition], ...]]] = {(0, 0): (0, ())}
    processed = 0
    for weight in sorted(grouped, reverse=True):
        vector_components = tuple(sorted(grouped[weight], key=lambda item: item.component_id))
        options = _vector_assignment_options(
            vector_components,
            seed=seed,
            component_fingerprint=component_fingerprint,
            max_val_items=targets["val"][label] // weight,
            max_test_items=targets["test"][label] // weight,
        )
        processed += weight * len(vector_components)
        next_states: dict[tuple[int, int], tuple[int, tuple[tuple[str, Partition], ...]]] = {}
        for (val_count, test_count), (tie_score, mapping_lex) in states.items():
            for (val_items, test_items), (option_score, option_mapping) in options.items():
                state = (
                    val_count + val_items * weight,
                    test_count + test_items * weight,
                )
                train_count = processed - state[0] - state[1]
                if any(
                    value > targets[partition][label]
                    for value, partition in (
                        (train_count, "train"),
                        (state[0], "val"),
                        (state[1], "test"),
                    )
                ):
                    continue
                candidate = (
                    tie_score + option_score,
                    tuple(sorted((*mapping_lex, *option_mapping))),
                )
                previous = next_states.get(state)
                if previous is None or candidate < previous:
                    next_states[state] = candidate
        states = next_states
        if not states:
            raise ManifestValidationError("V2 component target_counts 不存在精确可行解。")
    return states


def _vector_assignment_options(
    components: Sequence[_Component],
    *,
    seed: int,
    component_fingerprint: str,
    max_val_items: int | None = None,
    max_test_items: int | None = None,
) -> dict[tuple[int, int], tuple[int, tuple[tuple[str, Partition], ...]]]:
    ordered = tuple(
        sorted(
            components,
            key=lambda item: (min(item.groups), item.groups, item.component_id),
        )
    )
    choice_node = tuple[object, Partition] | None
    states: dict[tuple[int, int], tuple[int, choice_node]] = {(0, 0): (0, None)}
    for component in ordered:
        next_states: dict[tuple[int, int], tuple[int, choice_node]] = {}
        for (val_count, test_count), (tie_score, previous_node) in states.items():
            for partition in PARTITIONS:
                state = (
                    val_count + int(partition == "val"),
                    test_count + int(partition == "test"),
                )
                if max_val_items is not None and state[0] > max_val_items:
                    continue
                if max_test_items is not None and state[1] > max_test_items:
                    continue
                score = tie_score + _tie_value(
                    seed, component_fingerprint, component.component_id, partition
                )
                node: choice_node = (previous_node, partition)
                previous = next_states.get(state)
                if previous is None or score < previous[0]:
                    next_states[state] = (score, node)
                elif score == previous[0] and _choice_sequence(node) < _choice_sequence(
                    previous[1]
                ):
                    next_states[state] = (score, node)
        states = next_states
    result = {}
    for state, (score, node) in states.items():
        choices = _choice_sequence(node)
        mapping = tuple(
            sorted(
                (group_id, partition)
                for component, partition in zip(ordered, choices, strict=True)
                for group_id in component.groups
            )
        )
        result[state] = score, mapping
    return result


def _choice_sequence(node: tuple[object, Partition] | None) -> tuple[Partition, ...]:
    reversed_choices: list[Partition] = []
    while node is not None:
        previous, partition = node
        reversed_choices.append(partition)
        node = previous if isinstance(previous, tuple) else None
    return tuple(reversed(reversed_choices))


def _normalize_target_counts(
    rows: Sequence[ValidatedKnownRecord],
    value: Mapping[str, Mapping[str, int]],
) -> dict[str, dict[str, int]]:
    if set(value) != set(PARTITIONS):
        raise ManifestValidationError("V2 target_counts 必须精确包含 train、val、test。")
    result: dict[str, dict[str, int]] = {}
    for partition in PARTITIONS:
        counts = value[partition]
        if not isinstance(counts, Mapping) or set(counts) != set(LABELS):
            raise ManifestValidationError(
                f"V2 target_counts.{partition} 必须包含 authentic/counterfeit。"
            )
        result[partition] = {}
        for label in LABELS:
            count = counts[label]
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                raise ManifestValidationError("V2 target_counts 必须是正整数。")
            result[partition][label] = count
    observed = defaultdict(int)
    for row in rows:
        observed[row.row.authenticity] += 1
    for label in LABELS:
        if sum(result[partition][label] for partition in PARTITIONS) != observed[label]:
            raise ManifestValidationError(f"V2 target_counts 的 {label} 总数与输入不一致。")
    return result


def _partition_label_counts(
    rows: Sequence[ValidatedKnownRecord], by_group: Mapping[str, Partition]
) -> dict[str, dict[str, int]]:
    result = {partition: {label: 0 for label in LABELS} for partition in PARTITIONS}
    for row in rows:
        result[by_group[row.row.group_id]][row.row.authenticity] += 1
    return result


def _independent_cross_partition_count(
    rows: Sequence[PartitionedImageRecord], maximum_distance: int
) -> int:
    count = 0
    for left_index, left in enumerate(rows):
        left_value = int(left.perceptual_hash, 16)
        for right in rows[left_index + 1 :]:
            if left.partition == right.partition:
                continue
            if (left_value ^ int(right.perceptual_hash, 16)).bit_count() <= maximum_distance:
                count += 1
    return count


def _tie_value(seed: int, fingerprint: str, component_id: str, partition: str) -> int:
    payload = f"{seed}|{fingerprint}|{component_id}|{partition}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest(), "big")


def _json_fingerprint(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _plain_component_counts(
    value: Mapping[str, Mapping[str, int]],
) -> dict[str, dict[str, int]]:
    return {
        component_id: {label: int(counts[label]) for label in LABELS}
        for component_id, counts in sorted(value.items())
    }


def _freeze_component_counts(
    value: Mapping[str, Mapping[str, int]],
) -> Mapping[str, Mapping[str, int]]:
    return MappingProxyType(
        {
            component_id: MappingProxyType({label: int(counts[label]) for label in LABELS})
            for component_id, counts in sorted(value.items())
        }
    )


def _component_counts_mapping(value: object) -> Mapping[str, Mapping[str, int]]:
    if not isinstance(value, Mapping):
        raise TypeError("component label counts 必须是映射")
    normalized: dict[str, dict[str, int]] = {}
    for component_id, counts in value.items():
        if not isinstance(counts, Mapping) or set(counts) != set(LABELS):
            raise TypeError("每个 component label counts 必须包含 authentic/counterfeit")
        normalized[str(component_id)] = {}
        for label in LABELS:
            count = counts[label]
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise TypeError("component label counts 必须是非负整数")
            normalized[str(component_id)][label] = count
    return _freeze_component_counts(normalized)


def _plain_counts(value: Mapping[str, Mapping[str, int]]) -> dict[str, dict[str, int]]:
    return {
        partition: {label: int(value[partition][label]) for label in LABELS}
        for partition in PARTITIONS
    }


def _freeze_counts(
    value: Mapping[str, Mapping[str, int]],
) -> Mapping[str, Mapping[str, int]]:
    return MappingProxyType(
        {
            partition: MappingProxyType({label: int(value[partition][label]) for label in LABELS})
            for partition in PARTITIONS
        }
    )


def _counts_mapping(value: object) -> Mapping[str, Mapping[str, int]]:
    if not isinstance(value, Mapping):
        raise TypeError("计数必须是映射")
    return _freeze_counts(value)


def _string_mapping(value: Mapping[object, object]) -> Mapping[str, str]:
    return MappingProxyType(dict(sorted((str(key), str(item)) for key, item in value.items())))


def _validate_ratios(ratios: Sequence[float]) -> tuple[float, float, float]:
    if len(ratios) != 3:
        raise ManifestValidationError("切分比例必须依次包含 train、val、test 三项。")
    normalized = tuple(float(value) for value in ratios)
    if any(not math.isfinite(value) or value <= 0 for value in normalized):
        raise ManifestValidationError("train、val、test 切分比例必须是正有限数。")
    if not math.isclose(sum(normalized), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ManifestValidationError("train、val、test 切分比例之和必须为 1。")
    return normalized[0], normalized[1], normalized[2]


def _validate_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ManifestValidationError("切分 seed 必须是整数。")


def _validate_perceptual_hash_distance(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 64:
        raise ManifestValidationError("切分感知哈希距离必须是 0 到 64 之间的整数。")


def _validate_known_rows(
    rows: Sequence[ValidatedImageRecord],
) -> tuple[ValidatedKnownRecord, ...]:
    if not rows:
        raise ManifestValidationError("known 数据为空，无法切分。")
    if any(not isinstance(record, ValidatedKnownRecord) for record in rows):
        raise ManifestValidationError("group-level 切分只接受已验证的 known 记录。")
    for record in rows:
        _perceptual_hash_value(record.perceptual_hash)
    return tuple(record for record in rows if isinstance(record, ValidatedKnownRecord))


def _collect_group_strata(
    rows: Sequence[ValidatedKnownRecord],
) -> dict[str, tuple[str, str]]:
    grouped_strata: dict[str, set[tuple[str, str]]] = defaultdict(set)
    sample_groups: dict[str, str] = {}
    hash_groups: dict[str, str] = {}
    for record in rows:
        group_id = record.row.group_id
        grouped_strata[group_id].add((record.row.claimed_herb_name, record.row.authenticity))
        _require_single_group(
            sample_groups,
            record.row.sample_id,
            group_id,
            "sample_id",
        )
        _require_single_group(hash_groups, record.sha256, group_id, "图片 SHA-256")

    result: dict[str, tuple[str, str]] = {}
    for group_id, strata in grouped_strata.items():
        if len(strata) != 1:
            raise ManifestValidationError(f"group_id={group_id} 是混合分层组：{sorted(strata)}")
        result[group_id] = next(iter(strata))
    return result


def _require_single_group(
    owners: dict[str, str],
    value: str,
    group_id: str,
    label: str,
) -> None:
    previous = owners.setdefault(value, group_id)
    if previous != group_id:
        raise ManifestValidationError(
            f"{label}={value} 同时属于 group_id={previous} 与 {group_id}，应先合并为同一 group。"
        )


def _build_group_mapping(
    group_strata: Mapping[str, tuple[str, str]],
    ratios: tuple[float, float, float],
    seed: int,
) -> dict[str, Partition]:
    strata: dict[tuple[str, str], list[str]] = defaultdict(list)
    for group_id, stratum in group_strata.items():
        strata[stratum].append(group_id)

    by_group: dict[str, Partition] = {}
    for stratum in sorted(strata):
        group_ids = sorted(strata[stratum])
        if len(group_ids) < 3:
            raise ManifestValidationError(
                f"分层 {stratum} 只有 {len(group_ids)} 个独立 group，至少需要 3 个。"
            )
        randomizer = random.Random(_stratum_seed(seed, stratum))
        randomizer.shuffle(group_ids)
        counts = _largest_remainder_counts(len(group_ids), ratios)
        offset = 0
        for partition, count in zip(PARTITIONS, counts, strict=True):
            for group_id in group_ids[offset : offset + count]:
                by_group[group_id] = partition
            offset += count
    return by_group


def _stratum_seed(seed: int, stratum: tuple[str, str]) -> int:
    payload = json.dumps([seed, *stratum], ensure_ascii=False, separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


def _largest_remainder_counts(
    total: int,
    ratios: tuple[float, float, float],
) -> tuple[int, int, int]:
    quotas = tuple(total * ratio for ratio in ratios)
    counts = [math.floor(quota) for quota in quotas]
    remaining = total - sum(counts)
    order = sorted(
        range(3),
        key=lambda index: (-(quotas[index] - counts[index]), index),
    )
    for index in order[:remaining]:
        counts[index] += 1

    for empty_index in tuple(index for index, count in enumerate(counts) if count == 0):
        donors = [index for index, count in enumerate(counts) if count > 1]
        if not donors:
            raise ManifestValidationError("无法保证 train、val、test 各至少一个 group。")
        donor = max(
            donors,
            key=lambda index: (
                counts[index] - quotas[index],
                counts[index],
                -index,
            ),
        )
        counts[donor] -= 1
        counts[empty_index] += 1
    return counts[0], counts[1], counts[2]


def _input_fingerprint(rows: Sequence[ValidatedKnownRecord]) -> str:
    normalized = [
        {
            "actual_species": tuple(sorted(record.row.actual_species)),
            "annotator": record.row.annotator,
            "authenticity": record.row.authenticity,
            "batch_id": record.row.batch_id,
            "capture_device": record.row.capture_device,
            "claimed_herb_name": record.row.claimed_herb_name,
            "collected_at": record.row.collected_at,
            "file_size": record.file_size,
            "group_id": record.row.group_id,
            "image_format": record.image_format,
            "image_path": record.row.image_path.as_posix(),
            "manifest_encoding": record.manifest_encoding,
            "notes": record.row.notes,
            "origin": record.row.origin,
            "perceptual_hash": record.perceptual_hash,
            "sample_id": record.row.sample_id,
            "sha256": record.sha256,
            "species_annotation_status": record.row.species_annotation_status,
            "width": record.width,
            "height": record.height,
        }
        for record in rows
    ]
    normalized.sort(key=lambda item: json.dumps(item, ensure_ascii=False, sort_keys=True))
    payload = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_partition_leakage(
    rows: Sequence[PartitionedImageRecord],
    maximum_distance: int,
) -> None:
    errors: list[str] = []
    checks: tuple[
        tuple[str, Callable[[PartitionedImageRecord], str]],
        ...,
    ] = (
        ("group_id", lambda item: item.row.group_id),
        ("sample_id", lambda item: item.row.sample_id),
        ("图片 SHA-256", lambda item: item.sha256),
    )
    for label, key in checks:
        owners: dict[str, Partition] = {}
        for item in rows:
            value = key(item)
            previous = owners.setdefault(value, item.partition)
            if previous != item.partition:
                errors.append(f"{label}={value} 跨 {previous}/{item.partition} 分区。")

    errors.extend(_near_duplicate_partition_errors(rows, maximum_distance=maximum_distance))
    if errors:
        raise ManifestValidationError("\n".join(sorted(set(errors))))


def _near_duplicate_partition_errors(
    rows: Sequence[PartitionedImageRecord],
    maximum_distance: int,
) -> list[str]:
    """用 d+1 段倒排索引查找 64 位哈希中距离不超过 d 的跨分区记录。"""

    buckets: dict[tuple[Partition, int, int], list[PartitionedImageRecord]] = defaultdict(list)
    errors: list[str] = []
    for item in rows:
        value = _perceptual_hash_value(item.perceptual_hash)
        candidates: dict[int, PartitionedImageRecord] = {}
        keys = _segment_keys(value, maximum_distance)
        for partition in PARTITIONS:
            if partition == item.partition:
                continue
            for segment_index, segment_value in keys:
                bucket_key = (partition, segment_index, segment_value)
                for candidate in buckets[bucket_key]:
                    candidates[id(candidate)] = candidate
                    if len(candidates) > MAX_CANDIDATES_PER_IMAGE:
                        raise ManifestValidationError(
                            "感知哈希候选数量过多，数据可能低熵或存在大规模近重复；"
                            "请检查采集数据与 group_id。"
                        )
        for candidate in candidates.values():
            distance = (_perceptual_hash_value(candidate.perceptual_hash) ^ value).bit_count()
            if distance <= maximum_distance:
                errors.append(
                    f"感知近重复图片跨 {candidate.partition}/{item.partition} "
                    f"分区（距离 {distance}）：{candidate.row.image_path} "
                    f"与 {item.row.image_path}"
                )
                if len(errors) >= MAX_NEAR_DUPLICATE_ERRORS:
                    errors.append("感知近重复错误过多，后续同类错误已截断。")
                    return errors
        for segment_index, segment_value in keys:
            buckets[(item.partition, segment_index, segment_value)].append(item)
    return errors


def _perceptual_hash_value(value: str) -> int:
    if re.fullmatch(r"[0-9a-f]{16}", value) is None:
        raise ManifestValidationError(f"感知哈希必须是 16 位小写十六进制：{value!r}")
    return int(value, 16)


def _segment_keys(value: int, maximum_distance: int) -> tuple[tuple[int, int], ...]:
    segment_count = maximum_distance + 1
    base_width, wider_segments = divmod(64, segment_count)
    keys: list[tuple[int, int]] = []
    offset = 0
    for index in range(segment_count):
        width = base_width + int(index < wider_segments)
        mask = (1 << width) - 1
        keys.append((index, (value >> offset) & mask))
        offset += width
    return tuple(keys)
