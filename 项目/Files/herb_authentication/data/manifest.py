"""Known/unknown 数据清单解析、图片检查与切分前隔离校验。"""

from __future__ import annotations

import hashlib
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeVar

from PIL import Image, UnidentifiedImageError

from ..encoding import DecodedCsv, read_csv_dicts, write_json_utf8
from ..errors import ManifestValidationError

AUTHENTICITY_VALUES = frozenset({"authentic", "counterfeit"})
SPECIES_STATUS_VALUES = frozenset({"verified", "unverified", "not_applicable"})
UNKNOWN_ROLE_VALUES = frozenset({"calibration", "test"})
ALLOWED_IMAGE_FORMATS = frozenset({"JPEG", "PNG", "WEBP", "BMP", "TIFF"})

KNOWN_REQUIRED_COLUMNS = frozenset(
    {"image_path", "group_id", "sample_id", "claimed_herb_name", "authenticity"}
)
KNOWN_OPTIONAL_COLUMNS = frozenset(
    {
        "actual_species",
        "species_annotation_status",
        "batch_id",
        "origin",
        "capture_device",
        "collected_at",
        "annotator",
        "notes",
    }
)
UNKNOWN_REQUIRED_COLUMNS = frozenset(
    {"image_path", "group_id", "sample_id", "unknown_type", "dataset_role"}
)


@dataclass(frozen=True, slots=True)
class KnownManifestRow:
    image_path: Path
    group_id: str
    sample_id: str
    claimed_herb_name: str
    authenticity: Literal["authentic", "counterfeit"]
    actual_species: tuple[str, ...]
    species_annotation_status: Literal["verified", "unverified", "not_applicable"]
    batch_id: str | None
    origin: str | None
    capture_device: str | None
    collected_at: str | None
    annotator: str | None
    notes: str | None
    source_line: int


@dataclass(frozen=True, slots=True)
class UnknownManifestRow:
    image_path: Path
    group_id: str
    sample_id: str
    unknown_type: str
    dataset_role: Literal["calibration", "test"]
    source_line: int


@dataclass(frozen=True, slots=True)
class LoadedKnownManifest:
    rows: tuple[KnownManifestRow, ...]
    source_path: Path
    actual_encoding: str
    used_encoding_fallback: bool


@dataclass(frozen=True, slots=True)
class LoadedUnknownManifest:
    rows: tuple[UnknownManifestRow, ...]
    source_path: Path
    actual_encoding: str
    used_encoding_fallback: bool


@dataclass(frozen=True, slots=True)
class ValidatedKnownRecord:
    row: KnownManifestRow
    sha256: str
    perceptual_hash: str
    image_format: str
    width: int
    height: int
    file_size: int
    manifest_encoding: str = "utf-8-sig"


@dataclass(frozen=True, slots=True)
class ValidatedUnknownRecord:
    row: UnknownManifestRow
    sha256: str
    perceptual_hash: str
    image_format: str
    width: int
    height: int
    file_size: int
    manifest_encoding: str = "utf-8-sig"


ValidatedImageRecord = ValidatedKnownRecord | ValidatedUnknownRecord


@dataclass(frozen=True, slots=True)
class ManifestBundle:
    known: tuple[ValidatedKnownRecord, ...]
    unknown: tuple[ValidatedUnknownRecord, ...]
    known_encoding: str
    unknown_encoding: str | None
    warnings: tuple[str, ...]


def load_known_manifest(
    path: str | Path,
    *,
    data_root: str | Path | None = None,
    encoding: str | None = None,
) -> LoadedKnownManifest:
    """解析 known `metadata.csv`，不执行图片解码。"""

    source_path = _require_file(path, "known manifest")
    root = _resolve_data_root(source_path, data_root)
    try:
        decoded = read_csv_dicts(source_path, encoding=encoding)
    except (OSError, UnicodeError, ValueError) as exc:
        raise ManifestValidationError(f"无法读取 known manifest {source_path}：{exc}") from exc
    _validate_columns(
        decoded,
        required=KNOWN_REQUIRED_COLUMNS,
        allowed=KNOWN_REQUIRED_COLUMNS | KNOWN_OPTIONAL_COLUMNS,
        manifest_name="known manifest",
    )

    rows: list[KnownManifestRow] = []
    errors: list[str] = []
    for line_number, raw in enumerate(decoded.rows, start=2):
        try:
            rows.append(_parse_known_row(raw, root, line_number))
        except ManifestValidationError as exc:
            errors.append(str(exc))
    _raise_if_errors(errors)
    return LoadedKnownManifest(
        rows=tuple(rows),
        source_path=source_path,
        actual_encoding=decoded.actual_encoding,
        used_encoding_fallback=decoded.used_fallback,
    )


def load_unknown_manifest(
    path: str | Path,
    *,
    data_root: str | Path | None = None,
    encoding: str | None = None,
) -> LoadedUnknownManifest:
    """解析独立的 `unknown_metadata.csv`，不执行图片解码。"""

    source_path = _require_file(path, "unknown manifest")
    root = _resolve_data_root(source_path, data_root)
    try:
        decoded = read_csv_dicts(source_path, encoding=encoding)
    except (OSError, UnicodeError, ValueError) as exc:
        raise ManifestValidationError(f"无法读取 unknown manifest {source_path}：{exc}") from exc
    _validate_columns(
        decoded,
        required=UNKNOWN_REQUIRED_COLUMNS,
        allowed=UNKNOWN_REQUIRED_COLUMNS,
        manifest_name="unknown manifest",
    )

    rows: list[UnknownManifestRow] = []
    errors: list[str] = []
    for line_number, raw in enumerate(decoded.rows, start=2):
        try:
            rows.append(_parse_unknown_row(raw, root, line_number))
        except ManifestValidationError as exc:
            errors.append(str(exc))
    _raise_if_errors(errors)
    return LoadedUnknownManifest(
        rows=tuple(rows),
        source_path=source_path,
        actual_encoding=decoded.actual_encoding,
        used_encoding_fallback=decoded.used_fallback,
    )


def validate_known_manifest(rows: Sequence[KnownManifestRow]) -> None:
    """校验 known 行的标签与 group/sample 一致性。"""

    _raise_if_errors(_known_semantic_errors(rows, require_minimum_groups=False))


def validate_global_isolation(
    known: Sequence[ValidatedKnownRecord],
    unknown: Sequence[ValidatedUnknownRecord],
    *,
    perceptual_hash_distance: int | None = None,
) -> None:
    """校验 known 与 unknown、unknown calibration 与 test 的全局隔离。"""

    errors = _global_isolation_errors(known, unknown, perceptual_hash_distance)
    _raise_if_errors(errors)


def validate_manifest_pair(
    known_manifest_path: str | Path,
    *,
    unknown_manifest_path: str | Path | None,
    data_root: str | Path | None = None,
    report_path: str | Path,
    encoding: str | None = None,
    max_image_bytes: int = 20 * 1024 * 1024,
    max_image_pixels: int = 40_000_000,
    perceptual_hash_distance: int | None = 4,
) -> ManifestBundle:
    """验证成对清单；无论成功失败都写 UTF-8 审计报告。"""

    errors: list[str] = []
    warnings: list[str] = []
    loaded_known: LoadedKnownManifest | None = None
    loaded_unknown: LoadedUnknownManifest | None = None

    try:
        loaded_known = load_known_manifest(
            known_manifest_path,
            data_root=data_root,
            encoding=encoding,
        )
    except ManifestValidationError as exc:
        errors.extend(_split_error(exc))

    if unknown_manifest_path is not None:
        try:
            loaded_unknown = load_unknown_manifest(
                unknown_manifest_path,
                data_root=data_root,
                encoding=encoding,
            )
        except ManifestValidationError as exc:
            errors.extend(_split_error(exc))

    known_records: tuple[ValidatedKnownRecord, ...] = ()
    unknown_records: tuple[ValidatedUnknownRecord, ...] = ()
    if loaded_known is not None:
        if loaded_known.used_encoding_fallback:
            warnings.append("known manifest 使用 GBK 回退解码；建议转换为 UTF-8-SIG。")
        semantic_errors, semantic_warnings = _known_semantic_errors_with_warnings(loaded_known.rows)
        errors.extend(semantic_errors)
        warnings.extend(semantic_warnings)
        known_records, image_errors = _inspect_known_images(
            loaded_known.rows,
            manifest_encoding=loaded_known.actual_encoding,
            max_image_bytes=max_image_bytes,
            max_image_pixels=max_image_pixels,
        )
        errors.extend(image_errors)

    if loaded_unknown is not None:
        if loaded_unknown.used_encoding_fallback:
            warnings.append("unknown manifest 使用 GBK 回退解码；建议转换为 UTF-8-SIG。")
        errors.extend(_unknown_semantic_errors(loaded_unknown.rows))
        unknown_records, image_errors = _inspect_unknown_images(
            loaded_unknown.rows,
            manifest_encoding=loaded_unknown.actual_encoding,
            max_image_bytes=max_image_bytes,
            max_image_pixels=max_image_pixels,
        )
        errors.extend(image_errors)

    if loaded_known is not None:
        errors.extend(_duplicate_record_errors(known_records, "known"))
    if loaded_unknown is not None:
        errors.extend(_duplicate_record_errors(unknown_records, "unknown"))
    if known_records and unknown_records:
        errors.extend(
            _global_isolation_errors(
                known_records,
                unknown_records,
                perceptual_hash_distance,
            )
        )

    errors = sorted(set(errors))
    warnings = sorted(set(warnings))
    report = {
        "schema_version": 1,
        "status": "invalid" if errors else "valid",
        "counts": {"known": len(known_records), "unknown": len(unknown_records)},
        "encodings": {
            "known": loaded_known.actual_encoding if loaded_known else None,
            "unknown": loaded_unknown.actual_encoding if loaded_unknown else None,
        },
        "errors": errors,
        "warnings": warnings,
    }
    write_json_utf8(report_path, report)
    _raise_if_errors(errors)

    if loaded_known is None:
        raise ManifestValidationError("known manifest 未成功加载。")
    return ManifestBundle(
        known=known_records,
        unknown=unknown_records,
        known_encoding=loaded_known.actual_encoding,
        unknown_encoding=loaded_unknown.actual_encoding if loaded_unknown else None,
        warnings=tuple(warnings),
    )


def _parse_known_row(raw: dict[str, str], root: Path, line: int) -> KnownManifestRow:
    prefix = f"known manifest 第 {line} 行"
    authenticity = _required(raw.get("authenticity"), "authenticity", prefix)
    if authenticity not in AUTHENTICITY_VALUES:
        raise ManifestValidationError(
            f"{prefix}：authenticity={authenticity!r}，只允许 authentic/counterfeit。"
        )
    status = _optional(raw.get("species_annotation_status")) or "unverified"
    if status not in SPECIES_STATUS_VALUES:
        raise ManifestValidationError(f"{prefix}：species_annotation_status={status!r} 不合法。")
    return KnownManifestRow(
        image_path=_safe_image_path(raw.get("image_path"), root, prefix),
        group_id=_required(raw.get("group_id"), "group_id", prefix),
        sample_id=_required(raw.get("sample_id"), "sample_id", prefix),
        claimed_herb_name=_required(raw.get("claimed_herb_name"), "claimed_herb_name", prefix),
        authenticity=authenticity,
        actual_species=_parse_species(raw.get("actual_species"), prefix),
        species_annotation_status=status,
        batch_id=_optional(raw.get("batch_id")),
        origin=_optional(raw.get("origin")),
        capture_device=_optional(raw.get("capture_device")),
        collected_at=_optional(raw.get("collected_at")),
        annotator=_optional(raw.get("annotator")),
        notes=_optional_multiline(raw.get("notes")),
        source_line=line,
    )


def _parse_unknown_row(raw: dict[str, str], root: Path, line: int) -> UnknownManifestRow:
    prefix = f"unknown manifest 第 {line} 行"
    role = _required(raw.get("dataset_role"), "dataset_role", prefix)
    if role not in UNKNOWN_ROLE_VALUES:
        raise ManifestValidationError(f"{prefix}：dataset_role={role!r}，只允许 calibration/test。")
    return UnknownManifestRow(
        image_path=_safe_image_path(raw.get("image_path"), root, prefix),
        group_id=_required(raw.get("group_id"), "group_id", prefix),
        sample_id=_required(raw.get("sample_id"), "sample_id", prefix),
        unknown_type=_required(raw.get("unknown_type"), "unknown_type", prefix),
        dataset_role=role,
        source_line=line,
    )


def _known_semantic_errors(
    rows: Sequence[KnownManifestRow], require_minimum_groups: bool
) -> list[str]:
    errors, _ = _known_semantic_errors_with_warnings(rows, require_minimum_groups)
    return errors


def _known_semantic_errors_with_warnings(
    rows: Sequence[KnownManifestRow], require_minimum_groups: bool = True
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    if not rows:
        errors.append("known manifest 没有数据行。")
    group_labels: dict[str, tuple[str, str]] = {}
    sample_labels: dict[str, tuple[str, str, str, tuple[str, ...], str]] = {}
    batch_groups: dict[str, str] = {}
    strata_groups: dict[tuple[str, str], set[str]] = defaultdict(set)
    paths: set[Path] = set()

    for row in rows:
        if row.image_path in paths:
            errors.append(f"known manifest 存在重复 image_path：{row.image_path}")
        paths.add(row.image_path)
        if row.species_annotation_status == "verified" and not row.actual_species:
            errors.append(f"known manifest 第 {row.source_line} 行：verified 实际物种不能为空。")

        group_value = (row.claimed_herb_name, row.authenticity)
        previous_group = group_labels.setdefault(row.group_id, group_value)
        if previous_group != group_value:
            errors.append(
                f"group_id={row.group_id} 同时对应 {previous_group} 与 {group_value}，必须拆组。"
            )

        if row.batch_id:
            previous_batch_group = batch_groups.setdefault(row.batch_id, row.group_id)
            if previous_batch_group != row.group_id:
                errors.append(
                    f"batch_id={row.batch_id} 跨 group_id="
                    f"{previous_batch_group}/{row.group_id}，必须合并。"
                )
        sample_value = (
            row.group_id,
            row.claimed_herb_name,
            row.authenticity,
            tuple(sorted(row.actual_species)),
            row.species_annotation_status,
        )
        previous_sample = sample_labels.setdefault(row.sample_id, sample_value)
        if previous_sample != sample_value:
            errors.append(
                f"sample_id={row.sample_id} 的 group/声称药材/真伪/实际物种/核验状态不一致。"
            )
        strata_groups[group_value].add(row.group_id)

    if require_minimum_groups:
        for stratum, groups in sorted(strata_groups.items()):
            if len(groups) < 3:
                errors.append(f"分层 {stratum} 只有 {len(groups)} 个独立 group，至少需要 3 个。")
            elif len(groups) < 10:
                warnings.append(f"分层 {stratum} 只有 {len(groups)} 个 group，评估可能不稳定。")
    return errors, warnings


def _unknown_semantic_errors(rows: Sequence[UnknownManifestRow]) -> list[str]:
    errors: list[str] = []
    sample_groups: dict[str, str] = {}
    group_roles: dict[str, str] = {}
    if not rows:
        errors.append("unknown manifest 没有数据行。")
    paths: set[Path] = set()
    for row in rows:
        if row.image_path in paths:
            errors.append(f"unknown manifest 存在重复 image_path：{row.image_path}")
        paths.add(row.image_path)
        previous_group = sample_groups.setdefault(row.sample_id, row.group_id)
        if previous_group != row.group_id:
            errors.append(f"unknown sample_id={row.sample_id} 对应多个 group_id。")
        previous_role = group_roles.setdefault(row.group_id, row.dataset_role)
        if previous_role != row.dataset_role:
            errors.append(f"unknown group_id={row.group_id} 跨 calibration/test。")
    return errors


def _inspect_known_images(
    rows: Sequence[KnownManifestRow],
    *,
    manifest_encoding: str,
    max_image_bytes: int,
    max_image_pixels: int,
) -> tuple[tuple[ValidatedKnownRecord, ...], list[str]]:
    return _inspect_images(
        rows, ValidatedKnownRecord, manifest_encoding, max_image_bytes, max_image_pixels
    )


def _inspect_unknown_images(
    rows: Sequence[UnknownManifestRow],
    *,
    manifest_encoding: str,
    max_image_bytes: int,
    max_image_pixels: int,
) -> tuple[tuple[ValidatedUnknownRecord, ...], list[str]]:
    return _inspect_images(
        rows, ValidatedUnknownRecord, manifest_encoding, max_image_bytes, max_image_pixels
    )


RowType = TypeVar("RowType", KnownManifestRow, UnknownManifestRow)
RecordType = TypeVar("RecordType", ValidatedKnownRecord, ValidatedUnknownRecord)


def _inspect_images(
    rows: Sequence[RowType],
    record_type: type[RecordType],
    manifest_encoding: str,
    max_image_bytes: int,
    max_image_pixels: int,
) -> tuple[tuple[RecordType, ...], list[str]]:
    records: list[RecordType] = []
    errors: list[str] = []
    for row in rows:
        try:
            image_info = _inspect_image(
                row.image_path,
                max_image_bytes=max_image_bytes,
                max_image_pixels=max_image_pixels,
            )
        except ManifestValidationError as exc:
            errors.append(str(exc))
            continue
        records.append(record_type(row=row, manifest_encoding=manifest_encoding, **image_info))
    return tuple(records), errors


def _inspect_image(path: Path, *, max_image_bytes: int, max_image_pixels: int) -> dict[str, object]:
    if not path.is_file():
        raise ManifestValidationError(f"图片不存在或不是文件：{path}")
    file_size = path.stat().st_size
    if file_size <= 0:
        raise ManifestValidationError(f"图片为空文件：{path}")
    if file_size > max_image_bytes:
        raise ManifestValidationError(
            f"图片超过大小上限 {max_image_bytes} 字节：{path}（{file_size} 字节）"
        )

    sha256 = _sha256_file(path)
    try:
        with Image.open(path) as image:
            image_format = (image.format or "").upper()
            width, height = image.size
            if width <= 0 or height <= 0:
                raise ManifestValidationError(f"图片尺寸非法：{path} ({width}x{height})")
            if width * height > max_image_pixels:
                raise ManifestValidationError(
                    f"图片像素超过上限 {max_image_pixels}：{path} ({width}x{height})"
                )
            if image_format not in ALLOWED_IMAGE_FORMATS:
                allowed = ", ".join(sorted(ALLOWED_IMAGE_FORMATS))
                raise ManifestValidationError(
                    f"图片格式 {image_format or 'unknown'} 不支持：{path}；允许 {allowed}"
                )
            image.load()
            perceptual_hash = _average_hash(image)
    except ManifestValidationError:
        raise
    except Image.DecompressionBombError as exc:
        raise ManifestValidationError(f"图片触发解压炸弹保护：{path}（{exc}）") from exc
    except (OSError, SyntaxError, UnidentifiedImageError) as exc:
        raise ManifestValidationError(f"图片无法解码：{path}（{exc}）") from exc

    return {
        "sha256": sha256,
        "perceptual_hash": perceptual_hash,
        "image_format": image_format,
        "width": width,
        "height": height,
        "file_size": file_size,
    }


def _average_hash(image: Image.Image) -> str:
    grayscale = image.convert("L").resize((8, 8), Image.Resampling.LANCZOS)
    pixels = tuple(grayscale.getdata())
    average = sum(pixels) / len(pixels)
    bits = 0
    for value in pixels:
        bits = (bits << 1) | int(value >= average)
    return f"{bits:016x}"


def _duplicate_record_errors(
    records: Sequence[ValidatedImageRecord], manifest_name: str
) -> list[str]:
    hashes: dict[str, Path] = {}
    errors: list[str] = []
    for record in records:
        previous = hashes.setdefault(record.sha256, record.row.image_path)
        if previous != record.row.image_path:
            errors.append(
                f"{manifest_name} 存在内容哈希重复图片：{previous} 与 {record.row.image_path}"
            )
    return errors


def _global_isolation_errors(
    known: Sequence[ValidatedKnownRecord],
    unknown: Sequence[ValidatedUnknownRecord],
    perceptual_hash_distance: int | None,
) -> list[str]:
    errors: list[str] = []
    known_groups = {record.row.group_id for record in known}
    known_samples = {record.row.sample_id for record in known}
    known_hashes = {record.sha256 for record in known}
    unknown_groups = {record.row.group_id for record in unknown}
    unknown_samples = {record.row.sample_id for record in unknown}
    unknown_hashes = {record.sha256 for record in unknown}

    _append_overlap(errors, "known/unknown", "group_id", known_groups, unknown_groups)
    _append_overlap(errors, "known/unknown", "sample_id", known_samples, unknown_samples)
    _append_overlap(errors, "known/unknown", "内容哈希", known_hashes, unknown_hashes)

    calibration = tuple(record for record in unknown if record.row.dataset_role == "calibration")
    test = tuple(record for record in unknown if record.row.dataset_role == "test")
    _append_overlap(
        errors,
        "unknown calibration/test",
        "group_id",
        {record.row.group_id for record in calibration},
        {record.row.group_id for record in test},
    )
    _append_overlap(
        errors,
        "unknown calibration/test",
        "sample_id",
        {record.row.sample_id for record in calibration},
        {record.row.sample_id for record in test},
    )
    _append_overlap(
        errors,
        "unknown calibration/test",
        "内容哈希",
        {record.sha256 for record in calibration},
        {record.sha256 for record in test},
    )

    if perceptual_hash_distance is not None:
        if not 0 <= perceptual_hash_distance <= 64:
            errors.append("perceptual_hash_distance 必须在 0 到 64 之间。")
        else:
            errors.extend(
                _near_duplicate_errors(
                    known,
                    unknown,
                    perceptual_hash_distance,
                    "known/unknown",
                )
            )
            errors.extend(
                _near_duplicate_errors(
                    calibration,
                    test,
                    perceptual_hash_distance,
                    "unknown calibration/test",
                )
            )
    return errors


@dataclass(slots=True)
class _HammingNode:
    value: int
    records: list[ValidatedImageRecord]
    children: dict[int, _HammingNode]


def _hamming_insert(root: _HammingNode, value: int, record: ValidatedImageRecord) -> None:
    node = root
    while True:
        distance = (value ^ node.value).bit_count()
        if distance == 0:
            node.records.append(record)
            return
        child = node.children.get(distance)
        if child is None:
            node.children[distance] = _HammingNode(value, [record], {})
            return
        node = child


def _hamming_query(
    root: _HammingNode, value: int, maximum_distance: int
) -> list[tuple[ValidatedImageRecord, int]]:
    matches: list[tuple[ValidatedImageRecord, int]] = []
    pending = [root]
    while pending:
        node = pending.pop()
        distance = (value ^ node.value).bit_count()
        if distance <= maximum_distance:
            matches.extend((record, distance) for record in node.records)
        minimum_edge = distance - maximum_distance
        maximum_edge = distance + maximum_distance
        pending.extend(
            node.children[edge]
            for edge in sorted(node.children, reverse=True)
            if minimum_edge <= edge <= maximum_edge
        )
    return matches


def _near_duplicate_errors(
    left: Sequence[ValidatedImageRecord],
    right: Sequence[ValidatedImageRecord],
    maximum_distance: int,
    label: str,
) -> list[str]:
    errors: list[str] = []
    if not left or not right:
        return errors

    first = right[0]
    root = _HammingNode(int(first.perceptual_hash, 16), [first], {})
    for record in right[1:]:
        _hamming_insert(root, int(record.perceptual_hash, 16), record)

    for left_record in left:
        value = int(left_record.perceptual_hash, 16)
        for right_record, distance in _hamming_query(root, value, maximum_distance):
            errors.append(
                f"{label} 存在感知近重复图片（距离 {distance}）："
                f"{left_record.row.image_path} 与 {right_record.row.image_path}"
            )
    return errors


def _append_overlap(
    errors: list[str], scope: str, label: str, left: set[str], right: set[str]
) -> None:
    overlap = sorted(left & right)
    if overlap:
        errors.append(f"{scope} {label} 发生交叉：{overlap}")


def _validate_columns(
    decoded: DecodedCsv,
    *,
    required: frozenset[str],
    allowed: frozenset[str],
    manifest_name: str,
) -> None:
    actual = set(decoded.fieldnames)
    missing = sorted(required - actual)
    unknown = sorted(actual - allowed)
    errors: list[str] = []
    if missing:
        errors.append(f"{manifest_name} 缺少必填列：{missing}")
    if unknown:
        errors.append(f"{manifest_name} 包含未知列：{unknown}")
    _raise_if_errors(errors)


def _required(value: str | None, field: str, prefix: str) -> str:
    normalized = _normalize_text(value)
    if not normalized:
        raise ManifestValidationError(f"{prefix}：{field} 不能为空。")
    return normalized


def _optional(value: str | None) -> str | None:
    normalized = _normalize_text(value)
    return normalized or None


def _optional_multiline(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = unicodedata.normalize("NFC", str(value)).replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.strip()
    controls = [
        character
        for character in normalized
        if unicodedata.category(character) in {"Cc", "Cf"} and character not in {"\n", "\t"}
    ]
    if controls:
        codepoints = ", ".join(f"U+{ord(character):04X}" for character in controls)
        raise ManifestValidationError(f"文本含禁止控制字符：{codepoints}")
    return normalized or None


def _normalize_text(value: str | None) -> str:
    if value is None:
        return ""
    normalized = unicodedata.normalize("NFC", str(value)).strip()
    controls = [
        character for character in normalized if unicodedata.category(character) in {"Cc", "Cf"}
    ]
    if controls:
        codepoints = ", ".join(f"U+{ord(character):04X}" for character in controls)
        raise ManifestValidationError(f"文本含禁止控制字符：{codepoints}")
    return normalized


def _parse_species(value: str | None, prefix: str) -> tuple[str, ...]:
    normalized = _normalize_text(value)
    if not normalized:
        return ()
    parts = tuple(_normalize_text(part) for part in normalized.split("|"))
    if any(not part for part in parts):
        raise ManifestValidationError(f"{prefix}：actual_species 包含空物种标签。")
    if len(set(parts)) != len(parts):
        raise ManifestValidationError(f"{prefix}：actual_species 包含重复物种标签。")
    return parts


def _safe_image_path(value: str | None, root: Path, prefix: str) -> Path:
    raw = _required(value, "image_path", prefix)
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()
    if not resolved.is_relative_to(root):
        raise ManifestValidationError(f"{prefix}：image_path 越过 data_root：{raw!r} -> {resolved}")
    return resolved


def _resolve_data_root(source_path: Path, data_root: str | Path | None) -> Path:
    if data_root is None:
        return source_path.parent.resolve()
    return Path(data_root).expanduser().resolve()


def _require_file(path: str | Path, name: str) -> Path:
    source_path = Path(path).expanduser().resolve()
    if not source_path.is_file():
        raise ManifestValidationError(f"{name} 不存在或不是文件：{source_path}")
    return source_path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _raise_if_errors(errors: Iterable[str]) -> None:
    normalized = sorted(set(errors))
    if normalized:
        raise ManifestValidationError("\n".join(normalized))


def _split_error(error: ManifestValidationError) -> list[str]:
    return [line for line in str(error).splitlines() if line]
