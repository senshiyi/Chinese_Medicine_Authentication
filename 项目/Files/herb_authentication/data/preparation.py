"""从受控原图目录原子生成稳定的闭集 known manifest。"""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from re import Pattern
from types import MappingProxyType

from PIL import Image, UnidentifiedImageError

from ..encoding import read_csv_dicts, write_csv_utf8_sig
from ..errors import ManifestValidationError

MANIFEST_FIELDNAMES = (
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
)
AUTHENTICITY_ORDER = {"authentic": 0, "counterfeit": 1}
SPECIES_STATUSES = frozenset({"verified", "unverified", "not_applicable"})


@dataclass(frozen=True, slots=True)
class ManifestSource:
    directory: Path
    filename_pattern: Pattern[str]
    id_prefix: str
    authenticity: str
    actual_species: str | None
    species_annotation_status: str


@dataclass(frozen=True, slots=True)
class ManifestBuildResult:
    path: Path
    rows: int
    counts: Mapping[str, int]
    sha256: str


def build_closed_set_manifest(
    sources: Sequence[ManifestSource],
    data_root: Path,
    output_path: Path,
    claimed_herb_name: str,
) -> ManifestBuildResult:
    """扫描来源目录，稳定排序并原子写出 UTF-8-SIG manifest。"""

    if not sources:
        raise ManifestValidationError("至少需要一个 manifest 来源目录。")
    if not claimed_herb_name.strip():
        raise ManifestValidationError("claimed_herb_name 不能为空。")

    root = Path(data_root).expanduser().resolve()
    discovered: list[tuple[ManifestSource, Path, int]] = []
    for source in sources:
        discovered.extend(_discover_source(source, root))
        _validate_closed_set_source(source, claimed_herb_name)

    discovered.sort(
        key=lambda item: (
            AUTHENTICITY_ORDER[item[0].authenticity],
            item[2],
            item[1].name,
        )
    )
    rows = [
        _row(source, image, index, root, claimed_herb_name)
        for source, image, index in discovered
    ]
    sample_ids = [row["sample_id"] for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ManifestValidationError(
            "来源配置生成了重复 sample_id/group_id，请使用唯一 id_prefix。"
        )

    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(destination)
    try:
        write_csv_utf8_sig(temporary, MANIFEST_FIELDNAMES, rows)
        decoded = read_csv_dicts(temporary, encoding="utf-8-sig")
        if decoded.fieldnames != MANIFEST_FIELDNAMES or list(decoded.rows) != rows:
            raise ManifestValidationError("临时 manifest 回读内容与待写数据不一致。")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)

    counts = Counter(row["authenticity"] for row in rows)
    return ManifestBuildResult(
        path=destination,
        rows=len(rows),
        counts=MappingProxyType(dict(sorted(counts.items()))),
        sha256=_sha256_file(destination),
    )


def _discover_source(
    source: ManifestSource, data_root: Path
) -> list[tuple[ManifestSource, Path, int]]:
    _validate_source(source)
    directory = Path(source.directory).expanduser()
    if directory.is_symlink():
        raise ManifestValidationError(f"来源目录不允许是符号链接：{directory}")
    resolved_directory = directory.resolve()
    _require_within_data_root(resolved_directory, data_root)
    if not resolved_directory.is_dir():
        raise ManifestValidationError(f"来源目录不存在或不是目录：{resolved_directory}")

    found: list[tuple[ManifestSource, Path, int]] = []
    indexes: dict[int, str] = {}
    for image in sorted(resolved_directory.iterdir(), key=lambda path: path.name):
        if image.is_symlink():
            raise ManifestValidationError(f"图片不允许是符号链接：{image}")
        if not image.is_file():
            raise ManifestValidationError(f"来源目录只允许普通 JPEG 文件：{image}")
        resolved_image = image.resolve()
        _require_within_data_root(resolved_image, data_root)
        match = source.filename_pattern.fullmatch(image.name)
        if match is None:
            detail = (
                "文件名缺少或包含非法数字索引"
                if image.suffix.lower() == ".jpg"
                else "文件名不符合约定"
            )
            raise ManifestValidationError(f"{detail}：{image.name}")
        try:
            raw_index = match.group(1)
            index = int(raw_index)
        except (IndexError, TypeError, ValueError) as exc:
            raise ManifestValidationError(f"文件名缺少合法数字索引：{image.name}") from exc
        if index in indexes:
            raise ManifestValidationError(
                f"来源目录存在重复数字索引 {index:04d}：{indexes[index]}、{image.name}"
            )
        _require_jpeg(resolved_image)
        indexes[index] = image.name
        found.append((source, resolved_image, index))

    expected_indexes = list(range(1, len(found) + 1))
    if sorted(indexes) != expected_indexes:
        raise ManifestValidationError(
            f"来源目录数字索引必须从 1 开始连续；实际 {sorted(indexes)}，预期 {expected_indexes}。"
        )
    return found


def _validate_source(source: ManifestSource) -> None:
    if source.authenticity not in AUTHENTICITY_ORDER:
        raise ManifestValidationError("authenticity 只允许 authentic/counterfeit。")
    if source.species_annotation_status not in SPECIES_STATUSES:
        raise ManifestValidationError("species_annotation_status 不合法。")
    if not source.id_prefix.strip():
        raise ManifestValidationError("id_prefix 不能为空。")


def _validate_closed_set_source(source: ManifestSource, claimed_herb_name: str) -> None:
    authentic = (
        source.actual_species == claimed_herb_name
        and source.species_annotation_status == "verified"
    )
    counterfeit = (
        source.actual_species is None
        and source.species_annotation_status == "unverified"
    )
    if source.authenticity == "authentic" and not authentic:
        raise ManifestValidationError(
            "闭集来源标签不合法：authentic 必须满足 "
            "actual_species=claimed_herb_name 且 species_annotation_status=verified。"
        )
    if source.authenticity == "counterfeit" and not counterfeit:
        raise ManifestValidationError(
            "闭集来源标签不合法：counterfeit 必须满足 "
            "actual_species=None 且 species_annotation_status=unverified。"
        )


def _require_within_data_root(path: Path, data_root: Path) -> None:
    try:
        path.relative_to(data_root)
    except ValueError as exc:
        raise ManifestValidationError(f"路径逃逸出 data_root：{path}") from exc


def _require_jpeg(path: Path) -> None:
    try:
        with Image.open(path) as image:
            if image.format != "JPEG":
                raise ManifestValidationError(f"文件内容不是 JPEG：{path}")
            image.verify()
    except ManifestValidationError:
        raise
    except (OSError, SyntaxError, UnidentifiedImageError) as exc:
        raise ManifestValidationError(f"JPEG 文件无法解码：{path}（{exc}）") from exc


def _row(
    source: ManifestSource,
    image: Path,
    index: int,
    data_root: Path,
    claimed: str,
) -> dict[str, str]:
    sample_id = f"{source.id_prefix}-{index:04d}"
    return {
        "image_path": image.resolve().relative_to(data_root.resolve()).as_posix(),
        "group_id": sample_id,
        "sample_id": sample_id,
        "claimed_herb_name": claimed,
        "authenticity": source.authenticity,
        "actual_species": source.actual_species or "",
        "species_annotation_status": source.species_annotation_status,
        "batch_id": "",
        "origin": "",
        "capture_device": "",
        "collected_at": "",
        "annotator": "",
        "notes": "independent physical sample; batch/source/device/session unavailable",
    }


def _temporary_path(destination: Path) -> Path:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    return Path(name)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
