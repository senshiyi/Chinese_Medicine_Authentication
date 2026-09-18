"""范围标注清单（manifest）读取与校验。

清单是训练 / 校准 / 评估 / 发布范围门控的唯一数据入口，列定义：

- ``image_path``：相对 ``--image-root`` 的图片路径。
- ``scope_label``：``in_scope`` / ``out_of_scope``（明确标注，绝不从闭集真伪标签推断）。
- ``split``：``train`` / ``calibration`` / ``test``。
- ``group_id``：去重内容 / 分组隔离键。
- ``source``：数据来源（溯源）。
- ``consent``：是否已获研究训练授权；未授权行会被拒绝参与训练。

不变量（训练前校验）：
1. 只有三个 split，且 scope_label 与 split 取值合法；
2. 每行都明确授权（consent 为真）；
3. group 在 train / calibration / test 之间互不重叠（避免分组泄漏）；
4. 任意两个样本内容（图片 SHA-256）不得重复（同一张图不允许被复用标注）。

本模块只依赖标准库，不导入 numpy / sklearn / torch。
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCOPE_IN_SCOPE = "in_scope"
SCOPE_OUT_OF_SCOPE = "out_of_scope"
SCOPE_LABELS = {SCOPE_IN_SCOPE, SCOPE_OUT_OF_SCOPE}

SPLIT_TRAIN = "train"
SPLIT_CALIBRATION = "calibration"
SPLIT_TEST = "test"
SPLITS = {SPLIT_TRAIN, SPLIT_CALIBRATION, SPLIT_TEST}

REQUIRED_COLUMNS = (
    "image_path",
    "scope_label",
    "split",
    "group_id",
    "source",
    "consent",
)

# 接受常见真值写法
_TRUE_TOKENS = {"1", "true", "yes", "y", "on", "t", "是", "同意", "已授权"}
_FALSE_TOKENS = {"0", "false", "no", "n", "off", "f", "否", "未同意", "未授权"}


class ManifestError(RuntimeError):
    """清单不满足范围训练前置条件。错误信息不带任何真伪结论。"""


@dataclass(frozen=True)
class ScopeManifestRow:
    image_path: str
    scope_label: str
    split: str
    group_id: str
    source: str
    consent: bool
    absolute_path: Path
    content_sha256: str | None = None  # 校验重复内容时填充


def _parse_consent(value: Any) -> bool:
    text = str(value).strip().lower()
    if text in _TRUE_TOKENS:
        return True
    if text in _FALSE_TOKENS:
        return False
    raise ManifestError(f"consent 字段无法解析为授权/未授权：{value!r}")


def _resolve(column: str, value: Any, image_root: Path) -> str:
    if value is None or not str(value).strip():
        raise ManifestError(f"{column} 为空")
    return str(value).strip()


def load_manifest(csv_path: str | Path, image_root: str | Path) -> list[ScopeManifestRow]:
    """读取并解析清单 CSV，校验列 / 枚举 / 授权 / 路径。

    只做结构与取值校验，不在此计算内容哈希（重复内容由 :func:`detect_duplicate_content` 处理）。
    """
    source_path = Path(csv_path).expanduser()
    if not source_path.is_file():
        raise ManifestError(f"清单文件不存在：{source_path}")
    root = Path(image_root).expanduser().resolve()
    if not root.is_dir():
        raise ManifestError(f"图片根目录不是有效目录：{root}")

    rows: list[ScopeManifestRow] = []
    with source_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [col for col in REQUIRED_COLUMNS if col not in (reader.fieldnames or [])]
        if missing:
            raise ManifestError(f"清单缺少必需列：{', '.join(missing)}")

        lines = 0
        for record in reader:
            lines += 1
            if record is None or not any(str(v).strip() for v in record.values()):
                continue  # 跳过空行

            image_path = _resolve("image_path", record.get("image_path"), root)
            scope_label = _resolve("scope_label", record.get("scope_label"), root)
            split = _resolve("split", record.get("split"), root)
            group_id = _resolve("group_id", record.get("group_id"), root)
            source = _resolve("source", record.get("source"), root)
            consent = _parse_consent(record.get("consent"))

            if scope_label not in SCOPE_LABELS:
                raise ManifestError(
                    f"scope_label 只允许 {sorted(SCOPE_LABELS)}，得到 {scope_label!r}"
                )
            if split not in SPLITS:
                raise ManifestError(f"split 只允许 {sorted(SPLITS)}，得到 {split!r}")

            absolute = (root / image_path).resolve()
            try:
                absolute.relative_to(root)
            except ValueError as exc:
                raise ManifestError(f"{image_path!r} 超出图片根目录：{root}") from exc
            if not absolute.is_file():
                raise ManifestError(f"图片不存在：{absolute}")

            rows.append(
                ScopeManifestRow(
                    image_path=image_path,
                    scope_label=scope_label,
                    split=split,
                    group_id=group_id,
                    source=source,
                    consent=consent,
                    absolute_path=absolute,
                )
            )
        if not rows:
            raise ManifestError("清单不包含任何可用的样本行")
    return rows


def detect_duplicate_content(
    rows: list[ScopeManifestRow], *, image_root: str | Path | None = None
) -> dict[str, list[tuple[str, str]]]:
    """检测重复内容（按图片字节 SHA-256）。

    返回 ``{sha256: [(split, image_path), ...]}``。``detect_duplicate_content`` 本身不抛错；
    由 :func:`validate_manifest` 决定是否因重复而失败。同图出现在不同样本即视为重复，
    因为同一张图不应用两个（可能冲突的）范围标签。
    """
    by_hash: dict[str, list[tuple[str, str]]] = {}
    for row in rows:
        digest = _sha256_file(row.absolute_path)
        row = _replace_content_sha(row, digest)
        by_hash.setdefault(digest, []).append((row.split, row.image_path))
    return by_hash


def validate_group_separation(rows: list[ScopeManifestRow]) -> dict[str, set[str]]:
    """校验 train / calibration / test 的 group 互不重叠；否则抛 ``ManifestError``。"""
    groups: dict[str, set[str]] = {split: set() for split in SPLITS}
    for row in rows:
        groups[row.split].add(row.group_id)
    for left in SPLITS:
        for right in SPLITS:
            if left < right:
                overlap = groups[left] & groups[right]
                if overlap:
                    raise ManifestError(
                        f"split {left!r} 与 {right!r} 存在重叠 group：{sorted(overlap)}"
                    )
    return groups


def validate_manifest(
    csv_path: str | Path,
    image_root: str | Path,
    *,
    require_split_samples: bool = True,
) -> dict[str, Any]:
    """完整校验清单：结构、授权、分组隔离、重复内容。

    返回一份校验报告（按 split 的样本数、group 集合、去重结果）；任何不满足前置条件的
    情况都抛出 :class:`ManifestError`，绝不放行到训练。
    """
    rows = load_manifest(csv_path, image_root)
    if not all(row.consent for row in rows):
        unconsented = [row.image_path for row in rows if not row.consent]
        raise ManifestError(f"存在未获授权的样本，拒绝参与训练：{unconsented[:5]}")

    groups = validate_group_separation(rows)

    duplicates = detect_duplicate_content(rows)
    duplicate_entries = {sha: items for sha, items in duplicates.items() if len(items) > 1}
    if duplicate_entries:
        detail = "; ".join(
            f"{sha[:8]}: {items}" for sha, items in sorted(duplicate_entries.items())[:5]
        )
        raise ManifestError(f"存在重复内容样本：{detail}")

    counts: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        split_rows = [row for row in rows if row.split == split]
        counts[split] = {
            "total": len(split_rows),
            "in_scope": sum(1 for r in split_rows if r.scope_label == SCOPE_IN_SCOPE),
            "out_of_scope": sum(1 for r in split_rows if r.scope_label == SCOPE_OUT_OF_SCOPE),
        }
        if require_split_samples and not split_rows:
            raise ManifestError(f"split {split!r} 没有样本，无法训练/校准/评估")

    return {
        "rows": len(rows),
        "counts": counts,
        "groups": {split: sorted(gs) for split, gs in groups.items()},
        "unique_content": len(duplicates),
    }


def _replace_content_sha(row: ScopeManifestRow, digest: str) -> ScopeManifestRow:
    # 重新构造以填充 content_sha256（dataclass 为 frozen，只能重建）
    return ScopeManifestRow(
        image_path=row.image_path,
        scope_label=row.scope_label,
        split=row.split,
        group_id=row.group_id,
        source=row.source,
        consent=row.consent,
        absolute_path=row.absolute_path,
        content_sha256=digest,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
