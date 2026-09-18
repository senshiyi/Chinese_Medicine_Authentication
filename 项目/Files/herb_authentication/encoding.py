"""显式且可审计的中文文本编码 I/O。"""

from __future__ import annotations

import csv
import io
import json
import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)
SUPPORTED_ENCODINGS = frozenset({"utf-8", "utf-8-sig", "gbk"})


@dataclass(frozen=True, slots=True)
class DecodedText:
    """文本内容及实际采用的解码方式。"""

    text: str
    actual_encoding: str
    used_fallback: bool


@dataclass(frozen=True, slots=True)
class DecodedCsv:
    """CSV 内容及可写入报告/checkpoint 的编码元数据。"""

    rows: tuple[dict[str, str], ...]
    fieldnames: tuple[str, ...]
    actual_encoding: str
    used_fallback: bool


def normalize_encoding(encoding: str) -> str:
    """规范化受支持编码名，不接受系统默认编码别名。"""

    canonical = encoding.strip().lower().replace("_", "-")
    aliases = {"utf8": "utf-8", "utf8-sig": "utf-8-sig", "cp936": "gbk"}
    canonical = aliases.get(canonical, canonical)
    if canonical not in SUPPORTED_ENCODINGS:
        supported = ", ".join(sorted(SUPPORTED_ENCODINGS))
        raise ValueError(f"不支持的文本编码 {encoding!r}；只允许：{supported}")
    return canonical


def read_text(path: str | Path, encoding: str | None = None) -> DecodedText:
    """读取文本；未指定时先 UTF-8-SIG，失败后才回退 GBK。"""

    source = Path(path)
    payload = source.read_bytes()
    if encoding is not None:
        actual_encoding = normalize_encoding(encoding)
        return DecodedText(
            text=payload.decode(actual_encoding),
            actual_encoding=actual_encoding,
            used_fallback=False,
        )

    try:
        return DecodedText(
            text=payload.decode("utf-8-sig"),
            actual_encoding="utf-8-sig",
            used_fallback=False,
        )
    except UnicodeDecodeError as utf8_error:
        try:
            text = payload.decode("gbk")
        except UnicodeDecodeError as gbk_error:
            raise utf8_error from gbk_error
        LOGGER.warning("%s 无法按 UTF-8-SIG 解码，已回退到 GBK；请尽快转换为 UTF-8。", source)
        return DecodedText(text=text, actual_encoding="gbk", used_fallback=True)


def read_csv_dicts(path: str | Path, encoding: str | None = None) -> DecodedCsv:
    """严格读取 DictReader CSV，并保留实际编码。"""

    decoded = read_text(path, encoding=encoding)
    reader = csv.DictReader(io.StringIO(decoded.text, newline=""))
    fieldnames = tuple(reader.fieldnames or ())
    duplicates = sorted({name for name in fieldnames if fieldnames.count(name) > 1})
    if duplicates:
        raise ValueError(f"CSV 存在重复表头：{duplicates}")

    rows: list[dict[str, str]] = []
    for row in reader:
        if None in row:
            raise ValueError(f"CSV 第 {reader.line_num} 行存在多余字段：{row[None]}")
        missing = [name for name, value in row.items() if value is None]
        if missing:
            raise ValueError(f"CSV 第 {reader.line_num} 行字段数量不足：{missing}")
        rows.append(
            {name: value for name, value in row.items() if name is not None and value is not None}
        )
    return DecodedCsv(
        rows=tuple(rows),
        fieldnames=fieldnames,
        actual_encoding=decoded.actual_encoding,
        used_fallback=decoded.used_fallback,
    )


def write_json_utf8(path: str | Path, value: Any) -> None:
    """使用 UTF-8 和原始中文写 JSON，并保留结尾换行。"""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    destination.write_text(payload, encoding="utf-8", newline="\n")


def write_csv_utf8_sig(
    path: str | Path,
    fieldnames: Sequence[str],
    rows: Iterable[Mapping[str, object]],
) -> None:
    """写出 Excel 可直接识别的 UTF-8-SIG CSV。"""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
