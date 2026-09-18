"""范围清单（manifest）校验测试：列、取值、授权、分组隔离、重复内容。"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import pytest

from drynaria_api.scope.manifest import (
    ManifestError,
    load_manifest,
    validate_manifest,
)


def _write_manifest(path: Path, rows: list[tuple[str, str, str, str, str, bool]]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_path", "scope_label", "split", "group_id", "source", "consent"])
        for row in rows:
            writer.writerow(row)
    return path


@pytest.fixture()
def manifest_dir(tmp_path):
    root = tmp_path / "images"
    root.mkdir()
    return root


def _valid_rows(root: Path) -> list[tuple[str, str, str, str, str, bool]]:
    names = []
    for split, group in (("train", "g_train"), ("calibration", "g_cal"), ("test", "g_test")):
        for i, label in enumerate(("in_scope", "out_of_scope")):
            name = f"{split}_{i}.img"
            (root / name).write_bytes(f"{split}-{label}-{group}-{i}".encode())
            names.append((name, label, split, group, "source-a", True))
    return names


def test_valid_manifest_passes_and_reports_counts(manifest_dir, tmp_path):
    rows = _valid_rows(manifest_dir)
    csv_path = _write_manifest(tmp_path / "m.csv", rows)
    report = validate_manifest(csv_path, manifest_dir)
    assert report["rows"] == 6
    assert report["counts"]["train"]["total"] == 2
    assert report["counts"]["train"]["in_scope"] == 1
    assert report["counts"]["calibration"]["out_of_scope"] == 1
    assert report["unique_content"] == 6


def test_missing_required_column_rejected(manifest_dir, tmp_path):
    rows = _valid_rows(manifest_dir)
    csv_path = tmp_path / "m.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_path", "scope_label", "split", "group_id", "consent"])  # 缺 source
        for r in rows:
            writer.writerow(list(r[:4]) + [r[4], r[5]])
    with pytest.raises(ManifestError, match="缺少必需列"):
        validate_manifest(csv_path, manifest_dir)


def test_invalid_scope_label_rejected(manifest_dir, tmp_path):
    rows = _valid_rows(manifest_dir)
    rows[0] = (rows[0][0], "unknown_label", rows[0][2], rows[0][3], rows[0][4], rows[0][5])
    csv_path = _write_manifest(tmp_path / "m.csv", rows)
    with pytest.raises(ManifestError, match="scope_label"):
        validate_manifest(csv_path, manifest_dir)


def test_invalid_split_rejected(manifest_dir, tmp_path):
    rows = _valid_rows(manifest_dir)
    rows[0] = (rows[0][0], rows[0][1], "val", rows[0][3], rows[0][4], rows[0][5])
    csv_path = _write_manifest(tmp_path / "m.csv", rows)
    with pytest.raises(ManifestError, match="split"):
        validate_manifest(csv_path, manifest_dir)


def test_unconsented_row_rejected(manifest_dir, tmp_path):
    rows = _valid_rows(manifest_dir)
    rows[0] = rows[0][:5] + (False,)
    csv_path = _write_manifest(tmp_path / "m.csv", rows)
    with pytest.raises(ManifestError, match="未获授权"):
        validate_manifest(csv_path, manifest_dir)


def test_group_separation_rejected_on_overlap(manifest_dir, tmp_path):
    rows = _valid_rows(manifest_dir)
    # 把 calibration 的 group 改成与 train 相同的 g_train
    for idx in range(2, 4):
        rows[idx] = (
            rows[idx][0], rows[idx][1], rows[idx][2], "g_train", rows[idx][4], rows[idx][5]
        )
    csv_path = _write_manifest(tmp_path / "m.csv", rows)
    with pytest.raises(ManifestError, match="重叠 group"):
        validate_manifest(csv_path, manifest_dir)


def test_duplicate_content_rejected(manifest_dir, tmp_path):
    rows = _valid_rows(manifest_dir)
    # 用同一个文件内容（相同字节）作为两个不同图像路径
    content = b"same-bytes"
    (manifest_dir / "a.img").write_bytes(content)
    (manifest_dir / "b.img").write_bytes(content)
    rows[0] = ("a.img", "in_scope", "train", "g_train", "source-a", True)
    rows[1] = ("b.img", "in_scope", "train", "g_train", "source-a", True)
    csv_path = _write_manifest(tmp_path / "m.csv", rows)
    with pytest.raises(ManifestError, match="重复内容"):
        validate_manifest(csv_path, manifest_dir)


def test_load_manifest_resolves_paths_and_hashes(manifest_dir, tmp_path):
    rows = _valid_rows(manifest_dir)
    csv_path = _write_manifest(tmp_path / "m.csv", rows)
    loaded = load_manifest(csv_path, manifest_dir)
    assert len(loaded) == 6
    assert all(row.absolute_path.is_file() for row in loaded)
    assert all(row.consent is True for row in loaded)
    # 相同文件内容的 SHA-256 对内容哈希决定；不同文件不同
    hashes = {hashlib.sha256(row.absolute_path.read_bytes()).hexdigest() for row in loaded}
    assert len(hashes) == 6


def test_missing_image_file_rejected(manifest_dir, tmp_path):
    rows = _valid_rows(manifest_dir)
    rows[0] = ("missing.img", rows[0][1], rows[0][2], rows[0][3], rows[0][4], rows[0][5])
    csv_path = _write_manifest(tmp_path / "m.csv", rows)
    with pytest.raises(ManifestError, match="不存在"):
        validate_manifest(csv_path, manifest_dir)
