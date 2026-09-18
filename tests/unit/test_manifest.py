from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import pytest
from herb_authentication.data.manifest import (
    KnownManifestRow,
    UnknownManifestRow,
    ValidatedKnownRecord,
    ValidatedUnknownRecord,
    load_known_manifest,
    validate_global_isolation,
    validate_known_manifest,
    validate_manifest_pair,
)
from herb_authentication.errors import ManifestValidationError
from PIL import Image


def make_known_row(**overrides: object) -> KnownManifestRow:
    values: dict[str, object] = {
        "image_path": Path("image.png"),
        "group_id": "g1",
        "sample_id": "s1",
        "claimed_herb_name": "当归",
        "authenticity": "authentic",
        "actual_species": ("当归",),
        "species_annotation_status": "verified",
        "batch_id": None,
        "origin": None,
        "capture_device": None,
        "collected_at": None,
        "annotator": None,
        "notes": None,
        "source_line": 2,
    }
    values.update(overrides)
    return KnownManifestRow(**values)


def test_manifest_requires_group_consistency() -> None:
    rows = [
        make_known_row(sample_id="s1", group_id="g1", claimed_herb_name="当归"),
        make_known_row(sample_id="s2", group_id="g1", claimed_herb_name="黄芪"),
    ]

    with pytest.raises(ManifestValidationError, match="group_id=g1"):
        validate_known_manifest(rows)


def test_verified_species_requires_nonempty_label() -> None:
    row = make_known_row(actual_species=(), species_annotation_status="verified")

    with pytest.raises(ManifestValidationError, match="verified"):
        validate_known_manifest([row])


def test_batch_id_cannot_span_multiple_groups() -> None:
    rows = [
        make_known_row(sample_id="s1", group_id="g1", batch_id="batch-a"),
        make_known_row(
            image_path=Path("two.png"), sample_id="s2", group_id="g2", batch_id="batch-a"
        ),
    ]

    with pytest.raises(ManifestValidationError, match="batch_id=batch-a"):
        validate_known_manifest(rows)


def test_sample_species_and_annotation_status_must_be_consistent() -> None:
    rows = [
        make_known_row(image_path=Path("one.png")),
        make_known_row(
            image_path=Path("two.png"),
            actual_species=("欧当归",),
            species_annotation_status="verified",
            source_line=3,
        ),
    ]

    with pytest.raises(ManifestValidationError, match="sample_id=s1.*实际物种"):
        validate_known_manifest(rows)


def test_unknown_and_known_content_hashes_must_not_overlap() -> None:
    known = ValidatedKnownRecord(
        row=make_known_row(),
        sha256="same",
        perceptual_hash="00",
        image_format="PNG",
        width=16,
        height=16,
        file_size=100,
    )
    unknown = ValidatedUnknownRecord(
        row=UnknownManifestRow(
            image_path=Path("unknown.png"),
            group_id="ug1",
            sample_id="us1",
            unknown_type="unknown_herb",
            dataset_role="calibration",
            source_line=2,
        ),
        sha256="same",
        perceptual_hash="ff",
        image_format="PNG",
        width=16,
        height=16,
        file_size=100,
    )

    with pytest.raises(ManifestValidationError, match="内容哈希"):
        validate_global_isolation((known,), (unknown,))


def test_manifest_path_cannot_escape_data_root(tmp_path) -> None:
    data_root = tmp_path / "images"
    data_root.mkdir()
    manifest = tmp_path / "metadata.csv"
    _write_csv(
        manifest,
        ("image_path", "group_id", "sample_id", "claimed_herb_name", "authenticity"),
        (
            {
                "image_path": "../outside.png",
                "group_id": "g1",
                "sample_id": "s1",
                "claimed_herb_name": "当归",
                "authenticity": "authentic",
            },
        ),
    )

    with pytest.raises(ManifestValidationError, match="越过 data_root"):
        load_known_manifest(manifest, data_root=data_root)


def test_invalid_image_writes_utf8_validation_report(tmp_path) -> None:
    data_root = tmp_path / "images"
    data_root.mkdir()
    (data_root / "broken.jpg").write_bytes(b"not-an-image")
    known_path = tmp_path / "metadata.csv"
    report_path = tmp_path / "validation_report.json"
    _write_known_csv(
        known_path,
        (
            {
                "image_path": "broken.jpg",
                "group_id": "g1",
                "sample_id": "s1",
                "claimed_herb_name": "当归",
                "authenticity": "authentic",
                "actual_species": "当归",
                "species_annotation_status": "verified",
            },
        ),
    )

    with pytest.raises(ManifestValidationError, match="无法解码"):
        validate_manifest_pair(
            known_path,
            unknown_manifest_path=None,
            data_root=data_root,
            report_path=report_path,
        )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "invalid"
    assert any("broken.jpg" in error for error in report["errors"])


def test_decompression_bomb_error_is_reported_as_manifest_error(tmp_path, monkeypatch) -> None:
    data_root = tmp_path / "images"
    data_root.mkdir()
    image_path = data_root / "bomb.png"
    _write_pattern_image(image_path, seed=3)
    known_path = tmp_path / "metadata.csv"
    report_path = tmp_path / "validation_report.json"
    _write_known_csv(
        known_path,
        (
            {
                "image_path": "bomb.png",
                "group_id": "g1",
                "sample_id": "s1",
                "claimed_herb_name": "当归",
                "authenticity": "authentic",
                "actual_species": "当归",
                "species_annotation_status": "verified",
            },
        ),
    )
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 10)

    with pytest.raises(ManifestValidationError, match="解压炸弹"):
        validate_manifest_pair(
            known_path,
            unknown_manifest_path=None,
            data_root=data_root,
            report_path=report_path,
            max_image_pixels=1000,
        )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "invalid"
    assert any("解压炸弹" in error for error in report["errors"])


def test_duplicate_csv_header_is_rejected(tmp_path) -> None:
    path = tmp_path / "metadata.csv"
    path.write_text(
        "image_path,group_id,sample_id,claimed_herb_name,authenticity,authenticity\n"
        "a.png,g1,s1,当归,authentic,counterfeit\n",
        encoding="utf-8",
    )

    with pytest.raises(ManifestValidationError, match="重复表头"):
        load_known_manifest(path)


def test_extra_csv_cell_is_rejected(tmp_path) -> None:
    path = tmp_path / "metadata.csv"
    path.write_text(
        "image_path,group_id,sample_id,claimed_herb_name,authenticity\n"
        "a.png,g1,s1,当归,authentic,unexpected\n",
        encoding="utf-8",
    )

    with pytest.raises(ManifestValidationError, match="多余字段"):
        load_known_manifest(path)


def test_empty_known_manifest_is_rejected_and_reported(tmp_path) -> None:
    known_path = tmp_path / "metadata.csv"
    report_path = tmp_path / "validation_report.json"
    _write_known_csv(known_path, ())

    with pytest.raises(ManifestValidationError, match="没有数据行"):
        validate_manifest_pair(
            known_path,
            unknown_manifest_path=None,
            data_root=tmp_path,
            report_path=report_path,
        )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "invalid"


def test_valid_known_and_unknown_manifests_return_auditable_bundle(tmp_path) -> None:
    data_root = tmp_path / "images"
    data_root.mkdir()
    known_rows: list[dict[str, str]] = []
    for index in range(6):
        name = f"known_{index}.png"
        _write_pattern_image(data_root / name, seed=index)
        authentic = index < 3
        known_rows.append(
            {
                "image_path": name,
                "group_id": f"g{index}",
                "sample_id": f"s{index}",
                "claimed_herb_name": "当归",
                "authenticity": "authentic" if authentic else "counterfeit",
                "actual_species": "当归" if authentic else "欧当归",
                "species_annotation_status": "verified",
            }
        )

    unknown_rows: list[dict[str, str]] = []
    for index, role in enumerate(("calibration", "test"), start=10):
        name = f"unknown_{index}.png"
        _write_pattern_image(data_root / name, seed=index)
        unknown_rows.append(
            {
                "image_path": name,
                "group_id": f"ug{index}",
                "sample_id": f"us{index}",
                "unknown_type": "unknown_herb",
                "dataset_role": role,
            }
        )

    known_path = tmp_path / "metadata.csv"
    unknown_path = tmp_path / "unknown_metadata.csv"
    report_path = tmp_path / "validation_report.json"
    _write_known_csv(known_path, known_rows)
    _write_csv(
        unknown_path,
        ("image_path", "group_id", "sample_id", "unknown_type", "dataset_role"),
        unknown_rows,
    )

    bundle = validate_manifest_pair(
        known_path,
        unknown_manifest_path=unknown_path,
        data_root=data_root,
        report_path=report_path,
        perceptual_hash_distance=None,
    )

    assert len(bundle.known) == 6
    assert len(bundle.unknown) == 2
    assert all(record.manifest_encoding == "utf-8-sig" for record in bundle.known)
    assert all(record.manifest_encoding == "utf-8-sig" for record in bundle.unknown)
    assert bundle.known_encoding == "utf-8-sig"
    assert bundle.unknown_encoding == "utf-8-sig"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "valid"
    assert report["counts"] == {"known": 6, "unknown": 2}


def test_gbk_fallback_is_recorded_in_records_and_report(tmp_path) -> None:
    data_root = tmp_path / "images"
    data_root.mkdir()
    rows: list[dict[str, str]] = []
    for index in range(3):
        name = f"known_{index}.png"
        _write_pattern_image(data_root / name, seed=index)
        rows.append(
            {
                "image_path": name,
                "group_id": f"g{index}",
                "sample_id": f"s{index}",
                "claimed_herb_name": "当归",
                "authenticity": "authentic",
                "actual_species": "当归",
                "species_annotation_status": "verified",
            }
        )

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=(
            "image_path",
            "group_id",
            "sample_id",
            "claimed_herb_name",
            "authenticity",
            "actual_species",
            "species_annotation_status",
        ),
    )
    writer.writeheader()
    writer.writerows(rows)
    known_path = tmp_path / "metadata.csv"
    known_path.write_bytes(buffer.getvalue().encode("gbk"))
    report_path = tmp_path / "validation_report.json"

    bundle = validate_manifest_pair(
        known_path,
        unknown_manifest_path=None,
        data_root=data_root,
        report_path=report_path,
        perceptual_hash_distance=None,
    )

    assert bundle.known_encoding == "gbk"
    assert all(record.manifest_encoding == "gbk" for record in bundle.known)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["encodings"]["known"] == "gbk"
    assert any("GBK" in warning for warning in report["warnings"])


def test_quoted_multiline_note_is_preserved(tmp_path) -> None:
    path = tmp_path / "metadata.csv"
    _write_csv(
        path,
        (
            "image_path",
            "group_id",
            "sample_id",
            "claimed_herb_name",
            "authenticity",
            "notes",
        ),
        (
            {
                "image_path": "a.png",
                "group_id": "g1",
                "sample_id": "s1",
                "claimed_herb_name": "当归",
                "authenticity": "authentic",
                "notes": "第一行\n第二行",
            },
        ),
    )

    loaded = load_known_manifest(path)

    assert loaded.rows[0].notes == "第一行\n第二行"


def test_duplicate_audit_continues_when_another_image_is_invalid(tmp_path) -> None:
    data_root = tmp_path / "images"
    data_root.mkdir()
    first = data_root / "first.png"
    duplicate = data_root / "duplicate.png"
    broken = data_root / "broken.png"
    _write_pattern_image(first, seed=1)
    duplicate.write_bytes(first.read_bytes())
    broken.write_bytes(b"broken")
    known_path = tmp_path / "metadata.csv"
    report_path = tmp_path / "validation_report.json"
    _write_known_csv(
        known_path,
        tuple(
            {
                "image_path": name,
                "group_id": f"g{index}",
                "sample_id": f"s{index}",
                "claimed_herb_name": "当归",
                "authenticity": "authentic",
                "actual_species": "当归",
                "species_annotation_status": "verified",
            }
            for index, name in enumerate(("first.png", "duplicate.png", "broken.png"))
        ),
    )

    with pytest.raises(ManifestValidationError):
        validate_manifest_pair(
            known_path,
            unknown_manifest_path=None,
            data_root=data_root,
            report_path=report_path,
            perceptual_hash_distance=None,
        )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert any("无法解码" in error for error in report["errors"])
    assert any("内容哈希重复" in error for error in report["errors"])


def test_perceptual_hash_distance_must_fit_64_bit_hash() -> None:
    known = ValidatedKnownRecord(
        row=make_known_row(),
        sha256="known",
        perceptual_hash="0000000000000000",
        image_format="PNG",
        width=16,
        height=16,
        file_size=100,
    )
    unknown = ValidatedUnknownRecord(
        row=UnknownManifestRow(
            image_path=Path("unknown.png"),
            group_id="ug1",
            sample_id="us1",
            unknown_type="unknown_herb",
            dataset_role="calibration",
            source_line=2,
        ),
        sha256="unknown",
        perceptual_hash="0000000000000001",
        image_format="PNG",
        width=16,
        height=16,
        file_size=100,
    )

    with pytest.raises(ManifestValidationError, match="0 到 64"):
        validate_global_isolation((known,), (unknown,), perceptual_hash_distance=65)


def _write_known_csv(path: Path, rows) -> None:
    _write_csv(
        path,
        (
            "image_path",
            "group_id",
            "sample_id",
            "claimed_herb_name",
            "authenticity",
            "actual_species",
            "species_annotation_status",
        ),
        rows,
    )


def _write_csv(path: Path, fieldnames, rows) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_pattern_image(path: Path, seed: int) -> None:
    payload = bytes((seed * 41 + index * 17 + index // 7) % 256 for index in range(16 * 16 * 3))
    Image.frombytes("RGB", (16, 16), payload).save(path)
