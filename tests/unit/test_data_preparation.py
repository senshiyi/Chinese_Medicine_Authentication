from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest
from herb_authentication.data.preparation import (
    ManifestSource,
    build_closed_set_manifest,
)
from herb_authentication.encoding import read_csv_dicts
from herb_authentication.errors import ManifestValidationError
from PIL import Image


def write_jpeg(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=(127, 64, 32)).save(path, format="JPEG")


def fixture_sources(authentic: Path, fake: Path) -> tuple[ManifestSource, ManifestSource]:
    return (
        ManifestSource(
            authentic,
            re.compile(r"Drynaria-fortunei_AUTHENTIC_(\d{4})\.jpg"),
            "df-auth",
            "authentic",
            "Drynaria_fortunei",
            "verified",
        ),
        ManifestSource(
            fake,
            re.compile(r"Drynaria-fortunei_FAKE_(\d{4})\.jpg"),
            "df-fake",
            "counterfeit",
            None,
            "unverified",
        ),
    )


def mark_path_as_symlink(
    monkeypatch: pytest.MonkeyPatch,
    symlink_path: Path,
) -> None:
    original_is_symlink = Path.is_symlink

    def is_symlink(path: Path) -> bool:
        return path == symlink_path or original_is_symlink(path)

    monkeypatch.setattr(Path, "is_symlink", is_symlink)


def test_build_closed_set_manifest_assigns_one_sample_and_group_per_file(
    tmp_path: Path,
) -> None:
    authentic = tmp_path / "raw" / "Drynaria_fortunei_authentic"
    fake = tmp_path / "raw" / "Drynaria_fortunei_fake"
    write_jpeg(authentic / "Drynaria-fortunei_AUTHENTIC_0001.jpg")
    write_jpeg(authentic / "Drynaria-fortunei_AUTHENTIC_0002.jpg")
    write_jpeg(fake / "Drynaria-fortunei_FAKE_0001.jpg")

    result = build_closed_set_manifest(
        sources=tuple(reversed(fixture_sources(authentic, fake))),
        data_root=tmp_path,
        output_path=tmp_path / "metadata.csv",
        claimed_herb_name="Drynaria_fortunei",
    )

    rows = read_csv_dicts(result.path, encoding="utf-8-sig").rows
    assert [row["sample_id"] for row in rows] == [
        "df-auth-0001",
        "df-auth-0002",
        "df-fake-0001",
    ]
    assert [row["group_id"] for row in rows] == [
        "df-auth-0001",
        "df-auth-0002",
        "df-fake-0001",
    ]
    assert rows[-1]["actual_species"] == ""
    assert rows[-1]["species_annotation_status"] == "unverified"
    assert result.rows == 3
    assert dict(result.counts) == {"authentic": 2, "counterfeit": 1}
    assert result.sha256 == hashlib.sha256(result.path.read_bytes()).hexdigest()
    assert result.path.read_bytes().startswith(b"\xef\xbb\xbf")


def test_build_closed_set_manifest_rejects_unexpected_filename(tmp_path: Path) -> None:
    authentic = tmp_path / "raw" / "Drynaria_fortunei_authentic"
    fake = tmp_path / "raw" / "Drynaria_fortunei_fake"
    write_jpeg(authentic / "Drynaria-fortunei_AUTHENTIC_0001.jpg")
    write_jpeg(authentic / "unrelated.jpg")
    write_jpeg(fake / "Drynaria-fortunei_FAKE_0001.jpg")

    with pytest.raises(ManifestValidationError, match="文件名"):
        build_closed_set_manifest(
            fixture_sources(authentic, fake), tmp_path, tmp_path / "metadata.csv", "Drynaria"
        )


def test_build_closed_set_manifest_rejects_missing_numeric_index(tmp_path: Path) -> None:
    authentic = tmp_path / "raw" / "Drynaria_fortunei_authentic"
    fake = tmp_path / "raw" / "Drynaria_fortunei_fake"
    write_jpeg(authentic / "Drynaria-fortunei_AUTHENTIC_.jpg")
    write_jpeg(fake / "Drynaria-fortunei_FAKE_0001.jpg")

    with pytest.raises(ManifestValidationError, match="数字索引"):
        build_closed_set_manifest(
            fixture_sources(authentic, fake), tmp_path, tmp_path / "metadata.csv", "Drynaria"
        )


def test_build_closed_set_manifest_rejects_duplicate_index(tmp_path: Path) -> None:
    source_dir = tmp_path / "raw" / "authentic"
    write_jpeg(source_dir / "AUTH_0001_a.jpg")
    write_jpeg(source_dir / "AUTH_0001_b.jpg")
    source = ManifestSource(
        source_dir,
        re.compile(r"AUTH_(\d{4})_[ab]\.jpg"),
        "auth",
        "authentic",
        "Drynaria_fortunei",
        "verified",
    )

    with pytest.raises(ManifestValidationError, match="重复.*0001"):
        build_closed_set_manifest((source,), tmp_path, tmp_path / "metadata.csv", "Drynaria")


def test_build_closed_set_manifest_rejects_path_escape(tmp_path: Path) -> None:
    data_root = tmp_path / "data_root"
    source_dir = tmp_path / "outside" / "authentic"
    write_jpeg(source_dir / "AUTH_0001.jpg")
    source = ManifestSource(
        source_dir,
        re.compile(r"AUTH_(\d{4})\.jpg"),
        "auth",
        "authentic",
        "Drynaria_fortunei",
        "verified",
    )

    with pytest.raises(ManifestValidationError, match="data_root"):
        build_closed_set_manifest((source,), data_root, tmp_path / "metadata.csv", "Drynaria")


def test_build_closed_set_manifest_rejects_symlinked_source_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_dir = tmp_path / "raw" / "authentic"
    source_dir.mkdir(parents=True)
    mark_path_as_symlink(monkeypatch, source_dir)
    source = ManifestSource(
        source_dir,
        re.compile(r"AUTH_(\d{4})\.jpg"),
        "auth",
        "authentic",
        "Drynaria_fortunei",
        "verified",
    )

    with pytest.raises(ManifestValidationError, match="\u7b26\u53f7\u94fe\u63a5"):
        build_closed_set_manifest(
            (source,), tmp_path, tmp_path / "metadata.csv", "Drynaria_fortunei"
        )


def test_build_closed_set_manifest_rejects_symlinked_image(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_dir = tmp_path / "raw" / "authentic"
    image = source_dir / "AUTH_0001.jpg"
    write_jpeg(image)
    mark_path_as_symlink(monkeypatch, image)
    source = ManifestSource(
        source_dir,
        re.compile(r"AUTH_(\d{4})\.jpg"),
        "auth",
        "authentic",
        "Drynaria_fortunei",
        "verified",
    )

    with pytest.raises(ManifestValidationError, match="\u7b26\u53f7\u94fe\u63a5"):
        build_closed_set_manifest(
            (source,), tmp_path, tmp_path / "metadata.csv", "Drynaria_fortunei"
        )


@pytest.mark.parametrize(
    ("authenticity", "actual_species", "annotation_status"),
    (
        ("authentic", None, "verified"),
        ("authentic", "Other_species", "verified"),
        ("authentic", "Drynaria_fortunei", "unverified"),
        ("counterfeit", "Drynaria_fortunei", "unverified"),
        ("counterfeit", None, "verified"),
    ),
)
def test_build_closed_set_manifest_rejects_invalid_closed_set_source_semantics(
    tmp_path: Path,
    authenticity: str,
    actual_species: str | None,
    annotation_status: str,
) -> None:
    source_dir = tmp_path / "raw" / "source"
    write_jpeg(source_dir / "SOURCE_0001.jpg")
    source = ManifestSource(
        source_dir,
        re.compile(r"SOURCE_(\d{4})\.jpg"),
        "source",
        authenticity,
        actual_species,
        annotation_status,
    )

    with pytest.raises(
        ManifestValidationError,
        match="\u95ed\u96c6\u6765\u6e90\u6807\u7b7e",
    ):
        build_closed_set_manifest(
            (source,),
            tmp_path,
            tmp_path / "metadata.csv",
            "Drynaria_fortunei",
        )


def test_build_closed_set_manifest_rejects_non_jpeg_payload(tmp_path: Path) -> None:
    source_dir = tmp_path / "raw" / "authentic"
    disguised_png = source_dir / "AUTH_0001.jpg"
    disguised_png.parent.mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(disguised_png, format="PNG")
    source = ManifestSource(
        source_dir,
        re.compile(r"AUTH_(\d{4})\.jpg"),
        "auth",
        "authentic",
        "Drynaria_fortunei",
        "verified",
    )

    with pytest.raises(ManifestValidationError, match="JPEG"):
        build_closed_set_manifest((source,), tmp_path, tmp_path / "metadata.csv", "Drynaria")


def test_build_closed_set_manifest_requires_contiguous_one_based_indexes(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "raw" / "authentic"
    write_jpeg(source_dir / "AUTH_0001.jpg")
    write_jpeg(source_dir / "AUTH_0003.jpg")
    source = ManifestSource(
        source_dir,
        re.compile(r"AUTH_(\d{4})\.jpg"),
        "auth",
        "authentic",
        "Drynaria_fortunei",
        "verified",
    )

    with pytest.raises(ManifestValidationError, match="连续"):
        build_closed_set_manifest((source,), tmp_path, tmp_path / "metadata.csv", "Drynaria")


def test_failed_atomic_replace_preserves_existing_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from herb_authentication.data import preparation

    source_dir = tmp_path / "raw" / "authentic"
    write_jpeg(source_dir / "AUTH_0001.jpg")
    output = tmp_path / "metadata.csv"
    output.write_bytes(b"existing-manifest")
    source = ManifestSource(
        source_dir,
        re.compile(r"AUTH_(\d{4})\.jpg"),
        "auth",
        "authentic",
        "Drynaria_fortunei",
        "verified",
    )

    def fail_replace(source_path: Path, destination_path: Path) -> None:
        raise OSError(f"simulated replace failure: {source_path} -> {destination_path}")

    monkeypatch.setattr(preparation.os, "replace", fail_replace)

    with pytest.raises(OSError, match="simulated replace failure"):
        build_closed_set_manifest((source,), tmp_path, output, "Drynaria_fortunei")

    assert output.read_bytes() == b"existing-manifest"
    assert not tuple(tmp_path.glob(".metadata.csv.*.tmp"))


def test_data_package_exports_preparation_api() -> None:
    import herb_authentication.data as data

    assert data.ManifestSource is ManifestSource
    assert data.build_closed_set_manifest is build_closed_set_manifest
