from __future__ import annotations

import csv
import json
import logging

import pytest
from herb_authentication.encoding import (
    read_text,
    write_csv_utf8_sig,
    write_json_utf8,
)


def test_read_text_supports_explicit_gbk(tmp_path) -> None:
    path = tmp_path / "labels.txt"
    path.write_bytes("当归".encode("gbk"))

    decoded = read_text(path, encoding="gbk")

    assert decoded.text == "当归"
    assert decoded.actual_encoding == "gbk"
    assert decoded.used_fallback is False


def test_default_read_falls_back_to_gbk_and_records_warning(tmp_path, caplog) -> None:
    path = tmp_path / "labels.txt"
    path.write_bytes("当归".encode("gbk"))

    with caplog.at_level(logging.WARNING):
        decoded = read_text(path)

    assert decoded.text == "当归"
    assert decoded.actual_encoding == "gbk"
    assert decoded.used_fallback is True
    assert "GBK" in caplog.text


def test_explicit_encoding_never_silently_falls_back(tmp_path) -> None:
    path = tmp_path / "labels.txt"
    path.write_bytes(b"\xff")

    with pytest.raises(UnicodeDecodeError):
        read_text(path, encoding="utf-8")


def test_json_keeps_chinese_and_csv_has_utf8_bom(tmp_path) -> None:
    json_path = tmp_path / "result.json"
    csv_path = tmp_path / "result.csv"

    write_json_utf8(json_path, {"label": "当归"})
    write_csv_utf8_sig(csv_path, fieldnames=("label",), rows=({"label": "当归"},))

    assert "当归" in json_path.read_text(encoding="utf-8")
    assert json.loads(json_path.read_text(encoding="utf-8")) == {"label": "当归"}
    assert csv_path.read_bytes().startswith(b"\xef\xbb\xbf")
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        assert list(csv.DictReader(handle)) == [{"label": "当归"}]
