from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_MANIFEST_FIELDS = (
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
_PARTITION_FIELDS = _MANIFEST_FIELDS + (
    "sha256",
    "perceptual_hash",
    "manifest_encoding",
    "partition",
)


def _manifest_row(*, authenticity: str = "authentic") -> dict[str, str]:
    return {
        "image_path": "a.jpg",
        "group_id": "g1",
        "sample_id": "s1",
        "claimed_herb_name": "Drynaria_fortunei",
        "authenticity": authenticity,
        "actual_species": "Drynaria_fortunei",
        "species_annotation_status": "verified",
        "batch_id": "",
        "origin": "",
        "capture_device": "",
        "collected_at": "",
        "annotator": "",
        "notes": "",
    }


def _partition_row(*, authenticity: str = "authentic") -> dict[str, str]:
    return {
        **_manifest_row(authenticity=authenticity),
        "sha256": "a" * 64,
        "perceptual_hash": "b" * 16,
        "manifest_encoding": "utf-8-sig",
        "partition": "test",
    }


def _write_csv(path: Path, fieldnames: tuple[str, ...], row: dict[str, str]) -> None:
    import csv

    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


def _load_evaluate_script():
    path = Path(__file__).resolve().parents[2] / "项目" / "Files" / "evaluate.py"
    spec = importlib.util.spec_from_file_location("evaluate_fingerprint_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_evaluate_split_loader_uses_public_payload_parser(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import herb_authentication.data as data_module

    expected = object()
    observed: dict[str, object] = {}

    def parse(payload):
        observed["payload"] = payload
        return expected

    monkeypatch.setattr(data_module, "split_assignment_from_payload", parse, raising=False)
    split = tmp_path / "splits.json"
    split.write_text('{"schema_version": 1, "marker": "v2"}', encoding="utf-8")

    assert _load_evaluate_script()._load_split_assignment(split) is expected
    assert observed == {"payload": {"schema_version": 1, "marker": "v2"}}


def test_locked_split_fingerprint_reads_recorded_input_fingerprint(tmp_path: Path) -> None:
    split = tmp_path / "splits.json"
    payload = json.dumps({"schema_version": 1, "input_fingerprint": "locked-input"})
    split.write_text(payload, encoding="utf-8")

    import hashlib

    assert (
        _load_evaluate_script()._locked_split_fingerprint(split)
        == hashlib.sha256(payload.encode()).hexdigest()
    )


def _closed_set_calibration(source_fingerprint: str, validation_fingerprint: str):
    return {
        "temperature": 1.5,
        "algorithm_version": "closed-set-temperature-v1",
        "validation_fingerprint": validation_fingerprint,
        "checkpoint_fingerprint": source_fingerprint,
        "nll_before": 0.5,
        "nll_after": 0.4,
        "brier_before": 0.3,
        "brier_after": 0.2,
        "ece_before": 0.1,
        "ece_after": 0.05,
    }


def test_locked_fingerprint_validation_does_not_read_validation_csv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hashlib

    module = _load_evaluate_script()
    monkeypatch.setattr(module, "_validate_current_closed_set_data", lambda **_kwargs: None)
    manifest = tmp_path / "metadata.csv"
    test = tmp_path / "test.csv"
    split = tmp_path / "splits.json"
    source = tmp_path / "candidate.pth"
    calibrated = tmp_path / "calibrated.pth"
    _write_csv(manifest, _MANIFEST_FIELDS, _manifest_row())
    _write_csv(test, _PARTITION_FIELDS, _partition_row())
    source.write_bytes(b"candidate")
    calibrated.write_bytes(b"calibrated")
    input_fingerprint = "1" * 64
    split.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "input_fingerprint": input_fingerprint,
                "by_group": {"g1": "test"},
            }
        ),
        encoding="utf-8",
    )
    manifest_fingerprint = hashlib.sha256(manifest.read_bytes()).hexdigest()
    split_fingerprint = hashlib.sha256(split.read_bytes()).hexdigest()
    source_fingerprint = hashlib.sha256(source.read_bytes()).hexdigest()
    validation_fingerprint = "2" * 64
    config_fingerprint = "3" * 64
    metadata = {
        "authenticity_classes": ("authentic", "counterfeit"),
        "data_fingerprint": input_fingerprint,
        "split_fingerprint": split_fingerprint,
        "training_config_fingerprint": config_fingerprint,
        "split_input_fingerprint": input_fingerprint,
        "split_artifact_fingerprint": split_fingerprint,
        "experimental_closed_set": True,
        "open_set_calibrated": False,
        "fingerprints": {
            "manifest": manifest_fingerprint,
            "known_validation": validation_fingerprint,
            "locked_split": split_fingerprint,
        },
        "closed_set_calibration": _closed_set_calibration(
            source_fingerprint, validation_fingerprint
        ),
    }

    bundle, calibration = module._validate_closed_set_fingerprints(
        metadata=metadata,
        manifest=manifest,
        partition_manifest=test,
        split=split,
        config_fingerprint=config_fingerprint,
        source_candidate=source,
        calibrated_checkpoint=calibrated,
        partition="locked-test",
        data_root=tmp_path,
        encoding="utf-8-sig",
        max_image_bytes=1_000_000,
        max_image_pixels=1_000_000,
    )

    assert bundle["manifest"] == manifest_fingerprint
    assert bundle["source_candidate"] == source_fingerprint
    assert calibration.temperature == 1.5
    assert not (tmp_path / "val.csv").exists()

    _write_csv(test, _PARTITION_FIELDS, _partition_row(authenticity="counterfeit"))
    with pytest.raises(ValueError, match="partition manifest"):
        module._validate_closed_set_fingerprints(
            metadata=metadata,
            manifest=manifest,
            partition_manifest=test,
            split=split,
            config_fingerprint=config_fingerprint,
            source_candidate=source,
            calibrated_checkpoint=calibrated,
            partition="locked-test",
            data_root=tmp_path,
            encoding="utf-8-sig",
            max_image_bytes=1_000_000,
            max_image_pixels=1_000_000,
        )


def test_closed_set_fingerprint_mismatch_fails_closed(tmp_path: Path) -> None:
    module = _load_evaluate_script()
    manifest = tmp_path / "metadata.csv"
    validation = tmp_path / "val.csv"
    split = tmp_path / "splits.json"
    source = tmp_path / "candidate.pth"
    calibrated = tmp_path / "calibrated.pth"
    for path in (manifest, validation, source, calibrated):
        path.write_bytes(path.name.encode())
    split.write_text(
        json.dumps({"schema_version": 1, "input_fingerprint": "1" * 64, "by_group": {}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="manifest 指纹"):
        module._validate_closed_set_fingerprints(
            metadata={"fingerprints": {"manifest": "0" * 64}},
            manifest=manifest,
            partition_manifest=validation,
            split=split,
            config_fingerprint="3" * 64,
            source_candidate=source,
            calibrated_checkpoint=calibrated,
            partition="validation",
            data_root=tmp_path,
            encoding="utf-8-sig",
            max_image_bytes=1_000_000,
            max_image_pixels=1_000_000,
        )


@pytest.mark.parametrize("field", ["actual_species", "species_annotation_status"])
def test_partition_manifest_rejects_species_field_tampering(tmp_path: Path, field: str) -> None:
    module = _load_evaluate_script()
    row = _manifest_row()
    selected_row = _partition_row()
    selected_row[field] = "tampered"
    manifest, selected = tmp_path / "metadata.csv", tmp_path / "test.csv"
    _write_csv(manifest, _MANIFEST_FIELDS, row)
    _write_csv(selected, _PARTITION_FIELDS, selected_row)
    with pytest.raises(ValueError, match="partition manifest"):
        module._validate_partition_manifest_rows(
            manifest, selected, {"by_group": {"g1": "test"}}, "test"
        )


@pytest.mark.parametrize("value", [0.79, 0.81])
def test_closed_set_evaluation_rejects_nonfixed_promotion_thresholds(value: float) -> None:
    module = _load_evaluate_script()
    with pytest.raises(ValueError, match="固定为 0.80"):
        module._require_fixed_promotion_thresholds(
            {"macro_f1": value, "balanced_accuracy": 0.8, "counterfeit_recall": 0.8}
        )


def test_partition_manifest_requires_complete_artifact_schema(tmp_path: Path) -> None:
    module = _load_evaluate_script()
    manifest = tmp_path / "metadata.csv"
    selected = tmp_path / "test.csv"
    manifest.write_text("image_path,group_id\na.jpg,g1\n", encoding="utf-8")
    selected.write_text("image_path,group_id,partition\na.jpg,g1,test\n", encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        module._validate_partition_manifest_rows(
            manifest, selected, {"by_group": {"g1": "test"}}, "test"
        )


def test_evaluate_v2_missing_split_fails_before_checkpoint_report_or_logits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import herb_authentication.config as config_module
    import herb_authentication.environment as environment_module

    module = _load_evaluate_script()
    config = type(
        "Config",
        (),
        {
            "calibration": type("Calibration", (), {"mode": "closed_set_temperature"})(),
            "data": type(
                "Data",
                (),
                {
                    "manifest_path": tmp_path / "data" / "metadata.csv",
                    "split_algorithm_version": ("group-stratified-near-duplicate-components-v2"),
                },
            )(),
            "promotion": type(
                "Promotion",
                (),
                {
                    "min_macro_f1": 0.8,
                    "min_balanced_accuracy": 0.8,
                    "min_counterfeit_recall": 0.8,
                },
            )(),
        },
    )()
    calls = {"checkpoint": 0, "logits": 0}

    def checkpoint_metadata(_path):
        calls["checkpoint"] += 1
        return {}

    def logits_collector(**_kwargs):
        calls["logits"] += 1
        raise AssertionError("logits must not run before split preflight")

    monkeypatch.setattr(config_module, "load_config", lambda _path: config)
    monkeypatch.setattr(
        environment_module,
        "run_bootstrap_preflight",
        lambda: type("Preflight", (), {"python_supported": True})(),
    )
    monkeypatch.setattr(module, "_checkpoint_metadata", checkpoint_metadata)

    result = module.main(
        [
            "--config",
            str(tmp_path / "config.yaml"),
            "--partition",
            "validation",
            "--run-dir",
            str(tmp_path / "run"),
            "--checkpoint",
            str(tmp_path / "candidate.pth"),
        ],
        closed_set_logits_collector=logits_collector,
    )

    assert result == 2
    assert "prepare_closed_set.py" in capsys.readouterr().err
    assert calls == {"checkpoint": 0, "logits": 0}


@pytest.mark.parametrize(
    ("configured", "recorded"),
    (
        ("group-stratified-v1", "group-stratified-near-duplicate-components-v2"),
        ("group-stratified-near-duplicate-components-v2", "group-stratified-v1"),
    ),
)
def test_evaluate_rejects_both_split_version_mismatch_directions(
    tmp_path: Path, configured: str, recorded: str
) -> None:
    module = _load_evaluate_script()
    config = type(
        "Config",
        (),
        {"data": type("Data", (), {"split_algorithm_version": configured})()},
    )()
    assignment = type("Assignment", (), {"algorithm_version": recorded})()

    with pytest.raises(ValueError, match="切分算法版本"):
        module._require_configured_split_version(config, assignment, tmp_path / "splits.json")


@pytest.mark.parametrize(
    ("configured", "recorded"),
    (
        ("group-stratified-v1", "group-stratified-near-duplicate-components-v2"),
        ("group-stratified-near-duplicate-components-v2", "group-stratified-v1"),
    ),
)
def test_evaluate_version_mismatch_fails_before_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    configured: str,
    recorded: str,
) -> None:
    import herb_authentication.config as config_module
    import herb_authentication.environment as environment_module

    split_path = tmp_path / "data" / "splits.json"
    split_path.parent.mkdir()
    split_path.write_text("{}", encoding="utf-8")
    config = type(
        "Config",
        (),
        {
            "calibration": type("Calibration", (), {"mode": "closed_set_temperature"})(),
            "data": type(
                "Data",
                (),
                {
                    "manifest_path": tmp_path / "data" / "metadata.csv",
                    "split_algorithm_version": configured,
                },
            )(),
        },
    )()
    module = _load_evaluate_script()
    monkeypatch.setattr(config_module, "load_config", lambda _path: config)
    monkeypatch.setattr(
        environment_module,
        "run_bootstrap_preflight",
        lambda: type("Preflight", (), {"python_supported": True})(),
    )
    monkeypatch.setattr(
        module,
        "_load_split_assignment",
        lambda _path: type("Assignment", (), {"algorithm_version": recorded})(),
    )
    monkeypatch.setattr(
        module,
        "_checkpoint_metadata",
        lambda _path: (_ for _ in ()).throw(
            AssertionError("checkpoint must not load before version preflight")
        ),
    )

    result = module.main(
        [
            "--config",
            str(tmp_path / "config.yaml"),
            "--partition",
            "validation",
            "--run-dir",
            str(tmp_path / "run"),
            "--checkpoint",
            str(tmp_path / "candidate.pth"),
        ]
    )

    assert result == 2


@pytest.mark.parametrize(
    "version",
    ("group-stratified-v1", "group-stratified-near-duplicate-components-v2"),
)
def test_evaluate_accepts_matching_split_versions(tmp_path: Path, version: str) -> None:
    module = _load_evaluate_script()
    config = type(
        "Config",
        (),
        {"data": type("Data", (), {"split_algorithm_version": version})()},
    )()
    assignment = type("Assignment", (), {"algorithm_version": version})()

    assert (
        module._require_configured_split_version(config, assignment, tmp_path / "splits.json")
        is assignment
    )
