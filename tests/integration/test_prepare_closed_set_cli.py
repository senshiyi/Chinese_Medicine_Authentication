from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PREPARE_CLOSED_SET = PROJECT_ROOT / "项目" / "Files" / "prepare_closed_set.py"


def load_script():
    spec = importlib.util.spec_from_file_location("prepare_closed_set_script", PREPARE_CLOSED_SET)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_pattern_jpeg(path: Path, identity: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(identity.encode("ascii")).digest()
    pixels = [255 if digest[index // 8] & (1 << (index % 8)) else 0 for index in range(64)]
    image = Image.new("L", (8, 8))
    image.putdata(pixels)
    image.convert("RGB").save(path, format="JPEG", quality=100, subsampling=0)


def write_sources(root: Path, authentic_count: int, counterfeit_count: int) -> None:
    raw = root / "references" / "raw"
    for index in range(1, authentic_count + 1):
        write_pattern_jpeg(
            raw / "Drynaria_fortunei_authentic" / f"Drynaria-fortunei_AUTHENTIC_{index:04d}.jpg",
            f"authentic-{index}",
        )
    for index in range(1, counterfeit_count + 1):
        write_pattern_jpeg(
            raw / "Drynaria_fortunei_fake" / f"Drynaria-fortunei_FAKE_{index:04d}.jpg",
            f"counterfeit-{index}",
        )


def write_config(
    root: Path,
    *,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
    split_algorithm_version: str = "group-stratified-v1",
) -> Path:
    config = root / "config.yaml"
    config.write_text(
        "\n".join(
            [
                'project_root: "."',
                "data:",
                '  manifest_path: "data/metadata.csv"',
                "  unknown_manifest_path: null",
                '  data_root: "references"',
                '  text_encoding: "utf-8-sig"',
                f"  train_ratio: {ratios[0]}",
                f"  val_ratio: {ratios[1]}",
                f"  test_ratio: {ratios[2]}",
                f"  seed: {seed}",
                f'  split_algorithm_version: "{split_algorithm_version}"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return config


def small_expected_counts() -> dict[str, dict[str, int]]:
    return {
        "all": {"authentic": 6, "counterfeit": 6, "total": 12},
        "train": {"authentic": 4, "counterfeit": 4, "total": 8},
        "val": {"authentic": 1, "counterfeit": 1, "total": 2},
        "test": {"authentic": 1, "counterfeit": 1, "total": 2},
    }


def test_prepare_closed_set_cli_writes_locked_split_and_utf8_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    write_sources(tmp_path, authentic_count=6, counterfeit_count=6)
    config = write_config(tmp_path)
    module = load_script()
    expected = small_expected_counts()
    monkeypatch.setattr(module, "EXPECTED", expected)

    assert module.main(["--config", str(config)]) == 0

    output = capsys.readouterr()
    assert output.err == ""
    report = json.loads(output.out)
    assert report["counts"] == expected
    assert (
        report["manifest_sha256"]
        == hashlib.sha256(Path(report["output_paths"]["manifest"]).read_bytes()).hexdigest()
    )
    assert (
        report["split_file_sha256"]
        == hashlib.sha256(Path(report["output_paths"]["splits_json"]).read_bytes()).hexdigest()
    )
    assert len(report["split_input_fingerprint"]) == 64
    assert (
        json.loads(Path(report["output_paths"]["splits_json"]).read_text(encoding="utf-8"))["seed"]
        == 42
    )
    for name in ("manifest", "train_csv", "val_csv", "test_csv"):
        assert Path(report["output_paths"][name]).read_bytes().startswith(b"\xef\xbb\xbf")


def test_prepare_closed_set_v2_writes_complete_component_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    write_sources(tmp_path, authentic_count=6, counterfeit_count=6)
    config = write_config(
        tmp_path,
        split_algorithm_version="group-stratified-near-duplicate-components-v2",
    )
    module = load_script()
    expected = small_expected_counts()
    monkeypatch.setattr(module, "EXPECTED", expected)

    assert module.main(["--config", str(config)]) == 0

    report = json.loads(capsys.readouterr().out)
    split_audit = report["split_audit"]
    assert split_audit["algorithm_version"] == ("group-stratified-near-duplicate-components-v2")
    assert split_audit["target_counts"] == {
        partition: {label: expected[partition][label] for label in ("authentic", "counterfeit")}
        for partition in ("train", "val", "test")
    }
    assert split_audit["actual_counts"] == split_audit["target_counts"]
    assert split_audit["solver_status"] == "optimal"
    assert split_audit["final_cross_partition_near_duplicate_count"] == 0
    for field in (
        "perceptual_hash_algorithm",
        "threshold_comparison",
        "component_algorithm_version",
        "component_fingerprint",
        "solver_algorithm_version",
        "objective_name",
        "tie_break",
        "mapping_fingerprint",
        "oracle_algorithm_version",
        "oracle_pair_count",
        "component_by_group",
        "component_sizes",
    ):
        assert field in split_audit


def test_prepare_closed_set_check_only_runs_full_pipeline_without_touching_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    write_sources(tmp_path, authentic_count=6, counterfeit_count=6)
    config = write_config(tmp_path)
    module = load_script()
    expected = small_expected_counts()
    monkeypatch.setattr(module, "EXPECTED", expected)
    output_dir = tmp_path / "data"
    output_dir.mkdir()
    artifacts = {
        name: output_dir / filename
        for name, filename in {
            "manifest": "metadata.csv",
            "validation_report": "manifest_validation.json",
            "train_csv": "train.csv",
            "val_csv": "val.csv",
            "test_csv": "test.csv",
            "splits_json": "splits.json",
        }.items()
    }
    before = {}
    for name, path in artifacts.items():
        payload = f"existing-{name}".encode("ascii")
        path.write_bytes(payload)
        before[name] = payload

    assert module.main(["--config", str(config), "--check-only"]) == 0

    output = capsys.readouterr()
    assert output.err == ""
    report = json.loads(output.out)
    assert report["check_only"] is True
    assert report["counts"] == expected
    assert len(report["manifest_sha256"]) == 64
    assert len(report["split_input_fingerprint"]) == 64
    assert len(report["split_file_sha256"]) == 64
    assert {name: path.read_bytes() for name, path in artifacts.items()} == before


def test_prepare_closed_set_cli_rejects_nonlocked_seed_before_scanning_sources(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = write_config(tmp_path, seed=7)
    module = load_script()

    assert module.main(["--config", str(config)]) == 2

    assert "seed=42" in capsys.readouterr().err
    assert not (tmp_path / "data" / "metadata.csv").exists()


def test_prepare_closed_set_cli_rejects_nonlocked_ratios_before_scanning_sources(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = write_config(tmp_path, ratios=(0.60, 0.20, 0.20))
    module = load_script()

    assert module.main(["--config", str(config)]) == 2

    assert "0.70/0.15/0.15" in capsys.readouterr().err
    assert not (tmp_path / "data" / "metadata.csv").exists()


def test_prepare_closed_set_cli_rejects_count_mismatch_before_split_writes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_sources(tmp_path, authentic_count=3, counterfeit_count=3)
    config = write_config(tmp_path)
    module = load_script()
    output_dir = tmp_path / "data"
    output_dir.mkdir()
    artifacts = {
        name: output_dir / name
        for name in (
            "metadata.csv",
            "manifest_validation.json",
            "train.csv",
            "val.csv",
            "test.csv",
            "splits.json",
        )
    }
    before = {}
    for name, path in artifacts.items():
        payload = f"existing-{name}".encode("ascii")
        path.write_bytes(payload)
        before[name] = payload

    assert module.main(["--config", str(config)]) == 2

    output = capsys.readouterr()
    assert "计数" in output.err
    assert {name: path.read_bytes() for name, path in artifacts.items()} == before


def test_publish_failure_restores_complete_previous_artifact_suite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_script()
    staging = tmp_path / "staging"
    destination = tmp_path / "destination"
    staging.mkdir()
    destination.mkdir()
    names = (
        "metadata.csv",
        "manifest_validation.json",
        "train.csv",
        "val.csv",
        "test.csv",
        "splits.json",
    )
    for name in names:
        (staging / name).write_bytes(f"new-{name}".encode("ascii"))
        (destination / name).write_bytes(f"old-{name}".encode("ascii"))

    real_replace = module.os.replace
    fail_once = {"done": False}

    def replace(source, target):
        if Path(source).name == "train.csv" and not fail_once["done"]:
            fail_once["done"] = True
            raise OSError("injected publication failure")
        return real_replace(source, target)

    monkeypatch.setattr(module.os, "replace", replace)
    with pytest.raises(OSError, match="injected publication failure"):
        module._publish_artifacts(
            staging,
            destination,
            manifest_name="metadata.csv",
        )

    assert {name: (destination / name).read_bytes() for name in names} == {
        name: f"old-{name}".encode("ascii") for name in names
    }


def test_formal_v2_target_matrix_is_exact() -> None:
    module = load_script()

    assert module._v2_target_counts() == {
        "train": {"authentic": 235, "counterfeit": 64},
        "val": {"authentic": 50, "counterfeit": 14},
        "test": {"authentic": 50, "counterfeit": 14},
    }
