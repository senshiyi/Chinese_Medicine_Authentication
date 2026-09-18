from __future__ import annotations

import json
import os
import re
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import MappingProxyType

import pytest
from herb_authentication.config import load_config
from herb_authentication.errors import ConfigurationError
from herb_authentication.training.run_artifacts import (
    create_run_context,
    sha256_file,
    write_run_metadata,
)


def make_config(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "# 中文配置快照\n"
        "project_root: '.'\n"
        "run:\n"
        "  root_dir: 'runs'\n",
        encoding="utf-8",
    )
    return load_config(path)


def test_create_run_context_writes_utf8_config_and_json_environment(tmp_path: Path) -> None:
    config = make_config(tmp_path)

    context = create_run_context(config, "audit-run-1")

    assert context.run_id == "audit-run-1"
    assert context.root == (tmp_path / "runs" / "audit-run-1").resolve()
    assert context.config_snapshot.read_text(encoding="utf-8").startswith("# 中文配置快照")
    environment = json.loads((context.root / "environment.json").read_text(encoding="utf-8"))
    assert environment["config_sha256"] == sha256_file(context.config_snapshot)
    assert environment["seed"] == config.data.seed
    assert isinstance(environment["command_arguments"], list)
    assert all(isinstance(argument, str) for argument in environment["command_arguments"])
    json.dumps(environment, ensure_ascii=False)


def test_create_run_context_rejects_existing_nonempty_run_directory(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    existing = config.run.root_dir / "existing-run"
    existing.mkdir(parents=True)
    (existing / "sentinel.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="非空"):
        create_run_context(config, "existing-run")

    assert (existing / "sentinel.txt").read_text(encoding="utf-8") == "keep"


def test_create_run_context_rejects_existing_empty_run_directory(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    existing = config.run.root_dir / "empty-run"
    existing.mkdir(parents=True)

    with pytest.raises(ConfigurationError, match="已存在"):
        create_run_context(config, "empty-run")


def test_create_run_context_rejects_existing_file_target(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    config.run.root_dir.mkdir(parents=True)
    target = config.run.root_dir / "file-run"
    target.write_text("keep", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="已存在"):
        create_run_context(config, "file-run")

    assert target.read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize(
    "run_id",
    [
        "../escape",
        "nested/run",
        "nested\\run",
        "",
        "CON",
        "con.txt",
        "LPT1",
        "audit-run.",
    ],
)
def test_create_run_context_rejects_unsafe_requested_run_id(tmp_path: Path, run_id: str) -> None:
    config = make_config(tmp_path)

    with pytest.raises(ConfigurationError, match="run ID"):
        create_run_context(config, run_id)


def test_create_run_context_rejects_final_component_reparse_point(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    config.run.root_dir.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    link = config.run.root_dir / "linked-run"
    _create_directory_link(link, outside)

    try:
        with pytest.raises(ConfigurationError, match="run ID|运行目录"):
            create_run_context(config, "linked-run")
    finally:
        os.rmdir(link)

    assert not (outside / "config.yaml").exists()
    assert not (outside / "environment.json").exists()


def test_create_run_context_claims_directory_once_under_concurrency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = make_config(tmp_path)
    run_root = config.run.root_dir.resolve()
    target = run_root / "concurrent-run"
    synchronized_paths = {
        target: threading.Barrier(2),
        target / "config.yaml": threading.Barrier(2),
        target / "environment.json": threading.Barrier(2),
    }
    original_exists = Path.exists

    def synchronized_exists(path: Path) -> bool:
        exists = original_exists(path)
        barrier = synchronized_paths.get(path)
        if barrier is not None:
            barrier.wait(timeout=5)
        return exists

    monkeypatch.setattr(Path, "exists", synchronized_exists)

    def create() -> object:
        try:
            return create_run_context(config, "concurrent-run")
        except ConfigurationError as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: create(), range(2)))

    assert sum(not isinstance(result, ConfigurationError) for result in results) == 1
    assert sum(isinstance(result, ConfigurationError) for result in results) == 1


@pytest.mark.parametrize("failure_point", ["write", "replace"])
def test_create_run_context_cleans_up_after_atomic_publish_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
) -> None:
    config = make_config(tmp_path)
    target = config.run.root_dir / "failed-run"
    _inject_io_failure(monkeypatch, failure_point)

    with pytest.raises(ConfigurationError, match="写入|发布"):
        create_run_context(config, "failed-run")

    assert not target.exists()


def test_create_run_context_auto_id_contains_timestamp_and_config_hash(tmp_path: Path) -> None:
    config = make_config(tmp_path)

    context = create_run_context(config)

    assert re.fullmatch(r"\d{8}T\d{6}Z-[0-9a-f]{8}", context.run_id)
    assert context.run_id.endswith(sha256_file(context.config_snapshot)[:8])


def test_write_run_metadata_is_utf8_and_immutable(tmp_path: Path) -> None:
    context = create_run_context(make_config(tmp_path), "metadata-run")

    metadata_path = write_run_metadata(context, {"message": "闭集实验", "count": 427})

    assert json.loads(metadata_path.read_text(encoding="utf-8")) == {
        "message": "闭集实验",
        "count": 427,
    }
    with pytest.raises(ConfigurationError, match="已存在"):
        write_run_metadata(context, {"message": "不得覆盖"})


def test_write_run_metadata_converts_mapping_to_plain_dict(tmp_path: Path) -> None:
    context = create_run_context(make_config(tmp_path), "mapping-run")

    metadata_path = write_run_metadata(
        context,
        MappingProxyType({"message": "只序列化一次", "count": 427}),
    )

    assert json.loads(metadata_path.read_text(encoding="utf-8")) == {
        "message": "只序列化一次",
        "count": 427,
    }


@pytest.mark.parametrize("invalid_value", [float("nan"), float("inf"), object()])
def test_write_run_metadata_rejects_unsafe_json_values(
    tmp_path: Path,
    invalid_value: object,
) -> None:
    context = create_run_context(make_config(tmp_path), "unsafe-json-run")

    with pytest.raises(ConfigurationError, match="安全 JSON"):
        write_run_metadata(context, {"value": invalid_value})

    assert not (context.root / "run_metadata.json").exists()


@pytest.mark.parametrize("failure_point", ["write", "replace"])
def test_write_run_metadata_cleans_up_after_atomic_publish_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
) -> None:
    context = create_run_context(make_config(tmp_path), "metadata-failure-run")
    _inject_io_failure(monkeypatch, failure_point)

    with pytest.raises(ConfigurationError, match="写入|发布"):
        write_run_metadata(context, {"message": "不得截断"})

    assert not (context.root / "run_metadata.json").exists()
    assert sorted(path.name for path in context.root.iterdir()) == [
        "config.yaml",
        "environment.json",
    ]


def test_sha256_file_matches_known_digest(tmp_path: Path) -> None:
    path = tmp_path / "payload.bin"
    path.write_bytes(b"abc")

    assert sha256_file(path) == (
        "ba7816bf8f01cfea414140de5dae2223"
        "b00361a396177a9cb410ff61f20015ad"
    )


def _create_directory_link(link: Path, target: Path) -> None:
    if os.name == "nt":
        result = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            pytest.skip(f"无法创建 junction：{result.stderr or result.stdout}")
        return
    link.symlink_to(target, target_is_directory=True)


def _inject_io_failure(
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
) -> None:
    def fail(*_args: object, **_kwargs: object) -> None:
        raise OSError(f"simulated {failure_point} failure")

    monkeypatch.setattr(os, failure_point, fail)
