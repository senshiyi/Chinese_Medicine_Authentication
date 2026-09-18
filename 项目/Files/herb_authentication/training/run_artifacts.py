"""不可变、可审计的训练运行目录与元数据。"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import secrets
import stat
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import torch

from ..config import AppConfig
from ..encoding import read_text
from ..errors import ConfigurationError

_SAFE_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_WINDOWS_RESERVED_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{number}" for number in range(1, 10)}
    | {f"LPT{number}" for number in range(1, 10)}
)


@dataclass(frozen=True, slots=True)
class RunContext:
    run_id: str
    root: Path
    config_snapshot: Path


def sha256_file(path: str | Path) -> str:
    """计算文件内容的 SHA-256。"""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_run_context(
    config: AppConfig,
    requested_run_id: str | None = None,
) -> RunContext:
    """独占创建运行目录，并原子保存 UTF-8 配置及环境快照。"""

    config_text = read_text(
        config.source_path,
        encoding=config.source_encoding,
    ).text
    config_bytes = config_text.encode("utf-8")
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    run_id = (
        f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{config_sha256[:8]}"
        if requested_run_id is None
        else requested_run_id
    )
    _validate_run_id(run_id, requested_run_id)
    environment_bytes = _serialize_json(
        {
            "python_version": platform.python_version(),
            "torch_version": str(torch.__version__),
            "torchvision_version": version("torchvision"),
            "operating_system": platform.platform(),
            "seed": config.data.seed,
            "deterministic_settings": {
                "torch_deterministic_algorithms": (
                    torch.are_deterministic_algorithms_enabled()
                ),
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
            },
            "config_sha256": config_sha256,
            "command_arguments": [str(argument) for argument in sys.argv],
        },
        "运行环境快照",
    )

    run_root = _prepare_run_root(config.run.root_dir)
    root = run_root / run_id
    _reject_existing_target(root)
    _require_contained(run_root, root)
    try:
        root.mkdir(exist_ok=False)
    except FileExistsError as exc:
        raise ConfigurationError(f"运行目录已存在：{root}") from exc
    except OSError as exc:
        raise ConfigurationError(f"无法独占创建运行目录 {root}：{exc}") from exc

    config_snapshot = root / "config.yaml"
    environment_path = root / "environment.json"
    try:
        _atomic_publish(config_snapshot, config_bytes)
        _atomic_publish(environment_path, environment_bytes)
    except ConfigurationError:
        _rollback_run_directory(root)
        raise

    return RunContext(
        run_id=run_id,
        root=root,
        config_snapshot=config_snapshot,
    )


def write_run_metadata(
    context: RunContext,
    payload: Mapping[str, Any],
) -> Path:
    """将 Mapping 一次性序列化并原子发布，不覆盖既有记录。"""

    try:
        plain_payload = dict(payload)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"运行元数据不是安全 JSON：{exc}") from exc
    payload_bytes = _serialize_json(plain_payload, "运行元数据")

    root = _require_context_root(context.root)
    destination = root / "run_metadata.json"
    claim = root / ".run_metadata.json.lock"
    claim_fd: int | None = None
    try:
        claim_fd = os.open(
            claim,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
            0o600,
        )
    except FileExistsError as exc:
        raise ConfigurationError(f"运行元数据写入正在进行或已存在：{destination}") from exc
    except OSError as exc:
        raise ConfigurationError(f"无法独占声明运行元数据 {destination}：{exc}") from exc
    finally:
        if claim_fd is not None:
            os.close(claim_fd)

    try:
        if destination.exists() or _is_reparse_point(destination):
            raise ConfigurationError(f"运行元数据已存在：{destination}")
        _atomic_publish(destination, payload_bytes)
    finally:
        try:
            claim.unlink(missing_ok=True)
        except OSError as exc:
            raise ConfigurationError(f"无法清理运行元数据声明文件 {claim}：{exc}") from exc
    return destination


def _prepare_run_root(configured_root: Path) -> Path:
    try:
        run_root = configured_root.resolve(strict=False)
        run_root.mkdir(parents=True, exist_ok=True)
        run_root = run_root.resolve(strict=True)
    except OSError as exc:
        raise ConfigurationError(f"无法准备运行根目录 {configured_root}：{exc}") from exc
    if not run_root.is_dir():
        raise ConfigurationError(f"运行根路径不是目录：{run_root}")
    return run_root


def _reject_existing_target(path: Path) -> None:
    try:
        status = path.lstat()
    except FileNotFoundError:
        return
    except OSError as exc:
        raise ConfigurationError(f"无法检查运行目录 {path}：{exc}") from exc

    if _status_is_reparse_point(status):
        raise ConfigurationError(
            f"运行目录 final component 不得是 symlink/junction/reparse point：{path}"
        )
    if stat.S_ISDIR(status.st_mode):
        try:
            nonempty = any(path.iterdir())
        except OSError as exc:
            raise ConfigurationError(f"无法检查运行目录 {path}：{exc}") from exc
        qualifier = "且非空" if nonempty else ""
        raise ConfigurationError(f"运行目录已存在{qualifier}：{path}")
    raise ConfigurationError(f"运行目标已存在且不是目录：{path}")


def _require_contained(run_root: Path, candidate: Path) -> None:
    try:
        canonical_candidate = candidate.resolve(strict=False)
        canonical_candidate.relative_to(run_root)
    except (OSError, ValueError) as exc:
        raise ConfigurationError(f"运行目录解析后逃出 canonical root：{candidate}") from exc
    if canonical_candidate.parent != run_root:
        raise ConfigurationError(f"运行目录解析后逃出 canonical root：{candidate}")


def _require_context_root(root: Path) -> Path:
    if _is_reparse_point(root):
        raise ConfigurationError(f"运行目录不得是 symlink/junction/reparse point：{root}")
    try:
        canonical = root.resolve(strict=True)
    except OSError as exc:
        raise ConfigurationError(f"无法解析运行目录 {root}：{exc}") from exc
    if not canonical.is_dir():
        raise ConfigurationError(f"运行路径不是目录：{canonical}")
    return canonical


def _is_reparse_point(path: Path) -> bool:
    try:
        return _status_is_reparse_point(path.lstat())
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise ConfigurationError(f"无法检查路径 {path}：{exc}") from exc


def _status_is_reparse_point(status: os.stat_result) -> bool:
    if stat.S_ISLNK(status.st_mode):
        return True
    attributes = getattr(status, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(attributes & reparse_flag)


def _serialize_json(value: Any, description: str) -> bytes:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{description}不是安全 JSON：{exc}") from exc
    return f"{payload}\n".encode()


def _atomic_publish(destination: Path, payload: bytes) -> None:
    temporary = destination.with_name(
        f".{destination.name}.{secrets.token_hex(8)}.tmp"
    )
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
            0o600,
        )
        remaining = memoryview(payload)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError("写入临时文件时未取得进展")
            remaining = remaining[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.replace(temporary, destination)
    except OSError as exc:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise ConfigurationError(f"无法写入或发布 {destination}：{exc}") from exc


def _rollback_run_directory(root: Path) -> None:
    for name in ("config.yaml", "environment.json"):
        try:
            (root / name).unlink(missing_ok=True)
        except OSError:
            pass
    try:
        root.rmdir()
    except OSError:
        pass


def _validate_run_id(run_id: object, requested_run_id: str | None) -> None:
    reserved_name = (
        run_id.split(".", maxsplit=1)[0].upper()
        if isinstance(run_id, str)
        else ""
    )
    if (
        not isinstance(run_id, str)
        or not run_id
        or run_id in {".", ".."}
        or run_id.endswith((".", " "))
        or reserved_name in _WINDOWS_RESERVED_NAMES
        or _SAFE_RUN_ID.fullmatch(run_id) is None
    ):
        displayed = requested_run_id if requested_run_id is not None else run_id
        raise ConfigurationError(f"不安全的 run ID：{displayed!r}")
