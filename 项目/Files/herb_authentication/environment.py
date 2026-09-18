"""只依赖标准库的运行环境预检。

本模块禁止在顶层导入 torch、torchvision、numpy 或 sklearn。重量级依赖统一在
隔离子进程中探测，使 OpenMP/DLL 冲突可以转换为结构化诊断结果。
"""

from __future__ import annotations

import json
import locale
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TextIO

HEAVY_MODULES = ("torch", "torchvision", "numpy", "sklearn")
SUPPORTED_PYTHON_MIN = (3, 11)
SUPPORTED_PYTHON_MAX_EXCLUSIVE = (3, 13)


@dataclass(frozen=True)
class EnvironmentReport:
    """不触发第三方导入的轻量环境报告。"""

    python_version: str
    python_executable: str
    python_supported: bool
    preferred_encoding: str
    filesystem_encoding: str
    stdout_encoding: str | None
    stderr_encoding: str | None
    internal_text_encoding: str
    imported_heavy_modules: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PackageProbe:
    """一个依赖包的版本与来源。"""

    version: str
    path: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class DependencyProbe:
    """隔离子进程中的重量级依赖探测结果。"""

    ok: bool
    returncode: int
    python_executable: str
    python_version: str
    packages: dict[str, PackageProbe]
    cuda_available: bool
    cuda_device_count: int
    kmp_duplicate_lib_ok: str | None
    error: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "returncode": self.returncode,
            "python_executable": self.python_executable,
            "python_version": self.python_version,
            "packages": {name: package.to_dict() for name, package in self.packages.items()},
            "cuda_available": self.cuda_available,
            "cuda_device_count": self.cuda_device_count,
            "kmp_duplicate_lib_ok": self.kmp_duplicate_lib_ok,
            "error": self.error,
        }


def run_bootstrap_preflight() -> EnvironmentReport:
    """在任何重量级依赖导入之前收集解释器与编码信息。"""

    return EnvironmentReport(
        python_version=platform.python_version(),
        python_executable=str(Path(sys.executable).resolve()),
        python_supported=(
            SUPPORTED_PYTHON_MIN
            <= sys.version_info[:2]
            < SUPPORTED_PYTHON_MAX_EXCLUSIVE
        ),
        preferred_encoding=locale.getpreferredencoding(False),
        filesystem_encoding=sys.getfilesystemencoding(),
        stdout_encoding=getattr(sys.stdout, "encoding", None),
        stderr_encoding=getattr(sys.stderr, "encoding", None),
        internal_text_encoding="utf-8",
        imported_heavy_modules=tuple(name for name in HEAVY_MODULES if name in sys.modules),
    )


def configure_utf8_stdio() -> None:
    """尽可能将 CLI 标准输出与错误输出配置为 UTF-8。"""

    _reconfigure_stream(sys.stdout)
    _reconfigure_stream(sys.stderr)


def _reconfigure_stream(stream: TextIO) -> None:
    reconfigure = getattr(stream, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(encoding="utf-8", errors="replace")


_PROBE_SCRIPT = r"""
import json
import os
import platform
import sys

payload = {
    "ok": False,
    "python_executable": sys.executable,
    "python_version": platform.python_version(),
    "packages": {},
    "cuda_available": False,
    "cuda_device_count": 0,
    "kmp_duplicate_lib_ok": os.environ.get("KMP_DUPLICATE_LIB_OK"),
    "error": None,
}

try:
    environment_root = os.path.normcase(os.path.realpath(sys.prefix))
    import torch
    import torchvision
    import numpy
    import sklearn
    from PIL import Image

    modules = {
        "torch": torch,
        "torchvision": torchvision,
        "numpy": numpy,
        "sklearn": sklearn,
        "Pillow": Image,
    }
    for name, module in modules.items():
        root = module
        if name == "Pillow":
            import PIL
            root = PIL
        payload["packages"][name] = {
            "version": str(getattr(root, "__version__", "unknown")),
            "path": str(getattr(root, "__file__", "")),
        }

    outside_environment = []
    for name, package in payload["packages"].items():
        package_path = os.path.normcase(os.path.realpath(package["path"]))
        try:
            inside = os.path.commonpath((environment_root, package_path)) == environment_root
        except ValueError:
            inside = False
        if not inside:
            outside_environment.append(f"{name}={package['path']}")
    if outside_environment:
        raise RuntimeError("依赖来自当前环境之外：" + ", ".join(outside_environment))
    payload["cuda_available"] = bool(torch.cuda.is_available())
    payload["cuda_device_count"] = int(torch.cuda.device_count())
    payload["ok"] = True
except BaseException as exc:
    payload["error"] = f"{type(exc).__name__}: {exc}"

print(json.dumps(payload, ensure_ascii=False))
raise SystemExit(0 if payload["ok"] else 3)
"""


def run_dependency_probe(timeout_seconds: float = 60.0) -> DependencyProbe:
    """在隔离子进程中导入重量级依赖并返回结构化结果。"""

    environment = os.environ.copy()
    environment.pop("KMP_DUPLICATE_LIB_OK", None)
    environment["PYTHONNOUSERSITE"] = "1"
    command = [sys.executable, "-I", "-X", "utf8", "-c", _PROBE_SCRIPT]

    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            env=environment,
        )
    except subprocess.TimeoutExpired as exc:
        return _failed_probe(returncode=124, error=f"依赖探测超时：{exc}")
    except OSError as exc:
        return _failed_probe(returncode=126, error=f"无法启动依赖探测子进程：{exc}")

    payload = _parse_probe_payload(completed.stdout)
    if payload is None:
        detail = completed.stderr.strip() or completed.stdout.strip() or "子进程没有输出"
        return _failed_probe(
            returncode=completed.returncode,
            error=f"依赖探测未返回合法 JSON：{detail}",
        )

    packages = {
        name: PackageProbe(version=str(value["version"]), path=str(value["path"]))
        for name, value in payload.get("packages", {}).items()
    }
    error = payload.get("error")
    if completed.returncode != 0 and not error:
        error = completed.stderr.strip() or f"依赖探测退出码：{completed.returncode}"

    return DependencyProbe(
        ok=bool(payload.get("ok")) and completed.returncode == 0,
        returncode=completed.returncode,
        python_executable=str(payload.get("python_executable", sys.executable)),
        python_version=str(payload.get("python_version", platform.python_version())),
        packages=packages,
        cuda_available=bool(payload.get("cuda_available", False)),
        cuda_device_count=int(payload.get("cuda_device_count", 0)),
        kmp_duplicate_lib_ok=payload.get("kmp_duplicate_lib_ok"),
        error=str(error) if error else None,
    )


def _parse_probe_payload(stdout: str) -> dict[str, Any] | None:
    for line in reversed(stdout.splitlines()):
        candidate = line.strip()
        if not candidate:
            continue
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _failed_probe(returncode: int, error: str) -> DependencyProbe:
    return DependencyProbe(
        ok=False,
        returncode=returncode,
        python_executable=str(Path(sys.executable).resolve()),
        python_version=platform.python_version(),
        packages={},
        cuda_available=False,
        cuda_device_count=0,
        kmp_duplicate_lib_ok=None,
        error=error,
    )
