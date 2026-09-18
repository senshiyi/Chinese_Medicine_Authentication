"""仓库级 pytest 配置：提供不依赖系统临时目录的 tmp_path 实现。

背景：在 DSH 文件沙箱内，python 进程对系统临时区（pytest-of-<user>）以及由
tempfile.mkdtemp 创建的目录不具备文件写入/枚举能力，内置 tmp_path 无法使用。
这里在仓库根的 testtmp/（已被 .gitignore 排除）下用普通 mkdir 逐测试建目录；
在普通环境同样生效（等价隔离）。
运行参数建议：python -m pytest -p no:cacheprovider -p no:tmpdir
"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent
_TMP_ROOT = _PROJECT_ROOT / "testtmp" / f"pytest-cases-{uuid.uuid4().hex[:12]}"


@pytest.fixture()
def tmp_path() -> Path:
    """逐测试隔离的临时目录；使用普通 mkdir，不依赖 pytest basetemp/tempfile。"""
    _TMP_ROOT.mkdir(parents=True, exist_ok=True)
    case_dir = _TMP_ROOT / f"case-{uuid.uuid4().hex[:10]}"
    case_dir.mkdir()
    yield case_dir
    shutil.rmtree(case_dir, ignore_errors=True)
