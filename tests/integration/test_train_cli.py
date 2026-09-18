from __future__ import annotations

import subprocess
import sys
from hashlib import sha256
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN = PROJECT_ROOT / "项目" / "Files" / "train.py"
LEGACY_MODEL = PROJECT_ROOT / "项目" / "Models" / "best_model.pth"


def test_train_cli_refuses_legacy_output_path() -> None:
    before = sha256(LEGACY_MODEL.read_bytes()).hexdigest()

    completed = subprocess.run(
        [sys.executable, "-I", "-X", "utf8", str(TRAIN), "--output", str(LEGACY_MODEL)],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
        timeout=30,
    )

    assert completed.returncode == 2
    assert "不允许覆盖旧模型" in completed.stderr
    assert sha256(LEGACY_MODEL.read_bytes()).hexdigest() == before


def test_train_cli_help_does_not_require_training_imports() -> None:
    completed = subprocess.run(
        [sys.executable, "-I", "-X", "utf8", str(TRAIN), "--help"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
        timeout=30,
    )

    assert completed.returncode == 0
    assert "--resume" in completed.stdout
    assert "--output" in completed.stdout
