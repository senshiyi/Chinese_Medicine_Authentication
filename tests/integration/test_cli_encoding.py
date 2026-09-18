from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FILES = PROJECT_ROOT / "项目" / "Files"


@pytest.mark.parametrize(
    ("script", "expected"),
    [
        ("doctor.py", "检查中药识别项目"),
        ("train.py", "中药图片三任务模型训练"),
        ("calibrate.py", "校准中药三任务模型"),
        ("evaluate.py", "生成中药识别评估审计报告"),
        ("predict.py", "中药图片三任务识别"),
    ],
)
def test_cli_help_is_utf8_without_python_utf8_flag(script: str, expected: str) -> None:
    environment = os.environ.copy()
    environment.pop("PYTHONUTF8", None)
    completed = subprocess.run(
        [sys.executable, "-I", str(FILES / script), "--help"],
        check=False,
        capture_output=True,
        timeout=30,
        env=environment,
    )

    assert completed.returncode == 0
    assert expected in completed.stdout.decode("utf-8", errors="strict")


def test_predict_experimental_help_is_utf8_without_python_utf8_flag() -> None:
    environment = os.environ.copy()
    environment.pop("PYTHONUTF8", None)

    completed = subprocess.run(
        [sys.executable, "-I", str(FILES / "predict.py"), "--help"],
        check=False,
        capture_output=True,
        timeout=30,
        env=environment,
    )

    stdout = completed.stdout.decode("utf-8", errors="strict")
    assert completed.returncode == 0
    assert "--experimental-closed-set" in stdout
    assert "实验闭集" in stdout


def test_predict_experimental_json_is_utf8_without_python_utf8_flag() -> None:
    environment = os.environ.copy()
    environment.pop("PYTHONUTF8", None)
    script = f"""
import importlib.util
from pathlib import Path

predict_path = Path({str(FILES / "predict.py")!r})
spec = importlib.util.spec_from_file_location("predict_cli_subprocess", predict_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
from herb_authentication.inference.closed_set import ClosedSetPredictionResult
from herb_authentication.inference.policy import ScoredLabel

class Service:
    def predict_experimental_closed_set(self, image):
        return ClosedSetPredictionResult(
            model_version="tiny-v1",
            authenticity=ScoredLabel("authentic", 0.91),
            probabilities={{"authentic": 0.91, "counterfeit": 0.09}},
            actual_species_verified=False,
            message="此药品为正品，有91.0%的可能该药材为正品。",
            model_fingerprint="a" * 64,
            data_fingerprint="b" * 64,
            split_fingerprint="c" * 64,
        )

raise SystemExit(module.main(
    ["--checkpoint", "model.pth", "--experimental-closed-set", "--json", "--image", "药材.png"],
    lambda _checkpoint, _device: Service(),
))
"""

    completed = subprocess.run(
        [sys.executable, "-I", "-c", script],
        check=False,
        capture_output=True,
        timeout=30,
        env=environment,
    )

    payload = json.loads(completed.stdout.decode("utf-8", errors="strict"))
    assert completed.returncode == 0
    assert payload["experimental_closed_set"] is True
    assert payload["open_set_calibrated"] is False
    assert payload["scope_warning"] == "仅适用于本数据集定义的骨碎补正品/伪品范围"
    assert payload["message"] == "此药品为正品，有91.0%的可能该药材为正品。"
