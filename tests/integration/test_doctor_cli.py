from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCTOR = PROJECT_ROOT / "项目" / "Files" / "doctor.py"
print("DEBUG file=", __file__, "root=", PROJECT_ROOT, "doctor=", DOCTOR, "exists=", DOCTOR.exists())


def test_doctor_outputs_utf8_json_from_isolated_environment() -> None:
    completed = subprocess.run(
        [sys.executable, "-I", "-X", "utf8", str(DOCTOR), "--json"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
        timeout=60,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["ok"] is True
    assert report["bootstrap"]["internal_text_encoding"] == "utf-8"
    assert report["dependencies"]["ok"] is True
    assert report["dependencies"]["packages"]["torch"]["version"].startswith("2.6.0")
