from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
from herb_authentication.evaluation import build_evaluation_report

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVALUATE = PROJECT_ROOT / "项目" / "Files" / "evaluate.py"


def load_script():
    spec = importlib.util.spec_from_file_location("evaluate_cli_under_test", EVALUATE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_synthetic_metrics_are_never_official() -> None:
    report = build_evaluation_report(
        data_provenance="synthetic",
        data_fingerprint="data",
        checkpoint_fingerprint="checkpoint",
        locked_split_fingerprint="locked",
        checkpoint_split_fingerprint="locked",
        is_calibrated=True,
        known_metrics={"sample_level": {"authenticity_macro_f1": 0.99}},
        unknown_metrics={"sample_level": {"reject_rate": 1.0}},
    )

    assert report["data_provenance"] == "synthetic"
    assert report["locked_split_compatible"] is True
    assert report["official_metrics"] is False


@pytest.mark.parametrize(
    ("provenance", "locked", "checkpoint", "official"),
    [
        ("real", "same", "same", False),
        ("real", "current", "trained", False),
        ("unknown", "same", "same", False),
    ],
)
def test_official_gate_requires_real_and_locked_compatible(
    provenance: str, locked: str, checkpoint: str, official: bool
) -> None:
    report = build_evaluation_report(
        data_provenance=provenance,
        data_fingerprint=None,
        checkpoint_fingerprint=None,
        checkpoint_provenance=provenance,
        locked_split_fingerprint=locked,
        checkpoint_split_fingerprint=checkpoint,
        is_calibrated=False,
    )

    assert report["official_metrics"] is official


def test_evaluate_cli_reports_provenance_and_writes_utf8_artifacts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        "\n".join(
            [
                'project_root: "."',
                "data:",
                '  provenance: "synthetic"',
                '  manifest_path: "metadata.csv"',
                '  data_root: "."',
                "evaluation:",
                '  report_dir: "reports"',
                "inference:",
                '  checkpoint_path: "missing.pth"',
            ]
        ),
        encoding="utf-8",
    )
    module = load_script()

    assert module.main(["--config", str(config), "--json"]) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["data_provenance"] == "synthetic"
    assert report["official_metrics"] is False
    assert report["metrics"]["known"]["status"] == "not_evaluated"
    assert (tmp_path / "reports" / "evaluation.json").read_text(encoding="utf-8")
    assert (tmp_path / "reports" / "evaluation.csv").read_bytes().startswith(b"\xef\xbb\xbf")


def test_evaluate_help_does_not_import_torch() -> None:
    completed = subprocess.run(
        [sys.executable, "-I", "-X", "utf8", str(EVALUATE), "--help"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
        timeout=30,
    )

    assert completed.returncode == 0
    assert "--config" in completed.stdout
    assert "--checkpoint" in completed.stdout
    assert "--partition" in completed.stdout
    assert "--run-dir" in completed.stdout
    assert "--json" in completed.stdout
