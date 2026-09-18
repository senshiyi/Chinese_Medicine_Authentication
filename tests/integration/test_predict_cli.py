from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
from herb_authentication.errors import CheckpointCompatibilityError
from herb_authentication.inference.closed_set import ClosedSetPredictionResult
from herb_authentication.inference.policy import ScoredLabel
from herb_authentication.inference.results import PredictionResult

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PREDICT = PROJECT_ROOT / "项目" / "Files" / "predict.py"


def load_script():
    spec = importlib.util.spec_from_file_location("predict_cli_under_test", PREDICT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def authentic_result() -> PredictionResult:
    return PredictionResult(
        model_version="tiny-v1",
        is_calibrated=True,
        claimed_herb=ScoredLabel("当归", 0.95),
        actual_species=(ScoredLabel("当归", 0.93),),
        authenticity=ScoredLabel("authentic", 0.9234),
        rejected=False,
        rejection_reasons=(),
        message="此药品为正品（置信度：92.3%）。",
    )


def closed_set_result() -> ClosedSetPredictionResult:
    return ClosedSetPredictionResult(
        model_version="tiny-v1",
        authenticity=ScoredLabel("authentic", 0.91),
        probabilities={"authentic": 0.91, "counterfeit": 0.09},
        actual_species_verified=False,
        message="此药品为正品，有91.0%的可能该药材为正品。",
        model_fingerprint="a" * 64,
        data_fingerprint="b" * 64,
        split_fingerprint="c" * 64,
    )


class FakeService:
    def predict_image(self, image: Path | bytes) -> PredictionResult:
        assert image == Path("药材.png")
        return authentic_result()

    def predict_experimental_closed_set(self, image: Path | bytes) -> ClosedSetPredictionResult:
        assert image == Path("药材.png")
        return closed_set_result()


def service_factory(_checkpoint: Path, _device: str) -> FakeService:
    return FakeService()


def test_predict_cli_prints_message_and_separate_disclaimer(
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = load_script()

    assert module.main(["药材.png", "--checkpoint", "model.pth"], service_factory) == 0

    assert capsys.readouterr().out.splitlines() == [
        "此药品为正品（置信度：92.3%）。",
        "模型结果不能替代专业鉴定。",
    ]


def test_predict_json_outputs_complete_schema_v1(
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = load_script()

    assert module.main(["药材.png", "--checkpoint", "model.pth", "--json"], service_factory) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == 1
    assert payload["message"] == "此药品为正品（置信度：92.3%）。"
    assert payload["disclaimer"] == "模型结果不能替代专业鉴定。"


def test_allow_uncalibrated_requires_json(capsys: pytest.CaptureFixture[str]) -> None:
    module = load_script()

    with pytest.raises(SystemExit) as error:
        module.main(
            [
                "药材.png",
                "--checkpoint",
                "model.pth",
                "--allow-uncalibrated",
            ],
            service_factory,
        )

    assert error.value.code == 2
    assert "--json" in capsys.readouterr().err


def test_predict_help_does_not_import_torch() -> None:
    completed = subprocess.run(
        [sys.executable, "-I", "-X", "utf8", str(PREDICT), "--help"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
        timeout=30,
    )

    assert completed.returncode == 0
    assert "--allow-uncalibrated" in completed.stdout
    assert "--checkpoint" in completed.stdout


def test_experimental_closed_set_requires_json(capsys: pytest.CaptureFixture[str]) -> None:
    module = load_script()

    with pytest.raises(SystemExit) as error:
        module.main(
            ["药材.png", "--checkpoint", "model.pth", "--experimental-closed-set"],
            service_factory,
        )

    assert error.value.code == 2
    assert "--experimental-closed-set 必须与 --json 同时使用" in capsys.readouterr().err


def test_experimental_closed_set_outputs_structured_json(
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = load_script()

    code = module.main(
        [
            "药材.png",
            "--checkpoint",
            "model.pth",
            "--experimental-closed-set",
            "--json",
        ],
        service_factory,
    )

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["experimental_closed_set"] is True
    assert payload["open_set_calibrated"] is False
    assert payload["scope_warning"] == "仅适用于本数据集定义的骨碎补正品/伪品范围"


def test_experimental_closed_set_missing_metadata_exits_two(
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = load_script()

    class MissingMetadataService(FakeService):
        def predict_experimental_closed_set(self, image: Path | bytes) -> ClosedSetPredictionResult:
            raise CheckpointCompatibilityError("缺少实验闭集元数据")

    code = module.main(
        [
            "药材.png",
            "--checkpoint",
            "model.pth",
            "--experimental-closed-set",
            "--json",
        ],
        lambda _checkpoint, _device: MissingMetadataService(),
    )

    assert code == 2
    assert "实验闭集" in capsys.readouterr().err


def test_experimental_closed_set_accepts_explicit_image_option(
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = load_script()

    code = module.main(
        [
            "--checkpoint",
            "model.pth",
            "--experimental-closed-set",
            "--json",
            "--image",
            "药材.png",
        ],
        service_factory,
    )

    assert code == 0
    assert json.loads(capsys.readouterr().out)["experimental_closed_set"] is True


def test_experimental_json_disallows_non_finite_numbers(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_script()
    original_dumps = module.json.dumps
    captured: dict[str, object] = {}

    def strictness_spy(value, **kwargs):
        captured.update(kwargs)
        return original_dumps(value, **kwargs)

    monkeypatch.setattr(module.json, "dumps", strictness_spy)

    assert (
        module.main(
            [
                "--checkpoint",
                "model.pth",
                "--experimental-closed-set",
                "--json",
                "--image",
                "药材.png",
            ],
            service_factory,
        )
        == 0
    )
    capsys.readouterr()
    assert captured["allow_nan"] is False
