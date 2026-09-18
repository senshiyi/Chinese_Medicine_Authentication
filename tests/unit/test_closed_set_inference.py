from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import pytest
import torch
from herb_authentication.errors import CheckpointCompatibilityError, InputImageError
from herb_authentication.inference.calibration import CalibrationBundle
from herb_authentication.inference.closed_set import (
    ClosedSetCalibration,
    ClosedSetPredictionResult,
    format_closed_set_result,
)
from herb_authentication.inference.policy import ScoredLabel
from herb_authentication.inference.service import InferenceService
from herb_authentication.models.multitask import MultiTaskLogits
from PIL import Image
from torch import nn

MODEL_FINGERPRINT = "a" * 64
DATA_FINGERPRINT = "b" * 64
SPLIT_FINGERPRINT = "c" * 64


class ClosedSetModel(nn.Module):
    def __init__(self, authenticity_logits: tuple[float, float]) -> None:
        super().__init__()
        self.register_buffer("authenticity_logits", torch.tensor([authenticity_logits]))

    def forward(self, images: torch.Tensor) -> MultiTaskLogits:
        batch = images.shape[0]
        return MultiTaskLogits(
            claimed_herb=torch.tensor([[5.0, 0.0]]).expand(batch, -1),
            actual_species=torch.tensor([[-5.0, 5.0]]).expand(batch, -1),
            authenticity=self.authenticity_logits.expand(batch, -1),
        )


def write_jpeg(path: Path) -> Path:
    Image.new("RGB", (8, 8), color=(80, 120, 40)).save(path, format="JPEG")
    return path


def calibration(temperature: float = 2.0) -> ClosedSetCalibration:
    return ClosedSetCalibration(
        temperature=temperature,
        algorithm_version="closed-set-temperature-v1",
        validation_fingerprint="d" * 64,
        checkpoint_fingerprint=MODEL_FINGERPRINT,
        nll_before=0.8,
        nll_after=0.7,
        brier_before=0.4,
        brier_after=0.3,
        ece_before=0.2,
        ece_after=0.1,
    )


def make_closed_set_service(
    authenticity_logits: tuple[float, float],
    counterfeit_species_verified: bool,
    *,
    temperature: float = 2.0,
    max_image_bytes: int = 1024 * 1024,
) -> InferenceService:
    return InferenceService(
        model=ClosedSetModel(authenticity_logits),
        preprocess=lambda image: torch.zeros(3, image.height, image.width),
        claimed_herb_classes=("当归", "黄芪"),
        actual_species_classes=("当归", "欧当归"),
        calibration=CalibrationBundle.uncalibrated(),
        model_version="tiny-v1",
        device="cpu",
        max_image_bytes=max_image_bytes,
        closed_set_calibration=calibration(temperature),
        counterfeit_species_verified=counterfeit_species_verified,
        experimental_closed_set=True,
        open_set_calibrated=False,
        model_fingerprint=MODEL_FINGERPRINT,
        data_fingerprint=DATA_FINGERPRINT,
        split_fingerprint=SPLIT_FINGERPRINT,
    )


def make_closed_set_result(
    authenticity: str,
    probability: float,
    actual_species_verified: bool,
) -> ClosedSetPredictionResult:
    other = 1.0 - probability
    message = (
        "此药品为正品，有91.0%的可能该药材为正品。"
        if authenticity == "authentic"
        else (
            "此药品为伪品，有93.0%的可能该药材为伪品，声称为‘当归’，实际应为‘欧当归’。"
            if actual_species_verified
            else "实际药材无法确定，有93.0%的可能该药材为伪品，请人工复核。"
        )
    )
    probabilities = (
        {"authentic": probability, "counterfeit": other}
        if authenticity == "authentic"
        else {"authentic": other, "counterfeit": probability}
    )
    return ClosedSetPredictionResult(
        model_version="tiny-v1",
        authenticity=ScoredLabel(authenticity, probability),
        probabilities=probabilities,
        actual_species_verified=actual_species_verified,
        message=message,
        model_fingerprint=MODEL_FINGERPRINT,
        data_fingerprint=DATA_FINGERPRINT,
        split_fingerprint=SPLIT_FINGERPRINT,
    )


def test_closed_set_path_and_bytes_return_same_json(tmp_path: Path) -> None:
    image = write_jpeg(tmp_path / "sample.jpg")
    service = make_closed_set_service(
        authenticity_logits=(4.0, 0.0), counterfeit_species_verified=True
    )

    assert (
        service.predict_experimental_closed_set(image).to_dict()
        == service.predict_experimental_closed_set(image.read_bytes()).to_dict()
    )


def test_authentic_closed_set_message_has_only_verdict_and_probability() -> None:
    result = make_closed_set_result(
        authenticity="authentic", probability=0.91, actual_species_verified=True
    )

    assert result.message == "此药品为正品，有91.0%的可能该药材为正品。"
    assert format_closed_set_result(result) == result.message
    assert "声称为" not in result.message
    assert "实际为" not in result.message


def test_unverified_actual_species_suppresses_counterfeit_verdict() -> None:
    result = make_closed_set_result(
        authenticity="counterfeit", probability=0.93, actual_species_verified=False
    )

    assert result.message == "实际药材无法确定，有93.0%的可能该药材为伪品，请人工复核。"
    assert result.authenticity.label == "counterfeit"
    assert result.authenticity.confidence == pytest.approx(0.93)


def test_verified_counterfeit_message_names_claimed_and_actual_species(tmp_path: Path) -> None:
    result = make_closed_set_service(
        authenticity_logits=(0.0, 5.173), counterfeit_species_verified=True
    ).predict_experimental_closed_set(write_jpeg(tmp_path / "sample.jpg"))

    assert result.message == (
        "此药品为伪品，有93.0%的可能该药材为伪品，声称为‘当归’，实际应为‘欧当归’。"
    )


def test_closed_set_json_has_native_versioned_audit_fields() -> None:
    result = make_closed_set_result(
        authenticity="counterfeit", probability=0.93, actual_species_verified=False
    )

    payload = result.to_dict()

    assert json.loads(json.dumps(payload, ensure_ascii=False)) == payload
    assert payload["schema_version"] == 1
    assert payload["experimental_closed_set"] is True
    assert payload["open_set_calibrated"] is False
    assert payload["scope_warning"] == "仅适用于本数据集定义的骨碎补正品/伪品范围"
    assert payload["model_fingerprint"] == MODEL_FINGERPRINT
    assert payload["data_fingerprint"] == DATA_FINGERPRINT
    assert payload["split_fingerprint"] == SPLIT_FINGERPRINT
    assert payload["authenticity"] == {"label": "counterfeit", "confidence": 0.93}
    assert payload["probabilities"] == pytest.approx({"authentic": 0.07, "counterfeit": 0.93})
    assert all(type(value) is float for value in payload["probabilities"].values())


@pytest.mark.parametrize(
    ("probabilities", "confidence", "match"),
    [
        ({"authentic": float("nan"), "counterfeit": float("nan")}, 0.9, "有限"),
        ({"authentic": 0.4, "counterfeit": 0.5}, 0.5, "和为 1"),
        ({"authentic": 0.4, "counterfeit": 0.6}, 0.5, "置信度"),
    ],
)
def test_closed_set_result_rejects_invalid_probability_contract(
    probabilities: dict[str, float], confidence: float, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        ClosedSetPredictionResult(
            model_version="tiny-v1",
            authenticity=ScoredLabel("counterfeit", confidence),
            probabilities=probabilities,
            actual_species_verified=False,
            message="message",
            model_fingerprint=MODEL_FINGERPRINT,
            data_fingerprint=DATA_FINGERPRINT,
            split_fingerprint=SPLIT_FINGERPRINT,
        )


def test_closed_set_applies_only_frozen_authenticity_temperature(tmp_path: Path) -> None:
    result = make_closed_set_service(
        authenticity_logits=(4.0, 0.0),
        counterfeit_species_verified=True,
        temperature=2.0,
    ).predict_experimental_closed_set(write_jpeg(tmp_path / "sample.jpg"))

    expected = torch.softmax(torch.tensor([2.0, 0.0]), dim=0)
    assert result.probabilities == {
        "authentic": pytest.approx(float(expected[0])),
        "counterfeit": pytest.approx(float(expected[1])),
    }
    assert result.authenticity.confidence == pytest.approx(float(expected[0]))


@pytest.mark.parametrize("payload", [b"not-an-image", b""])
def test_closed_set_reuses_safe_decode_for_corrupt_bytes(payload: bytes) -> None:
    with pytest.raises(InputImageError):
        make_closed_set_service((4.0, 0.0), True).predict_experimental_closed_set(payload)


def test_closed_set_reuses_safe_decode_for_oversized_bytes() -> None:
    stream = BytesIO()
    Image.new("RGB", (8, 8)).save(stream, format="PNG")

    with pytest.raises(InputImageError, match="字节"):
        make_closed_set_service(
            (4.0, 0.0), True, max_image_bytes=10
        ).predict_experimental_closed_set(stream.getvalue())


def test_experimental_prediction_fails_closed_when_metadata_is_missing() -> None:
    service = InferenceService(
        model=ClosedSetModel((4.0, 0.0)),
        preprocess=lambda _image: torch.zeros(3, 8, 8),
        claimed_herb_classes=("当归", "黄芪"),
        actual_species_classes=("当归", "欧当归"),
        calibration=CalibrationBundle.uncalibrated(),
        model_version="tiny-v1",
        device="cpu",
    )

    with pytest.raises(CheckpointCompatibilityError, match="实验闭集"):
        service.predict_experimental_closed_set(b"not-reached")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("temperature", 1e-4),
        ("temperature", 1e4),
        ("ece_before", 1.01),
        ("ece_after", 1.01),
        ("brier_before", 2.01),
        ("brier_after", 2.01),
    ],
)
def test_loaded_closed_set_calibration_rejects_semantically_invalid_ranges(
    field: str, value: float
) -> None:
    block = calibration().to_dict()
    block[field] = value

    with pytest.raises(ValueError):
        ClosedSetCalibration.from_dict(block)


def test_loaded_closed_set_calibration_rejects_worse_nll() -> None:
    block = calibration().to_dict()
    block["nll_after"] = block["nll_before"] + 1e-8

    with pytest.raises(ValueError, match="NLL"):
        ClosedSetCalibration.from_dict(block)


def test_malformed_closed_set_flag_fails_with_domain_error_before_decode() -> None:
    service = make_closed_set_service((4.0, 0.0), True)
    service.counterfeit_species_verified = []

    with pytest.raises(CheckpointCompatibilityError, match="实验闭集"):
        service.predict_experimental_closed_set(b"not-reached")


def test_closed_set_result_snapshots_mutable_probabilities() -> None:
    probabilities = {"authentic": 0.91, "counterfeit": 0.09}
    result = ClosedSetPredictionResult(
        model_version="tiny-v1",
        authenticity=ScoredLabel("authentic", 0.91),
        probabilities=probabilities,
        actual_species_verified=False,
        message="此药品为正品，有91.0%的可能该药材为正品。",
        model_fingerprint=MODEL_FINGERPRINT,
        data_fingerprint=DATA_FINGERPRINT,
        split_fingerprint=SPLIT_FINGERPRINT,
    )

    probabilities["authentic"] = float("nan")
    probabilities["counterfeit"] = 0.5

    assert result.to_dict()["probabilities"] == {
        "authentic": 0.91,
        "counterfeit": 0.09,
    }


@pytest.mark.parametrize(
    "tampered_probabilities",
    [
        {"authentic": float("nan"), "counterfeit": float("nan")},
        {"authentic": 0.4, "counterfeit": 0.5},
    ],
)
def test_closed_set_to_dict_revalidates_tampered_probabilities(
    tampered_probabilities: dict[str, float],
) -> None:
    result = make_closed_set_result("authentic", 0.91, False)
    object.__setattr__(result, "probabilities", tampered_probabilities)

    with pytest.raises(ValueError):
        result.to_dict()


def test_closed_set_to_dict_revalidates_tampered_confidence() -> None:
    result = make_closed_set_result("authentic", 0.91, False)
    object.__setattr__(result, "authenticity", ScoredLabel("authentic", 0.5))

    with pytest.raises(ValueError, match="置信度"):
        result.to_dict()


def test_closed_set_result_requires_probability_key_order() -> None:
    with pytest.raises(ValueError, match="顺序"):
        ClosedSetPredictionResult(
            model_version="tiny-v1",
            authenticity=ScoredLabel("authentic", 0.91),
            probabilities={"counterfeit": 0.09, "authentic": 0.91},
            actual_species_verified=False,
            message="此药品为正品，有91.0%的可能该药材为正品。",
            model_fingerprint=MODEL_FINGERPRINT,
            data_fingerprint=DATA_FINGERPRINT,
            split_fingerprint=SPLIT_FINGERPRINT,
        )


def test_closed_set_result_requires_exact_scope_warning() -> None:
    with pytest.raises(ValueError, match="scope_warning"):
        ClosedSetPredictionResult(
            model_version="tiny-v1",
            authenticity=ScoredLabel("authentic", 0.91),
            probabilities={"authentic": 0.91, "counterfeit": 0.09},
            actual_species_verified=False,
            message="此药品为正品，有91.0%的可能该药材为正品。",
            model_fingerprint=MODEL_FINGERPRINT,
            data_fingerprint=DATA_FINGERPRINT,
            split_fingerprint=SPLIT_FINGERPRINT,
            scope_warning="近似但未批准的范围文案",
        )


@pytest.mark.parametrize("schema_version", [True, 1.0])
def test_closed_set_result_requires_native_integer_schema_version(schema_version: object) -> None:
    with pytest.raises(ValueError, match="schema_version"):
        ClosedSetPredictionResult(
            model_version="tiny-v1",
            authenticity=ScoredLabel("authentic", 0.91),
            probabilities={"authentic": 0.91, "counterfeit": 0.09},
            actual_species_verified=False,
            message="此药品为正品，有91.0%的可能该药材为正品。",
            model_fingerprint=MODEL_FINGERPRINT,
            data_fingerprint=DATA_FINGERPRINT,
            split_fingerprint=SPLIT_FINGERPRINT,
            schema_version=schema_version,
        )


def test_closed_set_calibration_rejects_unknown_algorithm_version() -> None:
    block = calibration().to_dict()
    block["algorithm_version"] = "closed-set-temperature-v2"

    with pytest.raises(ValueError, match="algorithm_version"):
        ClosedSetCalibration.from_dict(block)
