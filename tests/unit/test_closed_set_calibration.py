from __future__ import annotations

import pytest
import torch
from herb_authentication.inference.closed_set import (
    ClosedSetCalibration,
    fit_closed_set_temperature,
)

FINGERPRINT_A = "a" * 64
FINGERPRINT_B = "b" * 64
ALGORITHM_VERSION = "closed-set-temperature-v1"


def _fit() -> ClosedSetCalibration:
    return fit_closed_set_temperature(
        torch.tensor([[8.0, 0.0], [7.0, 0.0], [0.0, 8.0], [0.0, 7.0]]),
        torch.tensor([0, 1, 1, 0]),
        validation_fingerprint=FINGERPRINT_A,
        checkpoint_fingerprint=FINGERPRINT_B,
        algorithm_version=ALGORITHM_VERSION,
    )


def test_closed_set_temperature_is_positive_deterministic_and_improves_nll() -> None:
    first = _fit()
    second = _fit()

    assert first == second
    assert first.temperature > 0
    assert first.nll_after <= first.nll_before + 1e-9
    assert ClosedSetCalibration.from_dict(first.to_dict()) == first


def test_closed_set_calibration_serialization_is_strict() -> None:
    block = _fit().to_dict()

    with pytest.raises(ValueError, match="字段"):
        ClosedSetCalibration.from_dict(block | {"threshold": 0.5})
    with pytest.raises(ValueError, match="字段"):
        ClosedSetCalibration.from_dict(
            {key: value for key, value in block.items() if key != "ece_after"}
        )
    with pytest.raises(ValueError, match="checkpoint"):
        ClosedSetCalibration.from_dict(block, checkpoint_fingerprint="c" * 64)


@pytest.mark.parametrize(
    ("logits", "targets", "message"),
    [
        (torch.ones(3, 2), torch.tensor([0, 0, 0]), "两个类别"),
        (torch.ones(3), torch.tensor([0, 1, 0]), r"\[N, 2\]"),
        (torch.ones(3, 3), torch.tensor([0, 1, 0]), r"\[N, 2\]"),
        (torch.ones(3, 2), torch.tensor([[0], [1], [0]]), r"\[N\]"),
        (torch.tensor([[float("nan"), 0.0], [0.0, 1.0]]), torch.tensor([0, 1]), "有限"),
        (torch.tensor([[float("inf"), 0.0], [0.0, 1.0]]), torch.tensor([0, 1]), "有限"),
        (torch.ones(2, 2), torch.tensor([0, 2]), "0 或 1"),
    ],
)
def test_closed_set_temperature_rejects_invalid_authenticity_data(
    logits: torch.Tensor,
    targets: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        fit_closed_set_temperature(
            logits,
            targets,
            validation_fingerprint=FINGERPRINT_A,
            checkpoint_fingerprint=FINGERPRINT_B,
            algorithm_version=ALGORITHM_VERSION,
        )


@pytest.mark.parametrize("field", ["validation_fingerprint", "checkpoint_fingerprint"])
@pytest.mark.parametrize("fingerprint", ["a" * 63, "g" * 64, "A" * 64, 64])
def test_closed_set_temperature_rejects_invalid_sha256_fingerprints(
    field: str,
    fingerprint: object,
) -> None:
    kwargs = {
        "validation_fingerprint": FINGERPRINT_A,
        "checkpoint_fingerprint": FINGERPRINT_B,
        "algorithm_version": ALGORITHM_VERSION,
    }
    kwargs[field] = fingerprint

    with pytest.raises(ValueError, match="SHA-256"):
        fit_closed_set_temperature(
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            torch.tensor([0, 1]),
            **kwargs,
        )


def test_closed_set_temperature_rejects_worse_post_calibration_nll(monkeypatch) -> None:
    import herb_authentication.inference.closed_set as closed_set

    monkeypatch.setattr(closed_set, "fit_positive_temperature", lambda *_args, **_kwargs: 1e-3)

    with pytest.raises(ValueError, match="NLL"):
        fit_closed_set_temperature(
            torch.tensor([[2.0, 0.0], [0.0, 2.0]]),
            torch.tensor([1, 0]),
            validation_fingerprint=FINGERPRINT_A,
            checkpoint_fingerprint=FINGERPRINT_B,
            algorithm_version=ALGORITHM_VERSION,
        )


@pytest.mark.parametrize(
    "field",
    [
        "temperature",
        "nll_before",
        "nll_after",
        "brier_before",
        "brier_after",
        "ece_before",
        "ece_after",
    ],
)
def test_closed_set_calibration_rejects_non_finite_numeric_metadata(field: str) -> None:
    block = _fit().to_dict()
    block[field] = float("nan")

    with pytest.raises(ValueError, match="有限"):
        ClosedSetCalibration.from_dict(block)
