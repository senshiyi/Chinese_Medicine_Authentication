from __future__ import annotations

import json
from types import SimpleNamespace

import preview_predict
import torch


class FakeService:
    model_version = "preview-test-v1"

    def __init__(self, authenticity_logits: tuple[float, float]) -> None:
        self.authenticity_logits = authenticity_logits

    def predict_logits(self, _image):
        return SimpleNamespace(
            authenticity=torch.tensor([self.authenticity_logits], dtype=torch.float32)
        )


def factory(logits: tuple[float, float]):
    return lambda _checkpoint, _device: FakeService(logits)


def test_authentic_preview_prints_requested_chinese_message(capsys) -> None:
    exit_code = preview_predict.main(
        ["sample.jpg", "--checkpoint", "candidate.pth"],
        service_factory=factory((2.0, 0.0)),
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "此药品为正品，有88.1%的可能该药材为正品。" in output
    assert "实验试运行" in output
    assert "未校准" in output


def test_counterfeit_preview_with_unknown_species_refuses_final_judgment(capsys) -> None:
    exit_code = preview_predict.main(
        ["sample.jpg", "--checkpoint", "candidate.pth"],
        service_factory=factory((0.0, 2.5867)),
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "实际药材无法确定，有93.0%的可能该药材为伪品，请人工复核。" in output


def test_preview_json_exposes_unofficial_uncalibrated_probabilities(capsys) -> None:
    exit_code = preview_predict.main(
        ["sample.jpg", "--checkpoint", "candidate.pth", "--json"],
        service_factory=factory((2.0, 0.0)),
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["official"] is False
    assert payload["probability_type"] == "uncalibrated_softmax"
    assert payload["authenticity"]["predicted_label"] == "authentic"
    assert payload["authenticity"]["authentic_probability"] == 0.880797
