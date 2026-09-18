from __future__ import annotations

import pytest
import torch
from herb_authentication.evaluation import (
    UnknownPrediction,
    summarize_known_predictions,
    summarize_unknown_predictions,
    write_evaluation_artifacts,
)
from herb_authentication.training.metrics import ImageLevelPredictions


def known_predictions() -> ImageLevelPredictions:
    return ImageLevelPredictions(
        claimed_probabilities=torch.tensor([[0.9, 0.1], [0.8, 0.2], [0.1, 0.9]]),
        species_probabilities=torch.tensor([[0.9, 0.1], [0.8, 0.2], [0.1, 0.9]]),
        authenticity_probabilities=torch.tensor([[0.9, 0.1], [0.8, 0.2], [0.1, 0.9]]),
        claimed_targets=torch.tensor([0, 0, 1]),
        species_targets=torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        authenticity_targets=torch.tensor([0, 0, 1]),
        species_mask=torch.tensor([True, True, True]),
        sample_ids=("s1", "s1", "s2"),
        group_ids=("g1", "g1", "g2"),
    )


def test_known_summary_reports_image_sample_and_group_scopes() -> None:
    summary = summarize_known_predictions(
        known_predictions(),
        min_species_supervision_coverage=0.2,
        species_threshold=0.75,
        bootstrap_samples=20,
        seed=7,
    )

    assert summary["counts"] == {"images": 3, "samples": 2, "groups": 2}
    assert summary["image_level"]["selection_score"] == pytest.approx(1.0)
    assert summary["sample_level"]["authenticity_macro_f1"] == pytest.approx(1.0)
    assert summary["group_bootstrap"]["samples"] == 20
    assert summary["group_bootstrap"]["median"] == pytest.approx(1.0)


def test_unknown_summary_fails_open_if_any_image_of_entity_is_accepted() -> None:
    summary = summarize_unknown_predictions(
        (
            UnknownPrediction("blurred", "s1", "g1", True),
            UnknownPrediction("blurred", "s1", "g1", False),
            UnknownPrediction("unknown_herb", "s2", "g2", True),
        )
    )

    assert summary["counts"] == {"images": 3, "samples": 2, "groups": 2}
    assert summary["image_level"]["reject_rate"] == pytest.approx(2 / 3)
    assert summary["sample_level"]["reject_rate"] == pytest.approx(0.5)
    assert summary["group_level"]["reject_rate"] == pytest.approx(0.5)
    assert summary["image_level"]["by_unknown_type"]["blurred"] == pytest.approx(0.5)


def test_artifact_publish_failure_preserves_existing_pair(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import herb_authentication.evaluation as evaluation

    output = tmp_path / "reports"
    output.mkdir()
    json_path = output / "evaluation.json"
    csv_path = output / "evaluation.csv"
    json_path.write_bytes(b"old-json")
    csv_path.write_bytes(b"old-csv")
    real_replace = evaluation.os.replace
    publishes = 0

    def fail_second_publish(source, destination):
        nonlocal publishes
        if destination in (json_path, csv_path):
            publishes += 1
            if publishes == 2:
                raise OSError("injected csv publish failure")
        return real_replace(source, destination)

    monkeypatch.setattr(evaluation.os, "replace", fail_second_publish)
    with pytest.raises(OSError, match="injected"):
        write_evaluation_artifacts({"metrics": {"value": 1}}, output)
    assert json_path.read_bytes() == b"old-json"
    assert csv_path.read_bytes() == b"old-csv"


def test_artifact_temp_failure_preserves_existing_pair_and_cleans_temp(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import herb_authentication.evaluation as evaluation

    output = tmp_path / "reports"
    output.mkdir()
    json_path = output / "evaluation.json"
    csv_path = output / "evaluation.csv"
    json_path.write_bytes(b"old-json")
    csv_path.write_bytes(b"old-csv")
    real_write_temp = evaluation._write_fsynced_temp
    writes = 0

    def fail_second_temp(directory, name, payload):
        nonlocal writes
        writes += 1
        if writes == 2:
            raise OSError("injected temp write failure")
        return real_write_temp(directory, name, payload)

    monkeypatch.setattr(evaluation, "_write_fsynced_temp", fail_second_temp)
    with pytest.raises(OSError, match="injected"):
        write_evaluation_artifacts({"metrics": {"value": 1}}, output)
    assert json_path.read_bytes() == b"old-json"
    assert csv_path.read_bytes() == b"old-csv"
    assert not tuple(output.glob("*.tmp"))


def test_artifact_rollback_failure_preserves_primary_and_attempts_both_restores(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import herb_authentication.evaluation as evaluation

    output = tmp_path / "reports"
    output.mkdir()
    json_path = output / "evaluation.json"
    csv_path = output / "evaluation.csv"
    json_path.write_bytes(b"old-json")
    csv_path.write_bytes(b"old-csv")
    real_replace = evaluation.os.replace
    primary = OSError("primary csv publish failure")
    rollback = OSError("json rollback failure")
    destinations = []

    def fail_publish_and_first_restore(source, destination):
        destinations.append(destination)
        if len(destinations) == 2:
            raise primary
        if len(destinations) == 3:
            raise rollback
        return real_replace(source, destination)

    monkeypatch.setattr(evaluation.os, "replace", fail_publish_and_first_restore)
    with pytest.raises(OSError, match="primary csv publish failure") as caught:
        write_evaluation_artifacts({"metrics": {"value": 1}}, output)

    assert caught.value is primary
    assert isinstance(caught.value.__cause__, ExceptionGroup)
    assert caught.value.__cause__.exceptions == (rollback,)
    assert destinations.count(csv_path) == 2
    assert csv_path.read_bytes() == b"old-csv"
    assert not tuple(output.glob("*.tmp"))
