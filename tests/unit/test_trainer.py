from __future__ import annotations

import pytest
import torch
from herb_authentication.errors import CheckpointCompatibilityError
from herb_authentication.models import MultiTaskHerbModel
from herb_authentication.training.checkpoints import CheckpointContract
from herb_authentication.training.losses import (
    MultiTaskMasks,
    MultiTaskTargets,
    compute_multitask_loss,
)
from herb_authentication.training.trainer import Trainer, compute_authenticity_class_weights
from torch import nn


class TinyEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(4, 3)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.projection(images)


def make_model() -> MultiTaskHerbModel:
    return MultiTaskHerbModel(
        encoder=TinyEncoder(),
        feature_dim=3,
        num_claimed=2,
        num_species=2,
        dropout=0.0,
    )


def make_loader() -> list[dict[str, object]]:
    return [
        {
            "image": torch.tensor(
                [[1.0, 0.0, 0.5, -0.5], [0.0, 1.0, -0.5, 0.5]],
                dtype=torch.float32,
            ),
            "claimed_target": torch.tensor([0, 1]),
            "actual_species_target": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            "authenticity_target": torch.tensor([0, 1]),
            "actual_species_mask": torch.tensor([True, True]),
        }
    ]


def optimizer_parameter_ids(trainer: Trainer) -> set[int]:
    return {
        id(parameter) for group in trainer.optimizer.param_groups for parameter in group["params"]
    }


def test_authenticity_class_weights_are_inverse_frequency_and_finite() -> None:
    weights = compute_authenticity_class_weights([0, 0, 0, 1])
    assert weights.tolist() == pytest.approx([2 / 3, 2.0])


def test_trainer_applies_authenticity_class_weights_only_to_authenticity_loss() -> None:
    model = make_model()
    weights = torch.tensor([0.5, 2.0])
    trainer = Trainer(
        model,
        epochs=1,
        learning_rate=1e-2,
        freeze_backbone=False,
        authenticity_class_weight=weights,
    )
    batch = make_loader()[0]
    model.eval()
    with torch.no_grad():
        logits = model(batch["image"])
        expected = compute_multitask_loss(
            logits,
            MultiTaskTargets(
                claimed_herb=batch["claimed_target"],
                actual_species=batch["actual_species_target"],
                authenticity=batch["authenticity_target"],
            ),
            MultiTaskMasks(actual_species=batch["actual_species_mask"]),
            trainer.loss_weights,
            authenticity_class_weight=weights,
        )

    result = trainer.run_epoch(0, [batch], training=False)

    assert trainer.authenticity_class_weight.device == trainer.device
    assert result.claimed_herb_loss == pytest.approx(float(expected.claimed_herb))
    assert result.actual_species_loss == pytest.approx(float(expected.actual_species))
    assert result.authenticity_loss == pytest.approx(float(expected.authenticity))


def test_encoder_unfreezes_at_epoch_two_and_is_added_to_optimizer() -> None:
    model = make_model()
    trainer = Trainer(
        model,
        epochs=3,
        learning_rate=1e-2,
        encoder_learning_rate=1e-3,
        freeze_backbone=True,
        unfreeze_epoch=2,
    )
    encoder_ids = {id(parameter) for parameter in model.encoder.parameters()}

    assert trainer.freeze_stage == "heads_only"
    assert not any(parameter.requires_grad for parameter in model.encoder.parameters())
    assert optimizer_parameter_ids(trainer).isdisjoint(encoder_ids)

    before = trainer.run_epoch(0, make_loader(), training=True)
    after = trainer.run_epoch(2, make_loader(), training=True)

    assert before.freeze_stage == "heads_only"
    assert after.freeze_stage == "full_model"
    assert all(parameter.requires_grad for parameter in model.encoder.parameters())
    assert encoder_ids <= optimizer_parameter_ids(trainer)
    assert trainer.optimizer.param_groups[-1]["lr"] == 1e-3


def test_freeze_backbone_false_trains_full_model_from_first_epoch() -> None:
    model = make_model()
    trainer = Trainer(
        model,
        epochs=1,
        learning_rate=1e-2,
        freeze_backbone=False,
    )
    encoder_ids = {id(parameter) for parameter in model.encoder.parameters()}

    result = trainer.run_epoch(0, make_loader(), training=True)

    assert trainer.freeze_stage == "full_model"
    assert result.freeze_stage == "full_model"
    assert all(parameter.requires_grad for parameter in model.encoder.parameters())
    assert encoder_ids <= optimizer_parameter_ids(trainer)


def test_fit_records_train_and_validation_losses_and_freeze_stage() -> None:
    trainer = Trainer(
        make_model(),
        epochs=2,
        learning_rate=1e-2,
        encoder_learning_rate=1e-3,
        freeze_backbone=True,
        unfreeze_epoch=1,
    )

    summary = trainer.fit(make_loader(), make_loader())

    assert len(summary.train_epochs) == 2
    assert len(summary.validation_epochs) == 2
    assert [row.freeze_stage for row in summary.train_epochs] == ["heads_only", "full_model"]
    assert [row.epoch for row in summary.validation_epochs] == [0, 1]
    assert all(row.batches == 1 and row.samples == 2 for row in summary.train_epochs)
    assert all(row.total_loss > 0 for row in summary.train_epochs)
    assert all(row.claimed_herb_loss > 0 for row in summary.validation_epochs)
    assert all(row.actual_species_loss > 0 for row in summary.validation_epochs)
    assert all(row.authenticity_loss > 0 for row in summary.validation_epochs)


def test_amp_request_is_disabled_on_cpu() -> None:
    trainer = Trainer(
        make_model(),
        epochs=1,
        learning_rate=1e-2,
        freeze_backbone=False,
        mixed_precision=True,
        device="cpu",
    )

    assert trainer.amp_enabled is False
    assert trainer.run_epoch(0, make_loader(), training=True).total_loss > 0


def checkpoint_contract() -> CheckpointContract:
    return CheckpointContract(
        schema_version=1,
        backbone="tiny",
        claimed_herb_classes=("当归", "黄芪"),
        actual_species_classes=("当归", "欧当归"),
        authenticity_classes=("authentic", "counterfeit"),
        image_size=2,
        preprocessing_version="test-v1",
        data_fingerprint="data",
        split_fingerprint="locked-split",
        training_config_fingerprint="training",
    )


def test_checkpoint_metadata_cannot_override_resume_contract(tmp_path) -> None:
    path = tmp_path / "model.pth"
    base = checkpoint_contract()
    strict = base
    trainer = Trainer(
        make_model(),
        epochs=1,
        learning_rate=1e-2,
        freeze_backbone=False,
        checkpoint_path=path,
        checkpoint_contract=strict,
        checkpoint_metadata={
            "split_fingerprint": "must-not-win",
            "provenance": "synthetic",
        },
    )

    trainer.fit(make_loader(), make_loader())

    state = torch.load(path, map_location="cpu", weights_only=True)
    assert state["metadata"]["split_fingerprint"] == "locked-split"
    assert state["metadata"]["data_fingerprint"] == "data"
    assert state["metadata"]["provenance"] == "synthetic"


def test_resume_rejects_contract_without_provenance_fingerprints(tmp_path) -> None:
    weak = CheckpointContract(
        schema_version=1,
        backbone="tiny",
        claimed_herb_classes=("当归", "黄芪"),
        actual_species_classes=("当归", "欧当归"),
        authenticity_classes=("authentic", "counterfeit"),
        image_size=2,
        preprocessing_version="test-v1",
    )
    trainer = Trainer(
        make_model(),
        epochs=1,
        learning_rate=1e-2,
        freeze_backbone=False,
        checkpoint_contract=weak,
    )

    with pytest.raises(CheckpointCompatibilityError, match="数据、完整切分和训练配置指纹"):
        trainer.fit(make_loader(), make_loader(), resume_from=tmp_path / "missing.pth")


def model_parameters(model: MultiTaskHerbModel) -> tuple[torch.Tensor, ...]:
    return tuple(parameter.detach().clone() for parameter in model.parameters())


def test_resume_matches_continuous_training_and_restores_training_state(tmp_path) -> None:
    torch.manual_seed(37)
    initial = make_model().state_dict()

    continuous_model = make_model()
    continuous_model.load_state_dict(initial)
    continuous = Trainer(
        continuous_model,
        epochs=3,
        learning_rate=1e-2,
        encoder_learning_rate=1e-3,
        freeze_backbone=True,
        unfreeze_epoch=1,
    )
    continuous.scheduler = torch.optim.lr_scheduler.StepLR(
        continuous.optimizer, step_size=1, gamma=0.5
    )
    torch.manual_seed(101)
    continuous.fit(make_loader(), make_loader())

    checkpoint_path = tmp_path / "resume.pth"
    interrupted_model = make_model()
    interrupted_model.load_state_dict(initial)
    interrupted = Trainer(
        interrupted_model,
        epochs=1,
        learning_rate=1e-2,
        encoder_learning_rate=1e-3,
        freeze_backbone=True,
        unfreeze_epoch=1,
        checkpoint_path=checkpoint_path,
        checkpoint_contract=checkpoint_contract(),
    )
    interrupted.scheduler = torch.optim.lr_scheduler.StepLR(
        interrupted.optimizer, step_size=1, gamma=0.5
    )
    torch.manual_seed(101)
    interrupted.fit(make_loader(), make_loader())

    resumed_model = make_model()
    resumed = Trainer(
        resumed_model,
        epochs=3,
        learning_rate=1e-2,
        encoder_learning_rate=1e-3,
        freeze_backbone=True,
        unfreeze_epoch=1,
        checkpoint_path=checkpoint_path,
        checkpoint_contract=checkpoint_contract(),
    )
    resumed.scheduler = torch.optim.lr_scheduler.StepLR(resumed.optimizer, step_size=1, gamma=0.5)
    summary = resumed.fit(make_loader(), make_loader(), resume_from=checkpoint_path)

    assert [row.epoch for row in summary.train_epochs] == [0, 1, 2]
    assert resumed.freeze_stage == "full_model"
    assert resumed.global_step == continuous.global_step
    assert resumed.scheduler.state_dict() == continuous.scheduler.state_dict()
    for actual, expected in zip(
        model_parameters(resumed_model), model_parameters(continuous_model), strict=True
    ):
        assert torch.equal(actual, expected)


@pytest.mark.parametrize("missing_key", ["resume_model_state", "rng_state"])
def test_resume_rejects_missing_deterministic_state(tmp_path, missing_key: str) -> None:
    checkpoint_path = tmp_path / "incomplete-resume.pth"
    interrupted = Trainer(
        make_model(),
        epochs=1,
        learning_rate=1e-2,
        freeze_backbone=False,
        checkpoint_path=checkpoint_path,
        checkpoint_contract=checkpoint_contract(),
    )
    interrupted.fit(make_loader(), make_loader())
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    del state[missing_key]
    torch.save(state, checkpoint_path)

    resumed = Trainer(
        make_model(),
        epochs=2,
        learning_rate=1e-2,
        freeze_backbone=False,
        checkpoint_path=checkpoint_path,
        checkpoint_contract=checkpoint_contract(),
    )

    with pytest.raises(CheckpointCompatibilityError, match=missing_key):
        resumed.fit(make_loader(), make_loader(), resume_from=checkpoint_path)


def test_resume_rejects_checkpoint_without_complete_history(tmp_path) -> None:
    checkpoint_path = tmp_path / "missing-history.pth"
    interrupted = Trainer(
        make_model(),
        epochs=1,
        learning_rate=1e-2,
        freeze_backbone=False,
        checkpoint_path=checkpoint_path,
        checkpoint_contract=checkpoint_contract(),
    )
    interrupted.fit(make_loader(), make_loader())
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state["training"].pop("history", None)
    torch.save(state, checkpoint_path)

    resumed = Trainer(
        make_model(),
        epochs=2,
        learning_rate=1e-2,
        freeze_backbone=False,
        checkpoint_path=checkpoint_path,
        checkpoint_contract=checkpoint_contract(),
    )

    with pytest.raises(CheckpointCompatibilityError, match="history"):
        resumed.fit(make_loader(), make_loader(), resume_from=checkpoint_path)


@pytest.mark.parametrize("same_output", [True, False])
def test_resume_rejects_noop_without_rewriting_parent_or_creating_child(
    tmp_path,
    same_output: bool,
) -> None:
    parent = tmp_path / "parent.pth"
    Trainer(
        make_model(),
        epochs=1,
        learning_rate=1e-2,
        freeze_backbone=False,
        checkpoint_path=parent,
        checkpoint_contract=checkpoint_contract(),
    ).fit(make_loader(), make_loader())
    parent_bytes = parent.read_bytes()
    output = parent if same_output else tmp_path / "child.pth"
    resumed = Trainer(
        make_model(),
        epochs=1,
        learning_rate=1e-2,
        freeze_backbone=False,
        checkpoint_path=output,
        checkpoint_contract=checkpoint_contract(),
    )

    with pytest.raises(ValueError, match=r"start_epoch=1.*epochs=1"):
        resumed.fit(make_loader(), make_loader(), resume_from=parent)

    assert parent.read_bytes() == parent_bytes
    if not same_output:
        assert not output.exists()
