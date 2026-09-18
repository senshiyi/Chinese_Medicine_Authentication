from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch
from herb_authentication.errors import ConfigurationError
from herb_authentication.models.multitask import (
    MultiTaskHerbModel,
    build_torchvision_encoder,
)
from torch import nn


class TinyEncoder(nn.Module):
    def __init__(self, feature_dim: int = 7) -> None:
        super().__init__()
        self.projection = nn.Linear(3 * 8 * 8, feature_dim)
        self.forward_calls = 0

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        self.forward_calls += 1
        return self.projection(images.flatten(start_dim=1))


class BatchNormEncoder(nn.Module):
    def __init__(self, feature_dim: int = 7) -> None:
        super().__init__()
        self.normalization = nn.BatchNorm1d(3 * 8 * 8)
        self.projection = nn.Linear(3 * 8 * 8, feature_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        flattened = images.flatten(start_dim=1)
        return self.projection(self.normalization(flattened))


def make_model() -> MultiTaskHerbModel:
    return MultiTaskHerbModel(
        TinyEncoder(),
        feature_dim=7,
        num_claimed=3,
        num_species=5,
        dropout=0.0,
    )


def test_multitask_output_shapes_and_shared_encoder_call() -> None:
    model = make_model()

    output = model(torch.randn(2, 3, 8, 8))

    assert output.claimed_herb.shape == (2, 3)
    assert output.actual_species.shape == (2, 5)
    assert output.authenticity.shape == (2, 2)
    assert model.encoder.forward_calls == 1


def test_freeze_then_unfreeze_changes_encoder_requires_grad() -> None:
    model = make_model()

    model.set_encoder_trainable(False)
    assert model.encoder_trainable is False
    assert not any(parameter.requires_grad for parameter in model.encoder.parameters())
    assert all(parameter.requires_grad for parameter in model.head_parameters())

    model.set_encoder_trainable(True)
    assert model.encoder_trainable is True
    assert all(parameter.requires_grad for parameter in model.encoder.parameters())


def test_frozen_encoder_stays_in_eval_mode_during_head_training() -> None:
    model = make_model()
    model.train()

    model.set_encoder_trainable(False)
    model.train()

    assert model.training is True
    assert model.encoder.training is False
    model.set_encoder_trainable(True)
    assert model.encoder.training is True


def test_frozen_encoder_does_not_update_batch_norm_statistics() -> None:
    encoder = BatchNormEncoder()
    model = MultiTaskHerbModel(
        encoder,
        feature_dim=7,
        num_claimed=3,
        num_species=5,
        dropout=0.0,
    )
    model.set_encoder_trainable(False)
    model.train()
    before = encoder.normalization.running_mean.detach().clone()

    model(torch.full((4, 3, 8, 8), 10.0))

    assert torch.equal(before, encoder.normalization.running_mean)


def test_unfreeze_can_add_encoder_to_optimizer_and_update_weights() -> None:
    model = make_model()
    model.set_encoder_trainable(False)
    optimizer = torch.optim.SGD(model.head_parameters(), lr=0.1)

    model.set_encoder_trainable(True)
    added = model.add_encoder_to_optimizer(optimizer, lr=0.1)
    before = model.encoder.projection.weight.detach().clone()
    output = model(torch.randn(2, 3, 8, 8))
    loss = output.claimed_herb.sum() + output.actual_species.sum() + output.authenticity.sum()
    loss.backward()
    optimizer.step()

    assert added == len(list(model.encoder.parameters()))
    assert not torch.equal(before, model.encoder.projection.weight)
    assert model.add_encoder_to_optimizer(optimizer, lr=0.1) == 0


def test_state_dict_restores_encoder_trainability_stage() -> None:
    frozen = make_model()
    frozen.set_encoder_trainable(False)
    state = deepcopy(frozen.state_dict())
    restored = make_model()

    restored.load_state_dict(state)

    assert restored.encoder_trainable is False
    assert not any(parameter.requires_grad for parameter in restored.encoder.parameters())


def test_forward_rejects_encoder_feature_dimension_mismatch() -> None:
    model = MultiTaskHerbModel(
        nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 6)),
        feature_dim=7,
        num_claimed=3,
        num_species=5,
    )

    with pytest.raises(RuntimeError, match="feature_dim=7"):
        model(torch.randn(2, 3, 8, 8))


@pytest.mark.parametrize(
    ("field", "value"),
    (("feature_dim", 0), ("num_claimed", 0), ("num_species", 0)),
)
def test_model_rejects_nonpositive_dimensions(field: str, value: int) -> None:
    kwargs = {"feature_dim": 7, "num_claimed": 3, "num_species": 5}
    kwargs[field] = value

    with pytest.raises(ValueError, match=field):
        MultiTaskHerbModel(TinyEncoder(), **kwargs)


def test_torchvision_factory_builds_encoder_without_pretrained_download() -> None:
    spec = build_torchvision_encoder("efficientnet_b0", pretrained=False)

    assert isinstance(spec.encoder, nn.Module)
    assert spec.feature_dim == 1280
    assert spec.backbone == "efficientnet_b0"
    assert spec.weights_source is None


def test_torchvision_factory_fails_if_pretrained_cache_is_missing(tmp_path) -> None:
    with pytest.raises(ConfigurationError, match="本地缓存"):
        build_torchvision_encoder(
            "efficientnet_b0",
            pretrained=True,
            weights_cache_dir=tmp_path,
        )


def test_torchvision_factory_supports_configured_efficientnet_b3() -> None:
    spec = build_torchvision_encoder("efficientnet_b3", pretrained=False)

    assert isinstance(spec.encoder, nn.Module)
    assert spec.feature_dim == 1536
    assert spec.backbone == "efficientnet_b3"


def test_torchvision_factory_rejects_mismatched_pretrained_hash(tmp_path: Path) -> None:
    from torchvision.models import EfficientNet_B0_Weights

    filename = EfficientNet_B0_Weights.DEFAULT.url.rsplit("/", 1)[-1]
    torch.save({}, tmp_path / filename)

    with pytest.raises(ConfigurationError, match="SHA-256"):
        build_torchvision_encoder("efficientnet_b0", pretrained=True, weights_cache_dir=tmp_path)


def test_torchvision_factory_rejects_unsupported_backbone() -> None:
    with pytest.raises(ConfigurationError, match="不支持的 backbone"):
        build_torchvision_encoder("unknown-network", pretrained=False)
