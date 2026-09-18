from __future__ import annotations

import torch
from herb_authentication.models import MultiTaskLogits
from herb_authentication.training.losses import (
    LossWeights,
    MultiTaskMasks,
    MultiTaskTargets,
    compute_multitask_loss,
    masked_species_loss,
)


def test_unverified_species_rows_do_not_change_loss() -> None:
    logits_a = torch.tensor([[0.0, 0.0], [100.0, -100.0]], requires_grad=True)
    logits_b = torch.tensor([[0.0, 0.0], [-100.0, 100.0]], requires_grad=True)
    target = torch.zeros_like(logits_a)
    mask = torch.tensor([True, False])

    assert torch.equal(
        masked_species_loss(logits_a, target, mask),
        masked_species_loss(logits_b, target, mask),
    )


def test_all_unverified_species_returns_differentiable_zero() -> None:
    logits = torch.randn(2, 3, requires_grad=True)
    loss = masked_species_loss(
        logits,
        torch.zeros_like(logits),
        torch.tensor([False, False]),
    )
    loss.backward()

    assert loss.item() == 0.0
    assert logits.grad is not None


def test_compute_multitask_loss_applies_explicit_weights() -> None:
    logits = MultiTaskLogits(
        claimed_herb=torch.tensor([[2.0, -1.0], [-1.0, 2.0]], requires_grad=True),
        actual_species=torch.zeros(2, 2, requires_grad=True),
        authenticity=torch.tensor([[1.0, -1.0], [-1.0, 1.0]], requires_grad=True),
    )
    targets = MultiTaskTargets(
        claimed_herb=torch.tensor([0, 1]),
        actual_species=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        authenticity=torch.tensor([0, 1]),
    )
    masks = MultiTaskMasks(actual_species=torch.tensor([True, False]))
    weights = LossWeights(claimed_herb=2.0, actual_species=3.0, authenticity=4.0)

    result = compute_multitask_loss(logits, targets, masks, weights)

    assert torch.allclose(
        result.total,
        2.0 * result.claimed_herb + 3.0 * result.actual_species + 4.0 * result.authenticity,
    )
    assert result.species_supervised_rows == 1
    result.total.backward()
