"""Weighted, token-normalized NPO over frozen causal-label weights."""

import torch
from torch.nn import functional as F

from trainer.masking import eligible_positions, nonempty_mean
from trainer.unlearn.npo import TokenNormalizedNPO


def compute_weighted_batch_nll(model, inputs, token_weights, *, outputs=None):
    """Aggregate a weighted sequence NLL per row using label-position weights."""
    eligible = eligible_positions(inputs)
    weights = token_weights.to(device=eligible.device, dtype=torch.float32)
    if weights.shape != eligible.shape or weights.requires_grad:
        raise ValueError("Frozen weights must match labels in shape and have no gradient.")
    if not torch.isfinite(weights).all() or (weights < 0).any() or (weights[~eligible] != 0).any():
        raise ValueError("Weights must be finite, nonnegative, and zero outside eligible positions.")
    mass = weights.sum(-1)
    valid = mass > 0
    if not torch.isclose(mass[valid], torch.ones_like(mass[valid]), rtol=0, atol=1e-6).all():
        raise ValueError("Each nonempty row of token weights must sum to one.")
    if outputs is None:
        outputs = model(**inputs)
    shifted_weights = weights[:, 1:]
    targets = inputs["labels"][:, 1:].masked_fill(shifted_weights == 0, -100)
    nll = F.cross_entropy(outputs.logits[:, :-1, :].float().transpose(1, 2), targets, reduction="none")
    return (nll * shifted_weights).sum(-1), valid, outputs


def compute_weighted_npo_loss(model, reference, inputs, weights, beta):
    if beta <= 0:
        raise ValueError("beta must be positive.")
    current, valid, outputs = compute_weighted_batch_nll(model, inputs, weights)
    with torch.no_grad():
        initial, _, _ = compute_weighted_batch_nll(reference, inputs, weights)
    return nonempty_mean(-2 / beta * F.logsigmoid(beta * (current - initial.to(current.device))), valid), outputs


class WeightedTokenNormalizedNPO(TokenNormalizedNPO):
    """Keep NPO's frozen reference and retain loss with a weighted forget NLL."""

    def __init__(self, *, mask=None, **kwargs):
        if mask is not None:
            raise ValueError("Weighted NPO requires frozen token weights.")
        super().__init__(mask=None, **kwargs)
        self.ref_model.requires_grad_(False)
        self.objective_name = "weighted token-normalized NPO"

    def compute_loss(self, model, inputs, return_outputs=False):
        forget = inputs["forget"]
        forward = {k: forget[k] for k in ("input_ids", "attention_mask", "labels", "position_ids") if k in forget}
        forget_loss, outputs = compute_weighted_npo_loss(model, self.ref_model, forward, forget["token_weights"], self.beta)
        retain = {k: inputs["retain"][k] for k in ("input_ids", "attention_mask", "labels")}
        loss = self.gamma * forget_loss + self.alpha * self.compute_retain_loss(model, retain)
        return (loss, outputs) if return_outputs else loss
