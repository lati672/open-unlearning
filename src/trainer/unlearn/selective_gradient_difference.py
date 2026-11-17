"""Selective gradient ascent with standard retain-set gradient descent."""

import math

import torch

from trainer.base import FinetuneTrainer
from trainer.masking import eligible_positions, nonempty_mean, sample_indices
from trainer.utils import compute_masked_batch_nll


class SelectiveGradientDifference(FinetuneTrainer):
    def __init__(self, *args, gamma_forget=1.0, alpha_retain=1.0, **kwargs):
        if not all(math.isfinite(value) and value >= 0 for value in (gamma_forget, alpha_retain)):
            raise ValueError("SU loss weights must be finite and nonnegative")
        super().__init__(*args, **kwargs)
        self.gamma_forget = gamma_forget
        self.alpha_retain = alpha_retain

    def compute_loss(self, model, inputs, return_outputs=False):
        forget = inputs["forget"]
        sample_indices(forget)
        eligible = eligible_positions(forget)
        mask = forget["reference_mask"].to(device=eligible.device, dtype=torch.bool)
        if mask.shape != eligible.shape or (mask & ~eligible).any():
            raise ValueError("SU requires a frozen mask restricted to eligible targets")

        fields = ("input_ids", "attention_mask", "labels")
        sums, outputs = compute_masked_batch_nll(model, {key: forget[key] for key in fields}, token_mask=mask)
        forget_loss = -nonempty_mean(sums, mask.any(dim=-1))

        retain = {key: inputs["retain"][key] for key in fields}
        retain_sums, _ = compute_masked_batch_nll(model, retain)
        retain_count = eligible_positions(retain).sum()
        retain_loss = retain_sums.sum() / retain_count.clamp_min(1)
        loss = self.gamma_forget * forget_loss + self.alpha_retain * retain_loss
        return (loss, outputs) if return_outputs else loss
