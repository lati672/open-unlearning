"""Adaptive RMU trainer that leverages token-level log probabilities."""

import torch

from data.utils import IGNORE_INDEX
from trainer.utils import build_adaptive_mask, compute_model_token_logprobs
from trainer.unlearn.rmu import RMU


class AdaptiveRMU(RMU):
    """
    RMU variant that adapts the forget-side activation loss to tokens whose log probability
    gap (stored vs. current model) falls below the average gap over supervised tokens,
    with an option to instead focus on tokens whose gap turns negative.
    """


    def __init__(self, *args, mask_negative_diff: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_negative_diff = bool(mask_negative_diff)
        self._adaptive_mask_cache = {}
        self._global_adaptive_mask = None

    def compute_forget_loss(self, model, forget_inputs):
        model_inputs = {
            "input_ids": forget_inputs["input_ids"],
            "attention_mask": forget_inputs["attention_mask"],
            "labels": forget_inputs["labels"],
        }

        model_forget_activations, forget_outputs = self.forward_with_cache(
            model, model_inputs, self.model_module, no_grad=False
        )
        control_vec = forget_inputs.get(
            "control_vec", self.get_control_vector(model_forget_activations.shape[-1])
        )
        model_logprobs = compute_model_token_logprobs(
            forget_outputs.logits, model_inputs["input_ids"]
        ).detach()

        attention_mask = forget_inputs.get("attention_mask")
        sample_indices = forget_inputs.get("index")
        if sample_indices is not None:
            if torch.is_tensor(sample_indices):
                sample_indices_list = sample_indices.view(-1).tolist()
            else:
                sample_indices_list = list(sample_indices)
        else:
            sample_indices_list = None

        adaptive_mask = None
        computed_mask = None

        if sample_indices_list is None:
            if self._global_adaptive_mask is None:
                computed_mask = build_adaptive_mask(
                    forget_inputs,
                    model_logprobs,
                    mask_negative_diff=self.mask_negative_diff,
                ).detach().cpu()
                self._global_adaptive_mask = computed_mask.clone()
            cached_mask = self._global_adaptive_mask
            batch_size = model_inputs["input_ids"].size(0)
            seq_len = model_inputs["input_ids"].size(1)
            if cached_mask.size(0) != batch_size or cached_mask.size(1) != seq_len:
                padded = torch.zeros(batch_size, seq_len, dtype=torch.bool)
                rows = min(batch_size, cached_mask.size(0))
                cols = min(seq_len, cached_mask.size(1))
                padded[:rows, :cols] = cached_mask[:rows, :cols]
                cached_mask = padded
            base_mask = forget_inputs["labels"].ne(IGNORE_INDEX)
            if attention_mask is not None:
                base_mask = base_mask & attention_mask.bool()
            adaptive_mask = (cached_mask & base_mask.cpu()).to(
                model_forget_activations.device
            )
        else:
            missing_rows = [
                row_idx
                for row_idx, sample_idx in enumerate(sample_indices_list)
                if sample_idx not in self._adaptive_mask_cache
            ]
            if missing_rows:
                computed_mask = build_adaptive_mask(
                    forget_inputs,
                    model_logprobs,
                    mask_negative_diff=self.mask_negative_diff,
                ).detach().cpu()

            seq_len = model_inputs["input_ids"].size(1)
            adaptive_rows = []
            labels = forget_inputs["labels"]

            for row_idx, sample_idx in enumerate(sample_indices_list):
                cached_row = self._adaptive_mask_cache.get(sample_idx)
                if cached_row is None:
                    row_mask = computed_mask[row_idx]
                    if attention_mask is not None:
                        valid_length = int(attention_mask[row_idx].sum().item())
                    else:
                        valid_length = row_mask.size(0)
                    cached_row = row_mask[:valid_length].to(torch.bool).clone().cpu()
                    self._adaptive_mask_cache[sample_idx] = cached_row
                padded_row = torch.zeros(seq_len, dtype=torch.bool)
                copy_len = min(seq_len, cached_row.size(0))
                if copy_len > 0:
                    padded_row[:copy_len] = cached_row[:copy_len]
                base_row = labels[row_idx, :seq_len].ne(IGNORE_INDEX).cpu()
                if attention_mask is not None:
                    base_row = base_row & attention_mask[row_idx, :seq_len].bool().cpu()
                padded_row = padded_row & base_row
                adaptive_rows.append(padded_row)

            adaptive_mask = torch.stack(adaptive_rows, dim=0).to(
                model_forget_activations.device
            )

        target_bs = model_forget_activations.size(0)
        if adaptive_mask.size(0) != target_bs:
            adaptive_mask = adaptive_mask[:target_bs]
            model_logprobs = model_logprobs[:target_bs]
            model_inputs = {k: v[:target_bs] for k, v in model_inputs.items()}

        control_vec = control_vec.to(
            dtype=model_forget_activations.dtype,
            device=model_forget_activations.device,
        )
        control_vec = control_vec.expand_as(model_forget_activations)
        if control_vec.size(0) != target_bs:
            control_vec = control_vec[:target_bs]

        forget_loss = self.compute_activation_loss(
            model_forget_activations, control_vec, adaptive_mask
        )

        return forget_loss, forget_outputs

    def compute_loss(self, model, inputs, return_outputs=False):
        forget_inputs = inputs["forget"]
        retain_raw_inputs = inputs["retain"]

        forget_loss, forget_outputs = self.compute_forget_loss(model, forget_inputs)

        retain_inputs = {
            "input_ids": retain_raw_inputs["input_ids"],
            "attention_mask": retain_raw_inputs["attention_mask"],
            "labels": retain_raw_inputs["labels"],
        }
        retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)
        loss = self.gamma * forget_loss + self.alpha * retain_loss
        return (loss, forget_outputs) if return_outputs else loss
