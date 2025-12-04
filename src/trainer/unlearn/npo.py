from typing import Optional

import torch

from data.utils import IGNORE_INDEX
from trainer.utils import (
    build_adaptive_mask,
    compute_dpo_loss,
    compute_masked_dpo_loss,
    compute_model_token_logprobs,
)
from trainer.unlearn.grad_diff import GradDiff


class NPO(GradDiff):

    def __init__(
        self,
        beta=1.0,
        *,
        mask: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.beta = beta
        mask_choice = None
        if mask is not None:
            mask_choice = str(mask).lower()
            if mask_choice not in {"adaptive", "entity"}:
                raise ValueError(
                    f"Unsupported mask type '{mask_choice}'. Expected 'adaptive', 'entity', or null."
                )
        self.mask_type = mask_choice
        self.apply_adaptive_mask = mask_choice == "adaptive"
        self.apply_entity_mask = mask_choice == "entity"
        self._adaptive_mask_cache = {}
        self._global_adaptive_mask = None
        if self.ref_model is None:
            self.ref_model = self._prepare_ref_model(self.model)

    def compute_loss(self, model, inputs, return_outputs=False):
        forget_inputs = inputs["forget"]
        forget_forward_inputs = {
            key: forget_inputs[key]
            for key in ("input_ids", "attention_mask", "labels", "position_ids")
            if key in forget_inputs
        }

        if self.apply_adaptive_mask:
            model_outputs = model(**forget_forward_inputs)
            model_logprobs = compute_model_token_logprobs(
                model_outputs.logits.detach(),
                forget_inputs["input_ids"],
            )
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
            logits_device = model_outputs.logits.device

            if sample_indices_list is None:
                if self._global_adaptive_mask is None:
                    computed_mask = build_adaptive_mask(
                        forget_inputs,
                        model_logprobs,
                        mask_negative_diff=False,
                    ).detach().cpu()
                    self._global_adaptive_mask = computed_mask.clone()
                cached_mask = self._global_adaptive_mask
                batch_size = forget_inputs["input_ids"].size(0)
                seq_len = forget_inputs["input_ids"].size(1)
                if cached_mask.size(0) != batch_size or cached_mask.size(1) != seq_len:
                    padded = torch.zeros(batch_size, seq_len, dtype=torch.bool)
                    rows = min(batch_size, cached_mask.size(0))
                    cols = min(seq_len, cached_mask.size(1))
                    padded[:rows, :cols] = cached_mask[:rows, :cols]
                    cached_mask = padded
                base_mask = forget_inputs["labels"].ne(IGNORE_INDEX)
                if attention_mask is not None:
                    base_mask = base_mask & attention_mask.bool()
                adaptive_mask = (cached_mask & base_mask.cpu()).to(logits_device)
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
                        mask_negative_diff=False,
                    ).detach().cpu()

                seq_len = forget_inputs["input_ids"].size(1)
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

                adaptive_mask = torch.stack(adaptive_rows, dim=0).to(logits_device)

            forget_loss, (_, forget_outputs) = compute_masked_dpo_loss(
                model=model,
                ref_model=self.ref_model,
                win_inputs=None,
                lose_inputs=forget_forward_inputs,
                beta=self.beta,
                lose_mask=adaptive_mask,
                lose_outputs=model_outputs,
            )
        elif self.apply_entity_mask:
            model_outputs = model(**forget_forward_inputs)
            attention_mask = forget_inputs.get("attention_mask")
            entity_mask = forget_inputs.get("entity_mask")
            if entity_mask is None:
                raise KeyError(
                    "Entity mask requested but 'entity_mask' missing from forget inputs."
                )
            labels = forget_inputs["labels"]
            base_mask = labels.ne(IGNORE_INDEX)
            if attention_mask is not None:
                base_mask = base_mask & attention_mask.bool()
            entity_mask = entity_mask.to(device=model_outputs.logits.device)
            if entity_mask.dtype != torch.bool:
                entity_mask = entity_mask.bool()
            entity_mask = entity_mask & base_mask.to(entity_mask.device)
            forget_loss, (_, forget_outputs) = compute_masked_dpo_loss(
                model=model,
                ref_model=self.ref_model,
                win_inputs=None,
                lose_inputs=forget_forward_inputs,
                beta=self.beta,
                lose_mask=entity_mask,
                lose_outputs=model_outputs,
            )
        else:
            forget_loss, (_, forget_outputs) = compute_dpo_loss(
                model=model,
                ref_model=self.ref_model,
                win_inputs=None,
                lose_inputs=forget_forward_inputs,
                beta=self.beta,
            )

        retain_inputs = inputs["retain"]
        retain_inputs = {
            "input_ids": retain_inputs["input_ids"],
            "attention_mask": retain_inputs["attention_mask"],
            "labels": retain_inputs["labels"],
        }
        retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)

        loss = self.gamma * forget_loss + self.alpha * retain_loss
        return (loss, forget_outputs) if return_outputs else loss
