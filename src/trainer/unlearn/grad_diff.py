import copy
from typing import Optional

import torch

from data.utils import IGNORE_INDEX
from trainer.utils import compute_kl_divergence, compute_masked_batch_nll
from trainer.unlearn.base import UnlearnTrainer


class GradDiff(UnlearnTrainer):
    def __init__(
        self,
        gamma=1.0,
        alpha=1.0,
        retain_loss_type="NLL",
        *trainer_args,
        mask: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(*trainer_args, **kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.retain_loss_type = retain_loss_type
        mask_choice = None
        if mask is not None:
            mask_choice = str(mask).lower()
            if mask_choice not in {"entity"}:
                raise ValueError(
                    f"Unsupported mask type '{mask_choice}'. Expected 'entity' or None."
                )
        self.mask_type = mask_choice
        self.apply_entity_mask = self.mask_type == "entity"
        self.ref_model = None
        if retain_loss_type == "KL" or retain_loss_type == "reverse_KL":
            self.ref_model = self._prepare_ref_model(self.model)

    def _prepare_ref_model(self, model):
        ref_model = copy.deepcopy(model).to(self.accelerator.device)
        ref_model.eval()
        if self.is_deepspeed_enabled:
            ref_model = self._prepare_deepspeed(ref_model)
        else:
            ref_model = self.accelerator.prepare_model(ref_model, evaluation_mode=True)
        return ref_model

    def compute_retain_loss(self, model, retain_inputs):
        retain_outputs = model(**retain_inputs)
        retain_loss = 0.0
        if self.retain_loss_type == "NLL":
            retain_loss += retain_outputs.loss
        elif self.retain_loss_type == "KL":
            kl_loss, retain_outputs = compute_kl_divergence(
                self.model, self.ref_model, retain_inputs
            )
            retain_loss += kl_loss
        else:
            raise NotImplementedError(
                f"{self.retain_loss_type} not implemented for retain set"
            )
        return retain_loss

    def compute_loss(self, model, inputs, return_outputs=False):
        forget_inputs = inputs["forget"]
        forget_inputs = {
            "input_ids": forget_inputs["input_ids"],
            "attention_mask": forget_inputs["attention_mask"],
            "labels": forget_inputs["labels"],
        }

        if self.apply_entity_mask:
            entity_mask = inputs["forget"].get("entity_mask")
            if entity_mask is None:
                raise KeyError(
                    "Entity mask requested but 'entity_mask' missing from forget inputs."
                )
            labels = forget_inputs["labels"]
            attention_mask = forget_inputs.get("attention_mask")
            base_mask = labels.ne(IGNORE_INDEX)
            if attention_mask is not None:
                base_mask = base_mask & attention_mask.bool()
            entity_mask = entity_mask.to(device=labels.device)
            if entity_mask.dtype != torch.bool:
                entity_mask = entity_mask.bool()
            token_mask = entity_mask & base_mask
            masked_loss_sums, forget_outputs = compute_masked_batch_nll(
                model,
                forget_inputs,
                token_mask=token_mask,
            )
            effective_mask = token_mask[..., 1:]
            token_counts = effective_mask.sum(dim=1)
            token_counts = token_counts.to(masked_loss_sums.dtype)
            valid = token_counts > 0
            if valid.any():
                per_sample_mean = masked_loss_sums / torch.clamp(token_counts, min=1.0)
                forget_loss = -per_sample_mean[valid].mean()
            else:
                forget_loss = masked_loss_sums.sum() * 0.0
        else:
            forget_outputs = model(**forget_inputs)
            forget_loss = -forget_outputs.loss

        retain_inputs = inputs["retain"]
        retain_inputs = {
            "input_ids": retain_inputs["input_ids"],
            "attention_mask": retain_inputs["attention_mask"],
            "labels": retain_inputs["labels"],
        }
        retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)

        loss = self.gamma * forget_loss + self.alpha * retain_loss

        return (loss, forget_outputs) if return_outputs else loss
