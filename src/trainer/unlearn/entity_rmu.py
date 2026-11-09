# trainer/unlearn/entity_rmu.py
import torch
from trainer.unlearn.rmu import RMU
from trainer.masking import eligible_positions

class EntityRMU(RMU):
    """
    RMU variant that restricts the forget-side activation loss to tokens
    where entity_mask == 1 (and labels != -100). Retain-side behavior unchanged.
    """

    @staticmethod
    def _safe_activation_loss(activation1: torch.Tensor,
                              activation2: torch.Tensor,
                              mask: torch.Tensor) -> torch.Tensor:
        """
        Match RMU's per-token loss semantics:
          - per-token MSE averaged over hidden dim
          - average over only the masked tokens
          - then mean over valid samples in the batch

        Shapes:
          activation*: [B, S, D]
          mask:       [B, S]  (bool)
        """
        return RMU.compute_activation_loss(None, activation1, activation2, mask)

    def compute_loss(self, model, inputs, return_outputs=False):
        # -------- forget branch (entity-masked) --------
        fg = inputs["forget"]
        # keep the forward inputs same as RMU, but we also expect entity_mask in fg
        forget_inputs = {
            "input_ids":      fg["input_ids"],
            "attention_mask": fg["attention_mask"],
            "labels":         fg["labels"],
        }

        model_forget_activations, forget_outputs = self.forward_with_cache(
            model, forget_inputs, self.model_module, no_grad=False
        )

        # control vector as in RMU
        control_vec = fg.get(
            "control_vec", self.get_control_vector(model_forget_activations.shape[-1])
        ).to(dtype=model_forget_activations.dtype, device=model_forget_activations.device)
        control_vec = control_vec.expand_as(model_forget_activations)

        # Build the entity-aware mask:
        #   - label-active (labels != -100)
        #   - attention_mask (optional but safe)
        #   - entity_mask == 1
        base_mask = eligible_positions(fg)
        if "entity_mask" not in fg:
            raise KeyError("EntityRMU requires entity_mask from the entity collator")
        forget_mask = (base_mask & fg["entity_mask"].to(base_mask.device).bool()).to(model_forget_activations.device)

        forget_loss = self._safe_activation_loss(
            model_forget_activations, control_vec, forget_mask
        )

        # -------- retain branch (unchanged) --------
        rt = inputs["retain"]
        retain_inputs = {
            "input_ids":      rt["input_ids"],
            "attention_mask": rt["attention_mask"],
            "labels":         rt["labels"],
        }
        retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)

        loss = self.gamma * forget_loss + self.alpha * retain_loss
        return (loss, forget_outputs) if return_outputs else loss
