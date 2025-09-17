# trainer/unlearn/entity_rmu.py
import torch
from trainer.unlearn.rmu import RMU  # or adjust the import path to where your RMU class lives

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
        # [B, S, D]  per-token squared diffs
        per_tok_mse = torch.nn.functional.mse_loss(
            activation1, activation2, reduction="none"
        ).mean(dim=2)  # -> [B, S]

        mask_f = mask.float()                # [B, S]
        sums   = (per_tok_mse * mask_f).sum(dim=1)      # [B]
        counts = mask_f.sum(dim=1)                        # [B]

        # Avoid div-by-zero; exclude samples with no masked tokens from the batch mean
        valid = counts > 0
        if valid.any():
            loss_per_sample = torch.zeros_like(sums)
            loss_per_sample[valid] = sums[valid] / counts[valid]
            
            return loss_per_sample[valid].mean()
        else:
            # No entities (or no label-active tokens) in this batch -> zero loss
            return torch.zeros((), dtype=per_tok_mse.dtype, device=per_tok_mse.device)

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
        base_mask   = (fg["labels"] != -100) & fg["attention_mask"].bool()
        if "entity_mask" in fg:
            ent_mask = fg["entity_mask"].bool()
            # Ensure device + shape alignment
            ent_mask = ent_mask.to(device=base_mask.device)
            forget_mask = base_mask & ent_mask
        else:
            # Fallback: if entity_mask not provided, behave like vanilla RMU
            forget_mask = base_mask

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
