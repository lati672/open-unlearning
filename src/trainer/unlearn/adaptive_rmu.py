"""Indexed adaptive and density-matched random RMU."""
from trainer.masking import MaskSelector, sample_indices
from trainer.utils import compute_model_token_logprobs
from trainer.unlearn.rmu import RMU


class AdaptiveRMU(RMU):
    def __init__(self, *args, mask_negative_diff=False, mask_kind="adaptive",
                 random_reference="adaptive", mask_seed=1729, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_selector = MaskSelector(mask_kind, random_reference, mask_seed, mask_negative_diff)

    def compute_forget_loss(self, model, forget):
        sample_indices(forget)
        forward = {k: forget[k] for k in ("input_ids", "attention_mask", "labels")}
        activations, outputs = self.forward_with_cache(model, forward, self.model_module, no_grad=False)
        source = self.mask_selector.random_reference if self.mask_selector.kind == "random" else self.mask_selector.kind
        logprobs = compute_model_token_logprobs(outputs.logits.detach(), forget["input_ids"]) if source == "adaptive" and "reference_mask" not in forget else None
        mask = self.mask_selector(forget, logprobs).to(activations.device)
        control = forget.get("control_vec", self.get_control_vector(activations.shape[-1]))
        control = control.to(dtype=activations.dtype, device=activations.device).expand_as(activations)
        return self.compute_activation_loss(activations, control, mask), outputs

    def compute_loss(self, model, inputs, return_outputs=False):
        forget_loss, outputs = self.compute_forget_loss(model, inputs["forget"])
        retain = {k: inputs["retain"][k] for k in ("input_ids", "attention_mask", "labels")}
        retain_loss = self.compute_retain_loss(model, retain)
        loss = self.gamma * forget_loss + self.alpha * retain_loss
        return (loss, outputs) if return_outputs else loss


class RandomRMU(AdaptiveRMU):
    def __init__(self, *args, mask_kind="random", **kwargs):
        if mask_kind != "random":
            raise ValueError("RandomRMU requires mask_kind=random")
        super().__init__(*args, mask_kind=mask_kind, **kwargs)
