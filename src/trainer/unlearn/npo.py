"""Summed masked NPO, token-normalized ablation, and matched random baseline."""
from trainer.masking import MaskSelector, sample_indices
from trainer.utils import compute_masked_dpo_loss, compute_model_token_logprobs
from trainer.unlearn.grad_diff import GradDiff


class NPO(GradDiff):
    def __init__(self, beta=1.0, *, mask=None, normalization="summed",
                 random_reference="adaptive", mask_seed=1729, **kwargs):
        super().__init__(**kwargs)
        if normalization not in {"summed", "token_mean"}:
            raise ValueError("normalization must be summed or token_mean")
        self.beta = beta
        self.normalization = normalization
        self.objective_name = "summed masked NPO" if normalization == "summed" else "token-normalized NPO"
        self.mask_type = mask
        self.mask_selector = MaskSelector(mask, random_reference, mask_seed)
        if self.ref_model is None:
            self.ref_model = self._prepare_ref_model(self.model)

    def compute_loss(self, model, inputs, return_outputs=False):
        forget = inputs["forget"]
        if self.mask_type in {"adaptive", "random"}:
            sample_indices(forget)
        forward = {k: forget[k] for k in ("input_ids", "attention_mask", "labels", "position_ids") if k in forget}
        outputs = model(**forward)
        source = self.mask_selector.random_reference if self.mask_type == "random" else self.mask_type
        logprobs = compute_model_token_logprobs(outputs.logits.detach(), forget["input_ids"]) if source == "adaptive" and "reference_mask" not in forget else None
        selected = self.mask_selector(forget, logprobs)
        forget_loss, _ = compute_masked_dpo_loss(
            model, self.ref_model, lose_inputs=forward, lose_mask=selected,
            lose_outputs=outputs, beta=self.beta, normalization=self.normalization,
        )
        retain = {k: inputs["retain"][k] for k in ("input_ids", "attention_mask", "labels")}
        retain_loss = self.compute_retain_loss(model, retain)
        loss = self.gamma * forget_loss + self.alpha * retain_loss
        return (loss, outputs) if return_outputs else loss


class TokenNormalizedNPO(NPO):
    def __init__(self, *args, normalization="token_mean", **kwargs):
        if normalization != "token_mean":
            raise ValueError("TokenNormalizedNPO requires normalization=token_mean")
        super().__init__(*args, normalization=normalization, **kwargs)


class RandomNPO(NPO):
    def __init__(self, *args, mask="random", **kwargs):
        if mask != "random":
            raise ValueError("RandomNPO requires mask=random")
        super().__init__(*args, mask=mask, **kwargs)
