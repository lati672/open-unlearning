"""Shared target-position semantics and deterministic per-sample mask caching."""

import hashlib
import numbers

import torch


def nonempty_mean(values, valid):
    """Sample mean with a graph-connected zero and correct DDP weighting.

    DDP averages rank gradients, so local sums are scaled by world_size /
    global_nonempty_count. The detached correction reports the global mean.
    """
    numerator = torch.where(valid, values, torch.zeros_like(values)).sum()
    count = valid.sum().to(device=values.device, dtype=torch.float32)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        global_count = count.clone()
        global_sum = numerator.detach().float().clone()
        torch.distributed.all_reduce(global_count)
        torch.distributed.all_reduce(global_sum)
        if global_count.item() == 0:
            return numerator * 0.0
        gradient_loss = numerator * (torch.distributed.get_world_size() / global_count)
        return gradient_loss + (global_sum / global_count - gradient_loss.detach())
    return numerator / count.clamp_min(1)


def eligible_positions(inputs):
    """Supervised, attended next-token targets, including real EOS, excluding j=0.

    NPO scores target j with logits[j-1]; RMU compares hidden[j]. Entity is
    intersected with this set. Adaptive thresholds finite discrepancies within
    this set and falls back to it only when no finite discrepancies exist.
    """
    mask = inputs["labels"].ne(-100)
    if "attention_mask" in inputs:
        mask = mask & inputs["attention_mask"].to(mask.device).bool()
    mask = mask.clone()
    mask[:, 0] = False
    if "attention_mask" in inputs:
        mask[:, 1:] &= inputs["attention_mask"][:, :-1].to(mask.device).bool()
    return mask


def sample_indices(inputs):
    if "index" not in inputs:
        raise KeyError("Adaptive/random masks require a stable per-sample 'index'.")
    indices = inputs["index"]
    if torch.is_tensor(indices):
        if indices.ndim != 1:
            raise ValueError("Sample index must be a one-dimensional integer batch.")
        indices = indices.detach().cpu().tolist()
    else:
        indices = list(indices)
    if len(indices) != inputs["input_ids"].shape[0]:
        raise ValueError("Sample index batch size does not match input_ids.")
    if any(isinstance(i, bool) or not isinstance(i, numbers.Integral) or i < 0 for i in indices):
        raise ValueError("Sample indices must be stable nonnegative integers.")
    return [int(i) for i in indices]


def density_matched_random_mask(eligible, reference_mask, indices, seed):
    """Uniform sampling without replacement, independent of global RNG/order."""
    eligible = eligible.bool().cpu()
    reference_mask = reference_mask.bool().cpu()
    if len(indices) != eligible.shape[0]:
        raise ValueError("One stable index is required per random-mask sample")
    if eligible.shape != reference_mask.shape or (reference_mask & ~eligible).any():
        raise ValueError("Reference mask must be a subset of eligible positions.")
    result = torch.zeros_like(eligible)
    for row, index in enumerate(indices):
        candidates = eligible[row].nonzero().flatten()
        count = int(reference_mask[row].sum())
        digest = hashlib.sha256(f"mask-rng-v1:{seed}:{index}".encode()).digest()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int.from_bytes(digest[:8], "little") % (2**63 - 1))
        chosen = candidates[torch.randperm(len(candidates), generator=generator)[:count]]
        result[row, chosen] = True
    return result


class MaskSelector:
    """Cache first-encounter masks by ID and reject IDs reused for other content.

    Cache entries contain unpadded positions, so batch size, order, left/right
    padding, and final short batches cannot change a sample's mask.
    """

    def __init__(self, kind=None, random_reference="adaptive", seed=1729,
                 mask_negative_diff=False):
        if kind not in {None, "adaptive", "entity", "random"}:
            raise ValueError(f"Unsupported mask kind: {kind}")
        if random_reference not in {"adaptive", "entity"}:
            raise ValueError("random_reference must be adaptive or entity")
        self.kind = kind
        self.random_reference = random_reference
        self.seed = int(seed)
        self.mask_negative_diff = mask_negative_diff
        self.cache = {}

    def __call__(self, inputs, model_logprobs=None):
        from trainer.utils import build_adaptive_mask

        eligible = eligible_positions(inputs)
        if self.kind is None:
            return eligible
        source = self.random_reference if self.kind == "random" else self.kind
        frozen = inputs.get("reference_mask")
        if frozen is not None:
            frozen = frozen.to(eligible.device).bool()
            if frozen.shape != eligible.shape or (frozen & ~eligible).any():
                raise ValueError("Frozen reference mask must be a subset of eligible positions")
        if source == "entity":
            if "entity_mask" not in inputs:
                raise KeyError("Entity masks require 'entity_mask' from the collator.")
            reference = eligible & inputs["entity_mask"].to(eligible.device).bool()
            if frozen is not None and not torch.equal(frozen, reference):
                raise ValueError("Frozen entity reference differs from the entity collator")
            if self.kind == "entity":
                return reference
        indices = sample_indices(inputs)
        computed = None
        rows = []
        for row, index in enumerate(indices):
            attended = inputs.get("attention_mask", torch.ones_like(inputs["input_ids"]))[row].bool()
            positions = attended.nonzero().flatten()
            ids = inputs["input_ids"][row, positions].detach().cpu()
            labels = inputs["labels"][row, positions].detach().cpu()
            signature = (tuple(ids.tolist()), tuple(labels.tolist()))
            entry = self.cache.get(index)
            if entry is not None and entry[0] != signature:
                raise ValueError(f"Sample index {index} was reused for different tokens/labels.")
            if entry is None:
                if computed is None:
                    if frozen is not None:
                        computed = frozen
                    elif source == "adaptive":
                        if model_logprobs is None:
                            raise ValueError("Adaptive masks require model token log probabilities.")
                        computed = build_adaptive_mask(inputs, model_logprobs,
                                                      mask_negative_diff=self.mask_negative_diff)
                    else:
                        computed = reference
                    if self.kind == "random":
                        computed = density_matched_random_mask(eligible, computed, indices, self.seed).to(eligible.device)
                compact = computed[row, positions].detach().cpu().clone()
                self.cache[index] = (signature, compact)
            restored = torch.zeros_like(eligible[row])
            restored[positions] = self.cache[index][1].to(restored.device)
            rows.append(restored & eligible[row])
        return torch.stack(rows)
