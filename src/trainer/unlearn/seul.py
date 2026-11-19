"""Paper-based SeUL (Wang et al., AAAI 2025), adapted to TOFU answer labels.

Eq. 1: MINIMIZE the SUM of selected token log probabilities per example.
Eq. 2: score with a FROZEN copy of the original, pre-unlearning target.
The prose specifies a mean threshold; the displayed sum lacks /T. We follow
the prose. Batch reduction/empty handling are explicit implementation choices.
"""
import copy
import math

import torch
import torch.nn.functional as F

from trainer.masking import eligible_positions, nonempty_mean, sample_indices
from trainer.unlearn.base import UnlearnTrainer


def label_logprobs(logits, labels):
    """Output at label position j is log p(label[j] | prefix[:j])."""
    values = -F.cross_entropy(logits[:, :-1].float().transpose(1, 2),
                              labels[:, 1:], ignore_index=-100, reduction="none")
    return F.pad(values, (1, 0), value=0.0)


def select_spans(scores, eligible, threshold="mean", alpha=None):
    """Strict threshold, then fill the one-token gap between distance-2 seeds.

    Fill is based on ORIGINAL selected endpoints, never iterated or extended
    across a prompt, padding or other ineligible label. Adjacent selected tokens
    form maximal continuous spans; a binary union prevents double counting.
    """
    if scores.shape != eligible.shape or scores.ndim != 2:
        raise ValueError("Expected equally shaped [batch, sequence] arrays")
    if threshold not in {"mean", "fixed"}:
        raise ValueError("threshold must be mean or fixed")
    if threshold == "fixed" and (alpha is None or not math.isfinite(float(alpha))):
        raise ValueError("fixed threshold requires finite log-probability alpha")
    valid = eligible.bool()
    if not torch.isfinite(scores[valid]).all():
        raise ValueError("Non-finite original-model score at an eligible label")
    count = valid.sum(-1)
    means = torch.where(valid, scores, 0.0).sum(-1) / count.clamp_min(1)
    cutoff = means if threshold == "mean" else torch.full_like(means, float(alpha))
    seeds = valid & (scores < cutoff[:, None])
    selected = seeds.clone()
    selected[:, 1:-1] |= seeds[:, :-2] & seeds[:, 2:] & valid[:, 1:-1]
    return selected & valid, seeds, cutoff


def span_intervals(mask):
    """Half-open maximal spans in the input-aligned label coordinate system."""
    result = []
    start = None
    for i, value in enumerate([*mask, False]):
        if value and start is None:
            start = i
        elif not value and start is not None:
            result.append([start, i])
            start = None
    return result


def selective_ga_loss(logits, labels, selected, eligible):
    if selected.shape != labels.shape or (selected & ~eligible).any():
        raise ValueError("SeUL selection must be a subset of eligible labels")
    logps = label_logprobs(logits, labels)
    sums = torch.where(selected, logps, 0.0).sum(-1)
    return nonempty_mean(sums, selected.any(-1))


class SeUL(UnlearnTrainer):
    """Original-model confidence + gap filling + summed selective GA, no retain loss."""

    def __init__(self, *args, threshold="mean", alpha=None, **kwargs):
        if threshold not in {"mean", "fixed"}:
            raise ValueError("Unknown SeUL threshold")
        if threshold == "fixed" and (alpha is None or not math.isfinite(float(alpha))):
            raise ValueError("A finite alpha is required for a fixed threshold")
        super().__init__(*args, **kwargs)
        self.threshold, self.alpha = threshold, alpha
        self.selection_model = copy.deepcopy(self.model)
        self.selection_model.requires_grad_(False)
        self.selection_model.eval()
        if self.is_deepspeed_enabled:
            self.selection_model = self._prepare_deepspeed(self.selection_model)
        else:
            self.selection_model = self.accelerator.prepare_model(
                self.selection_model, evaluation_mode=True)
        self.selection_audit = {}

    @torch.no_grad()
    def original_selection(self, forget, verify_original=False):
        if "reference_mask" in forget or "frozen_random_mask" in forget:
            raise ValueError("SeUL computes its own original-model selection")
        self.selection_model.eval()
        eligible = eligible_positions(forget)
        selected = torch.zeros_like(eligible)
        for row, index in enumerate(sample_indices(forget)):
            attended = forget.get("attention_mask", torch.ones_like(forget["labels"]))[row].bool()
            positions = attended.nonzero().flatten()
            if not len(positions) or not torch.equal(positions,torch.arange(
                    positions[0],positions[-1]+1,device=positions.device)):
                raise ValueError("SeUL expects a nonempty contiguous attended sequence")
            signature = {"doc_id": index,
                         "input_ids": forget["input_ids"][row, attended].tolist(),
                         "labels": forget["labels"][row, attended].tolist(),
                         "eligible": eligible[row, attended].tolist()}
            previous = self.selection_audit.get(index)
            if previous is not None and any(previous[k] != signature[k] for k in signature):
                raise ValueError(f"Sample ID {index} reused for different content/labels")
            if previous is None or verify_original:
                # Canonical single-example scoring removes BF16 padding/batch
                # dependence. Cache the fixed-original-model rule, never scores
                # from the evolving target. This also avoids repeated inference.
                compact = {"input_ids":forget["input_ids"][row,attended][None],
                           "attention_mask":torch.ones_like(forget["input_ids"][row,attended][None]),
                           "labels":forget["labels"][row,attended][None]}
                forward = {k:compact[k] for k in ("input_ids","attention_mask")}
                if "position_ids" in forget:
                    forward["position_ids"]=forget["position_ids"][row,attended][None]
                logits = self.selection_model(**forward,use_cache=False).logits
                scores = label_logprobs(logits,compact["labels"])
                valid = eligible_positions(compact)
                mask,seeds,cutoff = select_spans(scores,valid,self.threshold,self.alpha)
                record = dict(signature,
                    original_logprobs=[float(s) if v else None for s,v in zip(scores[0].tolist(),valid[0].tolist())],
                    mask=mask[0].int().tolist(),seed_mask=seeds[0].int().tolist(),threshold=float(cutoff[0]))
                record["spans"]=span_intervals(record["mask"])
                if previous is not None and any(previous[k]!=record[k] for k in ("mask","seed_mask","spans")):
                    raise ValueError(f"Frozen-original-model selection changed for sample {index}")
                self.selection_audit[index]=record
            selected[row,attended]=torch.tensor(self.selection_audit[index]["mask"],device=selected.device,dtype=torch.bool)
        return selected

    def compute_loss(self, model, inputs, return_outputs=False):
        forget = inputs["forget"]
        selected = self.original_selection(forget)
        forward = {k: forget[k] for k in ("input_ids", "attention_mask", "position_ids")
                   if k in forget}
        outputs = model(**forward, use_cache=False)
        loss = selective_ga_loss(outputs.logits, forget["labels"], selected,
                                 eligible_positions(forget))
        return (loss, outputs) if return_outputs else loss
