import random
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from data.utils import IGNORE_INDEX


def compute_model_token_logprobs(
    logits: torch.Tensor, input_ids: torch.Tensor
) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    batch_size, seq_len = input_ids.size()
    device = logits.device
    gathered = torch.full(
        (batch_size, seq_len),
        float("nan"),
        dtype=log_probs.dtype,
        device=device,
    )

    if seq_len > 1:
        token_indices = input_ids[:, 1:].unsqueeze(-1)
        gathered[:, 1:] = (
            log_probs[:, :-1]
            .gather(dim=2, index=token_indices)
            .squeeze(-1)
        )

    return gathered


def build_adaptive_mask(
    forget_inputs: dict,
    model_logprobs: torch.Tensor,
    *,
    mask_negative_diff: bool,
) -> torch.BoolTensor:
    required_field = "base_logprobs"
    if required_field not in forget_inputs:
        raise KeyError(
            f"AdaptiveRMU requires the '{required_field}' tensor from the data collator."
        )

    base_logprobs = forget_inputs[required_field].to(model_logprobs.device)
    labels = forget_inputs["labels"]
    attention_mask = forget_inputs.get("attention_mask")

    base_mask = labels.ne(IGNORE_INDEX)
    if attention_mask is not None:
        base_mask = base_mask & attention_mask.bool()

    diff = base_logprobs - model_logprobs
    valid_positions = base_mask & torch.isfinite(diff)
    valid_counts = valid_positions.sum(dim=1)
    has_valid = valid_counts > 0

    adaptive_mask = base_mask.clone()
    if has_valid.any():
        adaptive_mask[has_valid] = valid_positions[has_valid]
        if mask_negative_diff:
            negative_mask = valid_positions & (diff < 0)
            has_negative = negative_mask.any(dim=1)
            if has_negative.any():
                adaptive_mask[has_negative] = negative_mask[has_negative]
        else:
            masked_diff = diff.masked_fill(~valid_positions, 0.0)
            sum_diff = masked_diff.sum(dim=1)
            mean_diff = torch.zeros(
                diff.size(0), dtype=diff.dtype, device=diff.device
            )
            mean_diff[has_valid] = sum_diff[has_valid] / valid_counts[has_valid].to(
                diff.dtype
            )
            row_mask = valid_positions & (diff < mean_diff.unsqueeze(1))

            has_strict = row_mask.any(dim=1)
            if has_strict.any():
                adaptive_mask[has_strict] = row_mask[has_strict]

    return adaptive_mask


def compute_sequence_loss(model, target_model, inputs, beta=1):
    input_ids = inputs["input_ids"]

    with torch.no_grad():
        ref_outputs = target_model(**inputs)

    outputs = model(**inputs)
    ref_logits = ref_outputs.logits
    logits = outputs.logits

    probs = F.softmax(logits, dim=-1)  # shape: [batch_size, seq_len, vocab_size]

    batch_size, seq_len, vocab_size = probs.shape
    token_importance = torch.ones((batch_size, seq_len), device=probs.device)


    for i in range(1, seq_len):
        prev_token_ids = input_ids[:, i - 1]
        token_importance[:, i] += probs[torch.arange(batch_size), i, prev_token_ids] *  vocab_size * beta
    
    kl_per_token = F.kl_div(
        F.log_softmax(logits, dim=-1),
        F.log_softmax(ref_logits, dim=-1),
        reduction='none',
        log_target=True,
    ).sum(dim=-1)
    
    kl_per_token = kl_per_token * inputs["attention_mask"].float()  # shape: [batch_size, seq_len]
    weighted_token_loss = (token_importance * kl_per_token).mean()
    return weighted_token_loss, outputs

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_kl_divergence(model, target_model, inputs):
    with torch.no_grad():
        ref_outputs = target_model(**inputs)

    ref_probs = F.log_softmax(ref_outputs.logits, dim=-1)
    #ref_probs = F.log_softmax(ref_outputs.logits, dim=-1)
    ref_probs = ref_probs.view(-1, ref_outputs.logits.shape[-1])

    outputs = model(**inputs)
    current_probs = F.log_softmax(outputs.logits, dim=-1)
    current_probs = current_probs.view(-1, outputs.logits.shape[-1])

    # minimum KL divergence
    return nn.functional.kl_div(
        current_probs, ref_probs, reduction="batchmean", log_target=True
    ), outputs

def compute_weighted_kl(model, target_model, inputs):
    with torch.no_grad():
        ref_outputs = target_model(**inputs)

    ref_probs = F.log_softmax(ref_outputs.logits, dim=-1)
    ref_probs = ref_probs.view(-1, ref_outputs.logits.shape[-1])

    outputs = model(**inputs)
    current_probs = F.log_softmax(outputs.logits, dim=-1)
    current_probs = current_probs.view(-1, outputs.logits.shape[-1])

    # minimum KL divergence
    return nn.functional.kl_div(
        current_probs, ref_probs, reduction="batchmean", log_target=True
    ), outputs


def compute_batch_nll(model, inputs):
    # get the sum loss for each sequence in a batch
    # NOTE: not same as model(**inputs).loss but has sum loss for each seq in a batch
    outputs = model(**inputs)
    logits = outputs.logits
    labels = inputs["labels"]
    shifted_labels = labels[..., 1:].contiguous()
    logits = logits[..., :-1, :].contiguous()
    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    loss = loss_function(logits.transpose(-1, -2), shifted_labels).sum(dim=-1)
    return loss, outputs


def compute_dpo_loss(model, ref_model, win_inputs=None, lose_inputs=None, beta=1.0):
    if win_inputs is None and lose_inputs is None:
        raise ValueError("Both win_inputs and lose_inputs can't be None")

    win_log_ratio, lose_log_ratio = 0.0, 0.0
    win_outputs, lose_outputs = None, None

    if win_inputs is not None:
        win_loss, win_outputs = compute_batch_nll(model, win_inputs)
        with torch.no_grad():
            win_ref_loss, _ = compute_batch_nll(ref_model, win_inputs)
        win_log_ratio = -(win_loss - win_ref_loss)

    if lose_inputs is not None:
        lose_loss, lose_outputs = compute_batch_nll(model, lose_inputs)
        with torch.no_grad():
            lose_ref_loss, _ = compute_batch_nll(ref_model, lose_inputs)
        lose_log_ratio = -(lose_loss - lose_ref_loss)

    loss = -2 / beta * F.logsigmoid(beta * (win_log_ratio - lose_log_ratio)).mean()
    return loss, (win_outputs, lose_outputs)


def compute_masked_batch_nll(
    model,
    inputs,
    *,
    token_mask: Optional[torch.Tensor] = None,
    outputs=None,
):
    if outputs is None:
        outputs = model(**inputs)
    logits = outputs.logits
    labels = inputs["labels"]
    shifted_labels = labels[..., 1:].contiguous()
    shift_logits = logits[..., :-1, :].contiguous()
    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    per_token_loss = loss_function(
        shift_logits.transpose(-1, -2), shifted_labels
    )

    if token_mask is not None:
        effective_mask = token_mask[..., 1:]
        if effective_mask.dtype != per_token_loss.dtype:
            effective_mask = effective_mask.to(per_token_loss.dtype)
        per_token_loss = per_token_loss * effective_mask.to(per_token_loss.device)

    loss = per_token_loss.sum(dim=-1)
    return loss, outputs


def compute_masked_dpo_loss(
    model,
    ref_model,
    win_inputs=None,
    lose_inputs=None,
    beta=1.0,
    win_mask: Optional[torch.Tensor] = None,
    lose_mask: Optional[torch.Tensor] = None,
    win_outputs=None,
    lose_outputs=None,
):
    if win_inputs is None and lose_inputs is None:
        raise ValueError("Both win_inputs and lose_inputs can't be None")

    win_log_ratio, lose_log_ratio = 0.0, 0.0

    if win_inputs is not None:
        win_loss, win_outputs = compute_masked_batch_nll(
            model,
            win_inputs,
            token_mask=win_mask,
            outputs=win_outputs,
        )
        with torch.no_grad():
            win_ref_loss, _ = compute_masked_batch_nll(
                ref_model,
                win_inputs,
                token_mask=win_mask,
            )
        win_log_ratio = -(win_loss - win_ref_loss)

    if lose_inputs is not None:
        lose_loss, lose_outputs = compute_masked_batch_nll(
            model,
            lose_inputs,
            token_mask=lose_mask,
            outputs=lose_outputs,
        )
        with torch.no_grad():
            lose_ref_loss, _ = compute_masked_batch_nll(
                ref_model,
                lose_inputs,
                token_mask=lose_mask,
            )
        lose_log_ratio = -(lose_loss - lose_ref_loss)

    loss = -2 / beta * F.logsigmoid(beta * (win_log_ratio - lose_log_ratio)).mean()
    return loss, (win_outputs, lose_outputs)


def compute_undial_loss(model, ref_model, inputs, beta):
    # Forward pass on the student (trainable) model
    outputs = model(**inputs)
    logits = outputs.logits
    labels = inputs["labels"]

    shift_labels = labels[..., 1:].contiguous()
    shift_logits = logits[..., :-1, :].contiguous()

    # Forward pass on the teacher model (no grad)
    with torch.no_grad():
        teacher_logits = ref_model(**inputs).logits
    shift_teacher_logits = teacher_logits[..., :-1, :].contiguous()

    # Build the mask that identifies the tokens need to be unlearned
    mask = torch.zeros_like(shift_teacher_logits)
    batch_idx = torch.arange(mask.shape[0]).view(-1, 1, 1)
    seq_idx = torch.arange(mask.shape[1]).view(1, -1, 1)
    mask[batch_idx, seq_idx, shift_labels.unsqueeze(-1)] = 1.0

    # Adjust teacher logits: subtract di_strength on the correct token
    pre_softmax = shift_teacher_logits - mask * beta
    soft_label = F.softmax(pre_softmax, dim=-1)

    loss_fct = nn.CrossEntropyLoss(reduction="none")
    loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        soft_label.view(-1, soft_label.size(-1)),
    )
    return loss.mean(), outputs


def compute_wga_loss(model, inputs, beta):
    outputs = model(**inputs)
    labels = inputs["labels"]
    labels = labels.to(outputs.logits.device)

    shift_logits = outputs.logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    lm_loss = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")(
        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
    )
    weight_ce = ((-lm_loss).exp().detach()) ** beta
    forget_loss = -(weight_ce * lm_loss)[shift_labels.view(-1) != -100].mean()
    return forget_loss, outputs


def compute_satimp_loss(model, inputs, beta1, beta2):
    outputs = model(**inputs)
    labels = inputs["labels"]
    labels = labels.to(outputs.logits.device)

    shift_logits = outputs.logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    lm_loss = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")(
        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
    )
    weight_sat = ((-lm_loss).exp().detach()) ** beta1
    weight_imp = (1 - (-lm_loss).exp().detach()) ** beta2
    forget_loss = -((weight_sat * weight_imp) * lm_loss)[
        shift_labels.view(-1) != -100
    ].mean()
    return forget_loss, outputs
