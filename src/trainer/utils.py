import torch
from torch import nn
import torch.nn.functional as F


def compute_kl_divergence(model, target_model, inputs):
    with torch.no_grad():
        ref_outputs = target_model(**inputs)

    ref_probs = F.log_softmax(ref_outputs.logits, dim=-1)
    ref_probs = F.log_softmax(ref_outputs.logits, dim=-1)
    ref_probs = ref_probs.view(-1, ref_outputs.logits.shape[-1])

    outputs = model(**inputs)
    current_probs = F.log_softmax(outputs.logits, dim=-1)
    current_probs = current_probs.view(-1, outputs.logits.shape[-1])

    # minimum KL divergence
    return nn.functional.kl_div(
        current_probs, ref_probs, reduction="batchmean", log_target=True
    ), outputs

def compute_sequential_kl(model, target_model, inputs):
    """
    Compute KL divergence considering token dependencies.

    Args:
        model: The fine-tuned LLM.
        target_model: The reference/original LLM.
        inputs: A dictionary containing input tensors (e.g., input_ids, attention_mask).

    Returns:
        Tuple containing (KL loss, model outputs)
    """
    with torch.no_grad():
        ref_outputs = target_model(**inputs)
    
    model_outputs = model(**inputs)

    # Get log probabilities for both models
    log_probs_model = F.log_softmax(model_outputs.logits, dim=-1)
    log_probs_ref = F.log_softmax(ref_outputs.logits, dim=-1)

    # Get token probabilities (exp(log_probs) to get p(theta))
    probs_model = log_probs_model.exp()

    # Shift log probabilities left for next-token conditioning
    log_probs_model_shifted = log_probs_model[:, :-1, :]
    log_probs_ref_shifted = log_probs_ref[:, :-1, :]

    # Compute KL divergence for the first token in the sequence
    kl_first_token = F.kl_div(
        log_probs_model[:, 0, :], log_probs_ref[:, 0, :], reduction="batchmean", log_target=True
    )

    # Compute the expectation term for the second KL divergence
    kl_next_tokens = (probs_model[:, :-1, :] * (
        log_probs_model_shifted - log_probs_ref_shifted
    )).sum(dim=-1).mean()

    return kl_first_token + kl_next_tokens, model_outputs

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
