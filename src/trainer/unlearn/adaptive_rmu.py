"""Adaptive RMU trainer that leverages token-level log probabilities."""

import json
from pathlib import Path

import torch
import torch.nn.functional as F

from data.utils import IGNORE_INDEX
from trainer.unlearn.rmu import RMU


class AdaptiveRMU(RMU):
    """
    RMU variant that adapts the forget-side activation loss to tokens whose log probability
    gap (stored vs. current model) falls below the average gap over supervised tokens.
    """

    REQUIRED_FIELD = "base_logprobs"
    MAX_SAVED_SAMPLES = 20
    SAVE_DIR = Path("./saves")
    enable_adaptive_logging = False

    @staticmethod
    def _compute_model_token_logprobs(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
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
            gathered[:, 1:] = log_probs[:, :-1].gather(dim=2, index=token_indices).squeeze(-1)

        return gathered

    def _build_adaptive_mask(
        self,
        forget_inputs: dict,
        model_logprobs: torch.Tensor,
    ) -> torch.BoolTensor:
        if self.REQUIRED_FIELD not in forget_inputs:
            raise KeyError(
                f"AdaptiveRMU requires the '{self.REQUIRED_FIELD}' tensor from the data collator."
            )

        base_logprobs = forget_inputs[self.REQUIRED_FIELD].to(model_logprobs.device)
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
            masked_diff = diff.masked_fill(~valid_positions, 0.0)
            sum_diff = masked_diff.sum(dim=1)
            mean_diff = torch.zeros(
                diff.size(0), dtype=diff.dtype, device=diff.device
            )
            mean_diff[has_valid] = sum_diff[has_valid] / valid_counts[has_valid].to(diff.dtype)
            row_mask = valid_positions & (diff < mean_diff.unsqueeze(1))

            adaptive_mask[has_valid] = valid_positions[has_valid]
            has_strict = row_mask.any(dim=1)
            if has_strict.any():
                adaptive_mask[has_strict] = row_mask[has_strict]

        if getattr(self, "enable_adaptive_logging", False):
            tokenizer = getattr(self, "tokenizer", None)
            input_ids = forget_inputs.get("input_ids")
            save_limit = getattr(self, "_adaptive_save_limit", self.MAX_SAVED_SAMPLES)
            should_collect = (
                tokenizer is not None
                and input_ids is not None
                and getattr(self, "_adaptive_saved_samples", 0) < save_limit
            )
            if should_collect:
                adaptive_mask_cpu = adaptive_mask.detach().cpu()
                valid_positions_cpu = valid_positions.detach().cpu()
                diff_cpu = diff.detach().cpu()
                for row in range(base_mask.size(0)):
                    row_base_mask = base_mask[row]
                    sample_payload = {"row": int(row)}
                    row_input_ids = input_ids[row].detach().cpu()
                    base_indices = torch.where(row_base_mask)[0]
                    base_token_ids = row_input_ids[base_indices.detach().cpu()].tolist()
                    sample_payload["base_token_ids"] = base_token_ids
                    try:
                        sample_payload["base_tokens"] = tokenizer.convert_ids_to_tokens(base_token_ids)
                    except AttributeError:
                        sample_payload["base_tokens"] = base_token_ids
                    sample_payload["decoded_base"] = (
                        tokenizer.decode(base_token_ids, skip_special_tokens=False) if base_token_ids else ""
                    )

                    valid_positions_row = valid_positions_cpu[row]
                    if valid_positions_row.any():
                        valid_indices = torch.where(valid_positions_row)[0]
                        diffs_row = diff_cpu[row][valid_indices]
                        sort_order = torch.argsort(diffs_row)
                        sorted_indices = valid_indices[sort_order]
                        sorted_diffs = diff_cpu[row][sorted_indices].tolist()
                        sorted_token_ids = row_input_ids[sorted_indices.detach().cpu()].tolist()
                        sample_payload["sorted_token_ids"] = sorted_token_ids
                        sample_payload["sorted_diffs"] = sorted_diffs
                        try:
                            sample_payload["sorted_tokens"] = tokenizer.convert_ids_to_tokens(sorted_token_ids)
                        except AttributeError:
                            sample_payload["sorted_tokens"] = sorted_token_ids

                    adaptive_row_cpu = adaptive_mask_cpu[row]
                    if adaptive_row_cpu.any():
                        selected_indices = torch.where(adaptive_row_cpu)[0]
                        selected_token_ids = row_input_ids[selected_indices.detach().cpu()].tolist()
                        sample_payload["selected_token_ids"] = selected_token_ids
                        try:
                            sample_payload["selected_tokens"] = tokenizer.convert_ids_to_tokens(selected_token_ids)
                        except AttributeError:
                            sample_payload["selected_tokens"] = selected_token_ids
                        sample_payload["selected_diffs"] = diff_cpu[row][selected_indices].tolist()
                    self._save_adaptive_sample(sample_payload)

        return adaptive_mask

    def _save_adaptive_sample(self, payload: dict) -> None:
        if not getattr(self, "enable_adaptive_logging", False):
            return

        saved_count = getattr(self, "_adaptive_saved_samples", 0)
        save_limit = getattr(self, "_adaptive_save_limit", self.MAX_SAVED_SAMPLES)
        if saved_count >= save_limit:
            return

        save_dir = getattr(self, "_adaptive_save_dir", None)
        if save_dir is None:
            save_dir = self.SAVE_DIR
            setattr(self, "_adaptive_save_dir", save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        file_path = save_dir / f"adaptive_rmu_sample_{saved_count:02d}.json"
        with file_path.open("w", encoding="utf-8") as sink:
            json.dump(payload, sink, ensure_ascii=False, indent=2)

        setattr(self, "_adaptive_saved_samples", saved_count + 1)

    def compute_forget_loss(self, model, forget_inputs):
        model_inputs = {
            "input_ids": forget_inputs["input_ids"],
            "attention_mask": forget_inputs["attention_mask"],
            "labels": forget_inputs["labels"],
        }

        model_forget_activations, forget_outputs = self.forward_with_cache(
            model, model_inputs, self.model_module, no_grad=False
        )
        control_vec = forget_inputs.get(
            "control_vec", self.get_control_vector(model_forget_activations.shape[-1])
        )
        model_logprobs = self._compute_model_token_logprobs(
            forget_outputs.logits, model_inputs["input_ids"]
        ).detach()

        adaptive_mask = self._build_adaptive_mask(
            forget_inputs,
            model_logprobs,
        ).to(model_forget_activations.device)

        target_bs = model_forget_activations.size(0)
        if adaptive_mask.size(0) != target_bs:
            adaptive_mask = adaptive_mask[:target_bs]
            model_logprobs = model_logprobs[:target_bs]
            model_inputs = {k: v[:target_bs] for k, v in model_inputs.items()}

        control_vec = control_vec.to(
            dtype=model_forget_activations.dtype,
            device=model_forget_activations.device,
        )
        control_vec = control_vec.expand_as(model_forget_activations)
        if control_vec.size(0) != target_bs:
            control_vec = control_vec[:target_bs]

        forget_loss = self.compute_activation_loss(
            model_forget_activations, control_vec, adaptive_mask
        )

        return forget_loss, forget_outputs

    def compute_loss(self, model, inputs, return_outputs=False):
        forget_inputs = inputs["forget"]
        retain_raw_inputs = inputs["retain"]

        forget_loss, forget_outputs = self.compute_forget_loss(model, forget_inputs)

        retain_inputs = {
            "input_ids": retain_raw_inputs["input_ids"],
            "attention_mask": retain_raw_inputs["attention_mask"],
            "labels": retain_raw_inputs["labels"],
        }
        retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)
        loss = self.gamma * forget_loss + self.alpha * retain_loss
        return (loss, forget_outputs) if return_outputs else loss
