import re
import torch
import deepspeed
from trainer.unlearn.grad_diff import GradDiff


class EntityRMU(GradDiff):
    def __init__(
        self,
        module_regex=r"model\.layers\.7",
        trainable_params_regex=[r"model\.layers\.(5|6|7)\.mlp\.down_proj\.weight"],
        steering_coeff=20,
        *args,
        **kwargs,
    ):
        """
        RMU Trainer that fine-tunes only specific layers and parameters using regex-based filtering.
        Applies entity-aware masking ONLY to the forget loss.

        Args:
            module_regex (str): Regex pattern to match module names (single module expected).
            trainable_params_regex (list[str]): Regex patterns for trainable parameters.
            steering_coeff (float): L2 norm scale for the steering control vector.
        """
        super().__init__(*args, **kwargs)

        # Create reference model if not already set
        if self.ref_model is None:
            self.ref_model = self._prepare_ref_model(self.model)

        # Store selection rules
        self.trainable_params_regex = trainable_params_regex
        self.module_regex = module_regex

        # Resolve target module to steer in both student and ref
        self.model_module = self._get_matching_module(self.model, self.module_regex)
        self.ref_module = self._get_matching_module(self.ref_model, self.module_regex)

        self.steering_coeff = steering_coeff
        self.control_vec = None

    # ---------- optimizer & param selection ----------
    def create_optimizer(self):
        # Freeze everything; unfreeze only regex-matched params; then delegate
        self._freeze_all_params(self.model, False)
        self._set_trainable_params(self.model, self.trainable_params_regex, True)
        super().create_optimizer()
        # Keep the rest as in your original implementation:
        self._freeze_all_params(self.model, True)

    def _get_matching_module(self, model, module_regex):
        """Returns a single module matching the given regex from a DeepSpeed/DDP-wrapped model."""
        if isinstance(model, deepspeed.DeepSpeedEngine):
            model = model.module  # unwrap

        matched_modules = {
            name: module
            for name, module in model.named_modules()
            if re.fullmatch(module_regex, name)
        }

        if len(matched_modules) > 1:
            raise ValueError(
                f"More than one module matched with {module_regex}: {list(matched_modules.keys())}"
            )
        elif not matched_modules:
            raise ValueError(f"No module matched with {module_regex}")

        return next(iter(matched_modules.values()))

    def _freeze_all_params(self, model, requires_grad=True):
        """Freeze or unfreeze all parameters."""
        for param in model.parameters():
            param.requires_grad = requires_grad

    def _set_trainable_params(self, model, trainable_params_regex, requires_grad=True):
        """Unfreeze specific parameters that match the regex patterns."""
        for name, param in model.named_parameters():
            if any(re.fullmatch(pattern, name) for pattern in trainable_params_regex):
                param.requires_grad = requires_grad
                # print(f"{name}:requires_grad\t{requires_grad}")

    # ---------- forward & caches ----------
    def forward_with_cache(self, model, inputs, module, no_grad=True):
        """Performs a forward pass while caching the output of a specified module."""
        cache = []

        def hook(module, input, output):
            if isinstance(output, tuple):
                cache.append(output[0])
            else:
                cache.append(output)
            return None

        hook_handle = module.register_forward_hook(hook)
        with torch.set_grad_enabled(not (no_grad)):
            outputs = model(**inputs)
        hook_handle.remove()
        if not cache:
            raise RuntimeError("Hook did not capture any output from the target module.")
        return cache[0], outputs

    # ---------- steering vector ----------
    def get_control_vector(self, dim):
        if self.control_vec is None:
            random_vector = torch.rand(1, 1, dim)
            self.control_vec = (
                random_vector / torch.norm(random_vector) * self.steering_coeff
            )
        return self.control_vec

    # ---------- losses ----------
    def compute_activation_loss(self, activation1, activation2, mask):
        squared_diff = torch.nn.functional.mse_loss(
            activation1, activation2, reduction="none"
        )  # [B, S, D]
        expanded_mask = mask.unsqueeze(-1).expand_as(squared_diff)  # [B, S, D]
        squared_diff_sum = (squared_diff * expanded_mask).mean(dim=2).sum(dim=(1))  # [B]
        num_tokens = mask.sum(dim=-1, keepdim=True)  # [B, 1]
        return (squared_diff_sum / num_tokens.clamp_min(1)).mean()

    def compute_retain_loss(self, model, retain_inputs):
        retain_loss = 0.0

        if self.retain_loss_type == "EMBED_DIFF":
            model_retain_activations, _ = self.forward_with_cache(
                model, retain_inputs, module=self.model_module, no_grad=False
            )
            ref_retain_activations, _ = self.forward_with_cache(
                self.ref_model, retain_inputs, module=self.ref_module, no_grad=True
            )
            mask = retain_inputs["labels"] != -100  # [B, S]
            retain_loss = self.compute_activation_loss(
                model_retain_activations,
                ref_retain_activations.to(model_retain_activations.device),
                mask,
            )
        else:
            retain_loss = super().compute_retain_loss(model, retain_inputs)
        return retain_loss

    # ---------- main criterion ----------
    def compute_loss(self, model, inputs, return_outputs=False):
        # --- FORGET branch (entity-masked) ---
        forget_inputs = inputs["forget"]
        forget_inputs = {
            "input_ids": forget_inputs["input_ids"],
            "attention_mask": forget_inputs["attention_mask"],
            "labels": forget_inputs["labels"],
        }

        model_forget_activations, forget_outputs = self.forward_with_cache(
            model, forget_inputs, self.model_module, no_grad=False
        )

        control_vec = inputs["forget"].get(
            "control_vec", self.get_control_vector(model_forget_activations.shape[-1])
        )
        control_vec = control_vec.to(
            dtype=model_forget_activations.dtype, device=model_forget_activations.device
        )
        control_vec = control_vec.expand_as(model_forget_activations)

        # Entity-aware mask ONLY here (forget side)
        labels_mask = forget_inputs["labels"] != -100  # [B, S]
        ent_mask = inputs["forget"].get("entity_mask", labels_mask)  # [B, S] bool
        mask = labels_mask & ent_mask  # [B, S]

        forget_loss = self.compute_activation_loss(
            model_forget_activations, control_vec, mask
        )

        # --- RETAIN branch (unchanged) ---
        retain_inputs = inputs["retain"]
        retain_inputs = {
            "input_ids": retain_inputs["input_ids"],
            "attention_mask": retain_inputs["attention_mask"],
            "labels": retain_inputs["labels"],
        }
        retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)

        loss = self.gamma * forget_loss + self.alpha * retain_loss
        return (loss, forget_outputs) if return_outputs else loss
