import copy
from trainer.utils import compute_kl_divergence
from trainer.unlearn.grad_diff import GradDiff

class GradDiffRev(GradDiff):
    def __init__(self, gamma=1.0, alpha=1.0, retain_loss_type="reverse_KL", *args, **kwargs):
        """
        Initialize GradDiffRev, which focuses on Reverse KL divergence.

        Args:
            gamma (float): Scaling factor for the forget loss. Default is 1.0.
            alpha (float): Scaling factor for the retain loss. Default is 1.0.
            retain_loss_type (str): Specifies the loss type for retaining knowledge. Default is "reverse_KL".
        """
        super().__init__(gamma=gamma, alpha=alpha, retain_loss_type=retain_loss_type, *args, **kwargs)

    def compute_retain_loss(self, model, retain_inputs):
        """
        Computes the retain loss using Reverse KL divergence between the model and reference model.

        Args:
            model: The model being trained.
            retain_inputs: Inputs for the retain loss computation.

        Returns:
            retain_loss (float): The computed retain loss using reverse KL divergence.
        """
        retain_outputs = model(**retain_inputs)
        retain_loss = 0.0
        # Compute Reverse KL divergence between the ref_model and the model
        kl_loss, retain_outputs = compute_kl_divergence(
            self.ref_model, self.model, retain_inputs  # Reverse KL: ref_model vs model
        )

        retain_loss += kl_loss
        return retain_loss

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute the total loss, which includes both forget and retain losses.

        Args:
            model: The model being trained.
            inputs (dict): Contains both "forget" and "retain" data.
            return_outputs (bool): Whether to return the model outputs in addition to the loss.

        Returns:
            loss (float): The total loss for the current step.
            forget_outputs (optional): The model's outputs from the forget phase if return_outputs is True.
        """
        forget_inputs = inputs["forget"]
        forget_inputs = {
            "input_ids": forget_inputs["input_ids"],
            "attention_mask": forget_inputs["attention_mask"],
            "labels": forget_inputs["labels"],
        }

        forget_outputs = model(**forget_inputs)
        forget_loss = -forget_outputs.loss

        retain_inputs = inputs["retain"]
        retain_inputs = {
            "input_ids": retain_inputs["input_ids"],
            "attention_mask": retain_inputs["attention_mask"],
            "labels": retain_inputs["labels"],
        }
        retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)

        loss = self.gamma * forget_loss + self.alpha * retain_loss

        return (loss, forget_outputs) if return_outputs else loss
