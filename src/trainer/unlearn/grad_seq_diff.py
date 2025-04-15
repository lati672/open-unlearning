import copy
from trainer.utils import compute_sequence_loss
from trainer.unlearn.grad_diff import GradDiff

class GradSeqDiff(GradDiff):
    def __init__(self, gamma=1.0, alpha=1.0, beta=1.0, *args, **kwargs):
        """
        Initialize GradDiffRev, which focuses on Reverse KL divergence.

        Args:
            gamma (float): Scaling factor for the forget loss. Default is 1.0.
            alpha (float): Scaling factor for the retain loss. Default is 1.0.
        """
        # Make sure retain_loss_type is not passed again in kwargs
        super().__init__(gamma=gamma, alpha=alpha, *args, **kwargs)
        if beta > 1.0 or beta < 0.0:
            raise ValueError("beta must be between 0 and 1")
        self.beta = beta

    def compute_retain_loss(self, model, retain_inputs):
        """
        Computes the retain loss using Reverse KL divergence between the model and reference model.

        Args:
            model: The model being trained.
            retain_inputs: Inputs for the retain loss computation.

        Returns:
            retain_loss (float): The computed retain loss using reverse KL divergence.
        """
        #retain_outputs = model(**retain_inputs)
        loss, retain_outputs = compute_sequence_loss(
            self.model, self.ref_model, retain_inputs, self.beta
        )

        return loss

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
