"""Launch the frozen trainer with per-rank finite loss and gradient checks."""
from functools import wraps
from pathlib import Path

import torch

import train_mask_experiment as entry
from freeze_mask_protocol import save_json

original_load = entry.load_trainer
checks = {}


def checked_load(*args, **kwargs):
    trainer, training_args = original_load(*args, **kwargs)
    checks.update(loss=0, grad_norm=0)
    if training_args.max_grad_norm <= 0:
        raise ValueError("Numerical validation requires a gradient norm at each optimizer update")

    def validate(value, name):
        finite = torch.isfinite(torch.as_tensor(value).detach()).all()
        failed = (~finite).to(device=trainer.accelerator.device, dtype=torch.int32)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(failed, op=torch.distributed.ReduceOp.MAX)
        if failed.item():
            raise FloatingPointError(f"Nonfinite {name} before optimizer update {trainer.state.global_step + 1}")
        checks[name] += 1

    compute_loss = trainer.compute_loss
    clip_grad_norm = trainer.accelerator.clip_grad_norm_

    @wraps(compute_loss)
    def checked_loss(*args, **kwargs):
        result = compute_loss(*args, **kwargs)
        validate(result[0] if isinstance(result, tuple) else result, "loss")
        return result

    @wraps(clip_grad_norm)
    def checked_clip(*args, **kwargs):
        norm = clip_grad_norm(*args, **kwargs)
        validate(norm, "grad_norm")
        return norm

    trainer.compute_loss = checked_loss
    trainer.accelerator.clip_grad_norm_ = checked_clip
    checks.update(output_dir=training_args.output_dir, rank=training_args.process_index,
                  expected_loss=training_args.max_steps * training_args.gradient_accumulation_steps,
                  expected_grad_norm=training_args.max_steps)
    return trainer, training_args


if __name__ == "__main__":
    entry.load_trainer = checked_load
    entry.main()
    if checks["loss"] != checks["expected_loss"] or checks["grad_norm"] != checks["expected_grad_norm"]:
        raise RuntimeError(f"Incomplete numerical checks: {checks}")
    save_json(Path(checks["output_dir"]) / f"finite_checks_rank{checks['rank']}.json", checks)
