"""Run a prepared SU variant with open-unlearning's dataset, collator and trainer."""

import argparse
from functools import wraps
import json
import math
import os
from pathlib import Path
import time

from freeze_mask_protocol import save_json, sha256
from prepare_su_ngram import OUTPUT, ROOT, VARIANTS, check_config, read
from train_mask_experiment import evaluate

from omegaconf import OmegaConf
import torch

from data.collators import DataCollatorForSupervisedDataset, FrozenMaskCollator
from data.unlearn import ForgetRetainDataset
from model import get_model
from prepare_su_ngram import data_and_tokenizer
from trainer import load_trainer
from trainer.utils import seed_everything


def load_prepared(root, variant):
    manifest = read(root / "protocol_manifest.json")
    hashes = read(root / "hashes.json")
    relative_paths = ["protocol_manifest.json", f"configs/{variant}.json", f"masks/{variant}.jsonl",
                      f"masks/{variant}.metadata.json"]
    for relative in relative_paths:
        if sha256(root / relative) != hashes[relative]:
            raise ValueError(f"Prepared artifact SHA-256 differs: {relative}")
    for name, expected in manifest["source_sha256"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"Prepared source SHA-256 differs: {name}")
    cfg = read(root / f"configs/{variant}.json")
    check_config(cfg)
    mask_path = root / f"masks/{variant}.jsonl"
    if sha256(mask_path) != manifest["mask_sha256"][variant]:
        raise ValueError(f"Frozen mask SHA-256 differs: {variant}")
    return cfg, mask_path


def model_config(cfg, template, checkpoint=None):
    settings = cfg["model"]
    return OmegaConf.create({
        "model_args": {
            "pretrained_model_name_or_path": str(checkpoint) if checkpoint else settings["target_id"],
            **({} if checkpoint else {"revision": settings["target_revision"]}),
            "torch_dtype": settings["dtype"], "attn_implementation": settings["attention"],
            "local_files_only": True,
        },
        "tokenizer_args": {
            "pretrained_model_name_or_path": str(checkpoint) if checkpoint else settings["tokenizer_id"],
            **({} if checkpoint else {"revision": settings["tokenizer_revision"]}),
            "local_files_only": True,
        },
        "template_args": OmegaConf.to_container(template, resolve=True),
    })


def finite_checks(trainer):
    stats = {"loss": [], "grad_norm": [], "selected_forget_tokens": 0, "nonempty_forget_samples": 0,
             "forget_samples_seen": 0}
    original_loss, original_clip = trainer.compute_loss, trainer.accelerator.clip_grad_norm_

    def validate(value, key):
        value = torch.as_tensor(value).detach()
        failed = (~torch.isfinite(value).all()).int().to(trainer.accelerator.device)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(failed, op=torch.distributed.ReduceOp.MAX)
        if failed.item():
            raise FloatingPointError(f"Nonfinite SU {key} at optimizer step {trainer.state.global_step + 1}")
        stats[key].append(value.item())

    @wraps(original_loss)
    def checked_loss(model, inputs, *args, **kwargs):
        result = original_loss(model, inputs, *args, **kwargs)
        validate(result[0] if isinstance(result, tuple) else result, "loss")
        mask = inputs["forget"]["reference_mask"]
        stats["selected_forget_tokens"] += int(mask.sum())
        stats["nonempty_forget_samples"] += int(mask.any(dim=-1).sum())
        stats["forget_samples_seen"] += len(mask)
        return result

    @wraps(original_clip)
    def checked_clip(*args, **kwargs):
        norm = original_clip(*args, **kwargs)
        validate(norm, "grad_norm")
        return norm

    trainer.compute_loss = checked_loss
    trainer.accelerator.clip_grad_norm_ = checked_clip
    return stats


def smoke_evaluate(cfg, template, original_tokenizer, output, device):
    step = cfg["smoke"]["max_steps"]
    checkpoint = output / f"checkpoint-{step}"
    state = read(checkpoint / "trainer_state.json")
    if state["global_step"] != step or state["max_steps"] != step:
        raise ValueError("Smoke checkpoint global step differs")
    required = ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "trainer_state.json"]
    if any(not (checkpoint / name).is_file() for name in required):
        raise ValueError("Smoke checkpoint is missing tokenizer, config or state")
    weights = sorted(checkpoint.glob("*.safetensors"))
    if not weights:
        raise ValueError("Smoke checkpoint has no on-disk weights")
    cfg_model = model_config(cfg, template, checkpoint)
    cfg_model.model_args.device_map = {"": str(device)}
    model, tokenizer = get_model(cfg_model)
    for field in ("eos_token_id", "pad_token_id", "bos_token_id", "chat_template"):
        if getattr(tokenizer, field) != getattr(original_tokenizer, field):
            raise ValueError(f"Reloaded tokenizer differs at {field}")
    eval_dir = checkpoint / "evals"
    eval_dir.mkdir()
    summaries = {}
    for role in ["forget", "retain"]:
        split = cfg["dataset"][role]
        path = eval_dir / f"{split}.json"
        summaries[split] = evaluate(model, tokenizer, template, split, path, batch_size=4,
                                    generate=True, dataset_revision=cfg["dataset"]["revision"],
                                    limit=cfg["smoke"]["eval_samples_per_split"],
                                    max_new_tokens=cfg["smoke"]["eval_max_new_tokens"])
        rows = read(path)["rows"]
        if len(rows) != cfg["smoke"]["eval_samples_per_split"] or any(
            not all(isinstance(row[key], (int, float)) and math.isfinite(row[key])
                    for key in ("mean_nll", "answer_probability", "rougeL_recall")) or not isinstance(row["generation"], str)
            for row in rows
        ):
            raise ValueError(f"Incomplete or nonfinite smoke evaluation: {split}")
    return {"offline_checkpoint_reload": True, "global_step": step, "weight_sha256": {p.name: sha256(p) for p in weights},
            "evaluation_sha256": {p.name: sha256(p) for p in eval_dir.glob("*.json")}, "evaluation": summaries}


def run(args):
    cfg, mask_path = load_prepared(args.root, args.variant)
    training = cfg["training"]
    smoke = args.mode == "smoke"
    schedule = cfg["smoke"] if smoke else training
    seed = schedule["seed"] if smoke else args.seed
    if seed not in training["seeds"] or int(os.environ.get("WORLD_SIZE", "1")) != training["world_size"]:
        raise ValueError(f"Expected a configured seed and torchrun with {training['world_size']} processes")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    seed_everything(seed)
    output = args.root / ("smoke_tests" if smoke else "runs") / args.variant / f"seed{seed}"
    if output.exists():
        raise FileExistsError(output)
    tokenizer, template, splits = data_and_tokenizer(cfg)
    dataset = ForgetRetainDataset(**splits)
    collator = FrozenMaskCollator(DataCollatorForSupervisedDataset(tokenizer, index="index"), mask_path)
    if set(collator.records) != set(range(len(splits["forget"]))):
        raise ValueError("Prepared mask IDs differ from the training dataset")

    cfg_model = model_config(cfg, template)
    model, loaded_tokenizer = get_model(OmegaConf.create(OmegaConf.to_container(cfg_model, resolve=True)))
    if (getattr(model.config, "_commit_hash", None) != cfg["model"]["target_revision"]
            or tokenizer.get_vocab() != loaded_tokenizer.get_vocab()):
        raise ValueError("Target revision or tokenizer differs")
    trainer_cfg = OmegaConf.load(ROOT / "configs/trainer/SelectiveGradientDifference.yaml")
    trainer_cfg.method_args.gamma_forget = cfg["objective"]["gamma_forget"]
    trainer_cfg.method_args.alpha_retain = cfg["objective"]["alpha_retain"]
    trainer_cfg.args = {
        "output_dir": str(output), "max_steps": schedule["max_steps"], "save_steps": schedule["save_steps"],
        "warmup_steps": schedule["warmup_steps"], "per_device_train_batch_size": training["batch_size"],
        "gradient_accumulation_steps": training["gradient_accumulation_steps"], "learning_rate": training["learning_rate"],
        "weight_decay": training["weight_decay"], "max_grad_norm": training["max_grad_norm"], "optim": training["optim"],
        "seed": seed, "bf16": True, "gradient_checkpointing": True, "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "ddp_find_unused_parameters": False, "remove_unused_columns": False, "logging_steps": 1 if smoke else 5,
        "logging_nan_inf_filter": False, "save_only_model": True, "save_safetensors": True, "save_strategy": "steps",
        "report_to": [],
    }
    trainer, trainer_args = load_trainer(trainer_cfg, model, train_dataset=dataset, tokenizer=tokenizer,
                                         data_collator=collator, template_args=template)
    checks = finite_checks(trainer)
    if trainer.is_world_process_zero():
        snapshot = {"config": cfg, "model": OmegaConf.to_container(cfg_model, resolve=True),
                    "seed": seed, "mask_sha256": sha256(mask_path), "protocol_sha256": sha256(args.root / "protocol_manifest.json"),
                    "training_args": trainer_args.to_dict()}
        save_json(output / "resolved_config.json", snapshot)
    trainer.accelerator.wait_for_everyone()
    start = time.monotonic()
    result = trainer.train()
    seconds = time.monotonic() - start
    steps = schedule["max_steps"]
    if trainer.state.global_step != steps or len(checks["loss"]) != steps * training["gradient_accumulation_steps"] or len(checks["grad_norm"]) != steps:
        raise ValueError(f"Unexpected optimizer steps or finite checks: {checks}")
    if not math.isfinite(result.training_loss):
        raise FloatingPointError("Nonfinite final training loss")
    rank = trainer_args.process_index
    save_json(output / f"finite_checks_rank{rank}.json", checks)
    trainer.accelerator.wait_for_everyone()
    if trainer.is_world_process_zero():
        for step in range(schedule["save_steps"], steps + 1, schedule["save_steps"]):
            checkpoint = output / f"checkpoint-{step}"
            if read(checkpoint / "trainer_state.json")["global_step"] != step:
                raise ValueError(f"Wrong checkpoint global step at {checkpoint}")
            save_json(checkpoint / "resolved_config.json", snapshot)
        summary = {"variant": args.variant, "seed": seed, "global_step": steps, "train_wall_seconds": seconds,
                   "training_loss": result.training_loss, "mask_sha256": snapshot["mask_sha256"], "all_finite": True}
        if smoke:
            summary.update(smoke_evaluate(cfg, template, tokenizer, output, trainer.accelerator.device))
            rank_checks = [read(output / f"finite_checks_rank{i}.json") for i in range(training["world_size"])]
            for key in ("selected_forget_tokens", "nonempty_forget_samples", "forget_samples_seen"):
                summary[key] = sum(row[key] for row in rank_checks)
        save_json(output / "completed.json", summary)
        print(json.dumps(summary, indent=2), flush=True)
    trainer.accelerator.wait_for_everyone()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["smoke", "train"])
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--seed", type=int, help="Required for a formal run, unused for smoke")
    parser.add_argument("--root", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.root = args.root.resolve()
    if (args.mode == "train") != (args.seed is not None):
        parser.error("Use --seed only for train, which requires an explicit seed")
    run(args)


if __name__ == "__main__":
    main()
