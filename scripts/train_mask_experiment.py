"""One reusable TOFU paired experiment, launched via torchrun by the runner."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sys

from cleanup_model_weights import remove_model_weights
from freeze_mask_protocol import FORMAL_OUTPUT, check_frozen_run, fixed_rmu_vector, save_json, verify_dataset_revision

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from rouge_score.rouge_scorer import RougeScorer
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from transformers import TrainerCallback
from data import get_data, get_datasets, get_collators
from data.collators import DataCollatorForSupervisedDataset, FrozenMaskCollator
from model import get_model
from trainer import load_trainer
from trainer.masking import eligible_positions
from trainer.utils import seed_everything
from compute_logprobs import configure_logprob_collator


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def evaluate(model, tokenizer, template, split, output, batch_size=16, generate=False, dataset_revision=None):
    cfg = OmegaConf.load(ROOT / "configs/data/datasets/TOFU_QA_forget.yaml")
    cfg.TOFU_QA_forget.args.hf_args.name = split
    if dataset_revision is not None:
        cfg.TOFU_QA_forget.args.hf_args.revision = dataset_revision
    dataset = get_datasets(cfg, tokenizer=tokenizer, template_args=template)
    if dataset_revision is not None:
        verify_dataset_revision(dataset, dataset_revision)
    collator = DataCollatorForSupervisedDataset(tokenizer, index="index")
    device = next(model.parameters()).device
    rows = []
    model.eval()
    with torch.inference_mode():
        for batch in DataLoader(dataset, batch_size=batch_size, collate_fn=collator):
            forward = {k: batch[k].to(device) for k in ["input_ids", "attention_mask"]}
            logits = model(**forward).logits
            losses = F.cross_entropy(logits[:, :-1].float().transpose(1, 2), batch["labels"][:, 1:].to(device), ignore_index=-100, reduction="none")
            eligible = eligible_positions(batch)[:, 1:].to(device)
            sums = (losses * eligible).sum(1)
            counts = eligible.sum(1)
            for i, index in enumerate(batch["index"].tolist()):
                nll = float(sums[i] / counts[i]) if counts[i] else None
                rows.append({"doc_id": index, "mean_nll": nll, "answer_probability": math.exp(-nll) if nll is not None else None})
    if generate:
        generation_dataset = get_datasets(cfg, tokenizer=tokenizer, template_args=template, predict_with_generate=True)
        if dataset_revision is not None:
            verify_dataset_revision(generation_dataset, dataset_revision)
        generation_collator = DataCollatorForSupervisedDataset(tokenizer, padding_side="left", index="index")
        scorer = RougeScorer(["rougeL"], use_stemmer=True)
        by_index = {row["doc_id"]: row for row in rows}
        with torch.inference_mode():
            for batch in DataLoader(generation_dataset, batch_size=8, collate_fn=generation_collator):
                ids, attention = [batch[k].to(device) for k in ["input_ids", "attention_mask"]]
                generated = model.generate(input_ids=ids, attention_mask=attention, max_new_tokens=200,
                                           do_sample=False, use_cache=True, pad_token_id=tokenizer.pad_token_id,
                                           eos_token_id=tokenizer.eos_token_id)
                answers = tokenizer.batch_decode(generated[:, ids.shape[1]:], skip_special_tokens=True)
                for index, answer in zip(batch["index"].tolist(), answers):
                    gold = dataset.data[index]["answer"]
                    by_index[index].update(generation=answer, rougeL_recall=scorer.score(gold, answer)["rougeL"].recall)
    summary = {"samples": len(rows), "mean_nll": float(np.mean([r["mean_nll"] for r in rows if r["mean_nll"] is not None])),
               "answer_probability": float(np.mean([r["answer_probability"] for r in rows if r["answer_probability"] is not None]))}
    if generate:
        summary["rougeL_recall"] = float(np.mean([r["rougeL_recall"] for r in rows]))
    save(output, {"split": split, "summary": summary, "rows": rows})
    return summary


def evaluate_suite(model, tokenizer, template, root, dataset_revision=None):
    root.mkdir(parents=True, exist_ok=True)
    result = {}
    for split in ["forget05", "holdout05", "retain95"]:
        print("Evaluate", split, flush=True)
        result[split] = evaluate(model, tokenizer, template, split, root / f"{split}.json",
                                 generate=split == "forget05", dataset_revision=dataset_revision)
    forget = json.loads((root / "forget05.json").read_text())["rows"]
    holdout = json.loads((root / "holdout05.json").read_text())["rows"]
    scores = [-r["mean_nll"] for r in forget + holdout]
    result["loss_membership_auc"] = roc_auc_score([1] * len(forget) + [0] * len(holdout), scores)
    result["scope"] = "Raw TOFU QA answer probability, mean NLL, forget ROUGE-L recall (greedy max_new_tokens=200), loss membership AUC; not the complete official utility/forget-quality evaluation suite"
    save(root / "metrics.json", result)
    return result


class StepCheckpointEvaluation(TrainerCallback):
    """Save and evaluate the legacy step milestones without advancing training RNG."""

    def __init__(self, trainer, training, evaluate_checkpoint, output):
        self.trainer = trainer
        self.training = training
        self.evaluate_checkpoint = evaluate_checkpoint
        self.output = output
        self.results = {}

    def on_train_begin(self, args, state, control, **kwargs):
        if state.max_steps != self.training["max_steps"]:
            raise ValueError(f"Expected {self.training['max_steps']} training steps, got {state.max_steps}")

    def on_step_end(self, args, state, control, **kwargs):
        step = str(state.global_step)
        expected_epoch = self.training["checkpoints"].get(step)
        if expected_epoch is None:
            return
        if not math.isclose(state.epoch, expected_epoch, abs_tol=1e-8):
            raise ValueError(f"Checkpoint {step}: expected epoch {expected_epoch}, got {state.epoch}")
        self.trainer.accelerator.wait_for_everyone()
        if self.trainer.is_world_process_zero():
            self.save_and_evaluate(state)
        self.trainer.accelerator.wait_for_everyone()

    def save_and_evaluate(self, state):
        step = str(state.global_step)
        checkpoint = self.output / f"checkpoint-{step}"
        model = self.trainer.accelerator.unwrap_model(self.trainer.model)
        device = next(model.parameters()).device
        was_training = model.training
        python_rng, numpy_rng = random.getstate(), np.random.get_state()
        try:
            with torch.random.fork_rng(devices=[device.index] if device.type == "cuda" else []):
                self.trainer.save_model(str(checkpoint))
                state.save_to_json(str(checkpoint / "trainer_state.json"))
                weights = sorted(path.name for path in checkpoint.glob("*.safetensors"))
                if not weights:
                    raise RuntimeError(f"Checkpoint did not save Safetensors weights: {checkpoint}")
                metrics = self.evaluate_checkpoint(model, checkpoint / "evals")
        finally:
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
            model.train(was_training)
        remove_model_weights(checkpoint)
        result = {"step": state.global_step, "actual_epoch": state.epoch,
                  "checkpoint": checkpoint.name, "saved_weight_files": weights,
                  "weights_removed_after_eval": True, "metrics": metrics}
        save_json(checkpoint / "evaluated.json", result)
        self.results[step] = result


def validate_random_density(trainer, manifest, output, expected_masks=None):
    """Validate actual training caches, including agreement across DDP ranks."""
    local = {index: {"count": int(mask.sum()), "hash": hashlib.sha256(mask.numpy().tobytes()).hexdigest()}
             for index, (_, mask) in trainer.mask_selector.cache.items()}
    if torch.distributed.is_initialized():
        gathered = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gathered, local)
    else:
        gathered = [local]
    observed = {}
    for rank in gathered:
        for index, entry in rank.items():
            if index in observed and observed[index] != entry:
                raise AssertionError(f"Random mask differs across ranks for sample {index}")
            observed[index] = entry
    reference = {int(row["doc_id"]): sum(row["mask"]) for row in map(json.loads, manifest.open())}
    assert observed.keys() == reference.keys(), "Training did not cover all reference samples"
    assert all(observed[index]["count"] == count for index, count in reference.items())
    if expected_masks is not None and {str(i): entry for i, entry in observed.items()} != expected_masks:
        raise ValueError("Training Random masks differ from the frozen masks")
    if trainer.is_world_process_zero():
        save(output / "density_validation.json", {"per_example_exact_match": True,
             "aggregate_exact_match": True, "cross_rank_masks_identical": True,
             "sample_count": len(observed), "selected_tokens": sum(r["count"] for r in observed.values()),
             "reference_selected_tokens": sum(reference.values()), "per_example_counts": reference,
             "matches_frozen_mask_hashes": True if expected_masks is not None else None})


def validate_control_vector(trainer, output):
    vector = trainer.control_vec.detach().cpu()
    digest = hashlib.sha256(vector.numpy().tobytes()).hexdigest()
    hashes = [digest]
    if torch.distributed.is_initialized():
        hashes = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(hashes, digest)
    if trainer.is_world_process_zero():
        save(output / "control_vector.json", {"sha256": digest, "norm": float(vector.norm()),
            "cross_rank_identical": len(set(hashes)) == 1, "rank_hashes": hashes})
    assert len(set(hashes)) == 1, "RMU control vectors differ across ranks"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objective", choices=["NPO", "RMU"], required=True)
    parser.add_argument("--variant", choices=["base", "adaptive", "entity", "random_adaptive", "random_entity"], required=True)
    parser.add_argument("--normalization", choices=["summed", "token_mean"], default="summed")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--audit-root", type=Path, default=ROOT / "saves/audits/revision")
    parser.add_argument("--protocol", type=Path, help="Run the frozen multiseed protocol")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the resolved training configuration")
    parser.add_argument("--smoke-steps", type=int, default=0, help="GPU integration check; skips benchmark evaluation")
    parser.add_argument("--baseline-only", action="store_true", help="Evaluate the initial model without training")
    args = parser.parse_args()
    args.output_root = (args.output_root or (FORMAL_OUTPUT if args.protocol else ROOT / "saves/experiments/mask_controls")).resolve()
    protocol = json.loads(args.protocol.read_text()) if args.protocol else None
    freeze_manifest = None
    dataset_revision = None
    if protocol:
        training = protocol["training"]
        if args.epochs != training["nominal_epochs"] or args.seed not in training["seeds"]:
            raise ValueError("Epochs or seed differ from the frozen protocol")
        dataset_revision = protocol["dataset"]["revision"]
        if not args.dry_run:
            protocol, freeze_manifest = check_frozen_run(args.protocol, args.output_root)
        args.audit_root = args.output_root / "frozen_inputs"
    task = f"{args.objective}_{args.variant}_{args.normalization}_forget05_seed{args.seed}_E{args.epochs}"
    if args.smoke_steps:
        task += f"_smoke{args.smoke_steps}"
    output = args.output_root / task
    output.mkdir(parents=True, exist_ok=True)
    source = "entity" if "entity" in args.variant else "adaptive"
    collator_name = "DataCollatorForSupervisedDataset" if args.variant == "base" else "DataCollatorWithEntityMask" if source == "entity" else "DataCollatorWithLogProbs"
    trainer_name = args.objective
    if args.objective == "RMU" and args.variant in {"entity", "adaptive"}:
        trainer_name = "EntityRMU" if args.variant == "entity" else "AdaptiveRMU"
    if args.variant.startswith("random"):
        trainer_name = "Random" + args.objective
    if args.objective == "NPO" and args.normalization == "token_mean":
        trainer_name = "TokenNormalizedNPO"
    overrides = ["experiment=unlearn/tofu/default", f"trainer={trainer_name}", f"collator={collator_name}",
                 f"task_name={task}", "forget_split=forget05", "retain_split=retain95", "holdout_split=holdout05",
                 "model.model_args.attn_implementation=flash_attention_2", "trainer.args.per_device_train_batch_size=4",
                 "trainer.args.gradient_accumulation_steps=4", f"trainer.args.num_train_epochs={args.epochs}",
                 "trainer.args.learning_rate=1e-5", "trainer.args.weight_decay=0.01", f"trainer.args.seed={args.seed}",
                 "trainer.args.gradient_checkpointing=true", "+trainer.args.gradient_checkpointing_kwargs={use_reentrant:false}", "trainer.args.ddp_find_unused_parameters=true",
                 "trainer.args.do_eval=false", "trainer.args.eval_on_start=false", "trainer.args.eval_strategy=no"]
    if args.objective == "NPO":
        overrides.append(f"trainer.method_args.normalization={args.normalization}")
        if args.variant in {"entity", "adaptive"}:
            overrides.append(f"trainer.method_args.mask={args.variant}")
    if args.variant.startswith("random"):
        mask_seed = protocol["random_mask_seed"] if protocol else 1729
        overrides += [f"trainer.method_args.random_reference={source}", f"trainer.method_args.mask_seed={mask_seed}"]
        if trainer_name == "TokenNormalizedNPO":
            overrides.append("trainer.method_args.mask=random")
    if protocol:
        for key in ["per_device_train_batch_size", "gradient_accumulation_steps", "learning_rate", "weight_decay"]:
            overrides.append(f"trainer.args.{key}={training[key]}")
        for key in ["alpha", "gamma"] + (["beta"] if args.objective == "NPO" else []):
            overrides.append(f"trainer.method_args.{key}={training[key]}")
    if args.smoke_steps:
        overrides.append(f"+trainer.args.max_steps={args.smoke_steps}")
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        cfg = compose(config_name="unlearn", overrides=overrides)
    cfg.eval = None
    cfg.paths.output_dir = str(output)
    # Runtime cwd interpolation only exists under hydra.main; the runner has an explicit root.
    cfg.paths.work_dir = str(ROOT)
    if protocol:
        initial, reference = protocol["initial_model"], protocol["adaptive_reference"]
        OmegaConf.update(cfg, "model.model_args.pretrained_model_name_or_path", initial["id"])
        OmegaConf.update(cfg, "model.model_args.revision", initial["revision"], force_add=True)
        OmegaConf.update(cfg, "model.tokenizer_args.pretrained_model_name_or_path", reference["id"])
        OmegaConf.update(cfg, "model.tokenizer_args.revision", reference["revision"], force_add=True)
        for split in ["forget", "retain"]:
            key = f"data.{split}.TOFU_QA_{split}.args.hf_args"
            OmegaConf.update(cfg, f"{key}.revision", dataset_revision, force_add=True)
            OmegaConf.update(cfg, f"{key}.path", protocol["dataset"]["id"])
    seed_everything(args.seed)
    snapshot = OmegaConf.to_container(cfg, resolve=True)
    if args.dry_run:
        print(json.dumps(snapshot, indent=2))
        return
    template = OmegaConf.to_container(cfg.model.template_args)
    model, tokenizer = get_model(cfg.model)
    model.config.use_cache = False
    snapshot["model_commit"] = getattr(model.config, "_commit_hash", None)
    if protocol and snapshot["model_commit"] != protocol["initial_model"]["revision"]:
        raise ValueError("Loaded model differs from the frozen TOFU checkpoint")
    data = get_data(cfg.data, mode="unlearn", tokenizer=tokenizer, template_args=template)
    if protocol:
        for split in ["forget", "retain"]:
            dataset = getattr(data["train"], split)
            verify_dataset_revision(dataset, dataset_revision)
            if len(dataset) != protocol["dataset"][f"{split}_samples"]:
                raise ValueError(f"Training {split} split size differs from the frozen run")
    collator_kwargs = {"tokenizer": tokenizer}
    if source == "entity" and args.variant != "base":
        collator_kwargs["author_names"] = json.loads((ROOT / "configs/tofu_forget_authors.json").read_text())["forget05"]
    if collator_name == "DataCollatorWithLogProbs":
        if protocol:
            OmegaConf.update(cfg, "collator.DataCollatorWithLogProbs.args.logprob_path",
                             str(args.audit_root / "logprobs.json"), force_add=True)
        else:
            configure_logprob_collator(cfg.collator, cfg.data, cfg.model, forget_split="forget05", project_root=ROOT)
    collator = get_collators(cfg.collator, **collator_kwargs)
    if args.variant != "base":
        manifest = args.audit_root / f"masks_TOFU_forget05_{source}.jsonl"
        collator = FrozenMaskCollator(collator, manifest)
        if collator.metadata.get("mask_kind") != source:
            raise ValueError("Frozen mask provenance is missing or has the wrong mask kind; rerun audit_masks.py")
        if collator.metadata.get("original_model_commit") != snapshot["model_commit"]:
            raise ValueError("Frozen masks and training model use different checkpoint revisions")
        snapshot["frozen_reference_manifest"] = str(manifest)
        snapshot["frozen_reference_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
        snapshot["frozen_reference_metadata"] = collator.metadata
    trainer, _ = load_trainer(cfg.trainer, model, train_dataset=data["train"], tokenizer=tokenizer,
                              data_collator=collator, evaluators=None, template_args=template)
    snapshot["execution"] = {"backend": "DDP", "world_size": trainer.accelerator.num_processes,
                             "effective_batch_size": trainer.args.per_device_train_batch_size * trainer.args.gradient_accumulation_steps * trainer.accelerator.num_processes,
                             "normalization": args.normalization, "variant": args.variant,
                             "paired_masks": "Audited initialization masks replayed across ranks, epochs and normalization arms",
                             "mask_seed": protocol["random_mask_seed"] if protocol else 1729,
                             "seed": args.seed, "rmu_ddp_static_graph": args.objective == "RMU"}
    checkpoint_eval = None
    if protocol:
        if trainer.accelerator.num_processes != training["world_size"] or trainer.args.warmup_steps != training["warmup_steps"]:
            raise ValueError("World size or warmup steps differ from the frozen run")
        snapshot["frozen_protocol"] = {"commit": freeze_manifest["code_commit"],
            "sha256": freeze_manifest["protocol_sha256"], "checkpoints": training["checkpoints"],
            "objective_reference": protocol["objective_reference"]}
        if args.objective == "RMU":
            trainer.fixed_control_vec = fixed_rmu_vector(protocol)
        if not args.smoke_steps:
            checkpoint_eval = StepCheckpointEvaluation(trainer, training,
                lambda model, root: evaluate_suite(model, tokenizer, template, root, dataset_revision), output)
            trainer.add_callback(checkpoint_eval)
    if trainer.is_world_process_zero():
        save(output / "resolved_config.json", snapshot)
        save(output / "training_args.json", trainer.args.to_dict())
        baseline = args.output_root / "baseline"
        if not args.smoke_steps and not (baseline / "metrics.json").exists():
            evaluate_suite(trainer.accelerator.unwrap_model(trainer.model), tokenizer, template, baseline, dataset_revision)
    trainer.accelerator.wait_for_everyone()
    if args.baseline_only:
        return
    seed_everything(args.seed)  # Baseline evaluation must not perturb the paired training RNG stream.
    trainer.train()
    trainer.save_state()
    trainer.accelerator.wait_for_everyone()
    if args.objective == "RMU":
        validate_control_vector(trainer, output)
    if args.smoke_steps:
        if trainer.is_world_process_zero():
            save(output / "smoke_completed.json", {"global_step": trainer.state.global_step})
        trainer.accelerator.wait_for_everyone()
        return
    if args.variant.startswith("random"):
        expected_masks = json.loads((args.audit_root / "random_mask_expectations.json").read_text())[source] if protocol else None
        validate_random_density(trainer, manifest, output, expected_masks)
    if trainer.is_world_process_zero():
        result = {"task": task, "global_step": trainer.state.global_step, "actual_epoch": trainer.state.epoch}
        if checkpoint_eval:
            if checkpoint_eval.results.keys() != training["checkpoints"].keys():
                raise ValueError("Formal run did not evaluate every checkpoint")
            result["checkpoints"] = checkpoint_eval.results
            result["metrics"] = checkpoint_eval.results[str(training["max_steps"])]["metrics"]
            result["protocol_sha256"] = freeze_manifest["protocol_sha256"]
            result["code_commit"] = freeze_manifest["code_commit"]
        else:
            result["metrics"] = evaluate_suite(trainer.accelerator.unwrap_model(trainer.model), tokenizer, template, output / "evals")
            remove_model_weights(output)
        save_json(output / "completed.json", result)
    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
