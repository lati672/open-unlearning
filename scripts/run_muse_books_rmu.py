"""Freeze, smoke-test, train and evaluate corrected MUSE Books RMU."""

import argparse
import fcntl
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

from hydra import compose, initialize_config_dir
import numpy as np
from omegaconf import OmegaConf
from rouge_score.rouge_scorer import RougeScorer
from sklearn.metrics import roc_auc_score
import torch
from transformers import TrainerCallback, TrainingArguments

from prepare_muse_books_rmu import (
    ARMS, BASE, BASE_REV, COMMIT, DATA, DATA_PATH, DATA_REV, MASK_SEED, ROOT, SEEDS,
    SOURCE, TARGET, TARGET_PATH, TARGET_REV, TRAIN_SPLITS, WORKSPACE, digest,
    local_dataset, model_config, prepare_masks, read, require, save, train_datasets,
)

from data.collators import DataCollatorForSupervisedDataset, FrozenMaskCollator
from data.unlearn import ForgetRetainDataset
from evals import get_evaluators
from model import get_model
from trainer import load_trainer
from trainer.masking import eligible_positions
from trainer.utils import seed_everything


FROZEN = ROOT / "frozen_inputs"
CONFIGS = ROOT / "frozen_configs"
REFERENCE_SOURCE = WORKSPACE / "saves/eval/muse_Llama-2-7b-hf_Books_retrain/MUSE_EVAL.json"
REFERENCE = FROZEN / "retain_only_MUSE_EVAL.json"
NORMALIZATION_TEST = WORKSPACE / "tests/test_muse_rmu_normalization.py"
METRIC_NAMES = {"ES": "extraction_strength", "FK-R": "forget_knowmem_ROUGE", "FV-R": "forget_verbmem_ROUGE",
                "Ret-R": "retain_knowmem_ROUGE", "privleak_signed": "privleak"}
METRICS = (*METRIC_NAMES, "privleak_abs", "mia_min_k_auc")
PAPER_DIRECTIONS = {"ES": "lower", "FK-R": "lower", "FV-R": "lower", "Ret-R": "higher",
                    "privleak_signed": "toward_zero", "privleak_abs": "lower", "mia_min_k_auc": "diagnostic"}
EVAL_COUNTS = {"extraction_strength": 100, "forget_knowmem_ROUGE": 100, "forget_verbmem_ROUGE": 100,
               "retain_knowmem_ROUGE": 100, "mia_min_k/forget": 100, "mia_min_k/holdout": 100}
SCRIPTS = ("prepare_muse_books_rmu.py", "run_muse_books_rmu.py", "analyze_muse_books_rmu.py")
PROTOCOL_NAME = "MUSE Books corrected confirmatory protocol"
WORLD = 2
SMOKE_STEPS = 4
DEEPSPEED = SOURCE / "configs/accelerate/zero_stage3_offload_config.json"


def container(cfg):
    return OmegaConf.to_container(cfg, resolve=True)


def compose_config(name, overrides):
    with initialize_config_dir(version_base=None, config_dir=str(SOURCE / "configs")):
        return compose(config_name=name, overrides=overrides)


def evaluator_config():
    cfg = compose_config("eval", ["experiment=eval/muse/default", "task_name=books_rmu_eval", "data_split=Books",
                                   f"retain_logs_path={REFERENCE}", "paths.output_dir=."])
    config = container(cfg.eval.muse)
    config.update(overwrite=False, strict_outputs=True, output_dir=".")
    config["metrics"]["extraction_strength"]["record_details"] = True
    for spec in config["metrics"].values():
        for metric in (spec, *spec.get("pre_compute", {}).values()):
            for ds in metric.get("datasets", {}).values():
                for key in ("hf_args", "few_shot_dataset_hf_args"):
                    if key in ds["args"]:
                        ds["args"][key] = local_dataset(ds["args"][key])
    return config


def train_config(arm, settings):
    cfg = compose_config("unlearn", ["experiment=unlearn/muse/default", "data_split=Books", "task_name=books_rmu",
                                     "model=Llama-2-7b-hf", f"trainer={ARMS[arm]}",
                                     "collator=DataCollatorForSupervisedDataset", "paths.output_dir=."])
    cfg.model = model_config("target", "flash_attention_2")
    for role in TRAIN_SPLITS:
        spec = cfg.data[role][f"MUSE_{role}"].args
        spec.hf_args = local_dataset(container(spec.hf_args))
    cfg.trainer.method_args.update({"alpha": settings["alpha"], "steering_coeff": settings["steering_coeff"],
                                   "module_regex": settings["module_regex"],
                                   "trainable_params_regex": settings["trainable_params_regex"]})
    trainer = container(cfg.trainer)
    args = trainer['args']
    args.update({"per_device_train_batch_size": settings["micro_batch"], "gradient_accumulation_steps": settings["accumulation"],
                 "learning_rate": settings["learning_rate"], "num_train_epochs": settings["configured_epochs"],
                 "max_steps": settings["max_steps"], "ddp_find_unused_parameters": True, "gradient_checkpointing": True,
                 "save_strategy": "steps", "save_steps": settings["midpoint_step"], "save_total_limit": 2,
                 "eval_on_start": False, "do_eval": False, "eval_strategy": "no", "logging_nan_inf_filter": False,
                 "deepspeed": str(DEEPSPEED)})
    for key in ("warmup_steps", "warmup_ratio", "max_grad_norm", "dataloader_num_workers", "dataloader_drop_last"):
        args.setdefault(key, TrainingArguments.__dataclass_fields__[key].default)
    return {"model": container(cfg.model), "data": container(cfg.data), "trainer": trainer,
            "collator": container(cfg.collator), "distributed": read(DEEPSPEED)}


def freeze(args):
    require(not (ROOT / "protocol_manifest.json").exists(), "Protocol is already frozen")
    audit = prepare_masks()
    require(args.epochs > 0 and args.learning_rate > 0 and args.micro_batch > 0 and args.accumulation > 0,
            "The training budget and learning rate must be positive")
    tests = (ROOT / "logs/normalization_tests.log").read_text()
    require("Ran 2 tests" in tests and tests.strip().endswith("OK"), "RMU normalization tests did not pass")
    CONFIGS.mkdir(exist_ok=True)
    shutil.copy2(REFERENCE_SOURCE, REFERENCE)
    eval_cfg = evaluator_config()
    save(CONFIGS / "official_evaluator.json", eval_cfg)
    ref_log = read(REFERENCE)
    require(math.isclose(replay_auc(ref_log["mia_min_k"], check_split_means=False)[0], ref_log["mia_min_k"]["agg_value"], abs_tol=1e-14),
            "Archived retain-only reference AUC differs")
    batches = math.ceil(audit["samples"] / (args.micro_batch * WORLD))
    # Transformers 4.45.1 derives max_steps from floor updates per epoch.
    steps_per_epoch = max(1, batches // args.accumulation)
    midpoint_epochs = args.epochs / 2
    midpoint = math.ceil(midpoint_epochs * steps_per_epoch)
    steps = args.epochs * steps_per_epoch
    require(0 < midpoint < steps, "Training budget does not have an interior midpoint")
    settings = {"configured_epochs": args.epochs, "learning_rate": args.learning_rate, "micro_batch": args.micro_batch,
                "accumulation": args.accumulation, "world_size": WORLD, "global_effective_batch": args.micro_batch * args.accumulation * WORLD,
                "dataloader_batches_per_rank_per_epoch": batches, "trainer_budget_steps_per_epoch": steps_per_epoch,
                "budget_rule": "Transformers 4.45.1 Trainer uses floor(dataloader_batches_per_rank / gradient_accumulation_steps)",
                "actual_epoch_rule": "global_step * gradient_accumulation_steps / dataloader_batches_per_rank",
                "max_steps": steps, "midpoint_step": midpoint, "alpha": args.retain_alpha,
                "steering_coeff": args.steering_coeff, "module_regex": args.module_regex,
                "trainable_params_regex": [args.trainable_regex],
                "sampling": {"anchor": "forget", "forget": "seeded random shuffle, even distributed batches",
                             "retain": "uniform with replacement, using the seeded Torch stream",
                             "drop_last": False, "dataloader_workers": 0}}
    planned_epochs = {str(step): step * args.accumulation / batches for step in (midpoint, steps)}
    for arm in ARMS:
        save(CONFIGS / f"{arm}.json", train_config(arm, settings))
    config = read(CONFIGS / "base.json")
    settings.update({key: config["trainer"]["args"].get(key) for key in ("optim", "lr_scheduler_type", "weight_decay",
                     "warmup_steps", "warmup_ratio", "gradient_checkpointing", "bf16", "max_grad_norm")})
    settings.update({"max_sequence_length_parameter": config["data"]["forget"]["MUSE_forget"]["args"]["max_length"],
                     "observed_max_input_tokens": max(len(row["input_ids"]) for row in FrozenMaskCollator._read_records(FROZEN / "base.jsonl").values()),
                     "retain_loss": config["trainer"]["method_args"]["retain_loss_type"],
                     "forget_coefficient": config["trainer"]["method_args"]["gamma"]})
    gen = torch.Generator(device="cpu").manual_seed(args.control_seed)
    dim = read(TARGET_PATH / "config.json")["hidden_size"]
    control = torch.rand(1, 1, dim, generator=gen)
    control = control / control.norm() * args.steering_coeff
    save(FROZEN / "control_vector.json", {"shape": list(control.shape), "dtype": "float32", "values": control.flatten().tolist()})
    endpoints = {"midpoint": {"configured_epoch": midpoint_epochs, "optimizer_step": midpoint,
                               "expected_dataloader_epoch": planned_epochs[str(midpoint)]},
                 "primary": {"configured_epoch": args.epochs, "optimizer_step": steps,
                             "expected_dataloader_epoch": planned_epochs[str(steps)]}}
    plan = {"protocol_name": PROTOCOL_NAME, "seeds": SEEDS, "arms": ARMS, "primary_step": steps,
            "midpoint_step": midpoint, "max_steps": steps, "endpoints": endpoints,
            "planned_epochs": planned_epochs, "smoke_steps": SMOKE_STEPS,
            "runs": [{"arm": arm, "seed": seed, "checkpoint_steps": [midpoint, steps]} for seed in SEEDS for arm in ARMS]}
    save(ROOT / "run_plan.json", plan)
    sources = {name: digest(WORKSPACE / "scripts" / name) for name in SCRIPTS}
    manifest = {"protocol_name": PROTOCOL_NAME, "protocol_kind": "corrected_confirmatory",
                "code_commit": COMMIT, "experiment_script_hashes": sources, "budget_source": args.budget_evidence,
                "table6_usage": "Historical comparison of metric directions after the primary analysis",
                "endpoints": endpoints, "smoke_preflight": args.smoke_preflight,
                "models": {"target": {"id": TARGET, "revision": TARGET_REV},
                           "adaptive_base": {"id": BASE, "revision": BASE_REV},
                           "tokenizer": {"id": BASE, "revision": BASE_REV},
                           "frozen_rmu_retain_reference": {"source": "exact target copy at step 0", "revision": TARGET_REV},
                           "evaluation_reference": {"path": str(REFERENCE_SOURCE), "sha256": digest(REFERENCE)}},
                "target_snapshot_files": {p.name: digest(p) for p in sorted(TARGET_PATH.iterdir()) if p.is_file()},
                "dataset": {"id": DATA, "revision": DATA_REV, "train_splits": TRAIN_SPLITS,
                            "parquet_hashes": {p.relative_to(DATA_PATH).as_posix(): digest(p) for p in sorted(DATA_PATH.rglob("*.parquet"))}},
                "training": settings, "mask_hashes": audit["mask_hashes"], "random_mask_seed": MASK_SEED,
                "mask_audit_sha256": digest(ROOT / "mask_audit.json"),
                "control_vector": {"seed": args.control_seed, "shape": list(control.shape),
                                   "float32_sha256": hashlib.sha256(control.numpy().tobytes()).hexdigest(),
                                   "file_sha256": digest(FROZEN / "control_vector.json")},
                "config_hashes": {p.name: digest(p) for p in sorted(CONFIGS.glob("*.json"))},
                "run_plan_sha256": digest(ROOT / "run_plan.json"),
                "normalization_test_sha256": digest(NORMALIZATION_TEST),
                "normalization_test_log_sha256": digest(ROOT / "logs/normalization_tests.log"),
                "source_sha256": {p.relative_to(SOURCE).as_posix(): digest(p)
                                  for directory in ("src", "configs", "scripts", "tests")
                                  for p in sorted((SOURCE / directory).rglob("*"))
                                  if p.is_file() and "__pycache__" not in p.parts},
                "packages": {name: version(name) for name in ("transformers", "torch", "accelerate", "deepspeed",
                              "bitsandbytes", "flash-attn", "datasets", "huggingface-hub", "rouge-score", "scikit-learn")},
                "torch_cuda": torch.version.cuda, "metric_directions": PAPER_DIRECTIONS, "expected_eval_counts": EVAL_COUNTS}
    save(ROOT / "protocol_manifest.json", manifest)
    print(json.dumps({"plan": plan, "training": settings}), flush=True)


def locked():
    manifest = read(ROOT / "protocol_manifest.json")
    plan = read(ROOT / "run_plan.json")
    for name, sha in manifest["config_hashes"].items():
        require(digest(CONFIGS / name) == sha, f"Frozen config changed: {name}")
    for name, sha in manifest["experiment_script_hashes"].items():
        require(digest(WORKSPACE / "scripts" / name) == sha, f"Frozen runner changed: {name}")
    for name, sha in manifest["source_sha256"].items():
        require(digest(SOURCE / name) == sha, f"Frozen source changed: {name}")
    for arm, sha in manifest["mask_hashes"].items():
        require(digest(FROZEN / f"{arm}.jsonl") == sha, f"Frozen mask changed: {arm}")
    for name, sha in manifest["dataset"]["parquet_hashes"].items():
        require(digest(DATA_PATH / name) == sha, f"Frozen dataset changed: {name}")
    require(digest(ROOT / "run_plan.json") == manifest["run_plan_sha256"]
            and digest(FROZEN / "control_vector.json") == manifest["control_vector"]["file_sha256"]
            and digest(REFERENCE) == manifest["models"]["evaluation_reference"]["sha256"]
            and digest(ROOT / "mask_audit.json") == manifest["mask_audit_sha256"]
            and digest(NORMALIZATION_TEST) == manifest["normalization_test_sha256"]
            and digest(ROOT / "logs/normalization_tests.log") == manifest["normalization_test_log_sha256"],
            "Frozen plan, control vector, mask audit, normalization tests or reference changed")
    require(all(version(package) == value for package, value in manifest["packages"].items())
            and torch.version.cuda == manifest["torch_cuda"], "Frozen environment changed")
    return manifest, plan


def samples(log, size):
    entries = log["value_by_index"]
    require(set(entries) == set(map(str, range(size))), f"Expected {size} complete sample IDs")
    return entries.values()


def replay_auc(log, *, check_split_means=True):
    counts = {}
    values = []
    for split in ("forget", "holdout"):
        size = EVAL_COUNTS[f"mia_min_k/{split}"]
        scores = [row["score"] for row in samples(log[split], size)]
        require(all(math.isfinite(score) for score in scores), f"Nonfinite Min-K score in {split}")
        if check_split_means:
            require(math.isclose(float(np.mean(np.asarray(scores, dtype=np.float32))), log[split]["agg_value"], abs_tol=1e-13), f"Min-K mean differs in {split}")
        values.append(scores)
        counts[f"mia_min_k/{split}"] = size
    auc = float(roc_auc_score([0] * len(values[0]) + [1] * len(values[1]), values[0] + values[1]))
    require(math.isclose(auc, log["auc"], abs_tol=1e-14) and log["auc"] == log["agg_value"], "MUSE AUC replay differs")
    return auc, counts


def validate_eval(folder):
    logs = read(folder / "MUSE_EVAL.json")
    summary = read(folder / "MUSE_SUMMARY.json")
    require(set(logs) == set(METRIC_NAMES.values()) | {"mia_min_k"}
            and set(summary) == set(METRIC_NAMES.values()), "Unexpected official MUSE metric inventory")
    scorer = RougeScorer(["rougeL", "rouge1"], use_stemmer=True)
    values, counts = {}, {}
    for name in ("ES", "FK-R", "FV-R", "Ret-R"):
        key = METRIC_NAMES[name]
        counts[key] = EVAL_COUNTS[key]
        scores = []
        for row in samples(logs[key], counts[key]):
            if name == "ES":
                targets, preds = row["target_token_ids"], row["predicted_token_ids"]
                require(len(targets) == len(preds) > 0, "Incomplete extraction token details")
                last_wrong = max((i for i, (a, b) in enumerate(zip(targets, preds)) if a != b), default=-1)
                start = last_wrong + 1
                require(row["matching_suffix_start"] == (start if start < len(targets) else None), "Extraction suffix differs")
                # The frozen evaluator leaves k at n-1 when no nonempty suffix matches.
                score = 1 - min(start, len(targets) - 1) / len(targets)
                require(math.isclose(score, row["score"], abs_tol=1e-14), "Extraction replay differs")
            else:
                require(all(isinstance(row[k], str) for k in ("input", "ground_truth", "generation")), "Incomplete generation")
                result = scorer.score(row["ground_truth"], row["generation"])
                score = result["rougeL"].fmeasure
                for metric, expected in {"rougeL_f1": score, "rougeL_recall": result["rougeL"].recall,
                                         "rouge1_recall": result["rouge1"].recall}.items():
                    require(math.isclose(expected, row[metric], abs_tol=1e-14), f"{key}/{metric} replay differs")
            scores.append(score)
        values[name] = float(np.mean(scores))
    auc, sizes = replay_auc(logs["mia_min_k"])
    counts.update(sizes)
    ref_auc, _ = replay_auc(read(REFERENCE)["mia_min_k"], check_split_means=False)
    values["privleak_signed"] = ((1 - auc) - (1 - ref_auc)) / (1 - ref_auc + 1e-10) * 100
    values["privleak_abs"] = abs(values["privleak_signed"])
    values["mia_min_k_auc"] = auc
    require(counts == EVAL_COUNTS and all(math.isfinite(x) for x in values.values()), "MUSE metrics or sample counts differ")
    for name, key in METRIC_NAMES.items():
        require(summary[key] == logs[key]["agg_value"] and math.isclose(values[name], summary[key], rel_tol=1e-12, abs_tol=1e-14),
                f"Aggregate replay differs: {name}")
        values[name] = summary[key]
    return values, counts


def run_path(arm, seed, smoke=False):
    return ROOT / ("smoke" if smoke else "runs") / f"{arm}_seed{seed}"


def eval_checkpoint(arm, seed, step, smoke):
    manifest, plan = locked()
    run = run_path(arm, seed, smoke)
    checkpoint = run / f"checkpoint-{step}"
    state = read(checkpoint / "trainer_state.json")
    meta = read(checkpoint / "checkpoint_metadata.json")
    require(meta["seed"] == seed and meta["arm"] == arm and meta["optimizer_step"] == state["global_step"] == step
            and meta["actual_epoch"] == state["epoch"] and meta["protocol_sha256"] == digest(ROOT / "protocol_manifest.json")
            and meta["mask_sha256"] == manifest["mask_hashes"][arm] and meta["code_commit"] == COMMIT,
            f"Checkpoint provenance differs: {checkpoint}")
    folder = checkpoint / "evals"
    folder.mkdir(exist_ok=True)
    weights = sorted(checkpoint.glob("*.safetensors"))
    require(weights, f"Missing checkpoint weights: {checkpoint}")
    config = read(CONFIGS / "official_evaluator.json")
    config["output_dir"] = str(folder)
    reload_cfg = model_config("target", "sdpa", "cuda:0")
    reload_cfg.model_args.pretrained_model_name_or_path = str(checkpoint)
    reload_cfg.tokenizer_args.pop("revision")
    reload_cfg.tokenizer_args.pretrained_model_name_or_path = str(checkpoint)
    context = {**meta, "evaluator_config_sha256": manifest["config_hashes"]["official_evaluator.json"],
               "reload_model": container(reload_cfg), "weights_sha256": {p.name: digest(p) for p in weights}}
    if (folder / "evaluation_provenance.json").exists():
        require(read(folder / "evaluation_provenance.json") == context, "Previous evaluation used different weights or config")
    save(folder / "evaluation_provenance.json", context)
    save(folder / "resolved_evaluator.json", config)
    seed_everything(seed)
    model, tokenizer = get_model(reload_cfg)
    require(tokenizer.pad_token_id is not None and model.config.hidden_size == manifest["control_vector"]["shape"][-1],
            "Reloaded checkpoint model or tokenizer differs")
    get_evaluators(OmegaConf.create({"muse": config}))["muse"].evaluate(
        model=model, tokenizer=tokenizer, template_args=reload_cfg.template_args)
    metrics, counts = validate_eval(folder)
    save(folder / "official_metrics.json", metrics)
    artifacts = {p.name: digest(p) for p in sorted(folder.glob("*.json")) if p.name != "validated.json"}
    save(folder / "validated.json", {**context, "metrics": metrics, "sample_counts": counts,
                                    "metric_replay_passed": True, "checkpoint_reload_passed": True, "files_sha256": artifacts})
    print(f"Validated {arm} seed {seed} checkpoint {step}", flush=True)


class TrainingChecks(TrainerCallback):
    def __init__(self, trainer, steps):
        self.trainer = trainer
        self.steps = set(steps)
        self.losses = []
        self.grad_norms = []
        self.order = hashlib.sha256()
        self.current_mask = None
        self.loss_calls = 0
        self.original_loss = trainer.compute_loss
        self.activation_loss = trainer.compute_activation_loss
        trainer.compute_loss = self.compute_loss
        trainer.compute_activation_loss = self.compute_activation_loss

    def compute_loss(self, model, inputs, return_outputs=False):
        forget = inputs["forget"]
        for split in ("forget", "retain"):
            indices = inputs[split]["index"].detach().cpu().numpy().astype("<i8", copy=False)
            self.order.update(indices.tobytes())
        self.current_mask = forget.get("frozen_random_mask", forget["reference_mask"])
        self.loss_calls = 0
        result = self.original_loss(model, inputs, return_outputs)
        require(self.loss_calls == 2, "Expected one corrected forget loss and one corrected retain loss")
        value = (result[0] if isinstance(result, tuple) else result).detach()
        finite = torch.isfinite(value).int()
        torch.distributed.all_reduce(finite, op=torch.distributed.ReduceOp.MIN)
        require(finite.item(), "Nonfinite RMU loss")
        self.losses.append(value.item())
        return result

    def compute_activation_loss(self, actual, expected, mask):
        if self.loss_calls == 0:
            require(actual.shape[:2] == expected.shape[:2] == mask.shape
                    and torch.equal(mask, self.current_mask.to(mask.device)),
                    "RMU hidden-state positions differ from the frozen mask")
        self.loss_calls += 1
        return self.activation_loss(actual, expected, mask)

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        engine = self.trainer.model_wrapped
        require(engine.global_steps == state.global_step + 1, "DeepSpeed optimizer step differs from Trainer state")
        grad = float(engine.get_global_grad_norm())
        require(math.isfinite(grad) and grad > 0 and engine.skipped_steps == 0, "Invalid or skipped RMU optimizer step")
        self.grad_norms.append(grad)

    def on_step_end(self, args, state, control, **kwargs):
        control.should_save = state.global_step in self.steps
        return control


def weights_in_checkpoint(checkpoint):
    weights = sorted(checkpoint.glob("*.safetensors"))
    for name in ("config.json", "tokenizer_config.json", "trainer_state.json", "training_args.bin"):
        require((checkpoint / name).is_file(), f"Missing {name} in {checkpoint}")
    require(weights and ((checkpoint / "tokenizer.json").exists() or (checkpoint / "tokenizer.model").exists()),
            f"Reload files missing: {checkpoint}")
    if len(weights) > 1:
        index = read(checkpoint / "model.safetensors.index.json")
        require(set(index["weight_map"].values()) == {p.name for p in weights}, "Checkpoint weight shards differ")
    return weights


def train(arm, seed, smoke):
    manifest, plan = locked()
    output = run_path(arm, seed, smoke)
    require(not output.exists(), f"Run already exists: {output}")
    steps = [SMOKE_STEPS] if smoke else [plan["midpoint_step"], plan["primary_step"]]
    training = manifest["training"]
    config = read(CONFIGS / f"{arm}.json")
    trainer_cfg = OmegaConf.create(config["trainer"])
    trainer_cfg.args.update(seed=seed, output_dir=str(output), logging_dir=str(output / "tensorboard"))
    if smoke:
        trainer_cfg.args.update(max_steps=SMOKE_STEPS, save_steps=SMOKE_STEPS, logging_steps=1)
    seed_everything(seed)
    model, tokenizer = get_model(OmegaConf.create(config["model"]))
    model.config.use_cache = False
    data = train_datasets(OmegaConf.create(config["model"]), tokenizer)
    audit = read(ROOT / "mask_audit.json")
    require(len(data["forget"]) == audit["samples"] and len(data["retain"]) == audit["retain_samples"],
            "Training sample counts differ")
    dataset = ForgetRetainDataset(**data, anchor="forget")
    reference_mask = FROZEN / f"{'adaptive' if arm == 'random_adaptive' else arm}.jsonl"
    random_mask = FROZEN / "random_adaptive.jsonl" if arm == "random_adaptive" else None
    collator = FrozenMaskCollator(DataCollatorForSupervisedDataset(tokenizer, index="index"), reference_mask, random_mask)
    trainer, args = load_trainer(trainer_cfg, model, train_dataset=dataset, tokenizer=tokenizer, data_collator=collator,
                                 template_args=config["model"]["template_args"])
    require(trainer.is_deepspeed_enabled and args.world_size == WORLD
            and len(trainer.get_train_dataloader()) == training["dataloader_batches_per_rank_per_epoch"],
            "Expected the frozen two-rank DeepSpeed dataloader")
    control = read(FROZEN / "control_vector.json")
    vector = torch.tensor(control["values"], dtype=torch.float32).reshape(control["shape"])
    require(hashlib.sha256(vector.numpy().tobytes()).hexdigest() == manifest["control_vector"]["float32_sha256"]
            and math.isclose(vector.norm().item(), training["steering_coeff"], rel_tol=1e-6),
            "RMU steering vector differs")
    trainer.fixed_control_vec = vector
    trainer.ref_model.requires_grad_(False)
    trainer.ref_model.eval()
    require(not trainer.ref_model.module.training and not any(p.requires_grad for p in trainer.ref_model.parameters()),
            "RMU retain reference is not frozen")
    if smoke:
        from deepspeed.runtime.zero import GatheredParameters
        reference = dict(trainer.ref_model.module.named_parameters())
        active = dict(model.named_parameters())
        names = list(active)
        for name in (names[0], next(n for n in names if n.startswith("model.layers.7.")), names[-1]):
            with GatheredParameters([reference[name]]):
                require(torch.equal(reference[name].detach(), active[name].to(reference[name].device).detach()),
                        f"Initial RMU reference differs: {name}")
    check = TrainingChecks(trainer, steps)
    trainer.add_callback(check)
    resolved = {**config, "trainer": container(trainer_cfg), "arm": arm, "seed": seed,
                "target_revision": TARGET_REV, "dataset_revision": DATA_REV, "code_commit": COMMIT,
                "mask_sha256": manifest["mask_hashes"][arm], "control_sha256": manifest["control_vector"]["float32_sha256"],
                "protocol_sha256": digest(ROOT / "protocol_manifest.json")}
    if trainer.is_world_process_zero():
        output.mkdir(exist_ok=True, parents=True)
        save(output / "resolved_config.json", resolved)
        save(output / "training_args.json", args.to_dict())
    trainer.accelerator.wait_for_everyone()
    seed_everything(seed)
    result = trainer.train()
    trainer.save_state()
    engine = trainer.model_wrapped
    require(trainer.state.global_step == engine.global_steps == steps[-1] and engine.skipped_steps == 0
            and len(check.losses) == steps[-1] * training["accumulation"] and len(check.grad_norms) == steps[-1]
            and math.isfinite(result.training_loss), f"Incomplete numeric checks for {output}")
    rank_checks = {"microbatch_losses": check.losses, "optimizer_grad_norms": check.grad_norms,
                   "sampling_order_sha256": check.order.hexdigest(), "gpu": torch.cuda.get_device_name(args.device),
                   "gpu_peak_bytes": torch.cuda.max_memory_allocated(args.device),
                   "reference_probes_equal": smoke, "no_skipped_steps": True}
    save(output / f"checks_rank{args.process_index}.json", rank_checks)
    trainer.accelerator.wait_for_everyone()
    if trainer.is_world_process_zero():
        actual = {}
        for step in steps:
            checkpoint = output / f"checkpoint-{step}"
            weights_in_checkpoint(checkpoint)
            state = read(checkpoint / "trainer_state.json")
            predicted = step * training["accumulation"] / training["dataloader_batches_per_rank_per_epoch"]
            require(state["global_step"] == step and math.isclose(state["epoch"], predicted, abs_tol=1e-12),
                    f"Unexpected checkpoint step or epoch: {checkpoint}")
            actual[str(step)] = state["epoch"]
            save(checkpoint / "checkpoint_metadata.json", {k: v for k, v in resolved.items()
                 if k in {"arm", "seed", "target_revision", "dataset_revision", "code_commit",
                          "mask_sha256", "control_sha256", "protocol_sha256"}} |
                 {"optimizer_step": step, "actual_epoch": state["epoch"],
                  "resolved_training_config_sha256": digest(output / "resolved_config.json")})
        save(output / "training_completed.json", {"global_step": steps[-1], "actual_epochs": actual,
             "loss": result.training_loss, "resolved_config_sha256": digest(output / "resolved_config.json"),
             "optimizer_class": type(engine.optimizer.optimizer).__name__,
             "scheduler_class": type(trainer.lr_scheduler).__name__, "gradient_checkpointing": trainer.model.is_gradient_checkpointing})
    trainer.accelerator.wait_for_everyone()


def archive(arm, seed, step, smoke):
    run = run_path(arm, seed, smoke)
    checkpoint = run / f"checkpoint-{step}"
    folder = checkpoint / "evals"
    proof = read(folder / "validated.json")
    require(proof["checkpoint_reload_passed"] and proof["metric_replay_passed"], f"Eval not verified: {checkpoint}")
    require(all(digest(folder / name) == sha for name, sha in proof["files_sha256"].items()), "Evaluation artifact changed")
    metrics, counts = validate_eval(folder)
    require(metrics == proof["metrics"] and counts == proof["sample_counts"], "Evaluation replay differs")
    dest = ROOT / ("smoke_collected" if smoke else "collected_results") / run.name / checkpoint.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    stage = dest.with_name(dest.name + ".tmp")
    if not dest.exists():
        require(not stage.exists(), f"Partial archive requires inspection: {stage}")
        shutil.copytree(checkpoint, stage, ignore=shutil.ignore_patterns("*.safetensors"))
        for name in ("resolved_config.json", "training_args.json", "trainer_state.json", "training_completed.json",
                     "checks_rank0.json", "checks_rank1.json"):
            shutil.copy2(run / name, stage / f"run_{name}")
        hashes = {p.relative_to(stage).as_posix(): digest(p) for p in sorted(stage.rglob("*")) if p.is_file()}
        for name, sha in hashes.items():
            source = run / name.removeprefix("run_") if name.startswith("run_") else checkpoint / name
            require(digest(source) == sha, f"Archive copy differs: {name}")
        save(stage / "archive_hashes.json", hashes)
        stage.rename(dest)
    hashes = read(dest / "archive_hashes.json")
    require(all(digest(dest / name) == sha for name, sha in hashes.items()), f"Archive hash differs: {dest}")
    require(read(dest / "evals/validated.json") == proof, "Archived validation differs")
    index_path = ROOT / ("smoke_manifest.json" if smoke else "incremental_manifest.json")
    index = read(index_path) if index_path.exists() else {}
    key = f"{run.name}/{checkpoint.name}"
    record = {"archive": dest.relative_to(ROOT).as_posix(), "archive_hashes_sha256": digest(dest / "archive_hashes.json"),
              "validated_sha256": digest(dest / "evals/validated.json"), "weights_sha256": proof["weights_sha256"],
              "metrics": metrics, "sample_counts": counts, "actual_epoch": proof["actual_epoch"], "optimizer_step": step}
    require(key not in index or index[key] == record, "Checkpoint manifest differs")
    index[key] = record
    save(index_path, index)
    weights = {checkpoint / name: sha for name, sha in proof["weights_sha256"].items()}
    require(all(p.parent == checkpoint and p.suffix == ".safetensors" and
                (not p.exists() or (not p.is_symlink() and digest(p) == sha)) for p, sha in weights.items()),
            "Checkpoint weights changed after evaluation")
    for weight in weights:
        weight.unlink(missing_ok=True)
    print(f"Archived and verified {key}", flush=True)


def run_queue(smoke_only=False):
    manifest, plan = locked()
    require(smoke_only or not manifest["smoke_preflight"], "The provisional protocol is for smoke testing only")
    if not smoke_only:
        smoke_check = read(ROOT / "smoke_compatibility.json")
        require(smoke_check["passed"] and smoke_check["checkpoints"] == len(ARMS)
                and smoke_check["final_protocol_sha256"] == digest(ROOT / "protocol_manifest.json"),
                "Smoke evidence must match the frozen confirmatory protocol")
    base_command = [sys.executable, "-u", str(Path(__file__))]
    current = {}
    count = 0

    def status(phase, **fields):
        data = {"phase": phase, "formal_checkpoints": count, "formal_runs_completed": count // 2, **current, **fields}
        save(ROOT / "status.json", data)
        print(json.dumps(data), flush=True)

    def run_one(arm, seed, smoke):
        nonlocal count
        current.update(arm=arm, seed=seed, smoke=smoke)
        run = run_path(arm, seed, smoke)
        tag = f"{'smoke' if smoke else 'run'}_{run.name}"
        status("training")
        suffix = ["--smoke"] if smoke else []
        if not (run / "training_completed.json").exists():
            require(not run.exists(), f"Partial training requires inspection: {run}")
            with (ROOT / "logs" / f"{tag}.train.log").open("x") as log:
                subprocess.run([sys.executable, "-u", "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
                                *base_command[2:], "train", arm, str(seed), *suffix], cwd=SOURCE, stdout=log,
                               stderr=subprocess.STDOUT, check=True)
        steps = [SMOKE_STEPS] if smoke else [plan["midpoint_step"], plan["primary_step"]]
        completed = read(run / "training_completed.json")
        require(completed["global_step"] == steps[-1], "Training did not reach the planned optimizer step")
        status("evaluation", steps=steps)
        # Both evaluations are independent and each checkpoint fits one GPU.
        pending = [step for step in steps if not (run / f"checkpoint-{step}/evals/validated.json").exists()]
        for attempt in range(2):
            if not pending:
                break
            jobs = {}
            for gpu, step in enumerate(pending):
                with (ROOT / "logs" / f"{tag}.step{step}.eval.log").open("a") as log:
                    jobs[step] = subprocess.Popen([*base_command, "eval", arm, str(seed), "--step", str(step), *suffix],
                                                 cwd=SOURCE, stdout=log, stderr=subprocess.STDOUT,
                                                 env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)})
            pending = [step for step, proc in jobs.items() if proc.wait() != 0]
        require(not pending, f"Evaluation failed at {run.name} steps {pending}; checkpoint weights retained")
        for step in steps:
            archive(arm, seed, step, smoke)
            count += int(not smoke)
        status("run_complete")

    def paired_order(seed, smoke):
        for rank in range(WORLD):
            hashes = {read(run_path(arm, seed, smoke) / f"checks_rank{rank}.json")["sampling_order_sha256"] for arm in ARMS}
            require(len(hashes) == 1, f"Sampling orders differ for seed {seed}, rank {rank}")

    try:
        require(shutil.disk_usage(ROOT).free >= 28 * 1024**3, "Less than 28 GiB free for two BF16 checkpoints")
        if smoke_only:
            for arm in ARMS:
                run_one(arm, 0, True)
            paired_order(0, True)
            status("smoke_complete")
            return
        for seed in SEEDS:
            for arm in ARMS:
                require(shutil.disk_usage(ROOT).free >= 27 * 1024**3, "Insufficient space for both BF16 checkpoints")
                run_one(arm, seed, False)
            paired_order(seed, False)
        require(count == 18, "Formal checkpoint coverage differs")
        current.clear()
        status("analysis")
        subprocess.run([sys.executable, "-u", str(WORKSPACE / "scripts/analyze_muse_books_rmu.py")], cwd=SOURCE,
                       env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, check=True)
        status("complete")
    except Exception as error:
        status("failed", error=str(error))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    setup = sub.add_parser("freeze")
    setup.add_argument("--epochs", type=int, required=True)
    setup.add_argument("--learning-rate", type=float, required=True)
    setup.add_argument("--budget-evidence", required=True)
    setup.add_argument("--smoke-preflight", action="store_true")
    setup.add_argument("--micro-batch", type=int, default=2)
    setup.add_argument("--accumulation", type=int, default=4)
    setup.add_argument("--retain-alpha", type=float, default=3)
    setup.add_argument("--steering-coeff", type=float, default=1)
    setup.add_argument("--module-regex", default=r"model\.layers\.7")
    setup.add_argument("--trainable-regex", default=r"model\.layers\.(5|6|7)\..*")
    setup.add_argument("--control-seed", type=int, default=0)
    queue = sub.add_parser("queue")
    queue.add_argument("--smoke-only", action="store_true")
    for mode in ("train", "eval"):
        job = sub.add_parser(mode)
        job.add_argument("arm", choices=ARMS)
        job.add_argument("seed", type=int, choices=SEEDS)
        job.add_argument("--smoke", action="store_true")
        if mode == "eval":
            job.add_argument("--step", type=int, required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        freeze(args)
    elif args.command == "train":
        train(args.arm, args.seed, args.smoke)
    elif args.command == "eval":
        eval_checkpoint(args.arm, args.seed, args.step, args.smoke)
    else:
        with (ROOT / "queue.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            run_queue(args.smoke_only)


if __name__ == "__main__":
    main()
