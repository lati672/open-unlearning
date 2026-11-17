"""Run the LowConfidence-Matched v4 extension in a pinned v4 code snapshot."""

import argparse
from functools import wraps
import fcntl
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import shutil
from statistics import mean, stdev
import subprocess
import sys
import tempfile
from zipfile import ZIP_DEFLATED, ZipFile

from prepare_low_confidence import CONTROLS, MASK, NAME, OUTPUT, read, require, rows, write_csv
from freeze_mask_protocol import FORMAL_OUTPUT, ROOT, save_json, sha256, verify_dataset_revision
from train_mask_experiment import StepCheckpointEvaluation, evaluate as raw_qa_evaluate

import numpy as np
from omegaconf import OmegaConf
from scipy.stats import hmean, ks_2samp
from sklearn.metrics import roc_auc_score
import torch

from data import get_data
from data.collators import DataCollatorForSupervisedDataset, FrozenMaskCollator
from evaluate_tofu_checkpoint import METRICS, configure_eval, get_evaluators, get_model, validate_outputs, weight_hash
from trainer import load_trainer
from trainer.utils import seed_everything


VARIANT = "lowconfidence_matched"
STEPS = (60, 120)
SEEDS = (0, 1, 2)
ARTIFACTS = ("TOFU_EVAL.json", "TOFU_SUMMARY.json", "official_metrics.json", "evaluation_provenance.json", "validated.json")
TABLE_METRICS = [key for key in METRICS if key != "privleak"] + ["privleak_signed", "privleak_abs"]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_lock(root):
    lock = read(root / "protocol_manifest.json")
    for relative, checksum in lock["source_sha256"].items():
        require(sha256(ROOT / relative) == checksum, f"Frozen code differs: {relative}")
    for relative, key in [(MASK, "mask_sha256"), (Path(MASK).with_suffix(".metadata.json"), "mask_metadata_sha256"),
                          ("target_confidence.jsonl", "target_confidence_sha256"), ("existing_controls.csv", "controls_sha256"),
                          ("v4_evaluation_signature.json", "v4_evaluation_signature_sha256")]:
        require(sha256(root / relative) == lock[key], f"Frozen input differs: {relative}")
    return lock


def output_for(root, seed, smoke=False):
    return root / ("smoke_test" if smoke else f"runs/seed{seed}")


def instrument(trainer):
    checks = {"loss": [], "gradient_l2_preclip": []}

    def verify(value, key):
        tensor = torch.as_tensor(value).detach()
        failed = (~torch.isfinite(tensor).all()).int().to(trainer.accelerator.device)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(failed, op=torch.distributed.ReduceOp.MAX)
        if failed.item():
            raise FloatingPointError(f"Nonfinite {key} before step {trainer.state.global_step + 1}")
        checks[key].append(tensor.item())

    original_loss, original_clip = trainer.compute_loss, trainer.accelerator.clip_grad_norm_

    @wraps(original_loss)
    def loss(*args, **kwargs):
        result = original_loss(*args, **kwargs)
        verify(result[0] if isinstance(result, tuple) else result, "loss")
        return result

    @wraps(original_clip)
    def clip(*args, **kwargs):
        result = original_clip(*args, **kwargs)
        verify(result, "gradient_l2_preclip")
        return result

    trainer.compute_loss = loss
    trainer.accelerator.clip_grad_norm_ = clip
    return checks


def reload_model(snapshot, checkpoint, device):
    cfg = OmegaConf.create(snapshot["model"])
    for args in (cfg.model_args, cfg.tokenizer_args):
        args.pop("revision", None)
        args.pretrained_model_name_or_path = str(checkpoint)
        args.local_files_only = True
    cfg.model_args.device_map = {"": str(device)}
    return get_model(cfg)


def smoke_eval(root, output, snapshot, tokenizer, device, training):
    saved = {}
    for step in training["checkpoints"]:
        checkpoint = output / f"checkpoint-{step}"
        state = read(checkpoint / "trainer_state.json")
        require(state["global_step"] == int(step), f"Unexpected smoke checkpoint step: {step}")
        model, loaded = reload_model(snapshot, checkpoint, device)
        require(all(getattr(loaded, key) == getattr(tokenizer, key) for key in
                    ("pad_token_id", "eos_token_id", "bos_token_id", "chat_template")), "Reloaded tokenizer changed")
        directory = checkpoint / "smoke_evals"
        directory.mkdir()
        values = {}
        for split in ("forget05", "retain95"):
            path = directory / f"{split}.json"
            values[split] = raw_qa_evaluate(model, loaded, snapshot["model"]["template_args"], split, path, limit=8,
                                           dataset_revision=snapshot["data"]["forget"]["TOFU_QA_forget"]["args"]["hf_args"]["revision"])
            sample_rows = read(path)["rows"]
            require(len(sample_rows) == 8 and all(math.isfinite(row[key]) for row in sample_rows
                    for key in ("mean_nll", "answer_probability")), "Incomplete or nonfinite smoke evaluation")
        saved[step] = {"offline_reload": True, "qa": values, "weight_sha256": weight_hash(checkpoint / "model.safetensors")}
        del model
    save_json(output / "smoke_validation.json", {"passed": True, "max_steps": 4, "checkpoints": saved,
        "initial_reference_bitwise_equal": True, "mask_sha256": snapshot["mask_sha256"],
        "unit_tests_sha256": sha256(root / "unit_tests.log")})


def training(root, seed, smoke):
    lock = load_lock(root)
    protocol = lock["protocol"]
    training = dict(protocol["training"])
    if smoke:
        training.update(protocol["checkpoint_validation"])
    output = output_for(root, seed, smoke)
    require(not output.exists(), f"Training already exists: {output}")
    require(int(os.environ.get("WORLD_SIZE", "1")) == training["world_size"], "Run with both v4 GPU ranks")
    original = FORMAL_OUTPUT / "collected_results" / f"NPO_adaptive_token_mean_forget05_seed{seed}_steps120"
    snapshot = read(original / "resolved_config.json")
    source_args = read(original / "training_args.json")
    trainer_cfg = OmegaConf.create(snapshot["trainer"])
    trainer_cfg.args.output_dir = str(output)
    trainer_cfg.args.logging_dir = str(output / "logs")
    if smoke:
        trainer_cfg.args.max_steps = training["max_steps"]
        trainer_cfg.args.save_steps = training["save_steps"]
        trainer_cfg.args.num_train_epochs = math.ceil(training["max_steps"] / training["warmup_steps"])
        trainer_cfg.args.logging_steps = 1
    template = snapshot["model"]["template_args"]
    seed_everything(seed)
    model, tokenizer = get_model(OmegaConf.create(snapshot["model"]))
    model.config.use_cache = False
    require(model.config._commit_hash == protocol["target_model"]["revision"], "Target model revision differs")
    data = get_data(OmegaConf.create(snapshot["data"]), mode="unlearn", tokenizer=tokenizer, template_args=template)["train"]
    for split in ("forget", "retain"):
        dataset = getattr(data, split)
        verify_dataset_revision(dataset, protocol["dataset"]["revision"])
        require(len(dataset) == protocol["dataset"][f"{split}_samples"], f"Dataset size differs: {split}")
    mask_path = root / MASK
    collator = FrozenMaskCollator(DataCollatorForSupervisedDataset(tokenizer, index="index"), mask_path)
    trainer, args = load_trainer(trainer_cfg, model, train_dataset=data, tokenizer=tokenizer, data_collator=collator,
                                 template_args=template)
    require(trainer.normalization == "token_mean" and trainer.args.warmup_steps == training["warmup_steps"]
            and trainer.accelerator.num_processes == training["world_size"], "Frozen trainer settings differ")
    actual_args = args.to_dict()
    if trainer.is_world_process_zero() and not smoke:
        differences = {k: (source_args.get(k), actual_args.get(k)) for k in source_args.keys() | actual_args.keys()
                       if source_args.get(k) != actual_args.get(k) and k not in {"output_dir", "logging_dir", "run_name"}}
        require(not differences, f"v4 TrainingArguments mismatch: {differences}")
    if smoke:
        with torch.no_grad():
            require(all(torch.equal(p, q) for p, q in zip(model.parameters(), trainer.ref_model.parameters(), strict=True)),
                    "Step-0 NPO reference differs from the target initialization")
    snapshot.update(
        task_name=f"NPO_{VARIANT}_token_mean_forget05_seed{seed}", mask_file=str(mask_path), mask_sha256=lock["mask_sha256"],
        frozen_reference_manifest=str(mask_path), frozen_reference_sha256=lock["mask_sha256"],
        frozen_reference_metadata=read(mask_path.with_suffix(".metadata.json")),
        collator={"DataCollatorForSupervisedDataset": {"handler": "DataCollatorForSupervisedDataset",
                   "args": {"padding_side": "right", "index": "index"}}},
        trainer=OmegaConf.to_container(trainer_cfg, resolve=True),
    )
    snapshot["paths"]["output_dir"] = str(output)
    snapshot["paths"]["work_dir"] = str(ROOT)
    snapshot["execution"].update(variant=VARIANT, checkpoint_validation=smoke, mask_replay="frozen_reference_mask")
    snapshot["frozen_protocol"]["checkpoints"] = training["checkpoints"]
    snapshot["extension_manifest_sha256"] = sha256(root / "protocol_manifest.json")
    checks = instrument(trainer)
    callback = StepCheckpointEvaluation(trainer, training, None, output, resolved_config=snapshot, defer_evaluation=True)
    trainer.add_callback(callback)
    if trainer.is_world_process_zero():
        output.mkdir(exist_ok=True, parents=True)
        save_json(output / "resolved_config.json", snapshot)
        save_json(output / "training_args.json", actual_args)
    trainer.accelerator.wait_for_everyone()
    seed_everything(seed)
    result = trainer.train()
    trainer.save_state()
    steps = training["max_steps"]
    require(trainer.state.global_step == steps and math.isfinite(result.training_loss)
            and len(checks["loss"]) == steps * training["gradient_accumulation_steps"]
            and len(checks["gradient_l2_preclip"]) == steps, "Incomplete training finite checks")
    expected = collator.records
    for index, (_, mask) in trainer.mask_selector.cache.items():
        require(mask.int().tolist() == expected[index]["mask"], "NPO used a different frozen mask")
    checks["replayed_samples"] = len(trainer.mask_selector.cache)
    save_json(output / f"finite_checks_rank{args.process_index}.json", checks)
    trainer.accelerator.wait_for_everyone()
    if trainer.is_world_process_zero():
        require(set(callback.saved) == set(training["checkpoints"]), "Checkpoint schedule differs")
        for step in callback.saved:
            checkpoint = output / f"checkpoint-{step}"
            state = read(checkpoint / "trainer_state.json")
            current = int(step)
            logged = next(row for row in reversed(trainer.state.log_history) if row["step"] == current and "loss" in row)
            per_rank = [read(output / f"finite_checks_rank{rank}.json")["gradient_l2_preclip"][current - 1]
                        for rank in range(training["world_size"])]
            save_json(checkpoint / "training_diagnostics.json", {"global_step": current, "actual_epoch": state["epoch"],
                "logged_training_loss": logged["loss"], "learning_rate_after_step": logged["learning_rate"],
                "gradient_l2_preclip_by_rank": per_rank, "gradient_l2_preclip": per_rank[0]})
        if smoke:
            smoke_eval(root, output, snapshot, tokenizer, trainer.accelerator.device, training)
        save_json(output / "training_completed.json", {"global_step": steps, "seed": seed, "checkpoints": callback.saved,
            "training_loss": result.training_loss, "mask_sha256": lock["mask_sha256"], "code_commit": lock["code_commit"]})
    trainer.accelerator.wait_for_everyone()


def replay_metrics(cfg, logs, reference):
    """Independently aggregate the archived per-sample scores using v4 formulas."""
    def values(name, field):
        return np.asarray([row[field] for row in logs[name]["value_by_index"].values()], dtype=np.float64)

    def aggregation(name, spec):
        handler = spec["handler"]
        if handler in {"probability", "probability_w_options"}:
            return np.mean(values(name, "prob"))
        if handler == "rouge":
            return np.mean(values(name, spec["rouge_type"]))
        if handler == "truth_ratio":
            scores = values(name, "score")
            return np.mean(np.maximum(0, 1 - scores) if spec["aggregator"] == "true_better"
                           else np.minimum(scores, 1 / (scores + 1e-10)))
        raise ValueError(f"Unexpected replay handler: {handler}")

    metrics = cfg["metrics"]
    truth = values("forget_truth_ratio", "score")
    ref_truth = [row["score"] for row in reference["forget_truth_ratio"]["value_by_index"].values()]
    scores = [[row["score"] for row in logs["mia_min_k"][split]["value_by_index"].values()] for split in ("forget", "holdout")]
    auc = roc_auc_score([0] * len(scores[0]) + [1] * len(scores[1]), scores[0] + scores[1])
    ref_auc = reference["mia_min_k"]["agg_value"]
    return {
        "ES": float(np.mean(values("extraction_strength", "score"))),
        "forget_probability": float(aggregation("forget_Q_A_Prob", metrics["forget_Q_A_Prob"])),
        "forget_rouge": float(aggregation("forget_Q_A_ROUGE", metrics["forget_Q_A_ROUGE"])),
        "forget_ratio": float(aggregation("forget_truth_ratio", metrics["forget_truth_ratio"])),
        "forget_quality": float(ks_2samp(truth, ref_truth).pvalue),
        "utility": float(hmean([aggregation(name, spec) for name, spec in metrics["model_utility"]["pre_compute"].items()])),
        "privleak": float(((1 - auc) - (1 - ref_auc)) / (1 - ref_auc + 1e-10) * 100),
    }


def official_eval(root, seed, step):
    lock = load_lock(root)
    protocol = lock["protocol"]
    checkpoint = output_for(root, seed) / f"checkpoint-{step}"
    state = read(checkpoint / "trainer_state.json")
    metadata = read(checkpoint / "checkpoint_metadata.json")
    snapshot = read(checkpoint / "resolved_config.json")
    diagnostics = read(checkpoint / "training_diagnostics.json")
    require(state["global_step"] == metadata["global_step"] == step and state["epoch"] == metadata["actual_epoch"]
            and metadata["seed"] == seed and metadata["code_commit"] == lock["code_commit"]
            and metadata["protocol_sha256"] == lock["v4_protocol_sha256"] and metadata["mask_sha256"] == lock["mask_sha256"]
            and snapshot["extension_manifest_sha256"] == sha256(root / "protocol_manifest.json"), "Checkpoint provenance differs")
    reference = root / "retain_only_reference_TOFU_EVAL.json"
    reference_key = protocol["retain_only_model"]["evaluation_logs"]
    require(sha256(reference) == protocol["inputs"][reference_key], "Retain-only reference differs")
    output = checkpoint / "evals/official"
    output.mkdir(exist_ok=True, parents=True)
    config = configure_eval(protocol, output, reference)
    signature = {k: v for k, v in config.items() if k not in {"output_dir", "retain_logs_path"}}
    require(signature == read(root / "v4_evaluation_signature.json"), "Official evaluation differs from v4")
    weight_paths = [checkpoint / name for name in metadata["saved_weight_files"]]
    require(weight_paths and all(p.is_file() and not p.is_symlink() for p in weight_paths), "Expected checkpoint weights")
    provenance = {"checkpoint": metadata, "resolved_eval_config": config, "evaluator_sha256": lock["evaluator_sha256"],
                  "model": snapshot["model"], "dataset_revision": protocol["dataset"]["revision"],
                  "retain_only_reference_sha256": sha256(reference), "weight_sha256": {p.name: weight_hash(p) for p in weight_paths}}
    prior = output / "evaluation_provenance.json"
    if prior.exists():
        require(read(prior) == provenance, "Evaluation restart provenance differs")
    else:
        save_json(prior, provenance)
    if not (output / "validated.json").exists():
        model, tokenizer = reload_model(snapshot, checkpoint, "cuda:0")
        get_evaluators(OmegaConf.create({"tofu": config}))["tofu"].evaluate(
            model=model, tokenizer=tokenizer, template_args=snapshot["model"]["template_args"])
    logs = read(output / "TOFU_EVAL.json")
    aggregate, counts = validate_outputs(config, protocol["official_evaluation"], logs, read(output / "TOFU_SUMMARY.json"), reference)
    recomputed = replay_metrics(config, logs, read(reference))
    require(all(math.isclose(aggregate[key], recomputed[key], rel_tol=1e-10, abs_tol=1e-14) for key in aggregate),
            f"Cannot reproduce official metrics from per-sample outputs: {aggregate}, {recomputed}")
    save_json(output / "official_metrics.json", aggregate)
    save_json(output / "validated.json", {**metadata, "variant": VARIANT, "metrics": aggregate, "recomputed_metrics": recomputed,
        "sample_counts": counts, "training_diagnostics": diagnostics, "evaluator_sha256": lock["evaluator_sha256"],
        "artifact_sha256": {name: digest(output / name) for name in ARTIFACTS if name != "validated.json"}})
    print(f"VALIDATED seed={seed} step={step}", flush=True)


def copy_verified(source, dest):
    dest.parent.mkdir(exist_ok=True, parents=True)
    shutil.copy2(source, dest)
    checksum = digest(source)
    require(digest(dest) == checksum, f"Archive copy differs: {dest}")
    if dest.suffix == ".json":
        read(dest)
    return checksum


def archive(root, seed, step, manifest):
    run = output_for(root, seed)
    checkpoint = run / f"checkpoint-{step}"
    key = f"seed{seed}/checkpoint-{step}"
    dest = root / "collected_results" / key
    if key not in manifest:
        dest.parent.mkdir(exist_ok=True, parents=True)
        with tempfile.TemporaryDirectory(prefix=".collect-", dir=dest.parent) as directory:
            stage = Path(directory) / checkpoint.name
            files = {str(p.relative_to(checkpoint)): copy_verified(p, stage / p.relative_to(checkpoint))
                     for p in sorted(checkpoint.rglob("*")) if p.is_file() and not p.name.endswith((".safetensors", ".safetensors.index.json", ".tmp"))}
            folder = "evals/official"
            validated = read(stage / folder / "validated.json")
            require(validated["seed"] == seed and validated["step"] == step and set(validated["metrics"]) == set(METRICS),
                    f"Copied metrics differ: {key}")
            for name, checksum in validated["artifact_sha256"].items():
                require(files[f"{folder}/{name}"] == checksum, f"Copied evaluation hash differs: {key}/{name}")
            if not dest.exists():
                stage.rename(dest)
            else:
                require(all(digest(dest / file) == checksum for file, checksum in files.items()), f"Existing archive differs: {key}")
        manifest[key] = {"files": files, "validated": validated}
        save_json(root / "incremental_manifest.json", manifest)
    require(all(digest(dest / name) == checksum for name, checksum in manifest[key]["files"].items()),
            f"Archived checkpoint files changed: {key}")
    provenance = read(dest / "evals/official/evaluation_provenance.json")
    for filename, checksum in provenance["weight_sha256"].items():
        require(Path(filename).name == filename and filename.endswith(".safetensors"), "Unexpected checkpoint weight filename")
        path = checkpoint / filename
        if path.exists():
            require(not path.is_symlink() and weight_hash(path) == checksum, f"Checkpoint weights changed: {path}")
            path.unlink()
    print(f"ARCHIVED {key}", flush=True)


def comparison(root, manifest):
    keys = ["variant", "seed", "step", "actual_epoch", "selection_ratio", *TABLE_METRICS]
    result = [{key: row[key] for key in keys} for row in rows(root / "existing_controls.csv")]
    for item in manifest.values():
        validated = item["validated"]
        metrics = dict(validated["metrics"])
        signed = metrics.pop("privleak")
        result.append({**{k: validated[k] for k in ("variant", "seed", "step", "actual_epoch")},
                       "selection_ratio": 2417 / 7165, **metrics, "privleak_signed": signed, "privleak_abs": abs(signed)})
    write_csv(root / "official_metrics_long.csv", result)
    index = {(row["variant"], int(row["seed"]), int(row["step"])): row for row in result}
    require(len(index) == len(result) == 24, "Comparison matrix is incomplete")
    paired = []
    for step in STEPS:
        table = []
        for variant in (VARIANT, *CONTROLS):
            for metric in TABLE_METRICS:
                values = [float(index[variant, seed, step][metric]) for seed in SEEDS]
                table.append({"variant": variant, "step": step, "metric": metric,
                              **{f"seed_{s}": v for s, v in zip(SEEDS, values)}, "mean": mean(values), "sample_std": stdev(values), "n": 3})
                if variant == VARIANT:
                    for control in CONTROLS:
                        diff = [v - float(index[control, seed, step][metric]) for seed, v in zip(SEEDS, values)]
                        signs = [(v > 0) - (v < 0) for v in diff]
                        paired.append({"step": step, "lhs": VARIANT, "rhs": control, "metric": metric,
                            **{f"seed_{s}": v for s, v in zip(SEEDS, diff)}, "mean_difference": mean(diff),
                            "sample_std_difference": stdev(diff), "n": 3, "direction_consistent": len(set(signs)) == 1,
                            "direction": {1: "all_positive", 0: "all_zero", -1: "all_negative"}[signs[0]]
                                         if len(set(signs)) == 1 else "mixed"})
        write_csv(root / f"step{step}_mean_std.csv", table)
    write_csv(root / "paired_comparisons.csv", paired)


def pack(root):
    files = {}
    for path in sorted(root.iterdir()):
        if path.is_file() and path.name not in {"queue.log", "queue.lock", "queue.pid", "status.json"}:
            files[path.name] = path
    for name in ("mask_audit", "smoke_test", "collected_results", "logs"):
        for path in sorted((root / name).rglob("*")):
            if path.is_file() and not path.name.endswith((".safetensors", ".safetensors.index.json", ".tmp")):
                files[str(path.relative_to(root))] = path
    archive = root.parents[1] / "exports" / f"{NAME}.zip"
    archive.parent.mkdir(exist_ok=True)
    checksums = {name: digest(path) for name, path in files.items()}
    pending = archive.with_suffix(".zip.part")
    with ZipFile(pending, "w", ZIP_DEFLATED, compresslevel=4) as bundle:
        for name, path in files.items():
            bundle.write(path, f"{NAME}/{name}")
        bundle.writestr(f"{NAME}/MANIFEST.json", json.dumps({"files": checksums}, indent=2) + "\n")
    with ZipFile(pending) as bundle:
        require(len(bundle.namelist()) == len(files) + 1, "ZIP member count differs")
        for name, checksum in checksums.items():
            with bundle.open(f"{NAME}/{name}") as stream:
                require(hashlib.file_digest(stream, "sha256").hexdigest() == checksum, f"ZIP member SHA-256 differs: {name}")
    checksum = digest(pending)
    pending.replace(archive)
    archive.with_suffix(".zip.sha256").write_text(f"{checksum}  {archive.name}\n")
    return archive


def run_queue(root, code_commit):
    phase = "preparing"
    active = None
    manifest_path = root / "incremental_manifest.json"
    manifest = read(manifest_path) if manifest_path.exists() else {}
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--root", str(root)]
    logs = root / "logs"
    logs.mkdir(exist_ok=True)

    def status(new_phase, seed=None, error=None):
        nonlocal phase, active
        phase, active = new_phase, seed
        payload = {"phase": phase, "seed": active, "archived_official_checkpoints": len(manifest), "error": error}
        save_json(root / "status.json", payload)
        print(json.dumps(payload), flush=True)

    def train(seed, smoke=False):
        output = output_for(root, seed, smoke)
        if (output / "training_completed.json").exists():
            return
        require(not output.exists(), f"Partial training exists: {output}")
        torchrun = [sys.executable, "-u", "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
                    *command[2:], "--train", str(seed), *(["--smoke"] if smoke else [])]
        with (logs / ("smoke.train.log" if smoke else f"seed{seed}.train.log")).open("x") as log:
            subprocess.run(torchrun, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)

    try:
        status("unit_tests")
        with (root / "unit_tests.log").open("w") as log:
            subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_low_confidence.py", "-v"],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, check=True)
        status("mask_audit")
        if not (root / "protocol_manifest.json").exists():
            with (logs / "mask_audit.log").open("a") as log:
                subprocess.run([sys.executable, "-u", str(ROOT / "scripts/prepare_low_confidence.py"),
                                "--output", str(root), "--code-commit", code_commit], cwd=ROOT,
                               env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"}, stdout=log, stderr=subprocess.STDOUT, check=True)
        lock = load_lock(root)
        require(lock["code_commit"] == code_commit, "Unexpected v4 commit")
        status("smoke_test", 0)
        train(0, smoke=True)
        smoke = read(output_for(root, 0, True) / "smoke_validation.json")
        require(smoke["passed"] and set(smoke["checkpoints"]) == {"2", "4"} and smoke["mask_sha256"] == lock["mask_sha256"],
                "Smoke test did not validate both checkpoints")
        for seed in SEEDS:
            status("training", seed)
            train(seed)
            status("official_evaluation", seed)
            pending = [step for step in STEPS if f"seed{seed}/checkpoint-{step}" not in manifest]
            for attempt in range(2):
                jobs = {}
                for gpu, step in enumerate(pending):
                    with (logs / f"seed{seed}_step{step}.eval.log").open("a") as log:
                        jobs[step] = subprocess.Popen([*command, "--evaluate", str(seed), str(step)], cwd=ROOT,
                            env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}, stdout=log, stderr=subprocess.STDOUT)
                pending = [step for step, process in jobs.items() if process.wait() != 0]
                if not pending:
                    break
            require(not pending, f"Official checkpoint evaluation failed for seed {seed}, steps {pending}")
            status("archive", seed)
            for step in STEPS:
                archive(root, seed, step, manifest)
            for name in ["resolved_config.json", "training_args.json", "training_completed.json", "trainer_state.json",
                         "finite_checks_rank0.json", "finite_checks_rank1.json"]:
                copy_verified(output_for(root, seed) / name, root / "collected_results" / f"seed{seed}" / name)
        comparison(root, manifest)
        (root / "README.txt").write_text(
            "TOFU Forget05 LowConfidence-Matched selection-criterion baseline\n"
            "The initial target alone supplies absolute token log-probabilities. Adaptive supplies only per-example counts.\n"
            "The v4 TokenNormalizedNPO objective, dataset, model and training protocol are shared with the archived controls.\n"
            "The prepared mask is fixed across seeds. Step 120 is primary. Step 60 is a training-dynamics checkpoint.\n"
            "official_metrics_long.csv includes three new and nine existing runs. The paired differences are LowConfidence-Matched minus control.\n"
            "Both signed PrivLeak and absolute PrivLeak are saved. Absolute PrivLeak is the paper comparison magnitude.\n"
            "collected_results contains independently verified evaluations, training diagnostics, tokenizer and configs.\n"
            "The formal model weights were removed after validating the archived copies.\n"
        )
        output = pack(root)
        status("complete")
        print(output, flush=True)
    except Exception as error:
        status(f"failed_{phase}", active, str(error))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=OUTPUT)
    parser.add_argument("--code-commit")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--train", type=int)
    modes.add_argument("--evaluate", nargs=2, type=int, metavar=("SEED", "STEP"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    logging.basicConfig(level=logging.INFO)
    if args.train is not None:
        require(args.train in SEEDS and (not args.smoke or args.train == 0), "Unexpected training seed")
        training(root, args.train, args.smoke)
    elif args.evaluate:
        require(args.evaluate[0] in SEEDS and args.evaluate[1] in STEPS, "Unexpected evaluation target")
        official_eval(root, *args.evaluate)
    else:
        require(args.code_commit, "Supply the Git commit of the frozen v4 source snapshot")
        root.mkdir(exist_ok=True, parents=True)
        with (root / "queue.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            run_queue(root, args.code_commit)


if __name__ == "__main__":
    main()
