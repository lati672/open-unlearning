"""Run the predeclared SU-NGram-Paper P0 sweep, then confirm one selected prefix."""

import argparse
import csv
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
import time
from zipfile import ZIP_DEFLATED, ZipFile

from omegaconf import OmegaConf
import torch
from transformers import TrainerCallback

from evaluate_tofu_checkpoint import METRICS, configure_eval, get_evaluators, read, require, validate_outputs, weight_hash
from freeze_mask_protocol import DEFAULT_PROTOCOL, check_environment, save_json, sha256
from prepare_su_ngram import OUTPUT, ROOT, data_and_tokenizer
from run_su_ngram import finite_checks, get_model, load_prepared, model_config
from run_su_ngram_matrix import ARTIFACTS, digest, eval_protocol
from data.collators import DataCollatorForSupervisedDataset, FrozenMaskCollator
from data.unlearn import ForgetRetainDataset
from trainer import load_trainer
from trainer.utils import seed_everything


NAME = "tofu_forget05_su_ngram_p0"
SWEEP = OUTPUT / "p0_lr_sweep"
INITIAL = ROOT / "saves/eval/tofu_Llama-3.2-1B-Instruct_full/evals_forget05"
VARIANT = "SU-NGram-Paper"
RATES = ("1e-6", "3e-6", "1e-5")
STEPS = (5, 10, 20, 40, 60)
SCHEDULE_STEPS = 120
WARMUP_STEPS = 6
MIN_UTILITY_RETAINED = .90
MIN_FORGET_PROB_DROP = .10
TABLE_METRICS = [k for k in METRICS if k != "privleak"] + ["privleak_signed", "privleak_abs"]


def run_id(rate, seed):
    return f"lr_{rate}_seed{seed}"


def plan_path(root):
    return root / "run_plan.json"


def freeze(root):
    cfg, mask = load_prepared(OUTPUT, VARIANT)
    protocol, source_reference = eval_protocol(OUTPUT)
    check_environment(protocol)
    root.mkdir(parents=True, exist_ok=True)
    reference = root / "retain_only_reference_TOFU_EVAL.json"
    target = root / "initial_target_TOFU_SUMMARY.json"
    for source, dest in [(source_reference, reference), (INITIAL / "TOFU_SUMMARY.json", target)]:
        if not dest.exists():
            shutil.copy2(source, dest)
        require(sha256(source) == sha256(dest), f"Frozen evaluation input differs: {dest}")
    baseline = read(target)
    baseline_utility = baseline["model_utility"]
    baseline_forget_probability = baseline["forget_Q_A_Prob"]
    require(cfg["training"]["max_steps"] == SCHEDULE_STEPS and cfg["training"]["warmup_steps"] == WARMUP_STEPS,
            "Prepared training schedule differs")
    plan = {
        "method": VARIANT, "learning_rates": list(RATES), "sweep_seed": 0, "steps": list(STEPS),
        "confirmation_seeds": [0, 1, 2], "schedule_steps": SCHEDULE_STEPS, "warmup_steps": WARMUP_STEPS,
        "scheduler": "linear", "mask_sha256": sha256(mask), "prepared_config": cfg,
        "prepared_protocol_sha256": sha256(OUTPUT / "protocol_manifest.json"),
        "evaluator_protocol_sha256": sha256(DEFAULT_PROTOCOL), "evaluation_settings": protocol["official_evaluation"],
        "reference_sha256": sha256(reference), "target_summary_sha256": sha256(target),
        "runner_sha256": sha256(Path(__file__)), "evaluator_sha256": sha256(ROOT / "scripts/evaluate_tofu_checkpoint.py"),
        "baseline": {"utility": baseline_utility, "forget_probability": baseline_forget_probability},
        "selection_rule": {
            "minimum_utility_fraction": MIN_UTILITY_RETAINED,
            "utility_floor": MIN_UTILITY_RETAINED * baseline_utility,
            "minimum_relative_fprob_drop": MIN_FORGET_PROB_DROP,
            "fprob_ceiling": (1 - MIN_FORGET_PROB_DROP) * baseline_forget_probability,
            "score": "harmonic mean of min(1, Utility / initial Utility) and 1 - F-Prob / initial F-Prob",
            "tiebreakers": ["higher Utility", "fewer optimizer steps", "lower learning rate"],
            "eligible_rows": "all 15 seed-0 sweep checkpoints",
            "no_eligible_checkpoint": "stop without confirmation runs",
        },
    }
    path = plan_path(root)
    if path.exists():
        require(read(path) == plan, "Frozen sweep plan differs")
    else:
        save_json(path, plan)
    return plan


class Milestones(TrainerCallback):
    """Save chosen steps while leaving the full-horizon LR scheduler intact."""

    def __init__(self, steps):
        self.steps = tuple(steps)
        self.before = []
        self.norms = {}

    @torch.no_grad()
    def on_pre_optimizer_step(self, args, state, control, model=None, optimizer=None, **kwargs):
        if state.is_world_process_zero and state.global_step + 1 in self.steps:
            self.before = [(parameter, parameter.detach().clone()) for parameter in model.parameters() if parameter.requires_grad]
            self.lr = optimizer.param_groups[0]["lr"]

    @torch.no_grad()
    def on_optimizer_step(self, args, state, control, **kwargs):
        if not self.before:
            return
        device = self.before[0][0].device
        param_sum = torch.zeros((), dtype=torch.float64, device=device)
        update_sum = torch.zeros_like(param_sum)
        for parameter, previous in self.before:
            for after_chunk, prior_chunk in zip(parameter.detach().flatten().split(2**20), previous.flatten().split(2**20), strict=True):
                after = after_chunk.float()
                param_sum += torch.linalg.vector_norm(after).double().square()
                update_sum += torch.linalg.vector_norm(after - prior_chunk.float()).double().square()
        self.before = []
        self.norms[state.global_step + 1] = {
            "parameter_l2": float(param_sum.sqrt()), "optimizer_step_update_l2": float(update_sum.sqrt()), "lr_used": self.lr,
        }

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step in self.steps:
            control.should_save = True
        if state.global_step == self.steps[-1]:
            control.should_training_stop = True
        return control


def load_plan(root):
    plan = read(plan_path(root))
    require(plan["runner_sha256"] == sha256(Path(__file__)) and plan["evaluator_protocol_sha256"] == sha256(DEFAULT_PROTOCOL)
            and plan["evaluator_sha256"] == sha256(ROOT / "scripts/evaluate_tofu_checkpoint.py"), "Frozen sweep code changed")
    return plan


def steps_for(root, rate, seed):
    if seed == 0:
        return STEPS
    selection = read(root / "selection.json")
    require(selection["selected"] is not None and selection["selected"]["learning_rate_label"] == rate,
            "Confirmation can only use the selected configuration")
    return (selection["selected"]["step"],)


def train(root, rate, seed):
    plan = load_plan(root)
    steps = steps_for(root, rate, seed)
    cfg, mask = load_prepared(OUTPUT, VARIANT)
    require(cfg == plan["prepared_config"] and sha256(mask) == plan["mask_sha256"], "Prepared training input changed")
    training = cfg["training"]
    require(int(os.environ.get("WORLD_SIZE", "1")) == training["world_size"], "Use both configured GPU ranks")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    seed_everything(seed)
    output = root / "runs" / run_id(rate, seed)
    require(not output.exists(), f"Training directory already exists: {output}")
    tokenizer, template, splits = data_and_tokenizer(cfg)
    dataset = ForgetRetainDataset(**splits)
    collator = FrozenMaskCollator(DataCollatorForSupervisedDataset(tokenizer, index="index"), mask)
    require(set(collator.records) == set(range(len(splits["forget"]))), "Frozen mask IDs differ")
    mcfg = model_config(cfg, template)
    model, model_tokenizer = get_model(OmegaConf.create(OmegaConf.to_container(mcfg, resolve=True)))
    require(model.config._commit_hash == cfg["model"]["target_revision"] and tokenizer.get_vocab() == model_tokenizer.get_vocab(),
            "Initial model or tokenizer differs")
    trainer_cfg = OmegaConf.load(ROOT / "configs/trainer/SelectiveGradientDifference.yaml")
    for coefficient in ("gamma_forget", "alpha_retain"):
        trainer_cfg.method_args[coefficient] = cfg["objective"][coefficient]
    trainer_cfg.args = {
        "output_dir": str(output), "max_steps": SCHEDULE_STEPS, "warmup_steps": WARMUP_STEPS,
        "lr_scheduler_type": "linear", "learning_rate": float(rate), "per_device_train_batch_size": training["batch_size"],
        "gradient_accumulation_steps": training["gradient_accumulation_steps"], "weight_decay": training["weight_decay"],
        "max_grad_norm": training["max_grad_norm"], "optim": training["optim"], "seed": seed,
        "bf16": True, "gradient_checkpointing": True, "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "ddp_find_unused_parameters": False, "remove_unused_columns": False, "logging_steps": 1,
        "logging_nan_inf_filter": False, "save_only_model": True, "save_safetensors": True, "save_strategy": "no", "report_to": [],
    }
    trainer, args = load_trainer(trainer_cfg, model, train_dataset=dataset, tokenizer=tokenizer, data_collator=collator,
                                 template_args=template)
    milestones = Milestones(steps)
    trainer.add_callback(milestones)
    finite = finite_checks(trainer)
    snapshot = {"method": VARIANT, "seed": seed, "learning_rate_label": rate, "optimizer_stop_step": steps[-1],
                "checkpoint_steps": list(steps), "mask_sha256": plan["mask_sha256"],
                "sweep_protocol_sha256": sha256(plan_path(root)),
                "model": OmegaConf.to_container(mcfg, resolve=True), "training_args": args.to_dict()}
    if trainer.is_world_process_zero():
        output.mkdir(exist_ok=True, parents=True)
        save_json(output / "resolved_config.json", snapshot)
    trainer.accelerator.wait_for_everyone()
    start = time.monotonic()
    result = trainer.train()
    seconds = time.monotonic() - start
    require(trainer.state.global_step == steps[-1] and math.isfinite(result.training_loss)
            and len(finite["grad_norm"]) == steps[-1]
            and len(finite["loss"]) == steps[-1] * training["gradient_accumulation_steps"], "Training or finite-check coverage differs")
    save_json(output / f"finite_checks_rank{args.process_index}.json", finite)
    trainer.accelerator.wait_for_everyone()
    if trainer.is_world_process_zero():
        require(set(milestones.norms) == set(steps), "Missing optimizer norm callbacks")
        ranks = [read(output / f"finite_checks_rank{rank}.json") for rank in range(training["world_size"])]
        for step in steps:
            checkpoint = output / f"checkpoint-{step}"
            state = read(checkpoint / "trainer_state.json")
            require(state["global_step"] == step and state["max_steps"] == SCHEDULE_STEPS, f"Wrong scheduler or checkpoint step: {checkpoint}")
            norms = milestones.norms[step]
            norms["gradient_l2_preclip_by_rank"] = [r["grad_norm"][step - 1] for r in ranks]
            norms["gradient_l2_preclip"] = norms["gradient_l2_preclip_by_rank"][0]
            expected_lr = snapshot["training_args"]["learning_rate"] * (
                (step - 1) / WARMUP_STEPS if step - 1 < WARMUP_STEPS else (SCHEDULE_STEPS - step + 1) / (SCHEDULE_STEPS - WARMUP_STEPS)
            )
            require(math.isclose(norms["lr_used"], expected_lr, rel_tol=1e-8, abs_tol=1e-15), "Observed LR differs from frozen scheduler")
            require(all(math.isfinite(value) for key, value in norms.items() if key != "gradient_l2_preclip_by_rank"), "Nonfinite parameter/update norms")
            save_json(checkpoint / "optimizer_diagnostics.json", norms)
            save_json(checkpoint / "resolved_config.json", snapshot)
        save_json(output / "training_completed.json", {"seed": seed, "learning_rate_label": rate, "global_step": steps[-1],
                                                       "checkpoints": list(steps), "train_wall_seconds": seconds, "loss": result.training_loss,
                                                       "sweep_protocol_sha256": snapshot["sweep_protocol_sha256"]})
    trainer.accelerator.wait_for_everyone()


def evaluate(root, rate, seed, step):
    plan = load_plan(root)
    require(step in steps_for(root, rate, seed), "Unexpected evaluation step")
    run = root / "runs" / run_id(rate, seed)
    checkpoint = run / f"checkpoint-{step}"
    snapshot = read(checkpoint / "resolved_config.json")
    state = read(checkpoint / "trainer_state.json")
    require(snapshot["sweep_protocol_sha256"] == sha256(plan_path(root)) and snapshot["mask_sha256"] == plan["mask_sha256"]
            and snapshot["seed"] == seed and snapshot["learning_rate_label"] == rate
            and state["global_step"] == step and state["max_steps"] == SCHEDULE_STEPS
            and math.isclose(state["epoch"], .16 * step, abs_tol=1e-8), f"Checkpoint provenance differs: {checkpoint}")
    weights = sorted(checkpoint.glob("*.safetensors"))
    require(weights and all(not path.is_symlink() for path in weights), f"Missing checkpoint weights: {checkpoint}")
    reference = root / "retain_only_reference_TOFU_EVAL.json"
    require(sha256(reference) == plan["reference_sha256"], "Retain-only evaluation reference hash differs")
    protocol = read(DEFAULT_PROTOCOL)
    output = checkpoint / "evals" / "official"
    output.mkdir(parents=True, exist_ok=True)
    cfg = configure_eval(protocol, output, reference)
    meta = {"method": VARIANT, "seed": seed, "learning_rate_label": rate, "step": step, "actual_epoch": state["epoch"],
            "scheduler_horizon": SCHEDULE_STEPS, "mask_sha256": plan["mask_sha256"], "protocol_sha256": snapshot["sweep_protocol_sha256"],
            "code_commit": read(OUTPUT / "protocol_manifest.json")["git_head"],
            "optimizer_diagnostics": read(checkpoint / "optimizer_diagnostics.json")}
    provenance = {"checkpoint": meta, "resolved_eval_config": cfg, "retain_only_reference_sha256": plan["reference_sha256"],
                  "weight_sha256": {path.name: weight_hash(path) for path in weights}}
    prov_path = output / "evaluation_provenance.json"
    if prov_path.exists():
        require(read(prov_path) == provenance, f"Existing official evaluation provenance differs: {checkpoint}")
    else:
        save_json(prov_path, provenance)
    if not (output / "validated.json").exists():
        mcfg = OmegaConf.create(snapshot["model"])
        for args in [mcfg.model_args, mcfg.tokenizer_args]:
            args.pop("revision", None)
            args.pretrained_model_name_or_path = str(checkpoint)
            args.local_files_only = True
        mcfg.model_args.device_map = {"": "cuda:0"}
        model, tokenizer = get_model(mcfg)
        get_evaluators(OmegaConf.create({"tofu": cfg}))["tofu"].evaluate(model=model, tokenizer=tokenizer, template_args=mcfg.template_args)
    aggregate, counts = validate_outputs(cfg, plan["evaluation_settings"], read(output / "TOFU_EVAL.json"),
                                        read(output / "TOFU_SUMMARY.json"), reference)
    save_json(output / "official_metrics.json", aggregate)
    save_json(output / "validated.json", {**meta, "metrics": aggregate, "sample_counts": counts,
                                         "artifact_sha256": {name: digest(output / name) for name in ARTIFACTS if name != "validated.json"}})
    print(f"VALIDATED {run_id(rate, seed)} step {step}", flush=True)


def copy_verified(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    checksum = digest(source)
    require(digest(target) == checksum, f"Collected file hash differs: {target}")
    if target.suffix == ".json":
        read(target)
    return checksum


def archive_checkpoint(root, rate, seed, step, manifest):
    name = run_id(rate, seed)
    run = root / "runs" / name
    checkpoint = run / f"checkpoint-{step}"
    key = f"{name}/checkpoint-{step}"
    collected = root / "collected_results" / key
    if key not in manifest:
        collected.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".stage-", dir=collected.parent) as directory:
            staging = Path(directory) / checkpoint.name
            staging.mkdir()
            files = {str(path.relative_to(checkpoint)): copy_verified(path, staging / path.relative_to(checkpoint))
                     for path in sorted(checkpoint.rglob("*")) if path.is_file() and not path.name.endswith((".safetensors", ".safetensors.index.json", ".tmp"))}
            rel = "evals/official"
            validated = read(staging / rel / "validated.json")
            require(validated["seed"] == seed and validated["step"] == step and validated["learning_rate_label"] == rate
                    and set(validated["metrics"]) == set(METRICS) and all(math.isfinite(v) for v in validated["metrics"].values()),
                    f"Wrong collected evaluation: {key}")
            for item in ARTIFACTS:
                if item != "validated.json":
                    require(files[f"{rel}/{item}"] == validated["artifact_sha256"][item], f"Evaluation hash differs: {key}/{item}")
            if not collected.exists():
                staging.rename(collected)
            else:
                require(all(digest(collected / file) == checksum for file, checksum in files.items()), f"Existing copy differs: {key}")
        manifest[key] = {"files": files, "validated": validated}
        save_json(root / "incremental_manifest.json", manifest)
    record = manifest[key]
    for file, checksum in record["files"].items():
        require(digest(collected / file) == checksum, f"Collected result changed: {key}/{file}")
    provenance = read(collected / "evals/official/evaluation_provenance.json")
    weights = []
    for filename, expected in provenance["weight_sha256"].items():
        require(Path(filename).name == filename and filename.endswith(".safetensors"), "Unexpected model filename")
        path = checkpoint / filename
        if path.exists():
            require(not path.is_symlink() and weight_hash(path) == expected, f"Checkpoint hash changed before cleanup: {path}")
            weights.append(path)
    for path in weights:
        path.unlink()
    print(f"ARCHIVED {key} removed_weights={len(weights)}", flush=True)


def rows_from(manifest):
    rows = []
    for record in manifest.values():
        item = record["validated"]
        metrics = dict(item["metrics"])
        signed = metrics.pop("privleak")
        row = {"learning_rate_label": item["learning_rate_label"], "learning_rate": float(item["learning_rate_label"]),
               "seed": item["seed"], "step": item["step"], "actual_epoch": item["actual_epoch"],
               **metrics, "privleak_signed": signed, "privleak_abs": abs(signed)}
        row.update({key: item["optimizer_diagnostics"][key] for key in
                    ("gradient_l2_preclip", "parameter_l2", "optimizer_step_update_l2", "lr_used")})
        rows.append(row)
    return sorted(rows, key=lambda row: (row["seed"], row["learning_rate"], row["step"]))


def choose(plan, rows):
    baseline = plan["baseline"]
    rule = plan["selection_rule"]
    require(len(rows) == len(RATES) * len(STEPS)
            and {(row["learning_rate_label"], row["step"], row["seed"]) for row in rows}
            == {(rate, step, 0) for rate in RATES for step in STEPS}, "Selection requires the complete seed-0 sweep")
    scored = []
    for row in rows:
        retained = row["utility"] / baseline["utility"]
        drop = 1 - row["forget_probability"] / baseline["forget_probability"]
        reasons = []
        if row["utility"] < rule["utility_floor"]:
            reasons.append("utility_below_floor")
        if row["forget_probability"] > rule["fprob_ceiling"]:
            reasons.append("fprob_drop_below_floor")
        u = min(1, retained)
        score = None if reasons else 2 * u * drop / (u + drop)
        scored.append({**row, "utility_fraction": retained, "relative_fprob_drop": drop, "eligible": not reasons,
                       "tradeoff_score": score, "exclusions": "|".join(reasons)})
    eligible = [row for row in scored if row["eligible"]]
    selected = max(eligible, key=lambda r: (r["tradeoff_score"], r["utility"], -r["step"], -r["learning_rate"])) if eligible else None
    return selected, scored


def csv_write(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pack(root, final=False):
    target = ROOT / "saves/exports" / f"{NAME}{'' if final else '_partial'}.zip"
    target.parent.mkdir(exist_ok=True)
    files = {str(path.relative_to(root)): path for path in root.rglob("*") if path.is_file()
             and path.relative_to(root).parts[0] not in {"runs", "logs"}
             and path.name not in {"status.json", "queue.log", "queue.pid", "queue.lock"}
             and path.suffix != ".tmp"}
    pending = target.with_suffix(".zip.part")
    listing = {name: digest(path) for name, path in files.items()}
    with ZipFile(pending, "w", ZIP_DEFLATED, compresslevel=3) as archive:
        for name, path in sorted(files.items()):
            archive.write(path, f"{NAME}/{name}")
        archive.writestr(f"{NAME}/MANIFEST.json", json.dumps(listing, indent=2) + "\n")
    with ZipFile(pending) as archive:
        require(len(archive.namelist()) == len(listing) + 1, "Unexpected ZIP member count")
        for name, expected in listing.items():
            with archive.open(f"{NAME}/{name}") as stream:
                require(hashlib.file_digest(stream, "sha256").hexdigest() == expected, f"ZIP member hash differs: {name}")
    checksum = digest(pending)
    pending.replace(target)
    target.with_suffix(".zip.sha256").write_text(f"{checksum}  {target.name}\n")
    return target


def execute_run(root, rate, seed, manifest, status):
    name = run_id(rate, seed)
    run = root / "runs" / name
    logs = root / "logs"
    logs.mkdir(exist_ok=True)
    training_log = logs / f"{name}.train.log"
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--root", str(root)]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0,1"}
    if not (run / "training_completed.json").exists():
        require(not run.exists(), f"Partial training exists. Inspect the log: {training_log}")
        status("training", name)
        torchrun = [sys.executable, "-u", "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
                    *command[2:], "--train", rate, str(seed)]
        with training_log.open("a") as log:
            subprocess.run(torchrun, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    completed = read(run / "training_completed.json")
    steps = steps_for(root, rate, seed)
    require(completed["checkpoints"] == list(steps) and completed["global_step"] == steps[-1]
            and completed["sweep_protocol_sha256"] == sha256(plan_path(root)), f"Training provenance differs: {name}")
    for index in range(0, len(steps), 2):
        batch = steps[index:index + 2]
        pending = [step for step in batch if f"{name}/checkpoint-{step}" not in manifest]
        if pending:
            status("evaluation", name, pending)
        for attempt in range(2):
            if not pending:
                break
            jobs = {}
            for gpu, step in enumerate(pending):
                path = run / f"checkpoint-{step}" / "official_eval.log"
                with path.open("a") as log:
                    jobs[step] = subprocess.Popen([*command, "--evaluate", rate, str(seed), str(step)], cwd=ROOT,
                                                  env={**env, "CUDA_VISIBLE_DEVICES": str(gpu)}, stdout=log, stderr=subprocess.STDOUT)
            pending = [step for step, job in jobs.items() if job.wait() != 0]
            if pending:
                print(f"CHECKPOINT_EVAL_FAILURE {name} {pending} attempt={attempt + 1}", flush=True)
        require(not pending, f"Official evaluation failed for {name} at {pending}")
        for step in batch:
            archive_checkpoint(root, rate, seed, step, manifest)
    collected = root / "collected_results" / name
    for filename in ["training_completed.json", "resolved_config.json", "finite_checks_rank0.json", "finite_checks_rank1.json"]:
        copy_verified(run / filename, collected / filename)
    copy_verified(training_log, collected / "training.log")
    csv_write(root / "sweep_metrics.csv", [row for row in rows_from(manifest) if row["seed"] == 0])
    pack(root)


def matrix(root):
    plan = freeze(root)
    manifest_file = root / "incremental_manifest.json"
    manifest = read(manifest_file) if manifest_file.exists() else {}
    current = {}

    def status(phase, run=None, steps=None, error=None):
        current.update(phase=phase, run=run, steps=steps, archived_checkpoints=len(manifest), error=error)
        save_json(root / "status.json", current)
        print(json.dumps(current), flush=True)

    try:
        for rate in RATES:
            execute_run(root, rate, 0, manifest, status)
        selected, scored = choose(plan, [row for row in rows_from(manifest) if row["seed"] == 0])
        choice = {"selection_protocol_sha256": sha256(plan_path(root)), "selected": selected,
                  "eligible_count": sum(row["eligible"] for row in scored), "evaluated_count": len(scored)}
        choice_path = root / "selection.json"
        if choice_path.exists():
            require(read(choice_path) == choice, "Frozen selection differs")
        else:
            save_json(choice_path, choice)
        csv_write(root / "seed0_selection_table.csv", scored)
        if selected is None:
            pack(root, final=True)
            status("no_eligible_checkpoint")
            return
        print(f"SELECTED {selected['learning_rate_label']} at {selected['step']}", flush=True)
        for seed in (1, 2):
            execute_run(root, selected["learning_rate_label"], seed, manifest, status)
        confirmed = [row for row in rows_from(manifest) if row["learning_rate_label"] == selected["learning_rate_label"] and row["step"] == selected["step"]]
        require([row["seed"] for row in confirmed] == [0, 1, 2], "Incomplete selected-configuration seed coverage")
        csv_write(root / "selected_config_by_seed.csv", confirmed)
        metrics = []
        for metric in [*TABLE_METRICS, "gradient_l2_preclip", "parameter_l2", "optimizer_step_update_l2"]:
            values = [row[metric] for row in confirmed]
            metrics.append({"metric": metric, **{f"seed_{seed}": value for seed, value in enumerate(values)},
                            "mean": mean(values), "std": stdev(values), "n": len(values)})
        csv_write(root / "selected_config_mean_std.csv", metrics)
        pack(root, final=True)
        status("complete")
    except Exception as error:
        status("failed", current.get("run"), current.get("steps"), str(error))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=SWEEP)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--freeze-only", action="store_true")
    modes.add_argument("--train", nargs=2, metavar=("RATE", "SEED"))
    modes.add_argument("--evaluate", nargs=3, metavar=("RATE", "SEED", "STEP"))
    args = parser.parse_args()
    root = args.root.resolve()
    logging.basicConfig(level=logging.INFO)
    if args.train or args.evaluate:
        rate, raw_seed, *raw_step = args.train or args.evaluate
        seed = int(raw_seed)
        require(rate in RATES and seed in (0, 1, 2), "Unexpected sweep run")
        if args.train:
            train(root, rate, seed)
        else:
            evaluate(root, rate, seed, int(raw_step[0]))
    else:
        root.mkdir(exist_ok=True, parents=True)
        with (root / "queue.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if args.freeze_only:
                freeze(root)
                print(plan_path(root))
            else:
                matrix(root)


if __name__ == "__main__":
    main()
