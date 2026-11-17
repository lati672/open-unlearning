"""Run the approved SU matrix, archive official TOFU outputs and reclaim weights."""

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

from evaluate_tofu_checkpoint import (
    METRICS, configure_eval, get_evaluators, get_model, read, require, validate_outputs, weight_hash,
)
from freeze_mask_protocol import DEFAULT_PROTOCOL, FORMAL_OUTPUT, check_environment, save_json, sha256
from prepare_su_ngram import OUTPUT, ROOT, VARIANTS
from run_su_ngram import load_prepared
from omegaconf import OmegaConf


STEPS = (60, 120)
SEEDS = (0, 1, 2)
CASES = [(variant, seed) for seed in SEEDS for variant in VARIANTS]
ARTIFACTS = ("TOFU_EVAL.json", "TOFU_SUMMARY.json", "official_metrics.json", "evaluation_provenance.json", "validated.json")
BASENAME = "tofu_forget05_wan_su_reimplementation"


def run_name(variant, seed):
    return f"{variant}_seed{seed}"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def eval_protocol(root):
    protocol = read(DEFAULT_PROTOCOL)
    prepared = read(root / "protocol_manifest.json")
    for key in ("id", "revision", "forget", "retain"):
        require(prepared["dataset"][key] == protocol["dataset"][key], f"Evaluation dataset {key} differs")
    require(tuple(protocol["official_evaluation"]["steps"]) == STEPS, "Evaluation steps differ")
    reference_key = protocol["retain_only_model"]["evaluation_logs"]
    source = FORMAL_OUTPUT / "frozen_inputs" / Path(reference_key).name
    expected = protocol["inputs"][reference_key]
    require(sha256(source) == expected, "Pinned retain-only reference changed")
    frozen = root / "frozen_inputs"
    frozen.mkdir(exist_ok=True)
    reference = frozen / source.name
    if not reference.exists():
        shutil.copy2(source, reference)
    require(sha256(reference) == expected, "Copied retain-only reference hash differs")
    return protocol, reference


def training_check(root, variant, seed):
    run = root / "runs" / variant / f"seed{seed}"
    prepared = read(root / "protocol_manifest.json")
    frozen_cfg, mask = load_prepared(root, variant)
    cfg = read(run / "resolved_config.json")
    result = read(run / "completed.json")
    budget = prepared["training"]
    mask_hash = sha256(mask)
    protocol_hash = sha256(root / "protocol_manifest.json")
    require(cfg["config"] == frozen_cfg and cfg["seed"] == seed and result["seed"] == seed and result["variant"] == variant
            and cfg["protocol_sha256"] == protocol_hash and cfg["mask_sha256"] == result["mask_sha256"] == mask_hash
            and result["global_step"] == 120 and result["all_finite"] and math.isfinite(result["training_loss"]), f"Run provenance differs: {run}")
    mapping = {"per_device_train_batch_size": "batch_size", "warmup_steps": "warmup_steps", "max_steps": "max_steps",
               "save_steps": "save_steps", "gradient_accumulation_steps": "gradient_accumulation_steps",
               "learning_rate": "learning_rate", "weight_decay": "weight_decay", "max_grad_norm": "max_grad_norm"}
    actual = cfg["training_args"]
    require(all(actual[key] == budget[value] for key, value in mapping.items()) and actual["seed"] == seed
            and actual["save_total_limit"] is None, f"Training arguments differ: {run}")
    for rank in range(budget["world_size"]):
        finite = read(run / f"finite_checks_rank{rank}.json")
        require(len(finite["loss"]) == 480 and len(finite["grad_norm"]) == 120
                and all(math.isfinite(x) for x in finite["loss"] + finite["grad_norm"]), f"Incomplete numeric checks: {run}/{rank}")
    return run, cfg, prepared


def evaluate(root, variant, seed, step):
    run, cfg, prepared = training_check(root, variant, seed)
    checkpoint = run / f"checkpoint-{step}"
    state = read(checkpoint / "trainer_state.json")
    require(state["global_step"] == step and state["max_steps"] == 120 and math.isclose(state["epoch"], step * .16, abs_tol=1e-8)
            and read(checkpoint / "resolved_config.json") == cfg, f"Checkpoint state differs: {checkpoint}")
    protocol, reference = eval_protocol(root)
    required = ("config.json", "tokenizer.json", "tokenizer_config.json", "trainer_state.json", "training_args.bin")
    require(all((checkpoint / name).is_file() for name in required), f"Incomplete checkpoint: {checkpoint}")
    weights = sorted(checkpoint.glob("*.safetensors"))
    require(weights and all(p.is_file() and not p.is_symlink() for p in weights), f"Missing checkpoint weights: {checkpoint}")
    metadata = {"variant": variant, "seed": seed, "step": step, "global_step": step, "actual_epoch": state["epoch"],
                "code_commit": prepared["git_head"], "training_source_sha256": prepared["source_sha256"],
                "protocol_sha256": cfg["protocol_sha256"], "mask_sha256": cfg["mask_sha256"],
                "saved_weight_files": [p.name for p in weights]}
    save_json(checkpoint / "checkpoint_metadata.json", metadata)
    output = checkpoint / "evals/official"
    output.mkdir(parents=True, exist_ok=True)
    eval_cfg = configure_eval(protocol, output, reference)
    provenance = {"context": {"checkpoint": metadata, "resolved_eval_config": eval_cfg,
                              "retain_only_logs_sha256": sha256(reference)},
                  "weight_sha256": {p.name: weight_hash(p) for p in weights}}
    provenance_path = output / "evaluation_provenance.json"
    if provenance_path.exists():
        require(read(provenance_path) == provenance, f"Existing evaluation belongs to different weights: {checkpoint}")
    else:
        save_json(provenance_path, provenance)
    if not (output / "validated.json").exists():
        model_cfg = OmegaConf.create(cfg["model"])
        for args in (model_cfg.model_args, model_cfg.tokenizer_args):
            args.pop("revision", None)
            args.pretrained_model_name_or_path = str(checkpoint)
            args.local_files_only = True
        model_cfg.model_args.device_map = {"": "cuda:0"}
        model, tokenizer = get_model(model_cfg)
        evaluator = get_evaluators(OmegaConf.create({"tofu": eval_cfg}))["tofu"]
        evaluator.evaluate(model=model, tokenizer=tokenizer, template_args=model_cfg.template_args)
    logs, summary = output / "TOFU_EVAL.json", output / "TOFU_SUMMARY.json"
    metrics, counts = validate_outputs(eval_cfg, protocol["official_evaluation"], read(logs), read(summary), reference)
    save_json(output / "official_metrics.json", metrics)
    result = {**metadata, "evaluation_profile": "official_tofu", "metrics": metrics, "sample_counts": counts,
              "artifact_sha256": {name: sha256(output / name) for name in ARTIFACTS if name != "validated.json"}}
    save_json(output / "validated.json", result)
    print(f"VALIDATED {variant} seed {seed} step {step}", flush=True)


def collect(root, variant, seed):
    run, _, _ = training_check(root, variant, seed)
    name = run_name(variant, seed)
    destination = root / "collected_results"
    destination.mkdir(exist_ok=True)
    target = destination / name
    files = {}
    checkpoints = {}
    with tempfile.TemporaryDirectory(prefix=f".{name}-", dir=destination) as directory:
        stage = Path(directory) / name
        stage.mkdir()
        training_log = root / "formal_queue" / f"{name}.train.log"
        shutil.copy2(training_log, stage / "training.log")
        files["training.log"] = digest(training_log)
        require(files["training.log"] == digest(stage / "training.log"), f"Collected training log differs: {name}")
        for source in sorted(run.rglob("*")):
            if not source.is_file() or source.name.endswith((".safetensors", ".safetensors.index.json", ".tmp")):
                continue
            relative = source.relative_to(run)
            path = stage / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, path)
            checksum = digest(source)
            require(checksum == digest(path), f"Collected copy hash differs: {path}")
            if path.suffix == ".json":
                read(path)
            files[str(relative)] = checksum
        for step in STEPS:
            rel = f"checkpoint-{step}/evals/official"
            validation = read(stage / rel / "validated.json")
            require(validation["seed"] == seed and validation["global_step"] == step and set(validation["metrics"]) == set(METRICS)
                    and all(math.isfinite(value) for value in validation["metrics"].values()), f"Invalid copied evaluation: {name}/{step}")
            for artifact in ARTIFACTS:
                checksum = files[f"{rel}/{artifact}"]
                if artifact != "validated.json":
                    require(checksum == validation["artifact_sha256"][artifact], f"Copy disagrees with validation: {name}/{artifact}")
            checkpoints[str(step)] = validation
        if target.exists():
            require(all(digest(target / path) == checksum for path, checksum in files.items()), f"Existing collected copy differs: {target}")
        else:
            stage.rename(target)
    return {"variant": variant, "seed": seed, "files": files, "checkpoints": checkpoints}


def cleanup(root, name, record):
    run = root / "runs" / record["variant"] / f"seed{record['seed']}"
    collected = root / "collected_results" / name
    to_remove = []
    for step, validation in record["checkpoints"].items():
        relative = f"checkpoint-{step}/evals/official"
        for filename in ARTIFACTS:
            path = collected / relative / filename
            require(digest(path) == record["files"][f"{relative}/{filename}"], f"Collected hash differs before cleanup: {path}")
            read(path)
        provenance = read(collected / relative / "evaluation_provenance.json")
        for filename in validation["saved_weight_files"]:
            require(Path(filename).name == filename and filename.endswith(".safetensors"), f"Unsafe weight filename: {filename}")
            path = run / f"checkpoint-{step}" / filename
            if path.exists():
                require(not path.is_symlink() and weight_hash(path) == provenance["weight_sha256"][filename],
                        f"Weight SHA-256 differs before cleanup: {path}")
                to_remove.append(path)
    for path in to_remove:
        path.unlink()
    print(f"CLEANUP {name} removed={len(to_remove)}", flush=True)


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_tables(root, manifest):
    rows = []
    for record in manifest["runs"].values():
        for step, validation in record["checkpoints"].items():
            metrics = dict(validation["metrics"])
            signed = metrics.pop("privleak")
            rows.append({"variant": record["variant"], "seed": record["seed"], "step": int(step),
                         "actual_epoch": validation["actual_epoch"], **metrics,
                         "privleak_signed": signed, "privleak_abs": abs(signed)})
    write_csv(root / "official_metrics.csv", rows)
    if len(manifest["runs"]) == len(CASES):
        grouped = []
        for variant in VARIANTS:
            for step in STEPS:
                group = [r for r in rows if r["variant"] == variant and r["step"] == step]
                require([r["seed"] for r in group] == list(SEEDS), "Wrong seed order or coverage")
                for metric in [key for key in METRICS if key != "privleak"] + ["privleak_signed", "privleak_abs"]:
                    values = [r[metric] for r in group]
                    grouped.append({"variant": variant, "step": step, "metric": metric,
                                    **{f"seed_{seed}": value for seed, value in zip(SEEDS, values)},
                                    "mean": mean(values), "std": stdev(values), "n": len(values)})
        write_csv(root / "mean_std.csv", grouped)


def pack(root, manifest, final):
    collected = root / "collected_results"
    export = ROOT / "saves/exports" / f"{BASENAME}{'' if final else '_partial'}.zip"
    export.parent.mkdir(exist_ok=True)
    files = {f"runs/{name}/{path}": (collected / name / path, checksum)
             for name, run in manifest["runs"].items() for path, checksum in run["files"].items()}
    for folder in ("assistants", "masks", "configs", "audits", "frozen_inputs"):
        for path in (root / folder).rglob("*"):
            if path.is_file():
                files[str(path.relative_to(root))] = (path, digest(path))
    for relative in ["assumptions.md", "implementation_report.md", "protocol_manifest.json", "hashes.json",
                     "official_metrics.csv", "formal_queue/run_plan.json", "collected_results/incremental_manifest.json",
                     *(["mean_std.csv"] if final else [])]:
        path = root / relative
        files[relative] = (path, digest(path))
    pending = export.with_suffix(".zip.part")
    listing = {"runs": len(manifest["runs"]), "evaluated_checkpoints": 2 * len(manifest["runs"]),
               "files": {name: checksum for name, (_, checksum) in files.items()}}
    with ZipFile(pending, "w", compression=ZIP_DEFLATED, compresslevel=4) as archive:
        for name, (path, _) in sorted(files.items()):
            archive.write(path, f"{BASENAME}/{name}")
        archive.writestr(f"{BASENAME}/MANIFEST.json", json.dumps(listing, indent=2) + "\n")
    with ZipFile(pending) as archive:
        require(len(archive.namelist()) == len(files) + 1, "ZIP member count differs")
        for name, (_, checksum) in files.items():
            with archive.open(f"{BASENAME}/{name}") as stream:
                require(hashlib.file_digest(stream, "sha256").hexdigest() == checksum, f"ZIP member SHA-256 differs: {name}")
    checksum = digest(pending)
    pending.replace(export)
    export.with_suffix(".zip.sha256").write_text(f"{checksum}  {export.name}\n")
    return export


def alive(pid):
    path = Path(f"/proc/{pid}/stat")
    try:
        return path.read_text().split(") ", 1)[1][0] != "Z"
    except FileNotFoundError:
        return False


def train(root, queue, variant, seed, env):
    run = root / "runs" / variant / f"seed{seed}"
    if (variant, seed) == CASES[0] and (queue / "first_training.pid").exists():
        marker = queue / "first_training.exit"
        pid = int((queue / "first_training.pid").read_text())
        while not marker.exists() and alive(pid):
            time.sleep(5)
        require(marker.is_file() and marker.read_text().strip() == "0", "First training process failed")
        return
    if (run / "completed.json").exists():
        return
    require(not run.exists(), f"Incomplete training already exists. Inspect without repeating: {run}")
    command = [sys.executable, "-u", "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
               str(ROOT / "scripts/run_su_ngram.py"), "train", "--variant", variant, "--seed", str(seed), "--root", str(root)]
    with (queue / f"{run_name(variant, seed)}.train.log").open("x") as log:
        subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


def matrix(root):
    queue = root / "formal_queue"
    queue.mkdir(exist_ok=True)
    collected = root / "collected_results"
    collected.mkdir(exist_ok=True)
    manifest_path = collected / "incremental_manifest.json"
    plan_path = queue / "run_plan.json"
    protocol, reference = eval_protocol(root)
    check_environment(protocol)
    for variant in VARIANTS:
        load_prepared(root, variant)
    plan = {"cases": [run_name(v, s) for v, s in CASES], "seeds": SEEDS, "max_steps": 120, "checkpoints": STEPS,
            "warmup_steps": 6, "evaluation": "official_tofu", "evaluation_protocol_sha256": sha256(DEFAULT_PROTOCOL),
            "retain_only_reference_sha256": sha256(reference), "prepared_protocol_sha256": sha256(root / "protocol_manifest.json"),
            "queue_source_sha256": sha256(Path(__file__)), "evaluator_source_sha256": sha256(ROOT / "scripts/evaluate_tofu_checkpoint.py")}
    if plan_path.exists():
        require(read(plan_path) == json.loads(json.dumps(plan)), "Queue plan changed")
    else:
        save_json(plan_path, plan)
    manifest = read(manifest_path) if manifest_path.exists() else {"plan_sha256": sha256(plan_path), "runs": {}}
    require(manifest["plan_sha256"] == sha256(plan_path), "Collected runs belong to a different plan")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0,1"}
    status = queue / "status.json"

    def update(phase, name=None, step=None, error=None):
        save_json(status, {"phase": phase, "run": name, "step": step, "completed_runs": len(manifest["runs"]),
                           "total_runs": len(CASES), "error": error})
        print(f"{phase.upper()} {name or ''} {step or ''}", flush=True)

    name, step = None, None
    try:
        for variant, seed in CASES:
            name = run_name(variant, seed)
            if name in manifest["runs"]:
                write_tables(root, manifest)
                pack(root, manifest, final=False)
                cleanup(root, name, manifest["runs"][name])
                continue
            update("training", name)
            train(root, queue, variant, seed, env)
            run, _, _ = training_check(root, variant, seed)
            for step in STEPS:
                update("evaluation", name, step)
                command = [sys.executable, "-u", str(Path(__file__).resolve()), "--root", str(root),
                           "--evaluate", variant, str(seed), str(step)]
                checkpoint = run / f"checkpoint-{step}"
                with (checkpoint / "official_eval.log").open("a") as log:
                    for attempt in range(2):
                        result = subprocess.run(command, cwd=ROOT, env={**env, "CUDA_VISIBLE_DEVICES": "0"},
                                                stdout=log, stderr=subprocess.STDOUT)
                        if result.returncode == 0:
                            break
                        print(f"EVAL_RETRY {name} {step} attempt={attempt + 1}", flush=True)
                    require(result.returncode == 0, f"Evaluation failed: {checkpoint}/official_eval.log")
            step = None
            update("archiving", name)
            manifest["runs"][name] = collect(root, variant, seed)
            save_json(manifest_path, manifest)
            write_tables(root, manifest)
            pack(root, manifest, final=False)
            cleanup(root, name, manifest["runs"][name])
            update("completed_run", name)
        export = pack(root, manifest, final=True)
        update("complete")
        print(export, flush=True)
    except Exception as error:
        update("failed", name, step, str(error))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=OUTPUT)
    parser.add_argument("--evaluate", nargs=3, metavar=("VARIANT", "SEED", "STEP"))
    args = parser.parse_args()
    root = args.root.resolve()
    logging.basicConfig(level=logging.INFO)
    if args.evaluate:
        variant, seed, step = args.evaluate
        require(variant in VARIANTS and int(seed) in SEEDS and int(step) in STEPS, "Invalid evaluation request")
        evaluate(root, variant, int(seed), int(step))
    else:
        queue = root / "formal_queue"
        queue.mkdir(exist_ok=True)
        with (queue / "matrix.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            matrix(root)


if __name__ == "__main__":
    main()
