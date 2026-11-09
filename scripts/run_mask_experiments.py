"""Run legacy seed-0 controls or the frozen TOFU forget05 multiseed protocol.

python scripts/run_mask_experiments.py --objectives NPO RMU
NPO: five mask variants x summed/token_mean. RMU: five variants.
Completed runs are skipped, and any failed run stops the suite.
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys

from freeze_mask_protocol import FORMAL_OUTPUT, check_frozen_run, run_name, save_json, sha256

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_KEYS = ["ES", "forget_probability", "forget_rouge", "forget_ratio", "forget_quality", "utility", "privleak"]


def read(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_official_run(run, training, manifest, completed):
    require(completed["code_commit"] == manifest["code_commit"]
            and completed["protocol_sha256"] == manifest["protocol_sha256"]
            and completed["global_step"] == training["max_steps"], f"Wrong training provenance: {run}")
    require(completed["checkpoints"].keys() == training["checkpoints"].keys(), f"Wrong checkpoints: {run}")
    config = read(run / "resolved_config.json")
    for rank in range(training["world_size"]):
        checks = read(run / f"finite_checks_rank{rank}.json")
        require(checks["loss"] == training["max_steps"] * training["gradient_accumulation_steps"]
                and checks["grad_norm"] == training["max_steps"], f"Incomplete numeric checks: {run}")
    for step, epoch in training["checkpoints"].items():
        checkpoint = run / f"checkpoint-{step}"
        result = read(checkpoint / "evaluated.json")
        require(result == completed["checkpoints"][step] and result["global_step"] == int(step)
                and result["actual_epoch"] == epoch and result["seed"] == config["execution"]["seed"]
                and result["mask_sha256"] == config["mask_sha256"]
                and result["evaluation_profile"] == "official_tofu", f"Wrong checkpoint result: {checkpoint}")
        require(result["weights_removed_after_eval"] and not list(checkpoint.glob("*.safetensors")),
                f"Weights not cleaned after evaluation: {checkpoint}")
        require(set(result["metrics"]) == set(OFFICIAL_KEYS) and all(math.isfinite(v) for v in result["metrics"].values()),
                f"Missing or invalid official metrics: {checkpoint}")
        for name, digest in result["artifact_sha256"].items():
            require(sha256(checkpoint / "evals" / "official" / name) == digest, f"Changed official artifact: {checkpoint}/{name}")


def finish_official_run(run, training, manifest, protocol_path, output_root, env, gpu_ids, smoke):
    pending = read(run / "training_completed.json")
    require(pending["code_commit"] == manifest["code_commit"] and pending["protocol_sha256"] == manifest["protocol_sha256"]
            and pending["global_step"] == training["max_steps"]
            and pending["checkpoints"].keys() == training["checkpoints"].keys(), f"Incomplete training: {run}")
    for rank in range(training["world_size"]):
        checks = read(run / f"finite_checks_rank{rank}.json")
        require(checks["loss"] == training["max_steps"] * training["gradient_accumulation_steps"]
                and checks["grad_norm"] == training["max_steps"], f"Incomplete numeric checks: {run}")
    if smoke and not any(run.glob("checkpoint-*/evals/official/validated.json")):
        require(all((run / f"checkpoint-{step}" / name).is_file()
                    for step, record in pending["checkpoints"].items() for name in record["saved_weight_files"]),
                "Short run did not retain both checkpoint weights until evaluation")
    jobs = []
    for step, gpu in zip(training["checkpoints"], gpu_ids):
        checkpoint = run / f"checkpoint-{step}"
        with (checkpoint / "official_eval.log").open("a") as log:
            command = [sys.executable, str(ROOT / "scripts/evaluate_tofu_checkpoint.py"), str(checkpoint),
                       "--protocol", str(protocol_path), "--output-root", str(output_root)]
            print(f"EVAL official {run.name} step {step} on GPU {gpu}", flush=True)
            job = subprocess.Popen(command, cwd=ROOT, env={**env, "CUDA_VISIBLE_DEVICES": gpu}, stdout=log, stderr=subprocess.STDOUT)
            jobs.append((step, job))
    failures = [(step, code) for step, job in jobs if (code := job.wait()) != 0]
    require(not failures, f"Official evaluation failed at {run}: {failures}. See checkpoint official_eval.log")
    checkpoints = {step: read(run / f"checkpoint-{step}/evaluated.json") for step in training["checkpoints"]}
    result = {**pending, "checkpoints": checkpoints, "metrics": checkpoints[str(training["max_steps"])]["metrics"]}
    if smoke:
        result["both_checkpoints_retained_until_evaluation"] = True
    verify_official_run(run, training, manifest, result)
    save_json(run / ("checkpoint_validation.json" if smoke else "completed.json"), result)


def write_official_table(output_root):
    temporary = output_root / "official_metrics.csv.tmp"
    with temporary.open("w", newline="") as stream:
        fields = ["run", "objective", "variant", "normalization", "seed", "step", *OFFICIAL_KEYS]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for marker in sorted(output_root.glob("*/completed.json")):
            config = read(marker.parent / "resolved_config.json")["execution"]
            objective = marker.parent.name.split("_", 1)[0]
            for step, result in read(marker)["checkpoints"].items():
                writer.writerow({"run": marker.parent.name, "objective": objective,
                                 "normalization": "token_mean" if objective == "RMU" else config["normalization"],
                                 "variant": config["variant"], "seed": config["seed"], "step": int(step), **result["metrics"]})
    temporary.replace(output_root / "official_metrics.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objectives", nargs="+", choices=["NPO", "RMU"], default=["NPO", "RMU"])
    variants = ["base", "entity", "adaptive", "random_entity", "random_adaptive"]
    parser.add_argument("--variants", nargs="+", choices=variants, default=variants)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--protocol", type=Path, help="Use a frozen multiseed protocol")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--epochs", type=int, help="Legacy controls only")
    parser.add_argument("--seeds", nargs="+", type=int, help="Subset of the frozen seeds")
    parser.add_argument("--dry-run", action="store_true", help="Print the planned runs without training")
    parser.add_argument("--checkpoint-smoke", action="store_true", help="Validate one case using steps 2 and 4")
    args = parser.parse_args()
    args.output_root = (args.output_root or (FORMAL_OUTPUT if args.protocol else ROOT / "saves/experiments/mask_controls")).resolve()
    gpu_ids = args.gpus.split(",")
    seeds = [0]
    manifest = None
    official = False
    budget = f"E{args.epochs or 20}"
    run_root = args.output_root
    if args.checkpoint_smoke and not args.protocol:
        parser.error("--checkpoint-smoke requires --protocol")
    if args.protocol:
        args.protocol = args.protocol.resolve()
        protocol, manifest = check_frozen_run(args.protocol, args.output_root)
        training = protocol["training"]
        seeds = args.seeds or training["seeds"]
        require(len(set(seeds)) == len(seeds) and set(seeds) <= set(training["seeds"]), "Invalid frozen seed subset")
        official = protocol.get("evaluation_profile") == "official_tofu"
        if args.epochs is not None or len(set(gpu_ids)) != training["world_size"] or len(gpu_ids) != training["world_size"]:
            raise ValueError("The frozen protocol controls the step budget and GPU count")
        if args.checkpoint_smoke:
            training = {**training, **protocol["checkpoint_validation"]}
            run_root = args.output_root / "validation"
        budget = f"steps{training['max_steps']}"
    plan = [
        (seed, objective, normalization, variant,
         run_name(objective, variant, normalization, seed, budget))
        for seed in seeds for objective in args.objectives
        for normalization in (["summed", "token_mean"] if objective == "NPO" else ["summed"])
        for variant in args.variants
    ]
    if args.checkpoint_smoke:
        plan = plan[:1]
    if args.dry_run:
        print(json.dumps({"runs": [case[-1] for case in plan], "count": len(plan),
                          "seeds": seeds, "checkpoints": training["checkpoints"] if manifest else None,
                          "max_steps": training["max_steps"] if manifest else None,
                          "checkpoint_steps": list(map(int, training["checkpoints"])) if manifest else None,
                          "warmup_steps": training["warmup_steps"] if manifest else None}, indent=2))
        return
    if manifest and not args.checkpoint_smoke:
        validated = False
        for path in (args.output_root / "validation").glob("*/checkpoint_validation.json"):
            result = json.loads(path.read_text())
            if (result.get("code_commit") == manifest["code_commit"]
                    and result.get("protocol_sha256") == manifest["protocol_sha256"]
                    and result.get("both_checkpoints_retained_until_evaluation")
                    and result.get("global_step") == protocol["checkpoint_validation"]["max_steps"]):
                validated = True
                if official:
                    smoke_training = {**training, **protocol["checkpoint_validation"]}
                    verify_official_run(path.parent, smoke_training, manifest, result)
                break
        if not validated:
            raise ValueError("Run --checkpoint-smoke successfully before launching the formal matrix")
        if official and any(seed != 0 for seed in seeds):
            for obj in ["NPO", "RMU"]:
                for norm in ["summed", "token_mean"] if obj == "NPO" else ["summed"]:
                    for variant in variants:
                        run = args.output_root / run_name(obj, variant, norm, 0, budget)
                        require((run / "completed.json").exists(), f"Complete the official seed-0 matrix first: {run.name}")
                        verify_official_run(run, training, manifest, read(run / "completed.json"))
    run_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpus
    env["TOKENIZERS_PARALLELISM"] = "false"
    for seed, objective, normalization, variant, task in plan:
        marker = run_root / task / ("checkpoint_validation.json" if args.checkpoint_smoke else "completed.json")
        if marker.exists():
            if manifest:
                completed = json.loads(marker.read_text())
                if (completed.get("protocol_sha256") != manifest["protocol_sha256"]
                        or completed.get("code_commit") != manifest["code_commit"]
                        or completed.get("checkpoints", {}).keys() != training["checkpoints"].keys()):
                    raise ValueError(f"Existing result does not match the frozen protocol: {marker}")
                if official:
                    verify_official_run(marker.parent, training, manifest, completed)
            print("SKIP completed", task, flush=True)
            continue
        with socket.socket() as listener:
            listener.bind(("", 0))
            port = listener.getsockname()[1]
        entry = "train_checked_mask_experiment.py" if official else "train_mask_experiment.py"
        command = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", str(len(gpu_ids)),
                   "--master_port", str(port), str(ROOT / "scripts" / entry),
                   "--objective", objective, "--variant", variant, "--normalization", normalization,
                   "--output-root", str(args.output_root), "--seed", str(seed)]
        if args.protocol:
            command += ["--protocol", str(args.protocol)]
        elif args.epochs is not None:
            command += ["--epochs", str(args.epochs)]
        if args.checkpoint_smoke:
            command.append("--checkpoint-smoke")
        print("RUN", task, flush=True)
        run = run_root / task
        if not official or not (run / "training_completed.json").exists():
            with (run_root / f"{task}.log").open("w") as stream:
                subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
        if official:
            finish_official_run(run, training, manifest, args.protocol, args.output_root, env, gpu_ids, args.checkpoint_smoke)
            if not args.checkpoint_smoke:
                write_official_table(args.output_root)
        print("DONE", task, flush=True)
    if not args.protocol:
        subprocess.run([sys.executable, str(ROOT / "scripts/analyze_mask_results.py"), "--experiment-root", str(args.output_root)], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
