"""Run Phase B with the unchanged Phase A source and frozen mask artifacts."""

import argparse
import fcntl
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys


WORKSPACE = Path(__file__).resolve().parents[1]
A = WORKSPACE / "saves/experiments/tofu_forget05_adaptive_density_phase_a"
B = WORKSPACE / "saves/experiments/tofu_forget05_adaptive_density_phase_b"
SOURCE = A / "source_v4"
sys.path.insert(0, str(SOURCE / "scripts"))

from prepare_density_ablation import FORMAL, LOW, STEPS
from prepare_low_confidence import controls, read, require, rows, save_json, sha256, write_csv
from run_density_ablation import bind, initialize_jobs, load_phase
import run_low_confidence as v4


SEEDS = (1, 2)
SHARED = ("masks", "mask_audit", "existing_controls.csv", "v4_evaluation_signature.json",
          "retain_only_reference_TOFU_EVAL.json", "initial_target_TOFU_SUMMARY.json", "unit_tests.log")


def verify_git_tree(commit):
    listed = subprocess.check_output(["git", "ls-tree", "-rz", commit, "--", "src", "configs", "scripts", "tests"], cwd=WORKSPACE)
    checked = 0
    for entry in listed.split(b"\0"):
        if not entry:
            continue
        header, filename = entry.split(b"\t", 1)
        mode, kind, oid = header.decode().split()
        require(kind == "blob", f"Unsupported source entry: {filename!r}")
        path = SOURCE / filename.decode()
        content = os.readlink(path).encode() if mode == "120000" else path.read_bytes()
        blob = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
        require(blob == oid, f"Phase A Git snapshot differs: {path}")
        checked += 1
    return checked


def frozen_controls(root, phase):
    old = read(v4.FORMAL_OUTPUT / "protocol_manifest.json")
    values, hashes, signature = controls(phase["protocol"], old)
    require(signature == read(A / "v4_evaluation_signature.json"), "Existing official evaluator signature differs")
    low = read(LOW / "protocol_manifest.json")
    require(low["v4_protocol_sha256"] == phase["v4_protocol_sha256"] and low["evaluator_sha256"] == phase["evaluator_sha256"],
            "Existing LowConfidence protocol differs")
    low_index = read(LOW / "incremental_manifest.json")
    low_rows = [r for r in rows(LOW / "official_metrics_long.csv") if r["variant"] == "lowconfidence_matched"]
    require(len(low_rows) == 6 and {(r["seed"], r["step"]) for r in low_rows} == {(str(s), str(t)) for s in range(3) for t in (60, 120)},
            "Existing LowConfidence checkpoints differ")
    for row in low_rows:
        key = f"seed{row['seed']}/checkpoint-{row['step']}"
        archived = low_index[key]
        validation = archived["validated"]
        require(validation["seed"] == int(row["seed"]) and validation["global_step"] == int(row["step"])
                and validation["mask_sha256"] == low["mask_sha256"] and validation["code_commit"] == phase["code_commit"]
                and all(float(row[name if name != "privleak" else "privleak_signed"]) == value
                        for name, value in validation["metrics"].items())
                and float(row["privleak_abs"]) == abs(validation["metrics"]["privleak"]), f"LowConfidence provenance differs: {key}")
        for name, checksum in archived["files"].items():
            require(v4.digest(LOW / "collected_results" / key / name) == checksum, f"LowConfidence artifact changed: {key}/{name}")
        hashes[f"LowConfidence/{key}"] = archived["files"]
        values.append({**row, "run": f"LowConfidence-Matched_seed{row['seed']}", "objective": "NPO", "normalization": "token_mean"})
    write_csv(root / "existing_controls_multiseed.csv", values)
    return hashes


def freeze(root):
    a_state = read(A / "status.json")
    require(a_state["phase"] == "training_complete" and a_state["official_checkpoints"] == 24, "Phase A results are incomplete")
    phase = load_phase(A)
    require(phase["formal_variants"] == list(FORMAL) and phase["checkpoint_steps"] == list(STEPS), "Phase A run plan differs")
    for name, checksum in phase["mask_hashes"].items():
        require(sha256(A / "masks" / f"{name}.jsonl") == checksum, f"Phase A mask changed: {name}")
    manifest = root / "protocol_manifest.json"
    if not manifest.exists():
        shutil.copy2(A / "protocol_manifest.json", manifest)
    require(sha256(manifest) == sha256(A / "protocol_manifest.json"), "Phase B v4 protocol is not byte-identical to Phase A")
    for name in SHARED:
        link = root / name
        source = A / name
        if not link.exists():
            link.symlink_to(source, target_is_directory=source.is_dir())
        require(link.is_symlink() and link.resolve() == source.resolve(), f"Shared Phase A input differs: {name}")
    plan_path = root / "phase_b_manifest.json"
    if not plan_path.exists():
        count = verify_git_tree(phase["code_commit"])
        a_records = {}
        for variant in FORMAL:
            manifest_a = A / "jobs" / variant / "incremental_manifest.json"
            require(set(read(manifest_a)) == {f"seed0/checkpoint-{step}" for step in STEPS}, f"Phase A archive incomplete: {variant}")
            a_records[variant] = sha256(manifest_a)
        save_json(plan_path, {"phase": "B", "training_git_commit": phase["code_commit"], "git_blob_count_verified": count,
            "phase_a_protocol_sha256": sha256(manifest), "phase_a_archives_sha256": a_records,
            "phase_a_seed0_metrics_sha256": sha256(A / "official_metrics_long.csv"),
            "masks_reused_sha256": phase["mask_hashes"], "formal_arms": FORMAL, "training_seeds": SEEDS,
            "summary_seeds": [0, 1, 2], "checkpoint_steps": STEPS, "primary_endpoint": 120,
            "secondary_training_dynamics_steps": [20, 40, 60], "density_selection": "none; report all six arms",
            "evaluator_sha256": phase["evaluator_sha256"], "phase_b_launcher_sha256": sha256(Path(__file__)),
            "phase_a_support_source_sha256": sha256(SOURCE / "scripts/prepare_low_confidence.py"),
            "control_artifacts": frozen_controls(root, phase),
            "control_table_sha256": sha256(root / "existing_controls_multiseed.csv")})
    frozen = read(plan_path)
    require(frozen["phase_a_protocol_sha256"] == sha256(manifest)
            and frozen["phase_b_launcher_sha256"] == sha256(Path(__file__))
            and frozen["phase_a_support_source_sha256"] == sha256(SOURCE / "scripts/prepare_low_confidence.py")
            and frozen["control_table_sha256"] == sha256(root / "existing_controls_multiseed.csv"),
            "Phase B frozen inputs changed")
    initialize_jobs(root, phase)
    for variant in FORMAL:
        require(sha256(root / "jobs" / variant / "protocol_manifest.json") ==
                sha256(A / "jobs" / variant / "protocol_manifest.json"), f"Per-arm protocol changed: {variant}")
    return phase


def compare_training(root, variant, seed, phase):
    job = root / "jobs" / variant
    trained = job / f"runs/seed{seed}"
    record = read(trained / "training_completed.json")
    require(record["global_step"] == 120 and set(record["checkpoints"]) == set(map(str, STEPS))
            and record["seed"] == seed and record["mask_sha256"] == phase["mask_hashes"][variant]
            and record["code_commit"] == phase["code_commit"], f"Checkpoint set or training provenance differs: {variant}/{seed}")
    before = A / "jobs" / variant / "collected_results/seed0"
    old, new = [read(path / "training_args.json") for path in (before, trained)]
    exclude = {"seed", "output_dir", "logging_dir", "run_name"}
    require({k: v for k, v in old.items() if k not in exclude} == {k: v for k, v in new.items() if k not in exclude}
            and new["seed"] == seed, f"Phase A optimizer arguments differ: {variant}/{seed}")
    old_cfg, new_cfg = [read(path / "resolved_config.json") for path in (before, trained)]
    for name in ("model", "data", "collator", "mask_sha256", "frozen_protocol"):
        require(old_cfg[name] == new_cfg[name], f"Phase A resolved configuration {name} differs")
    require(old_cfg["trainer"]["method_args"] == new_cfg["trainer"]["method_args"]
            and old_cfg["trainer"]["handler"] == new_cfg["trainer"]["handler"] == "TokenNormalizedNPO",
            "Phase A NPO or retain objective differs")
    return trained


def queue(root):
    stage = "preflight"
    variant = seed = None
    new_count = 0
    logs = root / "logs"
    logs.mkdir(exist_ok=True)
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--root", str(root)]

    def status(phase, steps=None, error=None):
        nonlocal stage
        stage = phase
        message = {"phase": stage, "variant": variant, "seed": seed, "steps": steps,
                   "new_official_checkpoints": new_count, "total_new_checkpoints": 48, "error": error}
        save_json(root / "status.json", message)
        print(json.dumps(message), flush=True)

    try:
        status("preflight")
        phase = freeze(root)
        for seed in SEEDS:
            for variant in FORMAL:
                job = bind(root, variant)
                run = job / f"runs/seed{seed}"
                status("training")
                name = f"{variant}_seed{seed}"
                if not (run / "training_completed.json").exists():
                    require(not run.exists(), f"Partial training at {run}. Inspect before attempting recovery.")
                    with (logs / f"{name}.train.log").open("x") as log:
                        subprocess.run([sys.executable, "-u", "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
                            *command[2:], "--train", variant, str(seed)], cwd=SOURCE, stdout=log, stderr=subprocess.STDOUT, check=True)
                compare_training(root, variant, seed, phase)
                manifest_path = job / "incremental_manifest.json"
                manifest = read(manifest_path) if manifest_path.exists() else {}
                for offset in (0, 2):
                    batch = STEPS[offset:offset + 2]
                    pending = [step for step in batch if f"seed{seed}/checkpoint-{step}" not in manifest]
                    for _ in range(2):
                        if not pending:
                            break
                        status("evaluation", pending)
                        jobs = {}
                        for gpu, step in enumerate(pending):
                            with (logs / f"{name}.step{step}.eval.log").open("a") as log:
                                jobs[step] = subprocess.Popen([*command, "--evaluate", variant, str(seed), str(step)], cwd=SOURCE,
                                    env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}, stdout=log, stderr=subprocess.STDOUT)
                        pending = [step for step, proc in jobs.items() if proc.wait() != 0]
                    require(not pending, f"Official evaluation failed for {name} at steps {pending}")
                    status("archiving", list(batch))
                    for step in batch:
                        v4.archive(job, seed, step, manifest)
                        new_count += 1
                for file in ("training_completed.json", "resolved_config.json", "training_args.json", "trainer_state.json",
                             "finite_checks_rank0.json", "finite_checks_rank1.json", "optimizer_update_norms.json"):
                    v4.copy_verified(run / file, job / f"collected_results/seed{seed}" / file)
                status("run_complete")
        require(new_count == 48, "Phase B checkpoint coverage differs")
        variant = seed = None
        status("training_complete")
    except Exception as error:
        status(f"failed_{stage}", error=str(error))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=B)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--train", nargs=2, metavar=("VARIANT", "SEED"))
    modes.add_argument("--evaluate", nargs=3, metavar=("VARIANT", "SEED", "STEP"))
    args = parser.parse_args()
    root = args.root.resolve()
    logging.basicConfig(level=logging.INFO)
    if args.train or args.evaluate:
        variant, seed, *step = args.train or args.evaluate
        seed = int(seed)
        require(variant in FORMAL and seed in SEEDS, "Unexpected Phase B training case")
        job = bind(root, variant)
        if args.train:
            v4.training(job, seed, False)
        else:
            require(int(step[0]) in STEPS, "Unexpected Phase B evaluation step")
            v4.official_eval(job, seed, int(step[0]))
    else:
        root.mkdir(parents=True, exist_ok=True)
        with (root / "queue.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            queue(root)


if __name__ == "__main__":
    main()
