"""Run the seed-0 influence mask through the pinned density experiment trainer."""

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from prepare_influence_adaptive import (
    A, CONTROLS, ROOT, SOURCE, STEPS, VARIANT, WORKSPACE, control_archive,
    digest, read, require, save_json, verify_git_tree,
)
import run_density_ablation as density
import run_low_confidence as v4
from resume_low_confidence import configure_with_v4_reference

SCRIPTS = ("prepare_influence_adaptive.py", "run_influence_adaptive.py", "analyze_influence_adaptive.py")
ORIGINAL_OUTPUT_FOR = v4.output_for


def verify_lock():
    lock = read(ROOT / "protocol_manifest.json")
    require(lock["steps"] == list(STEPS) and lock["seed"] == 0 and lock["max_steps"] == 120,
            "Unexpected frozen run plan")
    require(digest(A / "protocol_manifest.json") == lock["density_protocol_sha256"]
            and digest(SOURCE / "scripts/evaluate_tofu_checkpoint.py") == lock["evaluator_sha256"]
            and digest(WORKSPACE / "scripts" / SCRIPTS[0]) == lock["prepare_script_sha256"]
            and verify_git_tree(lock["code_commit"]) == lock["verified_git_blobs"], "Frozen source differs")
    for name, checksum in lock["input_hashes"].items():
        require(digest(ROOT / name) == checksum, f"Frozen input changed: {name}")
    for name, checksum in lock["control_artifact_hashes"].items():
        arm, relative = name.split("/", 1)
        if "/" not in relative:
            require(digest(control_archive(arm) / relative) == checksum, f"Control config changed: {name}")
    for name, checksum in read(ROOT / "execution_manifest.json")["sha256"].items():
        require(digest(Path(name)) == checksum, f"Execution source changed: {name}")
    return lock


def context(smoke):
    job = ROOT / ("smoke" if smoke else "formal")
    lock = verify_lock()
    job.mkdir(exist_ok=True)
    for name in ("protocol_manifest.json", "mask.jsonl", "mask.metadata.json", "v4_evaluation_signature.json",
                 "retain_only_reference_TOFU_EVAL.json", "unit_tests.log"):
        dest = job / name
        if not dest.exists():
            dest.symlink_to(ROOT / name)
        require(dest.resolve() == ROOT / name, f"Input link differs: {dest}")
    v4.load_lock = lambda _: lock
    v4.MASK = "mask.jsonl"
    v4.VARIANT = VARIANT
    v4.configure_eval = configure_with_v4_reference
    v4.instrument = density.instrument
    density.STEPS = STEPS
    v4.output_for = (lambda root, seed, smoke=False: root / "smoke_test") if smoke else ORIGINAL_OUTPUT_FOR
    return job, lock


def verify_training(job, lock, smoke):
    run = v4.output_for(job, 0)
    checkpoints = {"4": .64} if smoke else lock["protocol"]["training"]["checkpoints"]
    result = read(run / "training_completed.json")
    state = read(run / "trainer_state.json")
    require(result["seed"] == 0 and state["global_step"] == result["global_step"] == max(map(int, checkpoints))
            and result["mask_sha256"] == lock["mask_sha256"] and set(result["checkpoints"]) == set(checkpoints),
            "Run completion metadata differs")
    for rank in (0, 1):
        checks = read(run / f"finite_checks_rank{rank}.json")
        require(checks["replayed_samples"] > 0 and len(checks["loss"]) == result["global_step"] * 4
                and len(checks["gradient_l2_preclip"]) == result["global_step"]
                and all(map(math.isfinite, checks["loss"] + checks["gradient_l2_preclip"])), "Training finite/cache checks differ")
    new = read(run / "training_args.json")
    config = read(run / "resolved_config.json")
    excluded = {"output_dir", "logging_dir", "run_name"}
    if smoke:
        excluded |= {"max_steps", "save_steps", "logging_steps", "num_train_epochs"}
    for arm in CONTROLS:
        archived = control_archive(arm)
        before = read(archived / "training_args.json")
        baseline = read(archived / "resolved_config.json")
        require({k: v for k, v in new.items() if k not in excluded} == {k: v for k, v in before.items() if k not in excluded}
                and all(config[k] == baseline[k] for k in ("model", "data", "collator"))
                and config["trainer"]["handler"] == baseline["trainer"]["handler"]
                and config["trainer"]["method_args"] == baseline["trainer"]["method_args"]
                and config["frozen_protocol"]["models"] == baseline["frozen_protocol"]["models"]
                and config["mask_sha256"] == lock["mask_sha256"], f"Protocol mismatch against {arm}")
    return run


def queue():
    script = Path(__file__).resolve()
    command = [sys.executable, "-u", str(script)]
    def status(phase, **details):
        value = {"phase": phase, **details}
        save_json(ROOT / "status.json", value)
        print(json.dumps(value), flush=True)

    def launch(args, log_file, gpu=None):
        env = os.environ.copy()
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        with (ROOT / "logs" / log_file).open("a") as stream:
            return subprocess.Popen(args, cwd=SOURCE, env=env, stdout=stream, stderr=subprocess.STDOUT)

    try:
        for smoke in (True, False):
            label = "smoke" if smoke else "formal"
            job, lock = context(smoke)
            run = v4.output_for(job, 0)
            if not (run / "training_completed.json").exists():
                require(not run.exists(), f"Incomplete training directory requires inspection: {run}")
                status("training", stage=label)
                train_cmd = [sys.executable, "-u", "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
                             str(script), "train", label]
                require(launch(train_cmd, f"{label}.train.log").wait() == 0, f"{label} training failed")
            run = verify_training(job, lock, smoke)
            steps = (4,) if smoke else STEPS
            for offset in range(0, len(steps), 2):
                pending = tuple(step for step in steps[offset:offset + 2]
                                if not (run / f"checkpoint-{step}/evals/official/validated.json").is_file())
                if not pending:
                    continue
                status("evaluation", stage=label, steps=pending)
                for attempt in range(2):
                    procs = {step: launch([*command, "evaluate", label, str(step)], f"{label}.step{step}.eval.log", gpu=gpu)
                             for gpu, step in enumerate(pending)}
                    pending = [step for step, proc in procs.items() if proc.wait() != 0]
                    if not pending:
                        break
                require(not pending, f"Official evaluation failed at {label} steps {pending}")
            if smoke:
                from analyze_influence_adaptive import verify_evaluation
                proof = read(run / "smoke_validation.json")
                require(proof["passed"] and proof["mask_sha256"] == lock["mask_sha256"]
                        and proof["initial_reference_bitwise_equal"] and proof["checkpoints"]["4"]["offline_reload"],
                        "Smoke checkpoint did not reload")
                verify_evaluation(run, 4, lock, smoke=True)
                proof.update(official_metric_replay_passed=True, token_mean_and_same_mask_tests_passed=True,
                             official_validated_sha256=digest(run / "checkpoint-4/evals/official/validated.json"))
                save_json(ROOT / "smoke_verification.json", proof)
                status("smoke_complete")
        status("analysis")
        require(launch([sys.executable, "-u", str(WORKSPACE / "scripts" / SCRIPTS[2])], "analysis.log", gpu="").wait() == 0,
                "Analysis or packaging failed")
        status("complete", verified_checkpoints=3)
    except Exception as error:
        status("failed", error=str(error))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("queue", "train", "evaluate"))
    parser.add_argument("stage", choices=("formal", "smoke"), nargs="?")
    parser.add_argument("step", type=int, nargs="?")
    args = parser.parse_args()
    if args.action == "queue":
        with (ROOT / "queue.lock").open("a") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            queue()
    else:
        smoke = args.stage == "smoke"
        job, lock = context(smoke)
        if not smoke:
            proof = read(ROOT / "smoke_verification.json")
            require(proof["official_metric_replay_passed"] and proof["mask_sha256"] == lock["mask_sha256"], "Smoke gate failed")
        if args.action == "train":
            v4.training(job, 0, smoke=smoke)
        else:
            require(args.step in ((4,) if smoke else STEPS), "Unexpected evaluation step")
            v4.official_eval(job, 0, args.step)


if __name__ == "__main__":
    main()
