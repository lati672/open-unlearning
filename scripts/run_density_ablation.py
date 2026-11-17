"""Execute the six Phase A seed-0 variants in the frozen v4 code snapshot."""

import argparse
from copy import deepcopy
import fcntl
import json
import logging
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

from transformers import TrainerCallback
import torch

from prepare_density_ablation import FORMAL, OUTPUT, STEPS, prepare
from prepare_low_confidence import ROOT, read, require, save_json, sha256
from freeze_mask_protocol import DEFAULT_PROTOCOL
from resume_low_confidence import configure_with_v4_reference
import run_low_confidence as v4


ORIGINAL_INSTRUMENT = v4.instrument


class UpdateNorm(TrainerCallback):
    """Record the actual parameter update across each checkpoint optimizer step."""

    def __init__(self):
        self.before = []
        self.results = {}

    @torch.no_grad()
    def on_pre_optimizer_step(self, args, state, control, model=None, optimizer=None, **kwargs):
        if state.is_world_process_zero and state.global_step + 1 in STEPS:
            self.before = [(parameter, parameter.detach().clone()) for parameter in model.parameters() if parameter.requires_grad]
            self.lr = optimizer.param_groups[0]["lr"]

    @torch.no_grad()
    def on_optimizer_step(self, args, state, control, **kwargs):
        if not self.before:
            return
        device = self.before[0][0].device
        update_squared = torch.zeros((), dtype=torch.float64, device=device)
        parameter_squared = torch.zeros_like(update_squared)
        for parameter, previous in self.before:
            for after_chunk, prior_chunk in zip(parameter.detach().flatten().split(2**20), previous.flatten().split(2**20), strict=True):
                after = after_chunk.float()
                parameter_squared += torch.linalg.vector_norm(after).double().square()
                update_squared += torch.linalg.vector_norm(after - prior_chunk.float()).double().square()
        self.before = []
        self.results[str(state.global_step + 1)] = {"optimizer_step_update_l2": float(update_squared.sqrt()),
                                                    "parameter_l2": float(parameter_squared.sqrt()), "learning_rate_used": self.lr}

    def on_train_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            require(set(self.results) == set(map(str, STEPS)) and all(
                math.isfinite(value) for record in self.results.values() for value in record.values()),
                "Missing or nonfinite checkpoint update norm")
            save_json(Path(args.output_dir) / "optimizer_update_norms.json", self.results)


def instrument(trainer):
    if trainer.args.max_steps == 120:
        trainer.add_callback(UpdateNorm())
    return ORIGINAL_INSTRUMENT(trainer)


def load_phase(root):
    phase = read(root / "protocol_manifest.json")
    require(phase["belowmean_byte_identical"] and phase["matched_per_example"]
            and sha256(DEFAULT_PROTOCOL) == phase["v4_protocol_sha256"], "Frozen phase validation differs")
    for name, checksum in phase["source_sha256"].items():
        require(sha256(ROOT / name) == checksum, f"Frozen Phase A code differs: {name}")
    for name, checksum in [("existing_controls.csv", phase["controls_sha256"]),
                           ("v4_evaluation_signature.json", phase["evaluation_signature_sha256"])]:
        require(sha256(root / name) == checksum, f"Frozen analysis input differs: {name}")
    return phase


def bind(root, variant):
    phase = load_phase(root)
    mask = root / "masks" / f"{variant}.jsonl"
    checksum = phase["mask_hashes"][variant]
    require(sha256(mask) == checksum, f"Frozen mask differs: {variant}")
    job = root / "jobs" / variant
    protocol = deepcopy(phase["protocol"])
    protocol["training"]["checkpoints"] = phase["actual_epochs"]
    expected = {"name": variant, "protocol": protocol, "mask_sha256": checksum, "code_commit": phase["code_commit"],
                "v4_protocol_sha256": phase["v4_protocol_sha256"], "evaluator_sha256": phase["evaluator_sha256"],
                "phase_manifest_sha256": sha256(root / "protocol_manifest.json")}
    require(read(job / "protocol_manifest.json") == expected, "Per-job training config differs")
    for filename, source in [("mask.jsonl", mask), ("mask.metadata.json", mask.with_suffix(".metadata.json")),
                             ("retain_only_reference_TOFU_EVAL.json", root / "retain_only_reference_TOFU_EVAL.json"),
                             ("v4_evaluation_signature.json", root / "v4_evaluation_signature.json")]:
        require((job / filename).resolve() == source.resolve(), f"Frozen job input differs: {filename}")
    v4.load_lock = lambda _: expected
    v4.MASK = "mask.jsonl"
    v4.VARIANT = variant
    v4.configure_eval = configure_with_v4_reference
    v4.instrument = instrument
    return job


def initialize_jobs(root, phase):
    for variant in FORMAL:
        job = root / "jobs" / variant
        job.mkdir(parents=True, exist_ok=True)
        mask = root / "masks" / f"{variant}.jsonl"
        for name, source in [("mask.jsonl", mask), ("mask.metadata.json", mask.with_suffix(".metadata.json")),
                             ("retain_only_reference_TOFU_EVAL.json", root / "retain_only_reference_TOFU_EVAL.json"),
                             ("v4_evaluation_signature.json", root / "v4_evaluation_signature.json")]:
            path = job / name
            if not path.exists():
                path.symlink_to(source)
        if not (job / "unit_tests.log").exists():
            shutil.copy2(root / "unit_tests.log", job / "unit_tests.log")
        protocol = deepcopy(phase["protocol"])
        protocol["training"]["checkpoints"] = phase["actual_epochs"]
        target = job / "protocol_manifest.json"
        if not target.exists():
            save_json(target, {"name": variant, "protocol": protocol, "mask_sha256": phase["mask_hashes"][variant],
                "code_commit": phase["code_commit"], "v4_protocol_sha256": phase["v4_protocol_sha256"],
                "evaluator_sha256": phase["evaluator_sha256"], "phase_manifest_sha256": sha256(root / "protocol_manifest.json")})


def run_queue(root, commit):
    phase_name = "start"
    variant = None
    completed = 0
    logs = root / "logs"
    logs.mkdir(exist_ok=True)
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--root", str(root)]

    def status(name, error=None):
        nonlocal phase_name
        phase_name = name
        data = {"phase": name, "variant": variant, "official_checkpoints": completed, "error": error}
        save_json(root / "status.json", data)
        print(json.dumps(data), flush=True)

    def train(smoke=False):
        run = root / "jobs" / variant / ("smoke_test" if smoke else "runs/seed0")
        if (run / "training_completed.json").exists():
            return
        require(not run.exists(), f"Partial run exists at {run}")
        log = logs / f"{variant}.{'smoke' if smoke else 'train'}.log"
        with log.open("x") as stream:
            subprocess.run([sys.executable, "-u", "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
                            *command[2:], "--train", variant, *(["--smoke"] if smoke else [])],
                           cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)

    try:
        status("unit_tests")
        if not (root / "unit_tests.log").exists():
            with (root / "unit_tests.log").open("x") as log:
                for test in ("test_density_ablation.py", "test_low_confidence.py"):
                    subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", test, "-v"],
                                   cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, check=True)
        require((root / "unit_tests.log").read_text().count("\nOK") == 2, "Unit-test gate failed")
        status("mask_audit")
        if not (root / "protocol_manifest.json").exists():
            prepare(root, commit)
        phase = load_phase(root)
        require(phase["code_commit"] == commit, "Frozen Git commit differs")
        initialize_jobs(root, phase)
        for variant in ("Discrepancy-34", "Random-34"):
            status("smoke_test")
            train(smoke=True)
            record = read(root / "jobs" / variant / "smoke_test/smoke_validation.json")
            require(record["passed"] and record["initial_reference_bitwise_equal"]
                    and set(record["checkpoints"]) == {"2", "4"} and record["mask_sha256"] == phase["mask_hashes"][variant],
                    f"Smoke test failed for {variant}")
        for variant in FORMAL:
            status("training")
            train()
            job = bind(root, variant)
            run = job / "runs/seed0"
            manifest_path = job / "incremental_manifest.json"
            manifest = read(manifest_path) if manifest_path.exists() else {}
            status("evaluation")
            for offset in (0, 2):
                steps = STEPS[offset:offset+2]
                pending = [step for step in steps if f"seed0/checkpoint-{step}" not in manifest]
                for _ in range(2):
                    processes = {}
                    for gpu, step in enumerate(pending):
                        with (logs / f"{variant}.step{step}.eval.log").open("a") as log:
                            processes[step] = subprocess.Popen([*command, "--evaluate", variant, str(step)], cwd=ROOT,
                                env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}, stdout=log, stderr=subprocess.STDOUT)
                    pending = [step for step, job_process in processes.items() if job_process.wait() != 0]
                    if not pending:
                        break
                require(not pending, f"Official evaluations failed for {variant}, steps {pending}")
                for step in steps:
                    v4.archive(job, 0, step, manifest)
                    completed += 1
            for name in ("training_completed.json", "resolved_config.json", "training_args.json", "trainer_state.json",
                         "finite_checks_rank0.json", "finite_checks_rank1.json", "optimizer_update_norms.json"):
                v4.copy_verified(run / name, job / "collected_results/seed0" / name)
            status("run_complete")
        variant = None
        status("training_complete")
    except Exception as error:
        status(f"failed_{phase_name}", str(error))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=OUTPUT)
    parser.add_argument("--code-commit")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--train", choices=FORMAL)
    modes.add_argument("--evaluate", nargs=2, metavar=("VARIANT", "STEP"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    logging.basicConfig(level=logging.INFO)
    if args.train:
        v4.training(bind(root, args.train), 0, args.smoke)
    elif args.evaluate:
        variant, step = args.evaluate
        require(variant in FORMAL and int(step) in STEPS, "Invalid evaluation target")
        v4.official_eval(bind(root, variant), 0, int(step))
    else:
        require(args.code_commit, "Supply the frozen v4 Git commit")
        root.mkdir(exist_ok=True, parents=True)
        with (root / "queue.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            run_queue(root, args.code_commit)


if __name__ == "__main__":
    main()
