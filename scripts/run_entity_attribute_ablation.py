"""Run audited summed-NPO Entity+Attribute and matched Random arms."""

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from prepare_entity_attribute_ablation import (
    ARMS, ARG_PATHS, COMMIT, FORMAL, INPUTS, ROOT, SEEDS, SOURCE, STEPS, WORKSPACE,
    check_environment, digest, protocol_signature, read, require, run_name,
    save_json, verify_dataset_revision,
)

from data import get_data
from data.collators import DataCollatorForSupervisedDataset, FrozenMaskCollator
from model import get_model
from resume_low_confidence import configure_with_v4_reference
import run_low_confidence as v4
from trainer import load_trainer
from trainer.masking import eligible_positions
from trainer.utils import seed_everything
import torch
from omegaconf import OmegaConf
from train_mask_experiment import StepCheckpointEvaluation


SMOKE_STEPS = (4,)
COPIED = ("training_completed.json", "trainer_state.json", "resolved_config.json", "training_args.json",
          "finite_checks_rank0.json", "finite_checks_rank1.json")
WRAPPERS = ("scripts/prepare_entity_attribute_ablation.py", "scripts/run_entity_attribute_ablation.py",
            "scripts/analyze_entity_attribute_ablation.py", "tests/test_entity_attribute_ablation.py")
EXECUTION_SOURCES = ("scripts/freeze_mask_protocol.py", "scripts/evaluate_tofu_checkpoint.py",
                     "scripts/train_mask_experiment.py", "scripts/run_low_confidence.py",
                     "scripts/resume_low_confidence.py", "scripts/prepare_low_confidence.py")


def root_manifest():
    lock = read(ROOT / "protocol_manifest.json")
    require(lock["code_commit"] == COMMIT, "Unexpected training commit")
    for relative, checksum in lock["inputs_sha256"].items():
        require(digest(ROOT / relative) == checksum, f"Frozen input changed: {relative}")
    for relative, checksum in lock["source_sha256"].items():
        require(digest(SOURCE / relative) == checksum, f"Frozen code changed: {relative}")
    require(digest(SOURCE / "scripts/evaluate_tofu_checkpoint.py") == lock["evaluator_sha256"], "Frozen evaluator changed")
    check_environment(lock["protocol"])
    return lock


def execution_manifest(initialize=False):
    path = ROOT / "execution_manifest.json"
    checks = {"protocol_manifest_sha256": digest(ROOT / "protocol_manifest.json"),
              "script_sha256": {name: digest(WORKSPACE / name) for name in WRAPPERS},
              "pinned_runner_sha256": {name: digest(SOURCE / name) for name in EXECUTION_SOURCES},
              "tests_log_sha256": digest(ROOT / "logs/unit_tests.log")}
    require((ROOT / "logs/unit_tests.log").read_text().strip().endswith("OK"), "Unit tests did not pass")
    if initialize and not path.exists():
        save_json(path, checks)
    require(read(path) == checks, "Frozen execution code or unit-test evidence changed")
    return checks


def job_path(arm, smoke=False):
    return ROOT / ("smoke" if smoke else "jobs") / arm


def bind(arm, smoke, lock):
    job = job_path(arm, smoke)
    job.mkdir(parents=True, exist_ok=True)
    for name, path in {"protocol_manifest.json": ROOT / "protocol_manifest.json",
                       "v4_evaluation_signature.json": ROOT / "v4_evaluation_signature.json",
                       "retain_only_reference_TOFU_EVAL.json": INPUTS / "TOFU_EVAL.json"}.items():
        link = job / name
        if not link.exists():
            link.symlink_to(path)
        require(link.resolve() == path.resolve(), f"Unexpected evaluator path: {link}")
    binding = dict(lock, mask_sha256=lock["mask_hashes"][arm])
    v4.load_lock = lambda _: binding
    v4.VARIANT = arm
    v4.configure_eval = configure_with_v4_reference
    return job


class FrozenEntityCollator(FrozenMaskCollator):
    """Replay the audited union through the same Entity/Random-Entity selector."""

    def __init__(self, tokenizer, arm):
        frozen = ROOT / "frozen_inputs"
        random_path = frozen / f"{ARMS[1]}.jsonl" if arm == ARMS[1] else None
        super().__init__(DataCollatorForSupervisedDataset(tokenizer, index="index"),
                         frozen / f"{ARMS[0]}.jsonl", random_path)
        self.order = hashlib.sha256()
        self.lookups = 0
        self.seen = set()

    def __call__(self, instances):
        batch = super().__call__(instances)
        forget = batch["forget"]
        forget["entity_mask"] = forget["reference_mask"]
        allowed = eligible_positions(forget)
        source = forget["reference_mask"]
        chosen = forget.get("frozen_random_mask", source)
        require(source.shape == chosen.shape == allowed.shape and not ((source | chosen) & ~allowed).any()
                and torch.equal(source.sum(-1), chosen.sum(-1)), "Frozen Entity+Attribute batch misaligned")
        ids = forget["index"].tolist()
        for i, doc in enumerate(ids):
            selected = self.random_records[doc]["mask"] if self.random_records is not None else self.records[doc]["mask"]
            require(chosen[i, :len(selected)].int().tolist() == selected and not chosen[i, len(selected):].any(),
                    f"Frozen mask lookup differs at {doc}")
        return batch

    def record_consumed_batch(self, batch):
        ids = batch["forget"]["index"].tolist()
        retain_ids = batch["retain"]["index"].tolist()
        self.order.update(json.dumps([ids, retain_ids], separators=(",", ":")).encode())
        self.lookups += len(ids)
        self.seen.update(ids)


def train(arm, seed, smoke, lock):
    job = bind(arm, smoke, lock)
    output = v4.output_for(job, seed)
    require(not output.exists(), f"Run exists: {output}")
    protocol = lock["protocol"]
    training = dict(protocol["training"])
    if smoke:
        training.update(max_steps=4, save_steps=4, checkpoints={"4": 0.64})
    require(int(os.environ.get("WORLD_SIZE", "1")) == training["world_size"], "Expected the frozen two-GPU world size")
    source_arm = "entity" if arm == ARMS[0] else "random_entity"
    control = run_name(source_arm, seed)
    source = FORMAL / "collected_results" / control
    for name in ("resolved_config.json", "training_args.json"):
        require(digest(source / name) == lock["control_artifacts"][f"runs/{control}/{name}"],
                f"Source training config changed for {control}")
    snapshot = read(source / "resolved_config.json")
    original_args = read(source / "training_args.json")
    trainer_cfg = OmegaConf.create(snapshot["trainer"])
    trainer_cfg.args.output_dir = str(output)
    trainer_cfg.args.logging_dir = str(output / "logs")
    if smoke:
        trainer_cfg.args.update({"max_steps": 4, "save_steps": 4, "logging_steps": 1})
    seed_everything(seed)
    model, tokenizer = get_model(OmegaConf.create(snapshot["model"]))
    model.config.use_cache = False
    require(model.config._commit_hash == protocol["target_model"]["revision"], "Target checkpoint revision differs")
    template = snapshot["model"]["template_args"]
    dataset = get_data(OmegaConf.create(snapshot["data"]), mode="unlearn", tokenizer=tokenizer, template_args=template)["train"]
    for split in ("forget", "retain"):
        data = getattr(dataset, split)
        verify_dataset_revision(data, protocol["dataset"]["revision"])
        require(len(data) == protocol["dataset"][f"{split}_samples"], f"Training split size differs: {split}")
    collator = FrozenEntityCollator(tokenizer, arm)
    trainer, args = load_trainer(trainer_cfg, model, train_dataset=dataset, tokenizer=tokenizer,
                                 data_collator=collator, template_args=template)
    require(trainer.normalization == "summed" and trainer.accelerator.num_processes == training["world_size"]
            and args.warmup_steps == training["warmup_steps"], "Frozen summed NPO settings differ")
    current = args.to_dict()
    ignored = ARG_PATHS | ({"max_steps", "save_steps", "logging_steps"} if smoke else set())
    require(current["local_rank"] == int(os.environ["LOCAL_RANK"]) and original_args["local_rank"] == 0,
            "Unexpected distributed rank provenance")
    differences = {k: (original_args.get(k), current.get(k)) for k in original_args.keys() | current.keys()
                   if k not in ignored | {"local_rank"} and original_args.get(k) != current.get(k)}
    require(not differences, f"TrainingArguments differ from v4: {differences}")
    require(protocol_signature(snapshot, original_args) == read(ROOT / "training_signature.json"), "Training signature changed")
    if smoke:
        with torch.no_grad():
            require(all(torch.equal(a, b) for a, b in zip(model.parameters(), trainer.ref_model.parameters(), strict=True)),
                    "Step-0 NPO reference differs from the target checkpoint")
    snapshot.update(task_name=run_name(arm, seed), mask_file=str(ROOT / "frozen_inputs" / f"{arm}.jsonl"),
                    mask_sha256=lock["mask_hashes"][arm], trainer=OmegaConf.to_container(trainer_cfg, resolve=True),
                    collator={"handler": "FrozenEntityCollator", "padding_side": "right", "index": "index"},
                    extension_manifest_sha256=digest(ROOT / "protocol_manifest.json"),
                    execution_manifest_sha256=digest(ROOT / "execution_manifest.json"))
    snapshot["paths"]["output_dir"] = str(output)
    snapshot["execution"].update(variant=arm, checkpoint_validation=smoke, mask_replay="frozen Entity+Attribute")
    snapshot["frozen_reference_manifest"] = str(ROOT / "frozen_inputs" / f"{ARMS[0]}.jsonl")
    snapshot["frozen_reference_sha256"] = lock["mask_hashes"][ARMS[0]]
    snapshot["frozen_reference_metadata"] = read(ROOT / "frozen_inputs" / f"{ARMS[0]}.metadata.json")
    snapshot["frozen_protocol"]["checkpoints"] = training["checkpoints"]
    compute_loss = trainer.compute_loss

    def tracked_loss(model, inputs, return_outputs=False):
        # Accelerate may prefetch a batch beyond the final optimizer step.
        collator.record_consumed_batch(inputs)
        return compute_loss(model, inputs, return_outputs=return_outputs)

    trainer.compute_loss = tracked_loss
    checks = v4.instrument(trainer)
    callback = StepCheckpointEvaluation(trainer, training, None, output, resolved_config=snapshot, defer_evaluation=True)
    trainer.add_callback(callback)
    if trainer.is_world_process_zero():
        output.mkdir(parents=True, exist_ok=True)
        save_json(output / "resolved_config.json", snapshot)
        save_json(output / "training_args.json", current)
    trainer.accelerator.wait_for_everyone()
    seed_everything(seed)
    result = trainer.train()
    trainer.save_state()
    steps = training["max_steps"]
    require(trainer.state.global_step == steps and math.isfinite(result.training_loss)
            and len(checks["loss"]) == steps * training["gradient_accumulation_steps"]
            and len(checks["gradient_l2_preclip"]) == steps
            and collator.lookups == len(checks["loss"]) * args.per_device_train_batch_size, "Incomplete finite or mask checks")
    save_json(output / f"finite_checks_rank{args.process_index}.json",
              dict(checks, sample_order_sha256=collator.order.hexdigest(), sample_ids=sorted(collator.seen),
                   lookups=collator.lookups, actual_mask_sha256=lock["mask_hashes"][arm],
                   initial_reference_bitwise_equal=smoke, gpu=torch.cuda.get_device_name(args.device)))
    trainer.accelerator.wait_for_everyone()
    if trainer.is_world_process_zero():
        require(set(callback.saved) == set(training["checkpoints"]), "Checkpoint schedule changed")
        for step in callback.saved:
            checkpoint = output / f"checkpoint-{step}"
            state = read(checkpoint / "trainer_state.json")
            current_step = int(step)
            logged = next(row for row in reversed(trainer.state.log_history) if row["step"] == current_step and "loss" in row)
            rank_norms = [read(output / f"finite_checks_rank{rank}.json")["gradient_l2_preclip"][current_step - 1] for rank in (0, 1)]
            save_json(checkpoint / "training_diagnostics.json", {"global_step": current_step,
                "actual_epoch": state["epoch"], "logged_training_loss": logged["loss"],
                "learning_rate_after_step": logged["learning_rate"], "gradient_l2_preclip_by_rank": rank_norms,
                "gradient_l2_preclip": rank_norms[0]})
        save_json(output / "training_completed.json", {"global_step": steps, "seed": seed, "checkpoints": callback.saved,
            "training_loss": result.training_loss, "mask_sha256": lock["mask_hashes"][arm], "code_commit": COMMIT})
    trainer.accelerator.wait_for_everyone()


def paired_order(seed, smoke, lock):
    result = {}
    for rank in (0, 1):
        checks = [read(v4.output_for(job_path(arm, smoke), seed) / f"finite_checks_rank{rank}.json") for arm in ARMS]
        require(checks[0]["sample_order_sha256"] == checks[1]["sample_order_sha256"]
                and checks[0]["sample_ids"] == checks[1]["sample_ids"]
                and checks[0]["lookups"] == checks[1]["lookups"], f"Paired sample order differs for seed {seed}, rank {rank}")
        require([r["actual_mask_sha256"] for r in checks] == [lock["mask_hashes"][arm] for arm in ARMS],
                "A run did not use the frozen seed-independent mask")
        result[str(rank)] = checks[0]["sample_order_sha256"]
    if not smoke:
        for arm in ARMS:
            ids = {doc for rank in (0, 1) for doc in
                   read(v4.output_for(job_path(arm), seed) / f"finite_checks_rank{rank}.json")["sample_ids"]}
            require(ids == set(range(200)), f"Training did not look up all 200 IDs: {arm}/{seed}")
    return result


def queue(smoke, lock):
    count = 0
    current = {}
    command = [sys.executable, "-u", str(Path(__file__).resolve())]
    steps = SMOKE_STEPS if smoke else STEPS
    stage = "smoke" if smoke else "formal"
    def status(phase, **fields):
        value = {"phase": phase, "stage": stage, "verified_checkpoints": count, **current, **fields}
        save_json(ROOT / "status.json", value)
        print(json.dumps(value), flush=True)

    try:
        if not smoke:
            proof = read(ROOT / "smoke_verification.json")
            require(proof["passed"] and proof["validated_checkpoints"] == len(ARMS)
                    and proof["execution_manifest_sha256"] == digest(ROOT / "execution_manifest.json"),
                    "Formal training requires both audited smoke runs")
        for seed in ((0,) if smoke else SEEDS):
            for arm in ARMS:
                current.update(arm=arm, seed=seed)
                job = bind(arm, smoke, lock)
                run = v4.output_for(job, seed)
                tag = f"{stage}_{arm}_seed{seed}"
                flags = ["--smoke"] if smoke else []
                status("training")
                if not (run / "training_completed.json").exists():
                    require(not run.exists(), f"Partial run requires inspection: {run}")
                    with (ROOT / "logs" / f"{tag}.train.log").open("x") as log:
                        subprocess.run([sys.executable, "-u", "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
                                        *command[2:], "train", arm, str(seed), *flags], cwd=SOURCE, stdout=log,
                                       stderr=subprocess.STDOUT, check=True)
                completed = read(run / "training_completed.json")
                require(completed["global_step"] == steps[-1] and completed["mask_sha256"] == lock["mask_hashes"][arm],
                        f"Training completion proof differs: {tag}")
                path = job / "incremental_manifest.json"
                index = read(path) if path.exists() else {}
                pending = [step for step in steps if f"seed{seed}/checkpoint-{step}" not in index]
                for attempt in range(2):
                    if not pending:
                        break
                    status("evaluation", steps=pending)
                    processes = {}
                    for gpu, step in enumerate(pending):
                        with (ROOT / "logs" / f"{tag}.step{step}.eval.log").open("a") as log:
                            processes[step] = subprocess.Popen([*command, "evaluate", arm, str(seed), "--step", str(step), *flags],
                                cwd=SOURCE, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)})
                    pending = [step for step, p in processes.items() if p.wait() != 0]
                require(not pending, f"Official evaluation failed at {tag}: {pending}")
                for step in steps:
                    v4.archive(job, seed, step, index)
                    count += 1
                for name in COPIED:
                    v4.copy_verified(run / name, job / f"collected_results/seed{seed}" / name)
                status("run_complete")
            paired_order(seed, smoke, lock)
        current.clear()
        if smoke:
            save_json(ROOT / "smoke_verification.json", {"passed": True, "validated_checkpoints": count,
                "optimizer_steps_per_arm": 4, "sample_order_sha256_by_rank": paired_order(0, True, lock),
                "execution_manifest_sha256": digest(ROOT / "execution_manifest.json"), "mask_sha256": lock["mask_hashes"]})
            status("smoke_complete")
        else:
            require(count == 12, "Formal checkpoint coverage differs")
            status("analysis")
            subprocess.run([sys.executable, "-u", str(WORKSPACE / WRAPPERS[2])], cwd=SOURCE,
                           env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, check=True)
            status("complete")
    except Exception as exc:
        status("failed", error=str(exc))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for name in ("queue", "train", "evaluate"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--smoke", action="store_true")
        if name != "queue":
            cmd.add_argument("arm", choices=ARMS)
            cmd.add_argument("seed", type=int, choices=SEEDS)
        if name == "evaluate":
            cmd.add_argument("--step", type=int, required=True)
    args = parser.parse_args()
    lock = root_manifest()
    if args.action == "queue":
        with (ROOT / "queue.lock").open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            execution_manifest(initialize=args.smoke)
            queue(args.smoke, lock)
    else:
        execution_manifest()
        if args.action == "train":
            train(args.arm, args.seed, args.smoke, lock)
        else:
            require(args.step in (SMOKE_STEPS if args.smoke else STEPS), "Unexpected checkpoint step")
            job = bind(args.arm, args.smoke, lock)
            v4.official_eval(job, args.seed, args.step)


if __name__ == "__main__":
    main()
