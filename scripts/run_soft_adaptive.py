"""Smoke-test and run the single frozen Soft-Adaptive-ESS34 experiment."""

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from prepare_soft_adaptive import A, B, CONTROLS, NAME, ROOT, SOURCE, STEPS, WORKSPACE, control_archive, verify_git_tree
from prepare_influence_adaptive import digest, read, require, save_json
import run_density_ablation as density
import run_low_confidence as v4
from resume_low_confidence import configure_with_v4_reference

import torch
from omegaconf import OmegaConf
from data import get_data
from data.collators import DataCollatorForSupervisedDataset
from freeze_mask_protocol import verify_dataset_revision
from trainer import _register_trainer, load_trainer
from trainer.utils import seed_everything

# Load the two additive modules while every baseline module stays on the verified tree.
import data
import trainer.unlearn
data.__path__.append(str(WORKSPACE / "src/data"))
trainer.unlearn.__path__.append(str(WORKSPACE / "src/trainer/unlearn"))
from data.frozen_token_weights import FrozenTokenWeightCollator
from trainer.unlearn.weighted_npo import WeightedTokenNormalizedNPO

_register_trainer(WeightedTokenNormalizedNPO)


def verify_lock():
    lock = read(ROOT / "protocol_manifest.json")
    require(lock["seed"] == 0 and lock["max_steps"] == 120 and lock["steps"] == list(STEPS)
            and lock["phase_a_sha256"] == digest(A / "protocol_manifest.json")
            and lock["phase_b_manifest_sha256"] == digest(B / "phase_b_manifest.json")
            and lock["evaluator_sha256"] == digest(SOURCE / "scripts/evaluate_tofu_checkpoint.py")
            and lock["verified_git_blobs"] == verify_git_tree(lock["code_commit"]), "Frozen source or protocol changed")
    for name, checksum in lock["input_hashes"].items():
        require(digest(ROOT / name) == checksum, f"Frozen input changed: {name}")
    for name, checksum in lock["extension_hashes"].items():
        require(digest(WORKSPACE / name) == checksum, f"Extension source changed: {name}")
    for name, checksum in lock["support_hashes"].items():
        path = SOURCE / name
        require(digest(path if path.is_file() else WORKSPACE / name) == checksum, f"Support source changed: {name}")
    return lock


def output_for(smoke):
    return ROOT / ("smoke_test" if smoke else "runs/seed0")


def verify_replay(collator, forget, smoke):
    count = len(forget)

    def replay(indices, batch_size):
        restored = {}
        for start in range(0, count, batch_size):
            batch = collator([forget[i] for i in indices[start:start + batch_size]])
            for row, index in enumerate(batch["index"].tolist()):
                _, _, expected = collator.records[index]
                w = batch["token_weights"][row]
                require(torch.equal(w[:len(expected)], expected) and not w[len(expected):].any(),
                        f"Replay mismatch for sample {index}")
                restored[index] = hashlib.sha256(w[:len(expected)].numpy().tobytes()).hexdigest()
        return restored

    order = list(range(count))
    restored = replay(order, 4)
    if smoke:
        require(restored == replay(order[::-1], 7) ==
                replay(torch.randperm(count, generator=torch.Generator().manual_seed(173)).tolist(), 5),
                "Weights differ under reversed or shuffled order")
    peers = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(peers, restored)
    require(all(p == peers[0] for p in peers), "Frozen weights differ across GPU ranks")
    return {"replayed_samples": count, "order_invariant": smoke, "ranks_agree": True,
            "per_sample_fp32_hash": restored}


def verify_args(actual, snapshot, smoke):
    excluded = {"output_dir", "logging_dir", "run_name", "local_rank"}
    require(actual["local_rank"] == int(os.environ["LOCAL_RANK"]), "Unexpected per-process device rank")
    if smoke:
        excluded |= {"max_steps", "save_steps", "logging_steps", "num_train_epochs"}
    for arm in CONTROLS:
        archive = control_archive(arm)
        old_args, cfg = read(archive / "training_args.json"), read(archive / "resolved_config.json")
        different = [k for k in actual.keys() | old_args.keys() if k not in excluded and actual.get(k) != old_args.get(k)]
        require(not different and all(cfg[k] == snapshot[k] for k in ("model", "data", "frozen_protocol"))
                and (cfg["trainer"]["method_args"] | {"mask": None}) == snapshot["trainer"]["method_args"],
                f"Optimization or resolved protocol differs from {arm}: {different}")


def training(smoke):
    lock = verify_lock()
    protocol = lock["protocol"]
    training = dict(protocol["training"])
    if smoke:
        training.update(protocol["checkpoint_validation"])
    else:
        require(read(ROOT / "smoke_validation.json")["passed"], "Smoke gate not passed")
    output = output_for(smoke)
    require(not output.exists() and int(os.environ.get("WORLD_SIZE", "1")) == training["world_size"],
            "Training requires a fresh run directory and two GPU ranks")
    snapshot = read(control_archive(CONTROLS[0]) / "resolved_config.json")
    cfg = OmegaConf.create(snapshot["trainer"])
    cfg.handler = WeightedTokenNormalizedNPO.__name__
    cfg.method_args.mask = None
    cfg.args.output_dir, cfg.args.logging_dir = str(output), str(output / "logs")
    if smoke:
        cfg.args.max_steps = training["max_steps"]
        cfg.args.save_steps = training["save_steps"]
        cfg.args.num_train_epochs = math.ceil(training["max_steps"] / training["warmup_steps"])
        cfg.args.logging_steps = 1
    seed_everything(0)
    model, tokenizer = v4.get_model(OmegaConf.create(snapshot["model"]))
    model.config.use_cache = False
    require(model.config._commit_hash == protocol["target_model"]["revision"], "Target revision changed")
    template = snapshot["model"]["template_args"]
    train_data = get_data(OmegaConf.create(snapshot["data"]), mode="unlearn", tokenizer=tokenizer, template_args=template)["train"]
    for split in ("forget", "retain"):
        ds = getattr(train_data, split)
        verify_dataset_revision(ds, protocol["dataset"]["revision"])
        require(len(ds) == protocol["dataset"][f"{split}_samples"], f"Dataset size differs: {split}")
    weights = ROOT / "weights.jsonl"
    collator = FrozenTokenWeightCollator(DataCollatorForSupervisedDataset(tokenizer, index="index"), weights)
    trainer, args = load_trainer(cfg, model, train_dataset=train_data, tokenizer=tokenizer, data_collator=collator,
                                 template_args=template)
    require(args.warmup_steps == training["warmup_steps"] and trainer.accelerator.num_processes == training["world_size"]
            and not trainer.ref_model.training and not any(p.requires_grad for p in trainer.ref_model.parameters()),
            "Training setup or frozen reference differs")
    require(all(torch.equal(p, q) and p.data_ptr() != q.data_ptr()
                for p, q in zip(model.parameters(), trainer.ref_model.parameters(), strict=True)),
            "Step-0 reference must be an independent, bitwise-identical target copy")
    snapshot.update(task_name=f"NPO_{NAME}_seed0", trainer=OmegaConf.to_container(cfg, resolve=True),
                    mask_file=str(weights), mask_sha256=lock["weights_sha256"], frozen_reference_manifest=str(weights),
                    frozen_reference_sha256=lock["weights_sha256"],
                    frozen_reference_metadata={"shape": "unpadded label positions", "weights_sha256": lock["weights_sha256"]},
                    extension_manifest_sha256=digest(ROOT / "protocol_manifest.json"))
    snapshot["execution"].update(variant=NAME, checkpoint_validation=smoke, mask_replay="frozen_token_weights")
    snapshot["paths"]["output_dir"] = str(output)
    snapshot["collator"] = {"handler": "FrozenTokenWeightCollator", "base": snapshot["collator"], "path": str(weights)}
    verify_args(args.to_dict(), snapshot, smoke)
    snapshot["frozen_protocol"]["checkpoints"] = training["checkpoints"]
    replay = verify_replay(collator, train_data.forget, smoke)
    checks = density.instrument(trainer)
    if smoke:
        density.STEPS = (4,)
        trainer.add_callback(density.UpdateNorm())
    callback = v4.StepCheckpointEvaluation(trainer, training, None, output, resolved_config=snapshot, defer_evaluation=True)
    trainer.add_callback(callback)
    if trainer.is_world_process_zero():
        output.mkdir(parents=True, exist_ok=True)
        save_json(output / "resolved_config.json", snapshot)
        save_json(output / "training_args.json", args.to_dict())
        save_json(output / "replay_validation.json", replay | {"initial_reference_bitwise_equal": True})
    trainer.accelerator.wait_for_everyone()
    seed_everything(0)
    result = trainer.train()
    trainer.save_state()
    steps = training["max_steps"]
    require(trainer.state.global_step == steps and math.isfinite(result.training_loss)
            and len(checks["loss"]) == steps * training["gradient_accumulation_steps"]
            and len(checks["gradient_l2_preclip"]) == steps, "Training ended with incomplete diagnostics")
    save_json(output / f"finite_checks_rank{args.process_index}.json", checks)
    trainer.accelerator.wait_for_everyone()
    if trainer.is_world_process_zero():
        require(set(callback.saved) == set(training["checkpoints"]), "Saved checkpoint set differs")
        for step in callback.saved:
            checkpoint = output / f"checkpoint-{step}"
            step_num = int(step)
            state = read(checkpoint / "trainer_state.json")
            logged = next(row for row in reversed(trainer.state.log_history) if row["step"] == step_num and "loss" in row)
            grads = [read(output / f"finite_checks_rank{rank}.json")["gradient_l2_preclip"][step_num - 1]
                     for rank in range(training["world_size"])]
            save_json(checkpoint / "training_diagnostics.json", {"global_step": step_num, "actual_epoch": state["epoch"],
                "logged_training_loss": logged["loss"], "learning_rate_after_step": logged["learning_rate"],
                "gradient_l2_preclip_by_rank": grads, "gradient_l2_preclip": grads[0]})
        if smoke:
            v4.smoke_eval(ROOT, output, snapshot, tokenizer, trainer.accelerator.device, training)
        save_json(output / "training_completed.json", {"global_step": steps, "seed": 0, "checkpoints": callback.saved,
            "training_loss": result.training_loss, "mask_sha256": lock["weights_sha256"], "code_commit": lock["code_commit"]})
    trainer.accelerator.wait_for_everyone()


def evaluate(smoke, step):
    lock = verify_lock()
    require(step in ((4,) if smoke else STEPS), "Unexpected evaluation checkpoint")
    v4.load_lock = lambda root: lock
    v4.VARIANT = NAME
    v4.output_for = lambda root, seed: output_for(smoke)
    v4.configure_eval = configure_with_v4_reference
    v4.official_eval(ROOT, 0, step)


def verify_training(smoke, lock):
    run = output_for(smoke)
    complete = read(run / "training_completed.json")
    expected_steps = (4,) if smoke else STEPS
    require(complete["global_step"] == max(expected_steps) == read(run / "trainer_state.json")["global_step"]
            and set(complete["checkpoints"]) == set(map(str, expected_steps))
            and complete["mask_sha256"] == lock["weights_sha256"], "Training step or input hash differs")
    for rank in (0, 1):
        values = read(run / f"finite_checks_rank{rank}.json")
        require(len(values["loss"]) == max(expected_steps) * 4 and len(values["gradient_l2_preclip"]) == max(expected_steps)
                and all(map(math.isfinite, values["loss"] + values["gradient_l2_preclip"])), "Nonfinite training diagnostics")
    proof = read(run / "replay_validation.json")
    require(proof["replayed_samples"] == 200 and proof["ranks_agree"] and proof["initial_reference_bitwise_equal"],
            "Replay or reference proof missing")


def queue():
    lock = verify_lock()
    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    command = [sys.executable, "-u", str(Path(__file__).resolve())]

    def status(phase, **details):
        record = {"phase": phase, **details}
        save_json(ROOT / "status.json", record)
        print(json.dumps(record), flush=True)

    def launch(cmd, name, gpu=None):
        env = os.environ.copy()
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        with (logs / name).open("a") as stream:
            return subprocess.Popen(cmd, cwd=SOURCE, env=env, stdout=stream, stderr=subprocess.STDOUT)

    try:
        for smoke in (True, False):
            stage, run, steps = ("smoke" if smoke else "formal"), output_for(smoke), ((4,) if smoke else STEPS)
            if not (run / "training_completed.json").is_file():
                require(not run.exists(), f"Partial training directory requires inspection: {run}")
                status("training", stage=stage)
                torchrun = [*command[:2], "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2", *command[2:]]
                require(launch([*torchrun, "train", stage], f"{stage}.train.log").wait() == 0, f"{stage} training failed")
            verify_training(smoke, lock)
            for offset in range(0, len(steps), 2):
                pending = [s for s in steps[offset:offset+2] if not (run / f"checkpoint-{s}/evals/official/validated.json").is_file()]
                for _ in range(2):
                    if not pending:
                        break
                    status("evaluation", stage=stage, steps=pending)
                    procs = {s: launch([*command, "evaluate", stage, str(s)], f"{stage}.step{s}.eval.log", gpu)
                             for gpu, s in enumerate(pending)}
                    pending = [s for s, proc in procs.items() if proc.wait()]
                require(not pending, f"Evaluation failed for {stage}/{pending}")
            if smoke:
                from analyze_soft_adaptive import verify_evaluation
                proof = read(run / "smoke_validation.json")
                require(proof["passed"] and proof["checkpoints"]["4"]["offline_reload"]
                        and proof["mask_sha256"] == lock["weights_sha256"], "Smoke reload/weights failed")
                result = verify_evaluation(run, 4, lock, smoke=True)[1]
                save_json(ROOT / "smoke_validation.json", proof | {"official_replay": result,
                          "two_rank_replay": read(run / "replay_validation.json")["ranks_agree"],
                          "weighted_npo_unit_tests_passed": True})
                status("smoke_passed")
        status("analysis")
        require(launch([sys.executable, "-u", str(WORKSPACE / "scripts/analyze_soft_adaptive.py")], "analysis.log", "").wait() == 0,
                "Analysis/ZIP validation failed")
        status("complete", formal_checkpoints=len(STEPS))
    except Exception as error:
        status("failed", error=str(error))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("queue", "train", "evaluate"))
    parser.add_argument("stage", choices=("smoke", "formal"), nargs="?")
    parser.add_argument("step", type=int, nargs="?")
    options = parser.parse_args()
    if options.action == "queue":
        with (ROOT / "queue.lock").open("a") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            queue()
    elif options.action == "train":
        training(options.stage == "smoke")
    else:
        evaluate(options.stage == "smoke", options.step)


if __name__ == "__main__":
    main()
