"""Run legacy seed-0 controls or the frozen TOFU forget05 multiseed protocol.

python scripts/run_mask_experiments.py --objectives NPO RMU
NPO: five mask variants x summed/token_mean. RMU: five variants.
Completed runs are skipped, and any failed run stops the suite.
"""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

from freeze_mask_protocol import FORMAL_OUTPUT, check_frozen_run, run_name

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objectives", nargs="+", choices=["NPO", "RMU"], default=["NPO", "RMU"])
    variants = ["base", "entity", "adaptive", "random_entity", "random_adaptive"]
    parser.add_argument("--variants", nargs="+", choices=variants, default=variants)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--protocol", type=Path, help="Use a frozen multiseed protocol")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--epochs", type=int, help="Legacy controls only")
    parser.add_argument("--dry-run", action="store_true", help="Print the planned runs without training")
    parser.add_argument("--checkpoint-smoke", action="store_true", help="Validate one case using steps 2 and 4")
    args = parser.parse_args()
    args.output_root = (args.output_root or (FORMAL_OUTPUT if args.protocol else ROOT / "saves/experiments/mask_controls")).resolve()
    gpu_ids = args.gpus.split(",")
    seeds = [0]
    manifest = None
    budget = f"E{args.epochs or 20}"
    run_root = args.output_root
    if args.checkpoint_smoke and not args.protocol:
        parser.error("--checkpoint-smoke requires --protocol")
    if args.protocol:
        args.protocol = args.protocol.resolve()
        protocol, manifest = check_frozen_run(args.protocol, args.output_root)
        training = protocol["training"]
        seeds = training["seeds"]
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
                          "checkpoints": training["checkpoints"] if manifest else None}, indent=2))
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
                break
        if not validated:
            raise ValueError("Run --checkpoint-smoke successfully before launching the formal matrix")
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
            print("SKIP completed", task, flush=True)
            continue
        with socket.socket() as listener:
            listener.bind(("", 0))
            port = listener.getsockname()[1]
        command = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", str(len(gpu_ids)),
                   "--master_port", str(port), str(ROOT / "scripts/train_mask_experiment.py"),
                   "--objective", objective, "--variant", variant, "--normalization", normalization,
                   "--output-root", str(args.output_root), "--seed", str(seed)]
        if args.protocol:
            command += ["--protocol", str(args.protocol)]
        elif args.epochs is not None:
            command += ["--epochs", str(args.epochs)]
        if args.checkpoint_smoke:
            command.append("--checkpoint-smoke")
        print("RUN", task, flush=True)
        with (run_root / f"{task}.log").open("w") as stream:
            subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
        print("DONE", task, flush=True)
    if not args.protocol:
        subprocess.run([sys.executable, str(ROOT / "scripts/analyze_mask_results.py"), "--experiment-root", str(args.output_root)], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
