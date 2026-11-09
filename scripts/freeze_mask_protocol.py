"""Freeze the TOFU multiseed protocol, its inputs, and environment provenance."""

import argparse
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "configs/protocols/tofu_forget05_multiseed.json"
FORMAL_OUTPUT = ROOT / "saves/experiments/tofu_forget05_multiseed_v2"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def check_environment(protocol):
    import torch

    for name, version in protocol["packages"].items():
        actual = metadata.version(name)
        if actual != version:
            raise ValueError(f"{name} version {actual} differs from frozen version {version}")
    if torch.version.cuda != protocol["torch_cuda"]:
        raise ValueError(f"PyTorch CUDA runtime {torch.version.cuda} differs from frozen version {protocol['torch_cuda']}")


def check_frozen_run(protocol_path, output_root):
    protocol = json.loads(protocol_path.read_text())
    check_environment(protocol)
    manifest = json.loads((output_root / "protocol_manifest.json").read_text())
    if manifest["code_commit"] != git("rev-parse", "HEAD") or git("status", "--porcelain"):
        raise ValueError("Formal runs require the clean Git commit recorded in protocol_manifest.json")
    if manifest["protocol_sha256"] != sha256(protocol_path):
        raise ValueError("Protocol definition changed after freezing")
    frozen = output_root / "frozen_inputs"
    for source, expected in protocol["inputs"].items():
        if sha256(frozen / Path(source).name) != expected:
            raise ValueError(f"Frozen input checksum mismatch: {source}")
    if sha256(frozen / "random_mask_expectations.json") != manifest["random_mask_expectations_sha256"]:
        raise ValueError("Frozen Random mask expectations changed")
    return protocol, manifest


def fixed_rmu_vector(protocol):
    import torch

    cfg = protocol["rmu_control"]
    generator = torch.Generator(device="cpu").manual_seed(cfg["cpu_seed"])
    torch.rand(cfg["skip_uniform_draws"], generator=generator)
    vector = torch.rand(1, 1, cfg["hidden_size"], generator=generator)
    vector = vector / vector.norm() * cfg["steering_coeff"]
    if hashlib.sha256(vector.numpy().tobytes()).hexdigest() != cfg["sha256"]:
        raise ValueError("RMU control vector differs from the seed-0 experiment")
    return vector


def verify_dataset_revision(dataset, revision):
    # Offline datasets can fall back to the newest cache even with revision set.
    cache_files = dataset.data.cache_files
    if not cache_files or any(revision not in Path(item["filename"]).parts for item in cache_files):
        raise ValueError(f"Loaded TOFU data differs from frozen revision {revision}")


def validate_mask_inputs(protocol, frozen):
    sys.path.insert(0, str(ROOT / "src"))
    from omegaconf import OmegaConf
    import torch
    from torch.utils.data import DataLoader
    from data import get_datasets
    from data.collators import DataCollatorWithEntityMask, DataCollatorWithLogProbs, FrozenMaskCollator
    from model import get_tokenizer
    from trainer.masking import density_matched_random_mask, eligible_positions, sample_indices

    model_cfg = OmegaConf.load(ROOT / "configs/model/Llama-3.2-1B-Instruct.yaml")
    base_model = protocol["base_model"]
    logprob_metadata = json.loads((frozen / "logprobs.metadata.json").read_text())
    cached_base = logprob_metadata.get("base_model", logprob_metadata.get("model"))
    cached_revision = logprob_metadata.get("base_model_commit", logprob_metadata.get("model_commit"))
    if (cached_base, cached_revision) != (base_model["id"], base_model["revision"]):
        raise ValueError("Base model differs from frozen logprobs")
    tokenizer = get_tokenizer({"pretrained_model_name_or_path": base_model["id"], "revision": base_model["revision"]})
    ds = protocol["dataset"]
    config = OmegaConf.load(ROOT / "configs/data/datasets/TOFU_QA_forget.yaml")
    datasets = {}
    for split in ["forget", "retain", "holdout"]:
        config.TOFU_QA_forget.args.hf_args.update({"name": ds[split], "revision": ds["revision"], "path": ds["id"]})
        datasets[split] = get_datasets(config, tokenizer=tokenizer, template_args=model_cfg.template_args)
        verify_dataset_revision(datasets[split], ds["revision"])
        if len(datasets[split]) != ds[f"{split}_samples"]:
            raise ValueError(f"Unexpected size of frozen {split} split")
    dataset = datasets["forget"]
    names = json.loads((ROOT / "configs/tofu_forget_authors.json").read_text())[ds["forget"]]
    collators = {
        "adaptive": DataCollatorWithLogProbs(tokenizer, str(frozen / "logprobs.json")),
        "entity": DataCollatorWithEntityMask(tokenizer, names),
    }
    expectations, stats = {}, {}
    for kind, collator in collators.items():
        manifest = frozen / f'masks_TOFU_{ds["forget"]}_{kind}.jsonl'
        collator = FrozenMaskCollator(collator, manifest)
        if collator.metadata["target_model_commit"] != protocol["target_model"]["revision"]:
            raise ValueError(f"Wrong target checkpoint for {kind} mask")
        rows = {}
        selected, eligible_count, empty = 0, 0, 0
        for batch in DataLoader(dataset, batch_size=4, collate_fn=collator):
            eligible = eligible_positions(batch)
            reference = batch["reference_mask"].bool()
            if (reference & ~eligible).any():
                raise ValueError(f"Ineligible positions in {kind} mask")
            if kind == "entity" and not torch.equal(reference, eligible & batch["entity_mask"].bool()):
                raise ValueError("Entity mask differs from the frozen manifest")
            indices = sample_indices(batch)
            random_mask = density_matched_random_mask(eligible, reference, indices, protocol["random_mask_seed"])
            if not torch.equal(random_mask.sum(-1), reference.sum(-1)) or (random_mask & ~eligible).any():
                raise ValueError(f"Random {kind} mask does not match the reference density")
            selected += int(reference.sum())
            eligible_count += int(eligible.sum())
            empty += int((reference.sum(-1) == 0).sum())
            for i, index in enumerate(indices):
                if str(index) in rows:
                    raise ValueError(f"Duplicate sample index {index}")
                mask = random_mask[i, batch["attention_mask"][i].bool()]
                rows[str(index)] = {"count": int(mask.sum()), "hash": hashlib.sha256(mask.numpy().tobytes()).hexdigest()}
        if set(map(int, rows)) != collator.records.keys():
            raise ValueError(f"Frozen {kind} IDs do not cover the dataset")
        expectations[kind] = rows
        stats[kind] = {"samples": len(rows), "selected": selected, "eligible": eligible_count, "empty": empty}
    return expectations, stats


def existing_seed_zero():
    runs = []
    for marker in sorted((ROOT / "saves/experiments/mask_controls").glob("*/completed.json")):
        result = json.loads(marker.read_text())
        state = json.loads((marker.parent / "trainer_state.json").read_text())
        if "_seed0_" in marker.parent.name and state["global_step"] == 120 and state["epoch"] == 19.2:
            metrics_path = marker.parent / "evals/metrics.json"
            runs.append({"run": result["task"], "step": 120, "actual_epoch": 19.2,
                         "metrics_path": str(metrics_path.relative_to(ROOT)), "metrics_sha256": sha256(metrics_path)})
    return runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-root", type=Path, default=FORMAL_OUTPUT)
    args = parser.parse_args()
    manifest_path = args.output_root / "protocol_manifest.json"
    if manifest_path.exists():
        _, manifest = check_frozen_run(args.protocol, args.output_root)
        print(f"Already frozen at {manifest['code_commit']}: {manifest_path}")
        return
    if git("status", "--porcelain"):
        raise ValueError("Commit the experiment definition before freezing it")
    protocol = json.loads(args.protocol.read_text())
    if protocol["retain_only_model"]["evaluation_logs"] not in protocol["inputs"]:
        raise ValueError("Retain-only model evaluation logs must be pinned in the inputs")
    check_environment(protocol)
    fixed_rmu_vector(protocol)
    frozen = args.output_root / "frozen_inputs"
    frozen.mkdir(parents=True, exist_ok=True)
    for source, expected in protocol["inputs"].items():
        path = ROOT / source
        if sha256(path) != expected:
            raise ValueError(f"Source artifact checksum mismatch: {source}")
        shutil.copy2(path, frozen / path.name)
    expectations, stats = validate_mask_inputs(protocol, frozen)
    random_expectations = frozen / "random_mask_expectations.json"
    save_json(random_expectations, expectations)
    import torch

    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": dict(sorted((dist.metadata["Name"], dist.version) for dist in metadata.distributions() if dist.metadata["Name"])),
        "torch_cuda": torch.version.cuda,
        "gpus": subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], text=True).splitlines(),
    }
    save_json(args.output_root / "environment.json", environment)
    manifest = {
        "code_commit": git("rev-parse", "HEAD"),
        "code_branch": git("branch", "--show-current"),
        "protocol_sha256": sha256(args.protocol),
        "protocol": protocol,
        "random_mask_expectations_sha256": sha256(random_expectations),
        "mask_validation": stats,
        "seed0_existing_step120": existing_seed_zero(),
        "seed0_note": "Existing focused step-120 metrics are retained with provenance. Step 60 was never saved and requires a new run with the same 120-step schedule.",
    }
    save_json(manifest_path, manifest)
    print(json.dumps({"commit": manifest["code_commit"], "mask_validation": stats,
                      "existing_seed0_step120_runs": len(manifest["seed0_existing_step120"]),
                      "output": str(args.output_root)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
