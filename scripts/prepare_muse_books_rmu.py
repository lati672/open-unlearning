"""Score pinned MUSE Books models and freeze aligned RMU masks before training."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = WORKSPACE / "saves/experiments/muse_books_rmu_multiseed_v1"
ROOT = Path(os.environ.get("MUSE_BOOKS_RMU_ROOT", DEFAULT_ROOT)).resolve()
SOURCE = ROOT / "source"
HUB = Path("/workspace/.hf_home/hub")
COMMIT = "629363802f79b4ccb7b1d4e38bedc2980f6f47df"
TARGET = "muse-bench/MUSE-Books_target"
TARGET_REV = "c8dd3fb23a726762ec66d277933c7cff6767f3c2"
BASE = "meta-llama/Llama-2-7b-hf"
BASE_REV = "01c7f73d771dfac7d292323805ebc428287df4f9"
DATA = "muse-bench/MUSE-Books"
DATA_REV = "051ba90319e920d410d87cfdbd61f25843c1b892"
DATA_PATH = HUB / "datasets--muse-bench--MUSE-Books/snapshots" / DATA_REV
TARGET_PATH = HUB / "models--muse-bench--MUSE-books_target/snapshots" / TARGET_REV
SEEDS = (0, 1, 2)
ARMS = {"base": "RMU", "adaptive": "AdaptiveRMU", "random_adaptive": "RandomRMU"}
MASK_SEED = 1729
TRAIN_SPLITS = {"forget": "forget", "retain": "retain1"}

sys.path.insert(0, str(SOURCE / "src"))
from data import get_datasets
from data.collators import ATTENTION_MASK_POLICY, DataCollatorForSupervisedDataset
from model import get_model, get_tokenizer
from trainer.masking import density_matched_random_mask, eligible_positions
from trainer.utils import compute_model_token_logprobs


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def save(path, contents):
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(contents, indent=2, allow_nan=False) + "\n")
    pending.replace(path)


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def table(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def local_dataset(args):
    require(args["path"] == DATA, f"Unexpected dataset {args}")
    split = args["split"]
    files = list((DATA_PATH / args["name"]).glob(f"{split}-*.parquet"))
    require(len(files) == 1, f"Expected one pinned parquet shard for {args}")
    return {"path": "parquet", "data_files": str(files[0]), "split": "train"}


def model_config(role, attention, device=None):
    config = OmegaConf.load(SOURCE / "configs/model/Llama-2-7b-hf.yaml")
    config.tokenizer_args.update({"revision": BASE_REV, "local_files_only": True})
    config.model_args.update({"local_files_only": True, "attn_implementation": attention})
    if role == "target":
        config.model_args.pretrained_model_name_or_path = str(TARGET_PATH)
    else:
        config.model_args.revision = BASE_REV
    if device is not None:
        config.model_args.device_map = {"": device}
    return config


def train_datasets(config, tokenizer):
    datasets = {}
    for label, split in TRAIN_SPLITS.items():
        spec = OmegaConf.load(SOURCE / f"configs/data/datasets/MUSE_{label}.yaml")
        spec[f"MUSE_{label}"].args.hf_args = local_dataset({"path": DATA, "name": "raw", "split": split})
        datasets[label] = get_datasets(spec, tokenizer=tokenizer, template_args=config.template_args)
    return datasets


@torch.inference_mode()
def score(role):
    path = ROOT / "frozen_inputs" / f"{role}_logprobs.npz"
    if path.exists():
        print(f"Using completed scores {path}", flush=True)
        return
    config = model_config(role, "sdpa", "cuda:0")
    model, tokenizer = get_model(config)
    model.eval()
    if role == "base":
        require(model.config._commit_hash == BASE_REV, "Base checkpoint revision differs")
    data = train_datasets(config, tokenizer)["forget"]
    collator = DataCollatorForSupervisedDataset(tokenizer, index="index")
    logprobs, ids, offsets = [], [], [0]
    for start in range(0, len(data), 4):
        indices = range(start, min(start + 4, len(data)))
        batch = collator([data[i] for i in indices])
        tokens = batch["input_ids"].to("cuda")
        attention = batch["attention_mask"].to("cuda")
        outputs = model(input_ids=tokens, attention_mask=attention, use_cache=False)
        scores = compute_model_token_logprobs(outputs.logits, tokens).float().cpu()
        for row, doc_id in enumerate(batch["index"].tolist()):
            length = int(batch["attention_mask"][row].sum())
            require(doc_id == len(logprobs), "Scoring sample IDs are out of order")
            ids.extend(batch["input_ids"][row, :length].tolist())
            logprobs.append(scores[row, :length].numpy())
            offsets.append(offsets[-1] + length)
        if len(logprobs) % 100 == 0:
            print(f"{role}: {len(logprobs)}/{len(data)} scored samples", flush=True)
    pending = path.with_suffix(".tmp")
    with pending.open("wb") as handle:
        np.savez_compressed(handle, logprobs=np.concatenate(logprobs), input_ids=np.asarray(ids, dtype=np.int32),
                            offsets=np.asarray(offsets, dtype=np.int64))
    pending.replace(path)
    print(f"{role}: {len(logprobs)} samples, {digest(path)}", flush=True)


def distribution(values):
    return {"mean": float(np.mean(values)), "sample_std": float(np.std(values, ddof=1)),
            "median": float(np.median(values)),
            "quantiles": {str(q): float(np.percentile(values, q)) for q in (0, 1, 5, 25, 50, 75, 95, 99, 100)}}


def signs(values):
    return {"positive": int((values > 0).sum()), "negative": int((values < 0).sum()), "zero": int((values == 0).sum())}


def prepare_masks():
    frozen = ROOT / "frozen_inputs"
    audit_path = ROOT / "mask_audit.json"
    if audit_path.exists():
        audit = read(audit_path)
        for variant, checksum in audit["mask_hashes"].items():
            require(digest(frozen / f"{variant}.jsonl") == checksum, f"Frozen mask changed: {variant}")
        return audit
    config = model_config("target", "sdpa")
    tokenizer = get_tokenizer(config.tokenizer_args)
    datasets = train_datasets(config, tokenizer)
    data = datasets["forget"]
    collator = DataCollatorForSupervisedDataset(tokenizer, index="index")
    base_path, target_path = [frozen / f"{role}_logprobs.npz" for role in ("base", "target")]
    with np.load(base_path) as base, np.load(target_path) as target:
        for key in ("offsets", "input_ids"):
            require(np.array_equal(base[key], target[key]), f"Target and base inputs differ in {key}")
        require(len(base["offsets"]) == len(data) + 1, "Number of scored samples differs")
        offsets, ids = base["offsets"], base["input_ids"]
        discrepancy = base["logprobs"] - target["logprobs"]
    per_example, selected_deltas, all_deltas = [], [], []
    pending = {arm: frozen / f"{arm}.jsonl.tmp" for arm in ARMS}
    handles = {arm: path.open("w") for arm, path in pending.items()}
    try:
        for index in range(len(data)):
            batch = collator([data[index]])
            start, stop = offsets[index:index + 2]
            tokens = batch["input_ids"][0].tolist()
            require(tokens == ids[start:stop].tolist() and int(batch["index"][0]) == index,
                    f"Mismatch with frozen scored IDs at index {index}")
            eligible = eligible_positions(batch)
            delta = torch.from_numpy(discrepancy[start:stop]).unsqueeze(0)
            require(torch.isfinite(delta[eligible]).all() and eligible.any(), f"Nonfinite eligible discrepancy at {index}")
            finite = eligible & torch.isfinite(delta)
            values = delta[finite]
            average = values.mean()
            adaptive = finite & (delta < average)
            random = density_matched_random_mask(finite, adaptive, [index], MASK_SEED)
            require(random.sum() == adaptive.sum() and not (random & ~finite).any(), f"Random density differs at {index}")
            require(torch.equal(batch["labels"][finite], batch["input_ids"][finite]), f"Shifted label mismatch at {index}")
            all_deltas.append(values.numpy())
            selected_deltas.append(delta[adaptive].numpy())
            record = {"doc_id": index, "sequence_mean": float(average), "eligible": int(finite.sum())}
            for arm, mask in {"base": finite, "adaptive": adaptive, "random_adaptive": random}.items():
                handles[arm].write(json.dumps({"doc_id": index, "input_ids": tokens, "mask": mask[0].int().tolist()},
                                               separators=(",", ":")) + "\n")
                record[f"{arm}_selected"] = int(mask.sum())
                record[f"{arm}_ratio"] = int(mask.sum()) / record["eligible"]
            per_example.append(record)
    finally:
        for handle in handles.values():
            handle.close()
    for arm, path in pending.items():
        path.replace(frozen / f"{arm}.jsonl")
    table(ROOT / "mask_per_example.csv", per_example)
    all_values = np.concatenate(all_deltas)
    selected_values = np.concatenate(selected_deltas)
    sample_means = np.array([r["sequence_mean"] for r in per_example])
    summaries = {}
    for arm in ARMS:
        selected = [r[f"{arm}_selected"] for r in per_example]
        summaries[arm] = {"eligible_tokens": len(all_values), "selected_tokens": sum(selected),
                          "aggregate_selection_ratio": sum(selected) / len(all_values),
                          "empty_masks": selected.count(0),
                          "sample_selection_ratio": distribution([r[f"{arm}_ratio"] for r in per_example])}
    audit = {"samples": len(data), "retain_samples": len(datasets["retain"]), "arms": summaries,
             "model_revisions": {"base": BASE_REV, "target": TARGET_REV}, "dataset_revision": DATA_REV,
             "mask_hashes": {arm: digest(frozen / f"{arm}.jsonl") for arm in ARMS},
             "score_file_hashes": {"base": digest(base_path), "target": digest(target_path)},
             "per_example_sha256": digest(ROOT / "mask_per_example.csv"), "random_mask_seed": MASK_SEED,
             "attention_mask_policy": ATTENTION_MASK_POLICY,
             "eligible": "attended supervised next-token targets with attended predecessor and finite discrepancy",
             "adaptive": "strictly below the finite eligible sequence mean",
             "score_configuration": {"attention": "sdpa", "dtype": "bfloat16", "batch_size": 4},
             "target_to_hidden_alignment": "score from logits[t-1], compare hidden[t]",
             "all_eligible_discrepancies_finite": True, "labels_tokens_attention_alignment": True,
             "per_example_count_match": True, "empty_pattern_match": True,
             "all_discrepancies": {**distribution(all_values), "sign_counts": signs(all_values)},
             "selected_discrepancies": {**distribution(selected_values), "sign_counts": signs(selected_values)},
             "sequence_means": {**distribution(sample_means), "sign_counts": signs(sample_means)}}
    save(audit_path, audit)
    print(json.dumps({"samples": len(data), "retain": len(datasets["retain"]), "arms": summaries}), flush=True)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("target", "base", "masks"))
    args = parser.parse_args()
    (ROOT / "frozen_inputs").mkdir(exist_ok=True)
    if args.action == "masks":
        prepare_masks()
    else:
        score(args.action)


if __name__ == "__main__":
    main()
