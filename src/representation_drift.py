"""Measure layer-wise representation drift between a reference and fine-tuned LMs.

For each prompt, hidden states are pooled over non-padding tokens and compared with
the reference model using cosine distance. Models are loaded one at a time so that
the experiment only needs enough accelerator memory for a single model.
"""

import argparse
import csv
import gc
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer


def parse_model_spec(value: str) -> Tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("model must be written as LABEL=PATH")
    label, path = value.split("=", 1)
    if not label.strip() or not path.strip():
        raise argparse.ArgumentTypeError("model must have a non-empty label and path")
    return label.strip(), path.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot layer-wise cosine representation drift on shared prompts."
    )
    parser.add_argument("--reference-model", required=True)
    parser.add_argument(
        "--model",
        dest="models",
        action="append",
        type=parse_model_spec,
        required=True,
        help="Comparison model as LABEL=PATH; repeat for multiple checkpoints.",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Tokenizer path/name (defaults to --reference-model).",
    )
    parser.add_argument("--dataset-path", default="muse-bench/MUSE-Books")
    parser.add_argument("--dataset-name", default="raw")
    parser.add_argument("--dataset-split", default="retain1")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--num-prompts", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--pooling", choices=("mean", "last"), default="mean")
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cuda, cpu, mps, or a concrete device such as cuda:0.",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="auto",
    )
    parser.add_argument(
        "--include-embedding",
        action="store_true",
        help="Also report hidden_states[0], the embedding output.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("saves/analysis/muse_representation_drift"),
    )
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(requested: str, device: torch.device) -> torch.dtype:
    if requested != "auto":
        return {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[requested]
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device.type == "mps":
        return torch.float16
    return torch.float32


def load_prompts(args: argparse.Namespace) -> Tuple[List[str], List[int]]:
    dataset = load_dataset(
        args.dataset_path,
        args.dataset_name or None,
        split=args.dataset_split,
    )
    if args.text_column not in dataset.column_names:
        raise KeyError(
            f"Column {args.text_column!r} not found; available: {dataset.column_names}"
        )
    candidates = [
        (index, value)
        for index, value in enumerate(dataset[args.text_column])
        if isinstance(value, str) and value.strip()
    ]
    if not candidates:
        raise ValueError("No non-empty text prompts were found in the selected split.")
    generator = torch.Generator().manual_seed(args.seed)
    order = torch.randperm(len(candidates), generator=generator).tolist()
    chosen = [candidates[i] for i in order[: min(args.num_prompts, len(candidates))]]
    return [text for _, text in chosen], [index for index, _ in chosen]


def pool_hidden_state(
    hidden: torch.Tensor, attention_mask: torch.Tensor, pooling: str
) -> torch.Tensor:
    if pooling == "last":
        last_indices = attention_mask.sum(dim=1).sub(1).clamp_min(0)
        rows = torch.arange(hidden.shape[0], device=hidden.device)
        return hidden[rows, last_indices]
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)


@torch.inference_mode()
def extract_representations(
    model_path: str,
    token_batches: Sequence[Dict[str, torch.Tensor]],
    device: torch.device,
    dtype: torch.dtype,
    pooling: str,
    include_embedding: bool,
) -> torch.Tensor:
    print(f"Loading model: {model_path}", flush=True)
    model = AutoModel.from_pretrained(model_path, torch_dtype=dtype)
    model.to(device)
    model.eval()
    pooled_batches = []
    expected_layers = None
    for batch_index, cpu_batch in enumerate(token_batches, start=1):
        batch = {key: value.to(device) for key, value in cpu_batch.items()}
        outputs = model(
            **batch,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        states = outputs.hidden_states
        if not include_embedding:
            states = states[1:]
        if expected_layers is None:
            expected_layers = len(states)
        elif len(states) != expected_layers:
            raise RuntimeError("The model returned an inconsistent number of layers.")
        pooled = torch.stack(
            [pool_hidden_state(state, batch["attention_mask"], pooling) for state in states],
            dim=1,
        )
        pooled_batches.append(pooled.float().cpu())
        print(
            f"  extracted batch {batch_index}/{len(token_batches)}",
            end="\r" if batch_index < len(token_batches) else "\n",
            flush=True,
        )
        del outputs, states, pooled, batch
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()
    return torch.cat(pooled_batches, dim=0)


def tokenize_prompts(
    prompts: Sequence[str], tokenizer, batch_size: int, max_length: int
) -> List[Dict[str, torch.Tensor]]:
    batches = []
    for start in range(0, len(prompts), batch_size):
        batches.append(
            tokenizer(
                list(prompts[start : start + batch_size]),
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
        )
    return batches


def summarize_distances(
    reference: torch.Tensor, comparison: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if reference.shape != comparison.shape:
        raise ValueError(
            "Reference and comparison representations have different shapes: "
            f"{tuple(reference.shape)} vs {tuple(comparison.shape)}"
        )
    per_prompt = 1.0 - F.cosine_similarity(reference, comparison, dim=-1)
    mean = per_prompt.mean(dim=0)
    if per_prompt.shape[0] > 1:
        sem = per_prompt.std(dim=0, unbiased=True) / math.sqrt(per_prompt.shape[0])
    else:
        sem = torch.zeros_like(mean)
    return per_prompt, mean, sem


def save_results(
    args: argparse.Namespace,
    sample_indices: Sequence[int],
    results: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    layer_offset = -1 if args.include_embedding else 0
    summary_path = args.output_dir / "representation_drift.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("model", "layer_index", "hidden_state_index", "mean", "sem"),
        )
        writer.writeheader()
        for label, (_, means, sems) in results.items():
            for position, (mean, sem) in enumerate(zip(means.tolist(), sems.tolist())):
                writer.writerow(
                    {
                        "model": label,
                        "layer_index": position + layer_offset,
                        "hidden_state_index": position if args.include_embedding else position + 1,
                        "mean": mean,
                        "sem": sem,
                    }
                )

    detail_path = args.output_dir / "representation_drift_per_prompt.csv"
    with detail_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("model", "dataset_index", "layer_index", "cosine_distance"))
        for label, (distances, _, _) in results.items():
            for prompt_position, dataset_index in enumerate(sample_indices):
                for position, value in enumerate(distances[prompt_position].tolist()):
                    writer.writerow((label, dataset_index, position + layer_offset, value))

    metadata = {
        "reference_model": args.reference_model,
        "models": dict(args.models),
        "dataset": {
            "path": args.dataset_path,
            "name": args.dataset_name,
            "split": args.dataset_split,
            "text_column": args.text_column,
            "sample_indices": list(sample_indices),
        },
        "num_prompts": len(sample_indices),
        "seed": args.seed,
        "max_length": args.max_length,
        "pooling": args.pooling,
        "include_embedding": args.include_embedding,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    plt.figure(figsize=(10, 6))
    for label, (_, means, sems) in results.items():
        x = torch.arange(len(means)).numpy() + layer_offset
        y = means.numpy()
        error = sems.numpy()
        plt.plot(x, y, marker="o", markersize=3, linewidth=1.8, label=label)
        plt.fill_between(x, y - error, y + error, alpha=0.15)
    plt.xlabel(
        "Layer index (-1 = embedding)"
        if args.include_embedding
        else "Transformer layer index"
    )
    plt.ylabel("Representation drift (1 - cosine similarity)")
    plt.title(f"Representation drift on MUSE retain prompts (n={len(sample_indices)})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output_dir / "representation_drift.png", dpi=220)
    plt.savefig(args.output_dir / "representation_drift.pdf")
    plt.close()
    print(f"Saved results to {args.output_dir}")


def main() -> None:
    args = parse_args()
    if args.num_prompts <= 0 or args.max_length <= 0 or args.batch_size <= 0:
        raise ValueError("num-prompts, max-length, and batch-size must be positive.")
    labels = [label for label, _ in args.models]
    if len(labels) != len(set(labels)):
        raise ValueError("Each comparison model label must be unique.")

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    print(f"Using device={device}, dtype={dtype}, pooling={args.pooling}")
    prompts, sample_indices = load_prompts(args)
    print(f"Selected {len(prompts)} retain prompts with seed {args.seed}.")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.reference_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    token_batches = tokenize_prompts(prompts, tokenizer, args.batch_size, args.max_length)

    reference = extract_representations(
        args.reference_model,
        token_batches,
        device,
        dtype,
        args.pooling,
        args.include_embedding,
    )
    results = {}
    for label, model_path in args.models:
        comparison = extract_representations(
            model_path,
            token_batches,
            device,
            dtype,
            args.pooling,
            args.include_embedding,
        )
        results[label] = summarize_distances(reference, comparison)
        del comparison
        gc.collect()
    save_results(args, sample_indices, results)


if __name__ == "__main__":
    main()
