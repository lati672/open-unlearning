"""Diagnose layer-wise representation and utility drift after LLM unlearning."""

import argparse
import csv
import gc
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from scipy.stats import pearsonr, spearmanr
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class ChunkRecord:
    sample_id: int
    book_index: int
    chunk_index: int
    start_token: int
    source_tokens: int
    input_ids: List[int]


@dataclass
class ModelMetrics:
    pooled_drift: torch.Tensor
    token_drift: torch.Tensor
    norm_ratio: torch.Tensor
    loss: torch.Tensor
    loss_change: torch.Tensor
    cka_drift: torch.Tensor


def parse_model_spec(value: str) -> Tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("model must be written as LABEL=PATH")
    label, path = value.split("=", 1)
    if not label.strip() or not path.strip():
        raise argparse.ArgumentTypeError("model must have a non-empty label and path")
    return label.strip(), path.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure layer-wise representation and utility drift on retain chunks."
    )
    parser.add_argument("--reference-model", required=True)
    parser.add_argument(
        "--model", dest="models", action="append", type=parse_model_spec,
        required=True, help="Comparison model as LABEL=PATH; repeat as needed."
    )
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--dataset-path", default="muse-bench/MUSE-Books")
    parser.add_argument("--dataset-name", default="raw")
    parser.add_argument("--dataset-split", default="retain1")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--chunks-per-book", type=int, default=24)
    parser.add_argument(
        "--num-prompts", type=int, default=0,
        help="Optional cap after per-book sampling; 0 means no cap."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--pooling", choices=("mean", "last"), default="mean")
    parser.add_argument("--reference-device", default="auto")
    parser.add_argument("--comparison-device", default="auto")
    parser.add_argument(
        "--dtype", choices=("auto", "bfloat16", "float16", "float32"),
        default="auto"
    )
    parser.add_argument("--include-embedding", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-confidence", type=float, default=0.95)
    parser.add_argument(
        "--bootstrap-unit", choices=("book", "chunk"), default="book",
        help="Resample books by default so chunks from one book stay clustered."
    )
    parser.add_argument("--baseline-label", default="default-all-layers")
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("saves/analysis/muse_representation_drift")
    )
    return parser.parse_args()


def resolve_devices(
    reference_requested: str, comparison_requested: str
) -> Tuple[torch.device, torch.device]:
    if reference_requested != "auto":
        reference = torch.device(reference_requested)
    elif torch.cuda.is_available():
        reference = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        reference = torch.device("mps")
    else:
        reference = torch.device("cpu")
    if comparison_requested != "auto":
        comparison = torch.device(comparison_requested)
    elif torch.cuda.device_count() >= 2:
        comparison = torch.device("cuda:1")
    else:
        comparison = reference
    return reference, comparison


def resolve_dtype(requested: str, device: torch.device) -> torch.dtype:
    if requested != "auto":
        return {
            "bfloat16": torch.bfloat16, "float16": torch.float16,
            "float32": torch.float32,
        }[requested]
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device.type == "mps":
        return torch.float16
    return torch.float32


def sample_chunks(args: argparse.Namespace, tokenizer) -> List[ChunkRecord]:
    dataset = load_dataset(
        args.dataset_path, args.dataset_name or None, split=args.dataset_split
    )
    if args.text_column not in dataset.column_names:
        raise KeyError(
            f"Column {args.text_column!r} not found; available: {dataset.column_names}"
        )
    rng = random.Random(args.seed)
    prefix = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
    content_length = args.max_length - len(prefix)
    if content_length < 2:
        raise ValueError("max-length is too small to form a language-modeling chunk.")
    records: List[ChunkRecord] = []
    for book_index, text in enumerate(dataset[args.text_column]):
        if not isinstance(text, str) or not text.strip():
            continue
        token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(token_ids) < 2:
            continue
        # These fixed-width slots do not overlap. Shuffling samples positions
        # throughout each book without over-representing adjacent passages.
        starts = list(range(0, len(token_ids) - content_length + 1, content_length))
        if not starts:
            continue
        rng.shuffle(starts)
        for chunk_index, start in enumerate(starts[: args.chunks_per_book]):
            content = token_ids[start : start + content_length]
            records.append(
                ChunkRecord(
                    sample_id=-1, book_index=book_index, chunk_index=chunk_index,
                    start_token=start, source_tokens=len(content),
                    input_ids=prefix + content,
                )
            )
    if not records:
        raise ValueError("No valid chunks were produced from the selected split.")
    rng.shuffle(records)
    if args.num_prompts > 0:
        records = records[: args.num_prompts]
    for sample_id, record in enumerate(records):
        record.sample_id = sample_id
    return records


def make_batches(
    records: Sequence[ChunkRecord], pad_token_id: int, batch_size: int
) -> List[Dict[str, torch.Tensor]]:
    batches = []
    for start in range(0, len(records), batch_size):
        examples = records[start : start + batch_size]
        width = max(len(record.input_ids) for record in examples)
        input_ids = torch.full((len(examples), width), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for row, record in enumerate(examples):
            length = len(record.input_ids)
            input_ids[row, :length] = torch.tensor(record.input_ids)
            attention_mask[row, :length] = 1
        batches.append({"input_ids": input_ids, "attention_mask": attention_mask})
    return batches


def select_states(outputs, include_embedding: bool) -> Tuple[torch.Tensor, ...]:
    return outputs.hidden_states if include_embedding else outputs.hidden_states[1:]


def pool_hidden_state(
    hidden: torch.Tensor, attention_mask: torch.Tensor, pooling: str
) -> torch.Tensor:
    if pooling == "last":
        last_indices = attention_mask.sum(dim=1).sub(1).clamp_min(0)
        rows = torch.arange(hidden.shape[0], device=hidden.device)
        return hidden[rows, last_indices]
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)


def per_prompt_loss(logits: torch.Tensor, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    shift_logits = logits[:, :-1].float()
    targets = batch["input_ids"][:, 1:]
    mask = batch["attention_mask"][:, 1:].float()
    losses = F.cross_entropy(shift_logits.transpose(1, 2), targets, reduction="none")
    return (losses * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)


@torch.inference_mode()
def extract_reference_summary(
    model, batches: Sequence[Dict[str, torch.Tensor]], device: torch.device,
    pooling: str, include_embedding: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    pooled_batches, loss_batches = [], []
    for batch_index, cpu_batch in enumerate(batches, start=1):
        batch = {key: value.to(device) for key, value in cpu_batch.items()}
        outputs = model(
            **batch, output_hidden_states=True, return_dict=True, use_cache=False
        )
        states = select_states(outputs, include_embedding)
        pooled_batches.append(
            torch.stack(
                [pool_hidden_state(state, batch["attention_mask"], pooling) for state in states],
                dim=1,
            ).float().cpu()
        )
        loss_batches.append(per_prompt_loss(outputs.logits, batch).cpu())
        print(
            f"  reference batch {batch_index}/{len(batches)}",
            end="\r" if batch_index < len(batches) else "\n", flush=True,
        )
        del outputs, states, batch
    return torch.cat(pooled_batches), torch.cat(loss_batches)


def linear_cka_drift(reference: torch.Tensor, comparison: torch.Tensor) -> torch.Tensor:
    """Return 1-linear-CKA per layer for centered sample-by-feature matrices."""
    values = []
    for layer in range(reference.shape[1]):
        x = reference[:, layer].double()
        y = comparison[:, layer].double()
        x = x - x.mean(dim=0, keepdim=True)
        y = y - y.mean(dim=0, keepdim=True)
        k, l = x @ x.T, y @ y.T
        denominator = torch.linalg.vector_norm(k) * torch.linalg.vector_norm(l)
        similarity = (k * l).sum() / denominator.clamp_min(
            torch.finfo(torch.float64).eps
        )
        values.append(1.0 - similarity.clamp(0.0, 1.0))
    return torch.stack(values).float()


@torch.inference_mode()
def compare_model(
    reference_model, comparison_path: str,
    batches: Sequence[Dict[str, torch.Tensor]], reference_pooled: torch.Tensor,
    reference_loss: torch.Tensor, reference_device: torch.device,
    comparison_device: torch.device, dtype: torch.dtype, pooling: str,
    include_embedding: bool,
) -> ModelMetrics:
    print(f"Loading comparison model: {comparison_path}", flush=True)
    comparison_model = AutoModelForCausalLM.from_pretrained(
        comparison_path, torch_dtype=dtype
    ).to(comparison_device)
    comparison_model.eval()
    reference_base = reference_model.base_model
    pooled_all, token_all, norm_all, loss_all = [], [], [], []
    for batch_index, cpu_batch in enumerate(batches, start=1):
        ref_batch = {key: value.to(reference_device) for key, value in cpu_batch.items()}
        cmp_batch = {key: value.to(comparison_device) for key, value in cpu_batch.items()}
        ref_outputs = reference_base(
            **ref_batch, output_hidden_states=True, return_dict=True, use_cache=False
        )
        cmp_outputs = comparison_model(
            **cmp_batch, output_hidden_states=True, return_dict=True, use_cache=False
        )
        ref_states = select_states(ref_outputs, include_embedding)
        cmp_states = select_states(cmp_outputs, include_embedding)
        if len(ref_states) != len(cmp_states):
            raise ValueError("Reference and comparison models have different layer counts.")
        pooled_layers, token_layers, norm_layers = [], [], []
        for ref_state, cmp_state in zip(ref_states, cmp_states):
            ref_state = ref_state.to(comparison_device)
            mask = cmp_batch["attention_mask"].float()
            pooled_layers.append(
                pool_hidden_state(cmp_state, cmp_batch["attention_mask"], pooling)
            )
            cosine = F.cosine_similarity(ref_state.float(), cmp_state.float(), dim=-1)
            token_layers.append(((1.0 - cosine) * mask).sum(1) / mask.sum(1))
            ref_norm = torch.linalg.vector_norm(ref_state.float(), dim=-1)
            cmp_norm = torch.linalg.vector_norm(cmp_state.float(), dim=-1)
            mean_ref_norm = (ref_norm * mask).sum(1) / mask.sum(1)
            mean_cmp_norm = (cmp_norm * mask).sum(1) / mask.sum(1)
            norm_layers.append(mean_cmp_norm / mean_ref_norm.clamp_min(1e-12))
        pooled_all.append(torch.stack(pooled_layers, dim=1).float().cpu())
        token_all.append(torch.stack(token_layers, dim=1).cpu())
        norm_all.append(torch.stack(norm_layers, dim=1).cpu())
        loss_all.append(per_prompt_loss(cmp_outputs.logits, cmp_batch).cpu())
        print(
            f"  comparison batch {batch_index}/{len(batches)}",
            end="\r" if batch_index < len(batches) else "\n", flush=True,
        )
        del ref_outputs, cmp_outputs, ref_states, cmp_states, ref_batch, cmp_batch
    comparison_pooled = torch.cat(pooled_all)
    comparison_loss = torch.cat(loss_all)
    metrics = ModelMetrics(
        pooled_drift=1.0 - F.cosine_similarity(
            reference_pooled, comparison_pooled, dim=-1
        ),
        token_drift=torch.cat(token_all),
        norm_ratio=torch.cat(norm_all),
        loss=comparison_loss,
        loss_change=comparison_loss - reference_loss,
        cka_drift=linear_cka_drift(reference_pooled, comparison_pooled),
    )
    del comparison_model, comparison_pooled
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


def mean_sem(values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    mean = values.mean(dim=0)
    sem = (
        values.std(dim=0, unbiased=True) / math.sqrt(values.shape[0])
        if values.shape[0] > 1 else torch.zeros_like(mean)
    )
    return mean, sem


def safe_correlation(x: np.ndarray, y: np.ndarray) -> Tuple[float, ...]:
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return math.nan, math.nan, math.nan, math.nan
    pearson, spearman = pearsonr(x, y), spearmanr(x, y)
    return (
        float(pearson.statistic), float(pearson.pvalue),
        float(spearman.statistic), float(spearman.pvalue),
    )


def bootstrap_difference(
    difference: torch.Tensor, samples: int, confidence: float, seed: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(
        difference.shape[0], (samples, difference.shape[0]), generator=generator
    )
    bootstrap_means = difference[indices].mean(dim=1)
    alpha = (1.0 - confidence) / 2.0
    low = torch.quantile(bootstrap_means, alpha, dim=0)
    high = torch.quantile(bootstrap_means, 1.0 - alpha, dim=0)
    lower_count = (bootstrap_means <= 0).sum(dim=0)
    upper_count = (bootstrap_means >= 0).sum(dim=0)
    p_value = 2.0 * torch.minimum(
        (lower_count + 1).float() / (samples + 1),
        (upper_count + 1).float() / (samples + 1),
    )
    return difference.mean(0), low, high, p_value.clamp_max(1.0)


def layer_index(position: int, include_embedding: bool) -> int:
    return position - 1 if include_embedding else position


def save_csv_results(
    args: argparse.Namespace, records: Sequence[ChunkRecord],
    reference_loss: torch.Tensor, results: Dict[str, ModelMetrics],
) -> None:
    path = args.output_dir / "layer_metrics.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow((
            "model", "layer_index", "pooled_drift_mean", "pooled_drift_sem",
            "token_drift_mean", "token_drift_sem", "linear_cka_drift",
            "norm_ratio_mean", "norm_ratio_sem",
        ))
        for label, metrics in results.items():
            pooled_mean, pooled_sem = mean_sem(metrics.pooled_drift)
            token_mean, token_sem = mean_sem(metrics.token_drift)
            norm_mean, norm_sem = mean_sem(metrics.norm_ratio)
            for position in range(len(pooled_mean)):
                writer.writerow((
                    label, layer_index(position, args.include_embedding),
                    pooled_mean[position].item(), pooled_sem[position].item(),
                    token_mean[position].item(), token_sem[position].item(),
                    metrics.cka_drift[position].item(), norm_mean[position].item(),
                    norm_sem[position].item(),
                ))
    # Preserve the original pooled-drift outputs for downstream notebooks.
    path = args.output_dir / "representation_drift.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("model", "layer_index", "mean", "sem"))
        for label, metrics in results.items():
            mean, sem = mean_sem(metrics.pooled_drift)
            for position in range(len(mean)):
                writer.writerow((
                    label, layer_index(position, args.include_embedding),
                    mean[position].item(), sem[position].item(),
                ))
    path = args.output_dir / "per_prompt_metrics.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow((
            "model", "sample_id", "book_index", "chunk_index", "start_token",
            "source_tokens", "layer_index", "pooled_cosine_distance",
            "token_cosine_distance", "norm_ratio",
        ))
        for label, metrics in results.items():
            for sample, record in enumerate(records):
                for position in range(metrics.pooled_drift.shape[1]):
                    writer.writerow((
                        label, record.sample_id, record.book_index, record.chunk_index,
                        record.start_token, record.source_tokens,
                        layer_index(position, args.include_embedding),
                        metrics.pooled_drift[sample, position].item(),
                        metrics.token_drift[sample, position].item(),
                        metrics.norm_ratio[sample, position].item(),
                    ))
    path = args.output_dir / "representation_drift_per_prompt.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("model", "sample_id", "layer_index", "cosine_distance"))
        for label, metrics in results.items():
            for sample, record in enumerate(records):
                for position in range(metrics.pooled_drift.shape[1]):
                    writer.writerow((
                        label, record.sample_id,
                        layer_index(position, args.include_embedding),
                        metrics.pooled_drift[sample, position].item(),
                    ))
    path = args.output_dir / "utility_per_prompt.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow((
            "model", "sample_id", "book_index", "chunk_index", "start_token",
            "source_tokens", "reference_nll", "trained_nll", "nll_change",
        ))
        for label, metrics in results.items():
            for sample, record in enumerate(records):
                writer.writerow((
                    label, record.sample_id, record.book_index, record.chunk_index,
                    record.start_token, record.source_tokens, reference_loss[sample].item(),
                    metrics.loss[sample].item(), metrics.loss_change[sample].item(),
                ))
    path = args.output_dir / "utility_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow((
            "model", "reference_nll_mean", "trained_nll_mean", "nll_change_mean",
            "nll_change_sem",
        ))
        for label, metrics in results.items():
            change_mean, change_sem = mean_sem(metrics.loss_change[:, None])
            writer.writerow((
                label, reference_loss.mean().item(), metrics.loss.mean().item(),
                change_mean.item(), change_sem.item(),
            ))
    path = args.output_dir / "utility_drift_correlations.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow((
            "model", "metric", "layer_index", "pearson_r", "pearson_p",
            "spearman_rho", "spearman_p",
        ))
        for label, metrics in results.items():
            loss_change = metrics.loss_change.numpy()
            correlation_metrics = {
                "pooled_drift": metrics.pooled_drift,
                "token_drift": metrics.token_drift,
                "abs_norm_deviation": (metrics.norm_ratio - 1.0).abs(),
            }
            for metric_name, values in correlation_metrics.items():
                for position in range(values.shape[1]):
                    correlations = safe_correlation(
                        loss_change, values[:, position].numpy()
                    )
                    writer.writerow((
                        label, metric_name, layer_index(position, args.include_embedding),
                        *correlations,
                    ))
                global_correlations = safe_correlation(
                    loss_change, values.mean(dim=1).numpy()
                )
                writer.writerow((
                    label, metric_name, "global_mean", *global_correlations,
                ))


def save_bootstrap_results(
    args: argparse.Namespace, records: Sequence[ChunkRecord],
    results: Dict[str, ModelMetrics]
) -> Dict[Tuple[str, str], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    if args.baseline_label not in results:
        raise ValueError(
            f"Bootstrap baseline {args.baseline_label!r} not among {list(results)}"
        )
    plot_values = {}
    path = args.output_dir / "paired_bootstrap.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow((
            "baseline", "comparison", "metric", "layer_index", "mean_difference",
            "ci_low", "ci_high", "bootstrap_p_two_sided", "confidence",
            "bootstrap_samples", "bootstrap_unit",
        ))
        baseline = results[args.baseline_label]
        for comparison_label, comparison in results.items():
            if comparison_label == args.baseline_label:
                continue
            for metric_name in ("pooled_drift", "token_drift"):
                difference = getattr(baseline, metric_name) - getattr(
                    comparison, metric_name
                )
                if args.bootstrap_unit == "book":
                    book_indices = sorted(set(record.book_index for record in records))
                    difference = torch.stack([
                        difference[
                            torch.tensor([
                                index for index, record in enumerate(records)
                                if record.book_index == book_index
                            ])
                        ].mean(dim=0)
                        for book_index in book_indices
                    ])
                mean, low, high, p_value = bootstrap_difference(
                    difference, args.bootstrap_samples,
                    args.bootstrap_confidence, args.seed
                )
                plot_values[(comparison_label, metric_name)] = (mean, low, high)
                for position in range(len(mean)):
                    writer.writerow((
                        args.baseline_label, comparison_label, metric_name,
                        layer_index(position, args.include_embedding),
                        mean[position].item(), low[position].item(), high[position].item(),
                        p_value[position].item(), args.bootstrap_confidence,
                        args.bootstrap_samples, args.bootstrap_unit,
                    ))
                global_difference = difference.mean(dim=1, keepdim=True)
                g_mean, g_low, g_high, g_p = bootstrap_difference(
                    global_difference, args.bootstrap_samples,
                    args.bootstrap_confidence, args.seed
                )
                writer.writerow((
                    args.baseline_label, comparison_label, metric_name, "global_mean",
                    g_mean.item(), g_low.item(), g_high.item(), g_p.item(),
                    args.bootstrap_confidence, args.bootstrap_samples,
                    args.bootstrap_unit,
                ))
    return plot_values


def plot_results(
    args: argparse.Namespace, results: Dict[str, ModelMetrics],
    bootstrap_values: Dict[
        Tuple[str, str], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ],
) -> None:
    offset = -1 if args.include_embedding else 0
    figure, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    panels = (
        ("Mean-pooled cosine drift", "pooled_drift"),
        ("Token-wise cosine drift", "token_drift"),
        ("Linear CKA drift", "cka_drift"),
        ("Hidden-state norm ratio", "norm_ratio"),
    )
    for axis, (title, metric_name) in zip(axes.flat, panels):
        for label, metrics in results.items():
            values = getattr(metrics, metric_name)
            mean, sem = mean_sem(values) if values.ndim == 2 else (values, None)
            x = np.arange(len(mean)) + offset
            axis.plot(x, mean.numpy(), marker="o", markersize=2.5, label=label)
            if sem is not None:
                axis.fill_between(
                    x, (mean - sem).numpy(), (mean + sem).numpy(), alpha=0.13
                )
        if metric_name == "norm_ratio":
            axis.axhline(1.0, color="black", linestyle="--", linewidth=1)
        axis.set_title(title)
        axis.grid(True, alpha=0.3)
    xlabel = (
        "Layer index (-1 = embedding)" if args.include_embedding
        else "Transformer layer index"
    )
    for axis in axes[-1]:
        axis.set_xlabel(xlabel)
    for axis in axes[:, 0]:
        axis.set_ylabel("Value")
    axes[0, 0].legend(fontsize="small")
    n = next(iter(results.values())).loss.numel()
    figure.suptitle(f"MUSE retain representation diagnostics (n={n})")
    figure.tight_layout()
    figure.savefig(args.output_dir / "representation_metrics.png", dpi=220)
    figure.savefig(args.output_dir / "representation_metrics.pdf")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 6))
    for label, metrics in results.items():
        mean, sem = mean_sem(metrics.pooled_drift)
        x = np.arange(len(mean)) + offset
        axis.plot(x, mean.numpy(), marker="o", markersize=3, label=label)
        axis.fill_between(x, (mean - sem).numpy(), (mean + sem).numpy(), alpha=0.15)
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Representation drift (1 - cosine similarity)")
    axis.set_title(f"Representation drift on MUSE retain chunks (n={n})")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(args.output_dir / "representation_drift.png", dpi=220)
    figure.savefig(args.output_dir / "representation_drift.pdf")
    plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
    for axis, metric_name in zip(axes, ("pooled_drift", "token_drift")):
        for key, (mean, low, high) in bootstrap_values.items():
            comparison_label, stored_metric = key
            if stored_metric != metric_name:
                continue
            x = np.arange(len(mean)) + offset
            axis.plot(x, mean.numpy(), marker="o", markersize=2.5, label=comparison_label)
            axis.fill_between(x, low.numpy(), high.numpy(), alpha=0.15)
        axis.axhline(0.0, color="black", linestyle="--", linewidth=1)
        axis.set_title(f"Default minus conservative: {metric_name}")
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Paired mean difference")
        axis.grid(True, alpha=0.3)
    axes[0].legend(fontsize="small")
    figure.tight_layout()
    figure.savefig(args.output_dir / "paired_bootstrap_differences.png", dpi=220)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True)
    for axis, metric_name in zip(axes, ("pooled_drift", "token_drift")):
        for label, metrics in results.items():
            loss_change = metrics.loss_change.numpy()
            values = getattr(metrics, metric_name)
            correlations = [
                safe_correlation(loss_change, values[:, position].numpy())[0]
                for position in range(values.shape[1])
            ]
            x = np.arange(len(correlations)) + offset
            axis.plot(x, correlations, marker="o", markersize=2.5, label=label)
        axis.axhline(0.0, color="black", linestyle="--", linewidth=1)
        axis.set_title(f"NLL change vs. {metric_name}")
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Pearson r")
        axis.grid(True, alpha=0.3)
    axes[0].legend(fontsize="small")
    figure.tight_layout()
    figure.savefig(args.output_dir / "utility_drift_correlations.png", dpi=220)
    plt.close(figure)


def save_metadata(
    args: argparse.Namespace, records: Sequence[ChunkRecord],
    reference_device: torch.device, comparison_device: torch.device,
) -> None:
    metadata = {
        "reference_model": args.reference_model,
        "models": dict(args.models),
        "dataset": {
            "path": args.dataset_path, "name": args.dataset_name,
            "split": args.dataset_split, "text_column": args.text_column,
        },
        "sampling": {
            "num_chunks": len(records), "chunks_per_book": args.chunks_per_book,
            "total_cap": args.num_prompts, "seed": args.seed,
            "max_length": args.max_length,
            "chunks": [
                {key: value for key, value in asdict(record).items() if key != "input_ids"}
                for record in records
            ],
        },
        "metrics": {
            "pooling": args.pooling,
            "token_drift": "mean valid-token cosine distance per chunk",
            "norm_ratio": "mean trained token norm / mean reference token norm",
            "utility": "autoregressive mean token NLL; change = trained - reference",
            "cka": "1 - linear CKA over pooled chunk representations",
            "include_embedding": args.include_embedding,
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_confidence": args.bootstrap_confidence,
            "bootstrap_unit": args.bootstrap_unit,
        },
        "devices": {
            "reference": str(reference_device), "comparison": str(comparison_device)
        },
    }
    path = args.output_dir / "metadata.json"
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def validate_args(args: argparse.Namespace) -> None:
    if args.chunks_per_book <= 0 or args.max_length <= 1 or args.batch_size <= 0:
        raise ValueError(
            "chunks-per-book and batch-size must be positive; max-length must exceed 1."
        )
    if args.num_prompts < 0 or args.bootstrap_samples <= 0:
        raise ValueError(
            "num-prompts cannot be negative and bootstrap-samples must be positive."
        )
    if not 0.0 < args.bootstrap_confidence < 1.0:
        raise ValueError("bootstrap-confidence must be between 0 and 1.")
    labels = [label for label, _ in args.models]
    if len(labels) != len(set(labels)):
        raise ValueError("Each comparison model label must be unique.")


def main() -> None:
    args = parse_args()
    validate_args(args)
    reference_device, comparison_device = resolve_devices(
        args.reference_device, args.comparison_device
    )
    dtype = resolve_dtype(args.dtype, reference_device)
    print(
        f"reference_device={reference_device}, comparison_device={comparison_device}, "
        f"dtype={dtype}, pooling={args.pooling}"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.reference_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    records = sample_chunks(args, tokenizer)
    if len(records) < 2:
        raise ValueError("At least two sampled chunks are required for CKA.")
    batches = make_batches(records, tokenizer.pad_token_id, args.batch_size)
    print(
        f"Sampled {len(records)} chunks from "
        f"{len(set(record.book_index for record in records))} books."
    )
    print(f"Loading reference model: {args.reference_model}", flush=True)
    reference_model = AutoModelForCausalLM.from_pretrained(
        args.reference_model, torch_dtype=dtype
    ).to(reference_device)
    reference_model.eval()
    reference_pooled, reference_loss = extract_reference_summary(
        reference_model, batches, reference_device, args.pooling,
        args.include_embedding,
    )
    results = {}
    for label, model_path in args.models:
        results[label] = compare_model(
            reference_model, model_path, batches, reference_pooled, reference_loss,
            reference_device, comparison_device, dtype, args.pooling,
            args.include_embedding,
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_csv_results(args, records, reference_loss, results)
    bootstrap_values = save_bootstrap_results(args, records, results)
    plot_results(args, results, bootstrap_values)
    save_metadata(args, records, reference_device, comparison_device)
    print(f"Saved results to {args.output_dir}")


if __name__ == "__main__":
    main()
