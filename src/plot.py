"""
Generate per-metric line plots from TOFU experiment summaries.

Each experiment stores a TOFU_SUMMARY.json under::

    saves/unlearn/tofu_{model}_{forget_split}_{variant}_{trainer}_E{num_epochs}/evals

This script groups runs by forget split, draws one line per variant/trainer pair,
and saves one PNG per metric showing how it evolves across epoch counts.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # Enable plotting in headless environments.
import matplotlib.pyplot as plt  # noqa: E402


DEFAULT_MODEL = "Llama-3.2-1B-Instruct"
DEFAULT_ROOT = Path("saves/unlearn")
DEFAULT_OUTPUT = Path("saves/plot")


@dataclass(frozen=True)
class RunSummary:
    forget_split: str
    variant: str
    trainer: str
    epochs: int
    metrics: Mapping[str, float]
    task_dir: Path

    @property
    def variant_label(self) -> str:
        if self.variant == self.trainer:
            return self.variant
        return f"{self.variant} ({self.trainer})"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create per-metric line plots from TOFU summary JSON files."
    )
    parser.add_argument(
        "--saves-root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"Directory containing tofu_* runs (default: {DEFAULT_ROOT}).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model identifier embedded in run names (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--forget-splits",
        type=str,
        nargs="*",
        default=None,
        help="Optional subset of forget splits to plot (e.g., forget01 forget05).",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="*",
        default=None,
        help="Optional subset of metric keys to plot. Defaults to all found.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Folder to store generated figures (default: {DEFAULT_OUTPUT}).",
    )
    return parser.parse_args()


def discover_runs(root: Path, model: str, splits_filter: Optional[Sequence[str]]) -> List[RunSummary]:
    if not root.exists():
        raise FileNotFoundError(f"Saves root not found: {root}")
    prefix = f"tofu_{model}_"
    splits_set = set(splits_filter) if splits_filter else None
    summaries: List[RunSummary] = []
    for task_dir in sorted(root.glob(f"{prefix}*")):
        if not task_dir.is_dir():
            continue
        parsed = _parse_task_name(task_dir.name, model)
        if parsed is None:
            continue
        forget_split, variant, trainer, epochs = parsed
        if splits_set and forget_split not in splits_set:
            continue
        eval_path = task_dir / "evals" / "TOFU_SUMMARY.json"
        if not eval_path.exists():
            print(f"Skipping {task_dir}: missing {eval_path}")
            continue
        metrics = _load_metrics(eval_path)
        if not metrics:
            print(f"Skipping {task_dir}: no numeric metrics in {eval_path}")
            continue
        summaries.append(
            RunSummary(
                forget_split=forget_split,
                variant=variant,
                trainer=trainer,
                epochs=epochs,
                metrics=metrics,
                task_dir=task_dir,
            )
        )
    if not summaries:
        filter_msg = f" for splits {sorted(splits_set)}" if splits_set else ""
        raise ValueError(
            f"No runs discovered under {root} for model {model}{filter_msg}. "
            "Check --model/--forget-splits arguments."
        )
    return summaries


def _parse_task_name(task_name: str, model: str) -> Optional[tuple[str, str, str, int]]:
    prefix = f"tofu_{model}_"
    if not task_name.startswith(prefix):
        return None
    suffix = task_name[len(prefix) :]
    try:
        base, trainer, epoch_label = suffix.rsplit("_", 2)
        forget_split, variant = base.split("_", 1)
    except ValueError:
        print(f"Unable to parse task name '{task_name}'")
        return None
    if not epoch_label.startswith("E"):
        print(f"Task name missing epoch marker: '{task_name}'")
        return None
    try:
        epochs = int(epoch_label[1:])
    except ValueError:
        print(f"Invalid epoch count in '{task_name}'")
        return None
    return forget_split, variant, trainer, epochs


def _load_metrics(path: Path) -> Dict[str, float]:
    with path.open("r", encoding="utf-8") as handle:
        content = json.load(handle)
    metrics: Dict[str, float] = {}
    for key, value in content.items():
        if isinstance(value, (int, float)):
            metrics[key] = float(value)
    return metrics


def organize_by_split(
    runs: Iterable[RunSummary],
) -> Dict[str, Dict[str, Dict[int, Mapping[str, float]]]]:
    organized: Dict[str, Dict[str, Dict[int, Mapping[str, float]]]] = {}
    for run in runs:
        split_bucket = organized.setdefault(run.forget_split, {})
        variant_bucket = split_bucket.setdefault(run.variant_label, {})
        variant_bucket[run.epochs] = run.metrics
    return organized


def collect_metric_keys(
    runs: Iterable[RunSummary], filter_keys: Optional[Sequence[str]]
) -> List[str]:
    if filter_keys:
        return list(dict.fromkeys(filter_keys))
    keys: Dict[str, None] = {}
    for run in runs:
        for key in run.metrics:
            keys.setdefault(key, None)
    if not keys:
        raise ValueError("No numeric metrics found in provided runs.")
    return list(keys.keys())


def sanitize_filename(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")
    return cleaned or "metric"


def plot_metrics(
    organized: Mapping[str, Mapping[str, Mapping[int, Mapping[str, float]]]],
    metric_keys: Sequence[str],
    output_root: Path,
) -> None:
    for forget_split, variant_data in organized.items():
        split_dir = output_root / forget_split
        split_dir.mkdir(parents=True, exist_ok=True)
        for metric in metric_keys:
            plt.figure(figsize=(8, 5))
            plotted = False
            for variant_label, epochs_data in sorted(variant_data.items()):
                points = sorted(epochs_data.items())
                xs = [epoch for epoch, metrics in points if metric in metrics]
                ys = [metrics[metric] for _, metrics in points if metric in metrics]
                if not xs:
                    continue
                plotted = True
                plt.plot(xs, ys, marker="o", label=variant_label)
            if not plotted:
                plt.close()
                continue
            plt.title(f"{metric} vs epochs ({forget_split})")
            plt.xlabel("Epochs")
            plt.ylabel(metric)
            plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
            plt.legend(loc="best", fontsize="small")
            plt.tight_layout()
            file_name = f"{sanitize_filename(metric)}.png"
            output_path = split_dir / file_name
            plt.savefig(output_path, dpi=200)
            plt.close()
            print(f"Saved {output_path}")


def main() -> None:
    args = parse_args()
    runs = discover_runs(args.saves_root, args.model, args.forget_splits)
    metrics = collect_metric_keys(runs, args.metrics)
    organized = organize_by_split(runs)
    output_root = args.output_dir
    output_root.mkdir(parents=True, exist_ok=True)
    plot_metrics(organized, metrics, output_root)


if __name__ == "__main__":
    main()
