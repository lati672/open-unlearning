"""
Utility to visualize token-level log-probability gaps logged by the NPO trainer.

Example:
    python -m src.plot_logprob_diff \
        --run-dir saves/unlearn/tofu_Llama-3.2-1B-Instruct_forget01_NPO
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # Ensure plots work in headless environments.
import matplotlib.pyplot as plt  # noqa: E402

# Default configuration matching the current TOFU NPO run.

DEFAULT_RUN_DIR = Path("saves/unlearn/tofu_Llama-3.2-1B-Instruct_forget10_NPO")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot token-level log-probability differences across training steps."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help=(
            "Directory containing logprob_diff.jsonl (trainer output_dir). "
            f"Defaults to {DEFAULT_RUN_DIR}."
        ),
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Path to logprob_diff.jsonl. Defaults to <run-dir>/logprob_diff.jsonl.",
    )
    parser.add_argument(
        "--doc-ids",
        type=int,
        nargs="*",
        default=None,
        help="Optional subset of doc_ids to plot. Defaults to all present in the log file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for generated plots. Defaults to <run-dir>/logprob_plots.",
    )
    return parser.parse_args()


def load_records(log_path: Path) -> List[dict]:
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")
    records: List[dict] = []
    with log_path.open("r", encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Unable to parse JSON on line {line_num} of {log_path}") from exc
            required_keys = {"global_step", "doc_id", "positions", "diff_logprobs"}
            if not required_keys.issubset(record):
                missing = required_keys - set(record)
                raise KeyError(f"Missing keys {missing} in record on line {line_num}")
            records.append(record)
    if not records:
        raise ValueError(f"No logprob diff records found in {log_path}")
    return records


def _resolve_tokens(record: dict, count: int) -> Sequence[str]:
    if "tokens" in record and isinstance(record["tokens"], list):
        if len(record["tokens"]) == count:
            return record["tokens"]
        # If tokenizer output was truncated, fall back to string conversion.
        return record["tokens"][:count]
    # Fallback: use placeholder strings when token text is missing.
    return [f"<tok_{idx}>" for idx in range(count)]


def organize_records(
    records: Iterable[dict],
    filter_doc_ids: Iterable[int] = None,
) -> Dict[int, Dict[int, List[Tuple[int, float, str]]]]:
    """
    Returns a nested mapping:
        doc_id -> position -> list of (global_step, diff_value, token_str)
    """
    target_ids = set(filter_doc_ids) if filter_doc_ids is not None else None
    grouped: Dict[int, Dict[int, List[Tuple[int, float, str]]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        doc_id = int(record["doc_id"])
        if target_ids is not None and doc_id not in target_ids:
            continue
        step = int(record["global_step"])
        positions: Sequence[int] = record["positions"]
        diffs: Sequence[float] = record["diff_logprobs"]
        if len(positions) != len(diffs):
            raise ValueError(
                f"Inconsistent lengths in record for doc_id={doc_id}, step={step}: "
                f"positions={len(positions)}, diffs={len(diffs)}"
            )
        token_text = _resolve_tokens(record, len(positions))
        for pos, diff, token_str in zip(positions, diffs, token_text):
            grouped[doc_id][int(pos)].append((step, float(diff), str(token_str)))
    # Sort the (step, diff) sequences chronologically.
    for doc_data in grouped.values():
        for series in doc_data.values():
            series.sort(key=lambda item: item[0])
    return grouped


def plot_doc(
    doc_id: int,
    token_series: Dict[int, List[Tuple[int, float, str]]],
    output_dir: Path,
) -> Path:
    plt.figure(figsize=(12, 6))
    for position, history in sorted(token_series.items()):
        steps = [entry[0] for entry in history]
        diffs = [entry[1] for entry in history]
        labels = [entry[2] for entry in history]
        token_label = labels[-1] if labels else ""
        legend_label = f"{position}:{token_label}"
        plt.plot(steps, diffs, marker="o", label=legend_label)
    plt.title(f"Log-prob gap per token (doc_id={doc_id})")
    plt.xlabel("Training step")
    plt.ylabel("Base - Current logprob")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend(loc="best", fontsize="small", ncol=2)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"logprob_diff_doc{doc_id}.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def main() -> None:
    args = parse_args()
    log_path = args.log_file or (args.run_dir / "logprob_diff.jsonl")
    output_dir = args.output_dir or (args.run_dir / "logprob_plots")

    print(f"Reading logprob diff records from {log_path}")
    records = load_records(log_path)
    doc_ids = args.doc_ids
    groups = organize_records(records, doc_ids)

    if not groups:
        target_info = "specified doc_ids" if doc_ids else "log file"
        raise ValueError(f"No matching records found for {target_info}.")

    print(f"Generating plots in {output_dir}")
    for doc_id, series in groups.items():
        if not series:
            continue
        output_path = plot_doc(doc_id, series, output_dir)
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
