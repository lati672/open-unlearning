"""Compute, freeze and audit a target-only LowConfidence-Matched mask."""

import argparse
import csv
from html import escape
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

from freeze_mask_protocol import DEFAULT_PROTOCOL, FORMAL_OUTPUT, ROOT, check_environment, save_json, save_jsonl, sha256

sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader

from data import get_datasets
from data.collators import DataCollatorForSupervisedDataset, FrozenMaskCollator
from model import get_model
from trainer.masking import eligible_positions
from trainer.utils import compute_model_token_logprobs


NAME = "tofu_forget05_lowconfidence_matched"
OUTPUT = ROOT / "saves/experiments" / NAME
ANNOTATIONS = ROOT / "saves/audits/adaptive_relevance"
MASK = "mask_lowconfidence_matched.jsonl"
CONTROLS = ("adaptive", "random_adaptive", "base")
QUANTILES = (0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100)
CATEGORIES = ("factual_attribute", "entity_name", "other_content", "function", "punctuation")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def write_csv(path, data):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)


def rank_low_confidence(scores, eligible, count):
    require(len(scores) == len(eligible), "Score/eligibility lengths differ")
    positions = [index for index, valid in enumerate(eligible) if valid]
    require(0 <= count <= len(positions), "Matched count exceeds eligible positions")
    require(all(scores[pos] is not None and math.isfinite(scores[pos]) for pos in positions),
            "Nonfinite target log-probability at an eligible position")
    positions.sort(key=lambda pos: (scores[pos], pos))
    chosen = set(positions[:count])
    return [int(pos in chosen) for pos in range(len(scores))]


def controls(protocol, locked):
    root = FORMAL_OUTPUT
    source = root / "multiseed_export/multiseed_official_long.csv"
    table = [r for r in rows(source) if r["objective"] == "NPO" and r["normalization"] == "token_mean"
             and r["variant"] in CONTROLS]
    require(len(table) == 18 and len({(r["variant"], r["seed"], r["step"]) for r in table}) == 18, "Incomplete formal controls")
    independent = read(root / "collected_results/incremental_manifest.json")
    results = {}
    canonical = None
    for row in table:
        variant, seed, step = row["variant"], int(row["seed"]), int(row["step"])
        name = f"NPO_{variant}_token_mean_forget05_seed{seed}_steps120"
        require(row["run"] == name, "Control run identity differs")
        run = root / "collected_results" / name
        record = independent["runs"][name]
        prefix = f"checkpoint-{step}/evals/official"
        artifacts = ["TOFU_EVAL.json", "TOFU_SUMMARY.json", "official_metrics.json", "evaluation_provenance.json", "validated.json"]
        for relative in ["resolved_config.json", "training_args.json", f"checkpoint-{step}/trainer_state.json",
                         *[f"{prefix}/{artifact}" for artifact in artifacts]]:
            require(sha256(run / relative) == record["files"][relative]["sha256"], f"Control artifact hash differs: {name}/{relative}")
        config = read(run / "resolved_config.json")
        checkpoint = read(run / prefix / "validated.json")
        require(config["frozen_protocol"]["commit"] == locked["code_commit"]
                and config["frozen_protocol"]["sha256"] == locked["protocol_sha256"]
                and config["execution"]["world_size"] == protocol["training"]["world_size"]
                and config["execution"]["seed"] == seed and config["execution"]["normalization"] == "token_mean"
                and checkpoint["code_commit"] == locked["code_commit"] and checkpoint["protocol_sha256"] == locked["protocol_sha256"]
                and checkpoint["mask_sha256"] == config["mask_sha256"] and checkpoint["global_step"] == step
                and checkpoint["actual_epoch"] == protocol["training"]["checkpoints"][str(step)], f"Control provenance differs: {name}")
        mask_name = f"masks_TOFU_forget05_{variant}.jsonl"
        hashes = {Path(path).name: h for path, h in protocol["inputs"].items()} | locked["derived_inputs"]
        require(config["mask_sha256"] == hashes[mask_name], f"Control mask differs from frozen protocol: {name}")
        detail = read(run / prefix / "evaluation_provenance.json")["context"]
        evaluation = {key: val for key, val in detail["resolved_eval_config"].items() if key not in {"output_dir", "retain_logs_path"}}
        metrics = read(run / prefix / "official_metrics.json")
        require(metrics == checkpoint["metrics"] and all(
            math.isfinite(value) and float(row[key if key != "privleak" else "privleak_signed"]) == value for key, value in metrics.items()
        ) and float(row["privleak_abs"]) == abs(metrics["privleak"]), f"Control metric table differs: {name}")
        args = read(run / "training_args.json")
        compare_args = {k: v for k, v in args.items() if k not in {"seed", "output_dir", "logging_dir", "run_name"}}
        method = {k: v for k, v in config["trainer"]["method_args"].items() if k != "mask"}
        signature = (compare_args, method, config["model"], config["data"], evaluation, detail["retain_only_logs_sha256"])
        require(canonical is None or canonical == signature, f"Control optimization/evaluation differs: {name}")
        canonical = signature
        results[f"{name}/checkpoint-{step}"] = {"mask_sha256": checkpoint["mask_sha256"],
                                                 "official_metrics_sha256": sha256(run / prefix / "official_metrics.json"),
                                                 "resolved_config_sha256": sha256(run / "resolved_config.json")}
    return table, results, canonical[4]


def reference_masks(protocol, locked):
    hashes = {Path(path).name: digest for path, digest in protocol["inputs"].items()} | locked["derived_inputs"]
    result = {}
    for variant in (*CONTROLS, "entity"):
        filename = f"masks_TOFU_forget05_{variant}.jsonl"
        path = FORMAL_OUTPUT / "frozen_inputs" / filename
        require(sha256(path) == hashes[filename], f"Frozen mask SHA-256 differs: {variant}")
        data = [json.loads(line) for line in path.read_text().splitlines()]
        result[variant] = {row["doc_id"]: row for row in data}
        require(len(data) == 200 and set(result[variant]) == set(range(200)), f"Invalid stable IDs: {variant}")
    return result


@torch.inference_mode()
def score_samples(model, loader):
    model.eval()
    records = {}
    for batch in loader:
        require("index" in batch, "Confidence inference requires stable sample IDs")
        model_inputs = {key: batch[key].to("cuda:0") for key in ("input_ids", "attention_mask")}
        logits = model(**model_inputs, use_cache=False).logits
        scores = compute_model_token_logprobs(logits.float(), model_inputs["input_ids"]).cpu()
        eligibility = eligible_positions(batch)
        for row, doc in enumerate(batch["index"].tolist()):
            valid = batch["attention_mask"][row].bool()
            eligible = eligibility[row, valid].tolist()
            ids = batch["input_ids"][row, valid].tolist()
            labels = batch["labels"][row, valid].tolist()
            vals = [float(v) if use else None for v, use in zip(scores[row, valid], eligible, strict=True)]
            require(doc not in records and all(labels[pos] == ids[pos] and math.isfinite(vals[pos])
                    for pos, use in enumerate(eligible) if use), f"Invalid target confidence record: {doc}")
            records[doc] = {"doc_id": doc, "input_ids": ids, "labels": labels, "eligible": eligible,
                            "target_logprobs": vals}
    require(set(records) == set(range(200)), "Confidence inference did not cover the frozen 200 samples")
    return records


def make_mask(scores, refs):
    output = []
    for doc, sample in sorted(scores.items()):
        eligible = sample["eligible"]
        references = [masks[doc] for masks in refs.values()]
        require(all(row["input_ids"] == sample["input_ids"] for row in references)
                and list(map(int, eligible)) == refs["base"][doc]["mask"], f"Token/eligibility drift: {doc}")
        count = sum(refs["adaptive"][doc]["mask"])
        require(count == sum(refs["random_adaptive"][doc]["mask"]), f"Existing matched density differs: {doc}")
        mask = rank_low_confidence(sample["target_logprobs"], eligible, count)
        require(sum(mask) == count and all(not flag or eligible[pos] for pos, flag in enumerate(mask)), f"Invalid low-confidence mask: {doc}")
        output.append({"doc_id": doc, "input_ids": sample["input_ids"], "mask": mask})
    return output


def distribution(values):
    return {"n": len(values), "mean": float(np.mean(values)), "sample_std": float(np.std(values, ddof=1)),
            "median": float(np.median(values)), "min": min(values), "max": max(values),
            **{f"p{q:02}": float(value) for q, value in zip(QUANTILES, np.percentile(values, QUANTILES))}}


def audit(root, scores, masks, refs, tokenizer, dataset):
    category_rows = rows(ANNOTATIONS / "tokens.csv")
    category = {(int(r["doc_id"]), int(r["position"])): r for r in category_rows}
    require(len(category_rows) == len(category) == 7165, "Existing factual/category alignment is incomplete")
    masks = {r["doc_id"]: r["mask"] for r in masks}
    tokens, samples = [], []
    for doc, row in sorted(scores.items()):
        mask = masks[doc]
        eligible = row["eligible"]
        sample = {"doc_id": doc, "eligible_count": sum(eligible), "selected_count": sum(mask),
                  "adaptive_count": sum(refs["adaptive"][doc]["mask"]), "selection_ratio": sum(mask) / sum(eligible)}
        samples.append(sample)
        for pos, valid in enumerate(eligible):
            if not valid:
                continue
            prior = category[doc, pos]
            token_id = row["input_ids"][pos]
            require(int(prior["token_id"]) == token_id and (prior["adaptive"] == "True") == bool(refs["adaptive"][doc]["mask"][pos])
                    and (prior["entity"] == "True") == bool(refs["entity"][doc]["mask"][pos]), "Factual category/mask alignment differs")
            tokens.append({"doc_id": doc, "position": pos, "logit_position": pos - 1, "token_id": token_id,
                           "token": tokenizer.decode([token_id]), "label": row["labels"][pos], "target_logprob": row["target_logprobs"][pos],
                           "lowconfidence_matched": mask[pos], "category": prior["category"], "fact_scope": prior["fact_scope"],
                           **{name: refs[name][doc]["mask"][pos] for name in ("adaptive", "random_adaptive", "entity")}})
    selected = sum(r["lowconfidence_matched"] for r in tokens)
    require(len(tokens) == 7165 and selected == 2417 and all(r["selected_count"] == r["adaptive_count"] > 0 for r in samples),
            "Expected exact 2,417/7,165 density and no empty masks across 200 samples")
    overlap = {}
    for ref in ("adaptive", "random_adaptive", "entity"):
        both = sum(r["lowconfidence_matched"] and r[ref] for r in tokens)
        ref_count = sum(r[ref] for r in tokens)
        per_sample = []
        for doc in range(200):
            pairs = list(zip(masks[doc], refs[ref][doc]["mask"], strict=True))
            union = sum(a or b for a, b in pairs)
            if union:
                per_sample.append(sum(a and b for a, b in pairs) / union)
        overlap[ref] = {"intersection": both, "reference_selected": ref_count, "precision": both / selected,
                        "recall": both / ref_count, "micro_jaccard": both / (selected + ref_count - both),
                        "macro_jaccard": float(np.mean(per_sample))}
    categories, factual = [], []
    for name, predicate in [(cat, lambda r, c=cat: r["category"] == c) for cat in CATEGORIES] + [
        ("all_answer_bearing", lambda r: bool(r["fact_scope"])),
        ("direct", lambda r: r["fact_scope"] == "D"),
        ("descriptive", lambda r: r["fact_scope"] == "S"),
    ]:
        group = [r for r in tokens if predicate(r)]
        for method in ("lowconfidence_matched", "adaptive", "random_adaptive"):
            count = sum(r[method] for r in group)
            (categories if name in CATEGORIES else factual).append({
                "category": name, "mask": method, "eligible": len(group), "selected": count,
                "coverage": count / len(group), "selected_composition": count / selected,
            })
    output = root / "mask_audit"
    output.mkdir(exist_ok=True)
    write_csv(output / "per_example.csv", samples)
    write_csv(output / "eligible_tokens.csv", tokens)
    write_csv(output / "token_categories.csv", categories)
    write_csv(output / "factual_coverage.csv", factual)
    distribution_rows = [{"group": name, **distribution([r["target_logprob"] for r in tokens if r["lowconfidence_matched"] == flag])}
                         for name, flag in [("selected", 1), ("unselected", 0)]]
    write_csv(output / "target_logprob_distribution.csv", distribution_rows)
    page = ['<!doctype html><meta charset="utf-8"><title>LowConfidence-Matched examples</title>',
            '<style>body{max-width:940px;margin:35px auto;font:16px/1.6 system-ui}article{padding:12px 18px;border:1px solid #cedce4;border-radius:9px;margin:14px 0}mark{background:#ffe29c}span.adaptive{border-bottom:2px solid #44668f}.answer{white-space:pre-wrap}</style>',
            '<h1>LowConfidence-Matched · TOFU Forget05</h1><p>Yellow indicates LowConfidence selection. Blue underline indicates Adaptive selection. Hover for target log probability and category.</p>']
    chosen = sorted(range(200), key=lambda doc: (samples[doc]["selection_ratio"], doc))
    for doc in [chosen[index] for index in (0, 25, 50, 75, 100, 125, 150, 199)]:
        original = dataset.data[doc]
        snippets = []
        for token in (r for r in tokens if r["doc_id"] == doc):
            piece = escape(token["token"].replace("<|eot_id|>", " ⟨EOT⟩"))
            if token["lowconfidence_matched"]:
                piece = f"<mark>{piece}</mark>"
            css = "adaptive" if token["adaptive"] else ""
            tip = escape(f"position {token['position']} · target log p {token['target_logprob']:.4f} · {token['category']}")
            snippets.append(f'<span class="{css}" title="{tip}">{piece}</span>')
        page.append(f'<article><strong>Sample {doc} · {samples[doc]["selected_count"]}/{samples[doc]["eligible_count"]}</strong>'
                    f'<p>{escape(original["question"])}</p><p class="answer">{"".join(snippets)}</p></article>')
    (output / "decoded_examples.html").write_text("\n".join(page) + "\n")
    save_json(output / "summary.json", {"samples": 200, "eligible_tokens": len(tokens), "selected_tokens": selected,
        "selection_ratio": selected / len(tokens), "empty_masks": 0, "per_example_exact_match": True,
        "per_example_selection_ratio": distribution([r["selection_ratio"] for r in samples]),
        "target_logprob_distribution": distribution_rows, "overlap": overlap, "factual_coverage": factual})


def prepare(root, commit):
    root.mkdir(exist_ok=True, parents=True)
    require(not (root / "protocol_manifest.json").exists(), "Mask preparation is already frozen")
    protocol = read(DEFAULT_PROTOCOL)
    check_environment(protocol)
    locked = read(FORMAL_OUTPUT / "protocol_manifest.json")
    require(locked["code_commit"] == commit and locked["protocol_sha256"] == sha256(DEFAULT_PROTOCOL), "v4 code/protocol hash differs")
    table, control_hashes, eval_signature = controls(protocol, locked)
    refs = reference_masks(protocol, locked)
    annotation = read(ANNOTATIONS / "summary.json")
    portable = read(ROOT / "saves/exports/tofu_forget05_adaptive_mask_relevance/MANIFEST.json")
    require(annotation["protocol_sha256"] == locked["protocol_sha256"]
            and sha256(ANNOTATIONS / "answer_spans.tsv") == annotation["annotations_sha256"]
            and sha256(ANNOTATIONS / "tokens.csv") == portable["files"]["data/token_audit/tokens.csv"], "Frozen factual annotations differ")
    source_cfg = read(FORMAL_OUTPUT / "collected_results/NPO_adaptive_token_mean_forget05_seed0_steps120/resolved_config.json")
    model_cfg = OmegaConf.create(source_cfg["model"])
    model_cfg.model_args.local_files_only = True
    model_cfg.tokenizer_args.local_files_only = True
    model_cfg.model_args.device_map = {"": "cuda:0"}
    model, tokenizer = get_model(model_cfg)
    require(model.config._commit_hash == protocol["target_model"]["revision"], "Loaded target revision differs")
    template = source_cfg["model"]["template_args"]
    forget_cfg = OmegaConf.create(source_cfg["data"]["forget"])
    forget_cfg.TOFU_QA_forget.args.verify_revision = True
    dataset = get_datasets(forget_cfg, tokenizer=tokenizer, template_args=template)
    collator = DataCollatorForSupervisedDataset(tokenizer, index="index")
    loader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=collator)
    confidence = score_samples(model, loader)
    masks = make_mask(confidence, refs)
    regenerated = make_mask(score_samples(model, loader), refs)
    require(masks == regenerated, "Independent target forward pass changed the mask")
    mask_path = root / MASK
    save_jsonl(mask_path, masks)
    alternate = root / ".regenerated.jsonl"
    save_jsonl(alternate, regenerated)
    mask_hash = sha256(mask_path)
    require(sha256(alternate) == mask_hash, "Deterministic regeneration changed the mask SHA-256")
    alternate.unlink()
    frozen_collator = FrozenMaskCollator(collator, mask_path)
    for start in range(0, len(dataset), 7):
        sample_ids = list(reversed(range(start, min(start + 7, len(dataset)))))
        batch = frozen_collator([dataset[idx] for idx in sample_ids])
        require(batch["index"].tolist() == sample_ids and not (batch["reference_mask"] & ~eligible_positions(batch)).any(),
                "Padded replay selected an ineligible target or changed stable IDs")
        for row, doc in enumerate(sample_ids):
            require(batch["reference_mask"][row, batch["attention_mask"][row].bool()].int().tolist() == masks[doc]["mask"],
                    "Frozen mask reload changed an example")
    require(sha256(mask_path) == mask_hash, "Frozen mask hash changed after reload")
    audit(root, confidence, masks, refs, tokenizer, dataset)
    save_jsonl(root / "target_confidence.jsonl", [r for _, r in sorted(confidence.items())])
    save_json(mask_path.with_suffix(".metadata.json"), {"mask_kind": "lowconfidence_matched",
        "target_model": protocol["target_model"], "tokenizer": source_cfg["model"]["tokenizer_args"],
        "dataset_revision": protocol["dataset"]["revision"], "mask_sha256": mask_hash,
        "scoring": "direct target-only shifted log-softmax in float32 from frozen BF16 logits",
        "tie_break": "ascending token position", "density_reference": "frozen Adaptive counts", "independent_forward_regeneration": True})
    reference_key = protocol["retain_only_model"]["evaluation_logs"]
    reference = root / "retain_only_reference_TOFU_EVAL.json"
    shutil.copy2(FORMAL_OUTPUT / "frozen_inputs" / Path(reference_key).name, reference)
    require(sha256(reference) == protocol["inputs"][reference_key], "Pinned retain-only reference differs")
    write_csv(root / "existing_controls.csv", table)
    save_json(root / "v4_evaluation_signature.json", eval_signature)
    files = [Path(__file__), ROOT / "scripts/run_low_confidence.py", ROOT / "tests/test_low_confidence.py",
             ROOT / "scripts/train_mask_experiment.py", ROOT / "scripts/evaluate_tofu_checkpoint.py",
             ROOT / "src/trainer/unlearn/npo.py", ROOT / "src/trainer/utils.py", ROOT / "src/trainer/masking.py",
             ROOT / "src/data/collators.py"]
    save_json(root / "protocol_manifest.json", {"name": "LowConfidence-Matched", "code_commit": commit,
        "v4_protocol_sha256": locked["protocol_sha256"], "protocol": protocol, "mask_sha256": mask_hash,
        "mask_metadata_sha256": sha256(mask_path.with_suffix(".metadata.json")),
        "target_confidence_sha256": sha256(root / "target_confidence.jsonl"),
        "deterministic_regeneration": True, "controls_sha256": sha256(root / "existing_controls.csv"),
        "v4_evaluation_signature_sha256": sha256(root / "v4_evaluation_signature.json"),
        "control_artifacts": control_hashes, "evaluator_sha256": sha256(ROOT / "scripts/evaluate_tofu_checkpoint.py"),
        "annotations_sha256": annotation["annotations_sha256"], "annotation_tokens_sha256": sha256(ANNOTATIONS / "tokens.csv"),
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in files}})
    shutil.copy2(ANNOTATIONS / "answer_spans.tsv", root / "mask_audit/answer_spans.tsv")
    print(f"Frozen mask 2,417/7,165; all 200 counts match; SHA-256 {mask_hash}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--code-commit", required=True)
    args = parser.parse_args()
    prepare(args.output.resolve(), args.code_commit)


if __name__ == "__main__":
    main()
