"""Freeze the TOFU Forget05 discrepancy-density Phase A masks and audits."""

from collections import Counter
from html import escape
import json
import math
from pathlib import Path
import shutil

from prepare_low_confidence import (
    ANNOTATIONS, FORMAL_OUTPUT, ROOT, check_environment, controls, distribution, read,
    reference_masks, require, rows, save_json, save_jsonl, sha256, write_csv,
)
from freeze_mask_protocol import DEFAULT_PROTOCOL
import torch
from trainer.masking import density_matched_random_mask

NAME = "tofu_forget05_adaptive_density_phase_a"
OUTPUT = ROOT / "saves/experiments" / NAME
LOW = ROOT / "saves/experiments/tofu_forget05_lowconfidence_matched"
INITIAL = ROOT / "saves/eval/tofu_Llama-3.2-1B-Instruct_full/evals_forget05"
DENSITIES = (15, 34, 50)
RANDOM_SEED = 1729
STEPS = (20, 40, 60, 120)
FORMAL = tuple(f"{method}-{density}" for density in DENSITIES for method in ("Discrepancy", "Random"))
MASK_ONLY = ("BelowMean", "ZeroThreshold")


def count_at_density(n, density):
    require(n >= 0 and 0 < density <= 100, "Invalid eligible count or percentage")
    return max(1, (n * density + 50) // 100) if n else 0


def build_masks(records):
    result = {name: {} for name in (*FORMAL, *MASK_ONLY)}
    for doc, row in records.items():
        eligible = row["eligible"]
        positions = [p for p, valid in enumerate(eligible) if valid]
        delta = row["delta"]
        require(all(math.isfinite(delta[p]) for p in positions), f"Nonfinite eligible discrepancy {doc}")
        ranked = sorted(positions, key=lambda p: (delta[p], p))
        valid = torch.tensor([eligible], dtype=torch.bool)
        for density in DENSITIES:
            k = count_at_density(len(positions), density)
            chosen = set(ranked[:k])
            selected = [int(p in chosen) for p in range(len(eligible))]
            random = density_matched_random_mask(valid, torch.tensor([selected], dtype=torch.bool), [doc], RANDOM_SEED)
            require(sum(selected) == int(random.sum()) == k, f"Unmatched density {doc}/{density}")
            result[f"Discrepancy-{density}"][doc] = selected
            result[f"Random-{density}"][doc] = random[0].int().tolist()
        threshold = float(torch.tensor([delta[p] for p in positions], dtype=torch.float32).mean()) if positions else 0.0
        result["BelowMean"][doc] = [int(valid and delta[p] < threshold) for p, valid in enumerate(eligible)]
        result["ZeroThreshold"][doc] = [int(valid and delta[p] < 0) for p, valid in enumerate(eligible)]
    return result


def audit_masks(root, records, masks, reference, categories):
    summary, per_example, fact_rows, category_rows, overlap_rows, token_rows = {}, [], [], [], [], []
    audit = root / "mask_audit"
    audit.mkdir(exist_ok=True)
    targets = [(doc, p) for doc, row in records.items() for p, value in enumerate(row["eligible"]) if value]
    require(len(targets) == len(categories) == 7165 and set(targets) == set(categories), "Factual annotation alignment differs")
    page = ['<!doctype html><meta charset="utf-8"><title>Density Phase A mask examples</title>',
            '<style>body{max-width:920px;margin:30px auto;font:16px/1.55 system-ui}article{border:1px solid #cfdae0;padding:12px 16px;margin:12px 0}mark{background:#ffe39a}.answer{white-space:pre-wrap}</style>',
            '<h1>TOFU Forget05 · Phase A mask examples</h1><p>Highlighted pieces are selected eligible targets. Hover for discrepancy and token category.</p>']
    for doc, p in targets:
        r = records[doc]
        prior = categories[doc, p]
        token_rows.append({"doc_id": doc, "position": p, "logit_position": p - 1, "token_id": r["input_ids"][p],
                           "token": r["tokens"][p], "label": r["labels"][p], "delta": r["delta"][p],
                           "category": prior["category"], "fact_scope": prior["fact_scope"],
                           **{name: mask[doc][p] for name, mask in masks.items()}})
    for name, mask in masks.items():
        chosen = [(doc, p) for doc, p in targets if mask[doc][p]]
        require(len(chosen) == sum(sum(values) for values in mask.values()), f"Invalid selected target for {name}")
        example_ratios = []
        for doc, row in records.items():
            n = sum(row["eligible"])
            k = sum(mask[doc])
            example_ratios.append((k / n, doc))
            per_example.append({"mask": name, "doc_id": doc, "eligible": n, "selected": k, "ratio": k / n})
        signs = Counter((records[doc]["delta"][p] > 0) - (records[doc]["delta"][p] < 0) for doc, p in chosen)
        summary[name] = {"eligible": len(targets), "selected": len(chosen), "aggregate_ratio": len(chosen) / len(targets),
                         "per_sample": distribution([value for value, _ in example_ratios]),
                         "empty_masks": sum(not sum(values) for values in mask.values()),
                         "discrepancy_sign_fraction": {label: signs[sign] / len(chosen)
                                                       for sign, label in [(-1, "negative"), (0, "zero"), (1, "positive")]}}
        for label, predicate in [("all_answer_bearing", lambda r: bool(r["fact_scope"])),
                                  ("direct", lambda r: r["fact_scope"] == "D"),
                                  ("descriptive", lambda r: r["fact_scope"] == "S")]:
            cohort = [key for key in targets if predicate(categories[key])]
            k = sum(mask[doc][p] for doc, p in cohort)
            fact_rows.append({"mask": name, "scope": label, "eligible": len(cohort), "selected": k, "coverage": k / len(cohort)})
        for label in sorted({r["category"] for r in categories.values()}):
            cohort = [key for key in targets if categories[key]["category"] == label]
            k = sum(mask[doc][p] for doc, p in cohort)
            category_rows.append({"mask": name, "category": label, "eligible": len(cohort), "selected": k,
                                  "coverage": k / len(cohort), "selected_composition": k / len(chosen)})
        for label, ref in reference.items():
            counts = [(sum(a and b for a, b in zip(mask[doc], ref[doc], strict=True)),
                       sum(a or b for a, b in zip(mask[doc], ref[doc], strict=True))) for doc in records]
            intersection, union = map(sum, zip(*counts))
            reference_count = sum(sum(row) for row in ref.values())
            overlap_rows.append({"mask": name, "reference": label, "intersection": intersection,
                "precision": intersection / len(chosen), "recall": intersection / reference_count,
                "micro_jaccard": intersection / union,
                "macro_jaccard": sum(a / b for a, b in counts if b) / sum(b > 0 for _, b in counts)})
        page.append(f"<h2>{name}</h2>")
        ordered = sorted(example_ratios)
        for label, index in [("Low", 0), ("Median", len(ordered)//2), ("High", len(ordered)-1)]:
            ratio, doc = ordered[index]
            r = records[doc]
            snippets = []
            for p, valid in enumerate(r["eligible"]):
                if valid:
                    text = escape(r["tokens"][p].replace("<|eot_id|>", " ⟨EOT⟩"))
                    title = escape(f"position {p}, delta {r['delta'][p]:+.4f}, {categories[doc,p]['category']}")
                    tag = "mark" if mask[doc][p] else "span"
                    snippets.append(f'<{tag} title="{title}">{text}</{tag}>')
            page.append(f'<article><strong>{label} · sample {doc} · {ratio:.2%}</strong>'
                        f'<p>{escape(r["question"])}</p><p class="answer">{"".join(snippets)}</p></article>')
    for name, data in [("per_example", per_example), ("factual_coverage", fact_rows), ("token_categories", category_rows),
                       ("overlap", overlap_rows), ("eligible_tokens", token_rows)]:
        write_csv(audit / f"{name}.csv", data)
    (audit / "decoded_examples.html").write_text("\n".join(page) + "\n")
    return summary


def prepare(root, commit):
    lock = read(FORMAL_OUTPUT / "protocol_manifest.json")
    protocol = read(DEFAULT_PROTOCOL)
    check_environment(protocol)
    require(sha256(DEFAULT_PROTOCOL) == lock["protocol_sha256"] and lock["code_commit"] == commit, "v4 protocol or commit differs")
    old, old_hashes, signature = controls(protocol, lock)
    refs = reference_masks(protocol, lock)
    inputs = FORMAL_OUTPUT / "frozen_inputs"
    source = inputs / "tokens_TOFU_forget05.jsonl"
    require(sha256(source) == protocol["inputs"]["saves/audits/revision/tokens_TOFU_forget05.jsonl"], "Frozen discrepancy hash differs")
    low = read(LOW / "protocol_manifest.json")
    low_mask = LOW / "mask_lowconfidence_matched.jsonl"
    require(sha256(low_mask) == low["mask_sha256"] and low["v4_protocol_sha256"] == lock["protocol_sha256"]
            and low["evaluator_sha256"] == sha256(ROOT / "scripts/evaluate_tofu_checkpoint.py"), "LowConfidence provenance differs")
    low_manifest = read(LOW / "incremental_manifest.json")
    low_rows = [r for r in rows(LOW / "official_metrics_long.csv") if r["variant"] == "lowconfidence_matched" and r["seed"] == "0"]
    require({int(r["step"]) for r in low_rows} == {60, 120}, "LowConfidence seed 0 control missing")
    for r in low_rows:
        key = f"seed0/checkpoint-{r['step']}"
        val = low_manifest[key]["validated"]
        require(val["mask_sha256"] == low["mask_sha256"] and val["code_commit"] == commit
                and all(float(r[k if k != "privleak" else "privleak_signed"]) == v for k, v in val["metrics"].items()),
                "LowConfidence metric/provenance differs")
        old_hashes[key + "/LowConfidence"] = {name: digest for name, digest in low_manifest[key]["files"].items()}
        require(all(sha256(LOW / "collected_results" / key / name) == digest for name, digest in low_manifest[key]["files"].items()),
                "LowConfidence collected hashes differ")
    low_rows = [{**row, "run": "LowConfidence-Matched_seed0", "objective": "NPO", "normalization": "token_mean"}
                for row in low_rows]
    control_rows = [r for r in old if r["seed"] == "0"] + low_rows
    reference_low = {r["doc_id"]: r for r in map(json.loads, low_mask.read_text().splitlines())}
    references = {label: {i: row["mask"] for i, row in refs[key].items()}
                  for label, key in [("Adaptive", "adaptive"), ("Entity", "entity")]}
    references["LowConfidence-Matched"] = {i: r["mask"] for i, r in reference_low.items()}
    portable = read(ROOT / "saves/exports/tofu_forget05_adaptive_mask_relevance/MANIFEST.json")
    require(sha256(ANNOTATIONS / "tokens.csv") == portable["files"]["data/token_audit/tokens.csv"]
            and sha256(ANNOTATIONS / "answer_spans.tsv") == portable["files"]["data/token_audit/answer_spans.tsv"], "Factual annotation hash differs")
    categories = {(int(r["doc_id"]), int(r["position"])): r for r in rows(ANNOTATIONS / "tokens.csv")}
    records = {}
    for row in map(json.loads, source.read_text().splitlines()):
        doc = row["doc_id"]
        ids = refs["base"][doc]["input_ids"]
        tokens = row["tokens"]
        eligible = [t["eligible"] for t in tokens]
        require(doc not in records and list(map(int, eligible)) == refs["base"][doc]["mask"] and len(ids) == len(tokens)
                and reference_low[doc]["input_ids"] == ids, f"Frozen IDs or eligibility differ: {doc}")
        require(all(t["position"] == p and (not t["eligible"] or (p > 0 and t["label"] == ids[p]
                and int(categories[doc, p]["token_id"]) == ids[p] and float(categories[doc, p]["delta_base_minus_target"]) == t["delta"]))
                for p, t in enumerate(tokens)), f"Token/label/discrepancy alignment differs: {doc}")
        records[doc] = {"input_ids": ids, "eligible": eligible, "delta": [t["delta"] for t in tokens],
                        "labels": [t["label"] for t in tokens], "tokens": [t["token"] for t in tokens],
                        "question": row["question"], "answer": row["answer"]}
    require(set(records) == set(range(200)), "Missing Forget05 examples")
    masks = build_masks(records)
    require(masks == build_masks(dict(reversed(list(records.items())))), "Mask regeneration changes with sample order")
    for density in DENSITIES:
        for doc, rec in records.items():
            expected = count_at_density(sum(rec["eligible"]), density)
            require(sum(masks[f"Discrepancy-{density}"][doc]) == sum(masks[f"Random-{density}"][doc]) == expected,
                    f"Pair does not exactly match per-example count: {density}/{doc}")
    masks_dir = root / "masks"
    masks_dir.mkdir(exist_ok=True)
    hashes = {}
    for name, selected in masks.items():
        path = masks_dir / f"{name}.jsonl"
        payload = [{"doc_id": doc, "input_ids": records[doc]["input_ids"], "mask": selected[doc]} for doc in sorted(records)]
        save_jsonl(path, payload)
        require([json.loads(line) for line in path.read_text().splitlines()] == payload, f"Mask reload differs: {name}")
        hashes[name] = sha256(path)
        save_json(path.with_suffix(".metadata.json"), {"mask_kind": name, "mask_sha256": hashes[name],
            "source_discrepancy_sha256": sha256(source), "deterministic_regeneration": True,
            "random_mask_seed": RANDOM_SEED if name.startswith("Random") else None,
            "training": name in FORMAL})
    require((masks_dir / "BelowMean.jsonl").read_bytes() == (inputs / "masks_TOFU_forget05_adaptive.jsonl").read_bytes(),
            "BelowMean regeneration differs at the byte or token level. No training permitted.")
    stats = audit_masks(root, records, masks, references, categories)
    for name in stats:
        stats[name].update(mask_sha256=hashes[name], deterministic_regeneration=True)
    save_json(root / "mask_audit/summary.json", stats)
    shutil.copy2(ANNOTATIONS / "answer_spans.tsv", root / "mask_audit/answer_spans.tsv")
    write_csv(root / "existing_controls.csv", control_rows)
    save_json(root / "v4_evaluation_signature.json", signature)
    shutil.copy2(inputs / "TOFU_EVAL.json", root / "retain_only_reference_TOFU_EVAL.json")
    shutil.copy2(INITIAL / "TOFU_SUMMARY.json", root / "initial_target_TOFU_SUMMARY.json")
    save_json(root / "protocol_manifest.json", {
        "name": NAME, "phase": "A seed-0 screening", "code_commit": commit, "v4_protocol_sha256": lock["protocol_sha256"],
        "protocol": protocol, "seed": 0, "formal_variants": FORMAL, "mask_only_variants": MASK_ONLY,
        "checkpoint_steps": STEPS, "actual_epochs": {str(s): round(s*.16, 8) for s in STEPS},
        "random_mask_seed": RANDOM_SEED, "random_method": "v4 mask-rng-v1 seeded per sample; one nested permutation across densities",
        "rounding": "max(1, floor(n*p+0.5)) for nonempty examples", "mask_hashes": hashes,
        "source_discrepancy_sha256": sha256(source), "belowmean_byte_identical": True, "matched_per_example": True,
        "controls_sha256": sha256(root / "existing_controls.csv"), "verified_control_artifacts": old_hashes,
        "evaluator_sha256": sha256(ROOT / "scripts/evaluate_tofu_checkpoint.py"),
        "evaluation_signature_sha256": sha256(root / "v4_evaluation_signature.json"),
        "source_sha256": {name: sha256(ROOT / name) for name in ["scripts/prepare_density_ablation.py",
            "scripts/run_density_ablation.py", "scripts/run_low_confidence.py", "scripts/resume_low_confidence.py",
            "scripts/train_mask_experiment.py", "scripts/evaluate_tofu_checkpoint.py", "tests/test_density_ablation.py",
            "tests/test_low_confidence.py", "src/trainer/utils.py", "src/trainer/masking.py", "src/trainer/unlearn/npo.py"]}})
    print("Mask audit verified, including byte-identical BelowMean and every matched pair", flush=True)
