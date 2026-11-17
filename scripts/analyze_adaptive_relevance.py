"""Audit frozen TOFU Forget05 token relevance and paired official forgetting changes."""

import argparse
import csv
from hashlib import file_digest
from html import escape
import json
from pathlib import Path
import unicodedata

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
FORMAL = ROOT / "saves/experiments/tofu_forget05_multiseed_v4_official"
OUTPUT = ROOT / "saves/audits/adaptive_relevance"
TARGET_EVAL = ROOT / "saves/eval/tofu_Llama-3.2-1B-Instruct_full/evals_forget05/TOFU_EVAL.json"
AUTHORS = [
    "Hina Ameen", "Xin Lee Williams", "Moshe Ben-David", "Kalkidan Abera", "Takashi Nakamura",
    "Raven Marais", "Aysha Al-Hashim", "Edward Patrick Sullivan", "Basil Mahfouz Al-Kuwaiti", "Nikolai Abilov",
]
OBJECTIVES = [("Summed NPO", "NPO", "summed"), ("Token-normalized NPO", "NPO", "token_mean"), ("RMU", "RMU", "summed")]
CATEGORIES = ["entity_name", "factual_attribute", "other_content", "function", "punctuation"]
METRICS = {
    "nll_increase": ("forget_Q_A_Prob", "avg_loss", 1),
    "probability_drop": ("forget_Q_A_Prob", "prob", -1),
    "rougeL_recall_drop": ("forget_Q_A_ROUGE", "rougeL_recall", -1),
    "extraction_strength_drop": ("extraction_strength", "score", -1),
}
QUANTILES = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
EXAMPLES = 8


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def rows_by_id(path):
    with path.open() as stream:
        rows = [json.loads(line) for line in stream]
    indexed = {row["doc_id"]: row for row in rows}
    if len(rows) != 200 or set(indexed) != set(range(200)):
        raise ValueError(f"Expected 200 unique sample IDs in {path}")
    return indexed


def annotations(path):
    result = {}
    for line in path.read_text().splitlines():
        sample, scope, *spans = line.split("\t")
        sample = int(sample)
        if sample in result or scope not in {"D", "S", "-"} or bool(spans) != (scope != "-"):
            raise ValueError(f"Malformed or duplicate annotation on sample {sample}")
        result[sample] = scope, spans
    if set(result) != set(range(200)):
        raise ValueError("Annotations must cover every sample")
    return result


def token_data(formal, annotation_file):
    locked = read(formal / "protocol_manifest.json")
    protocol = locked["protocol"]
    frozen = formal / "frozen_inputs"
    hashes = {Path(path).name: digest for path, digest in protocol["inputs"].items()} | locked["derived_inputs"]
    files = {variant: f"masks_TOFU_forget05_{variant}.jsonl" for variant in ["base", "adaptive", "entity", "random_adaptive"]}
    for name in [*files.values(), "tokens_TOFU_forget05.jsonl", "logprobs.json"]:
        if sha(frozen / name) != hashes[name]:
            raise ValueError(f"Frozen input hash differs for {name}")
    checkpoint = formal / "collected_results/NPO_adaptive_summed_forget05_seed0_steps120/checkpoint-120"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    if not tokenizer.is_fast:
        raise ValueError("Fast tokenizer required for exact answer offsets")
    masks = {variant: rows_by_id(frozen / name) for variant, name in files.items()}
    audit = rows_by_id(frozen / "tokens_TOFU_forget05.jsonl")
    cache = rows_by_id(frozen / "logprobs.json")
    annotated = annotations(annotation_file)
    result, samples = [], []
    for doc, row in sorted(audit.items()):
        answer = row["answer"].strip()
        author = AUTHORS[doc // 20]
        if author.casefold() not in (row["question"] + "\n" + answer).casefold():
            raise ValueError(f"Author block check failed on sample {doc}")
        if (row["question"], row["answer"]) != (cache[doc]["question"], cache[doc]["answer"]):
            raise ValueError(f"Base log probability sample mismatch for {doc}")
        scope, spans = annotated[doc]
        intervals = []
        for span in spans:
            if answer.count(span) != 1:
                raise ValueError(f"Expected unique answer span in {doc}: {span!r}")
            start = answer.index(span)
            intervals.append((start, start + len(span)))
        encoded = tokenizer(answer, add_special_tokens=False, return_offsets_mapping=True)
        input_ids = masks["base"][doc]["input_ids"]
        if any(data[doc]["input_ids"] != input_ids for data in masks.values()):
            raise ValueError(f"Frozen mask input IDs differ for {doc}")
        eligible = [t for t in row["tokens"] if t["eligible"]]
        positions = [t["position"] for t in eligible]
        ids = [input_ids[pos] for pos in positions]
        if ids[:-1] != encoded["input_ids"] or ids[-1] not in tokenizer.all_special_ids:
            raise ValueError(f"Answer-token alignment failed for {doc}")
        if positions != [i for i, flag in enumerate(masks["base"][doc]["mask"]) if flag]:
            raise ValueError(f"Eligible positions differ for {doc}")
        selected = [i for i, flag in enumerate(masks["adaptive"][doc]["mask"]) if flag]
        entity = [i for i, flag in enumerate(masks["entity"][doc]["mask"]) if flag]
        random = [i for i, flag in enumerate(masks["random_adaptive"][doc]["mask"]) if flag]
        if selected != [t["position"] for t in eligible if t["selected"]] or entity != [t["position"] for t in eligible if t["entity"]]:
            raise ValueError(f"Audited masks differ for {doc}")
        if len(random) != len(selected) or not set(selected + entity + random) <= set(positions):
            raise ValueError(f"Mask density or eligibility differs for {doc}")
        offsets = encoded["offset_mapping"] + [(len(answer), len(answer))]
        baseline_tokens = []
        for audit_token, (start, end), token_id in zip(eligible, offsets, ids, strict=True):
            pos = audit_token["position"]
            delta = float(audit_token["delta"])
            baseline_tokens.append(cache[doc]["token_logprobs"][pos] - delta)
            if not np.isfinite([delta, baseline_tokens[-1]]).all():
                raise ValueError(f"Nonfinite eligible discrepancy on sample {doc}, token {pos}")
            is_special = token_id in tokenizer.all_special_ids
            piece = answer[start:end]
            has_span = any(max(start, left) < min(end, right) for left, right in intervals)
            category = "other_content"
            if pos in entity:
                category = "entity_name"
            elif is_special:
                category = "function"
            elif piece.strip() and all(c.isspace() or unicodedata.category(c)[0] in {"P", "S"} for c in piece):
                category = "punctuation"
            elif has_span:
                category = "factual_attribute"
            elif audit_token["category"] == "function":
                category = "function"
            result.append({
                "doc_id": doc, "author": author, "position": pos, "token_id": token_id, "token": audit_token["token"],
                "char_start": start, "char_end": end, "special": is_special, "delta_base_minus_target": delta,
                "sequence_mean": row["sequence_mean"], "adaptive": pos in selected, "entity": pos in entity,
                "random_adaptive": pos in random, "category": category, "fact_scope": scope if category == "factual_attribute" else "",
            })
        samples.append({"doc_id": doc, "author": author, "question": row["question"], "answer": answer, "spans": spans, "scope": scope,
                        "target_token_ids": ids, "eligible": len(ids), "adaptive_selected": len(selected),
                        "selection_ratio": len(selected) / len(ids), "target_cached_nll": -float(np.mean(baseline_tokens))})
    return result, samples


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def distribution(tokens):
    output = []
    for group, selection in [("selected", True), ("unselected", False)]:
        values = np.array([row["delta_base_minus_target"] for row in tokens if row["adaptive"] == selection])
        output.append({"group": group, "n": len(values), "mean": float(values.mean()), "std_sample": float(values.std(ddof=1)),
                       "fraction_negative": float((values < 0).mean()), "fraction_zero": float((values == 0).mean()),
                       "fraction_positive": float((values > 0).mean()),
                       **{f"p{q:02}": float(value) for q, value in zip(QUANTILES, np.percentile(values, QUANTILES))}})
    return output


def by_category(tokens):
    output = []
    for category in CATEGORIES + ["factual_attribute_direct_only", "factual_attribute_descriptive_only"]:
        if category == "factual_attribute_direct_only":
            rows = [r for r in tokens if r["fact_scope"] == "D"]
        elif category == "factual_attribute_descriptive_only":
            rows = [r for r in tokens if r["fact_scope"] == "S"]
        else:
            rows = [r for r in tokens if r["category"] == category]
        for mask in ["adaptive", "random_adaptive"]:
            selected = sum(r[mask] for r in rows)
            output.append({"category": category, "mask": mask, "eligible": len(rows), "selected": selected,
                           "eligible_composition": len(rows) / len(tokens), "coverage": selected / len(rows),
                           "selected_composition": selected / sum(r[mask] for r in tokens)})
    return output


def overlap(tokens):
    adaptive = np.array([r["adaptive"] for r in tokens])
    entity = np.array([r["entity"] for r in tokens])
    intersection, union = (adaptive & entity).sum(), (adaptive | entity).sum()
    samples = []
    for doc in range(200):
        rows = [r for r in tokens if r["doc_id"] == doc]
        selected = sum(r["adaptive"] for r in rows)
        named = sum(r["entity"] for r in rows)
        both = sum(r["adaptive"] and r["entity"] for r in rows)
        samples.append({"doc_id": doc, "author": rows[0]["author"], "adaptive": selected, "entity": named, "intersection": both,
                        "precision": both / selected if selected else None, "recall": both / named if named else None,
                        "jaccard": both / (selected + named - both) if selected + named - both else None})
    return {"adaptive_tokens": int(adaptive.sum()), "entity_tokens": int(entity.sum()), "intersection": int(intersection), "union": int(union),
            "micro_precision": float(intersection / adaptive.sum()), "micro_recall": float(intersection / entity.sum()),
            "micro_jaccard": float(intersection / union),
            "macro_jaccard": float(np.mean([r["jaccard"] for r in samples if r["jaccard"] is not None]))}, samples


def coverage_bootstrap(tokens, samples, cluster_draws):
    output = []
    for scope in ["all_answer_bearing", "direct_only", "descriptive_only"]:
        counts = np.zeros((10, 3), dtype=int)
        expected_random = 0.0
        for row in tokens:
            if not row["fact_scope"] or (scope == "direct_only" and row["fact_scope"] != "D") or (scope == "descriptive_only" and row["fact_scope"] != "S"):
                continue
            doc = row["doc_id"]
            counts[doc // 20] += [1, row["adaptive"], row["random_adaptive"]]
            expected_random += samples[doc]["selection_ratio"]
        n, selected, random = counts.sum(axis=0)
        draws = counts[cluster_draws].sum(axis=1)
        interval = np.quantile((draws[:, 1] - draws[:, 2]) / draws[:, 0], [0.025, 0.975])
        output.append({"scope": scope, "n": int(n), "adaptive": int(selected), "random_adaptive": int(random),
                       "adaptive_coverage": float(selected / n), "random_coverage": float(random / n),
                       "random_expected_coverage_given_counts": float(expected_random / n), "paired_gap": float((selected - random) / n),
                       "author_bootstrap_low": float(interval[0]), "author_bootstrap_high": float(interval[1])})
    return output


def eval_rows(path, samples, baseline=None):
    data = read(path)
    fields = ["forget_Q_A_ROUGE", *{name for name, _, _ in METRICS.values()}]
    result = {name: data[name]["value_by_index"] for name in fields}
    expected = {str(row["doc_id"]) for row in samples}
    if any(set(rows) != expected for rows in result.values()):
        raise ValueError(f"Sample ID mismatch in {path}")
    for row in samples:
        key = str(row["doc_id"])
        rouge = result["forget_Q_A_ROUGE"][key]
        prompt = rouge["input"]
        if rouge["ground_truth"].strip() != row["answer"] or row["question"].strip() not in prompt:
            raise ValueError(f"Question/answer differs for sample {key}: {path}")
        if baseline is not None and (prompt != baseline["forget_Q_A_ROUGE"][key]["input"]
                                     or result["forget_Q_A_Prob"][key]["target_token_ids"] != row["target_token_ids"]):
            raise ValueError(f"Evaluation prompt or target IDs differ for {key}: {path}")
    return result


def association(formal, samples, target_path, cluster_draws):
    target = eval_rows(target_path, samples)
    ids = [str(row["doc_id"]) for row in samples]
    ratio = np.array([row["selection_ratio"] for row in samples])
    nll = np.array([target["forget_Q_A_Prob"][key]["avg_loss"] for key in ids])
    differences = np.abs(nll - [row["target_cached_nll"] for row in samples])
    if differences.max() > 0.25:
        raise ValueError("Official target baseline NLL disagrees with the frozen log-probability audit")
    controls = np.column_stack([np.ones(len(samples)), np.log([r["eligible"] for r in samples]), nll])
    ratio_residual = ratio - controls @ np.linalg.lstsq(controls, ratio, rcond=None)[0]
    boot_ids = (cluster_draws[:, :, None] * 20 + np.arange(20)).reshape(len(cluster_draws), 200)
    correlations, changes = [], []
    for objective, method, norm in OBJECTIVES:
        for step in [60, 120]:
            paired = {}
            for seed in range(3):
                evaluations = {}
                for variant in ["adaptive", "random_adaptive"]:
                    run = f"{method}_{variant}_{norm}_forget05_seed{seed}_steps120"
                    path = formal / "collected_results" / run / f"checkpoint-{step}" / "evals/official/TOFU_EVAL.json"
                    marker = read(path.with_name("validated.json"))
                    if marker["artifact_sha256"]["TOFU_EVAL.json"] != sha(path):
                        raise ValueError(f"Official evaluation artifact differs for {path}")
                    evaluations[variant] = eval_rows(path, samples, target)
                for metric, (name, value, sign) in METRICS.items():
                    initial = np.array([target[name][key][value] for key in ids])
                    adaptive, random = [np.array([evaluations[v][name][key][value] for key in ids]) for v in evaluations]
                    if not np.isfinite([initial, adaptive, random]).all():
                        raise ValueError(f"Nonfinite per-sample values in {objective}, step {step}, seed {seed}, {metric}")
                    drop, control = sign * (adaptive - initial), sign * (random - initial)
                    for index, sample in enumerate(samples):
                        changes.append({"doc_id": sample["doc_id"], "author": sample["author"], "objective": objective,
                                        "step": step, "seed": seed, "metric": metric, "eligible": sample["eligible"],
                                        "selection_ratio": ratio[index], "target": initial[index], "adaptive": adaptive[index],
                                        "random_adaptive": random[index], "adaptive_forgetting": drop[index],
                                        "random_forgetting": control[index], "adaptive_minus_random": drop[index] - control[index]})
                    paired.setdefault((metric, "adaptive_vs_target"), []).append(drop)
                    paired.setdefault((metric, "adaptive_minus_random"), []).append(drop - control)
            for (metric, contrast), results in paired.items():
                for seed, y in list(enumerate(results)) + [("mean_of_3", np.mean(results, axis=0))]:
                    pearson = float(pearsonr(ratio, y).statistic)
                    rho = float(spearmanr(ratio, y).statistic)
                    outcome_residual = y - controls @ np.linalg.lstsq(controls, y, rcond=None)[0]
                    partial = float(pearsonr(ratio_residual, outcome_residual).statistic)
                    low = high = None
                    if seed == "mean_of_3":
                        boot = pearsonr(ratio[boot_ids], y[boot_ids], axis=1).statistic
                        low, high = np.quantile(boot[np.isfinite(boot)], [0.025, 0.975])
                    correlations.append({"objective": objective, "step": step, "seed": seed, "metric": metric, "contrast": contrast,
                                         "n_samples": len(samples), "n_authors": len(AUTHORS), "pearson_r": pearson, "spearman_rho": rho,
                                         "partial_pearson_log_length_initial_nll": partial,
                                         "pearson_author_bootstrap_low": low, "pearson_author_bootstrap_high": high})
    return correlations, changes, float(differences.max())


def save_figure(fig, output, name):
    fig.savefig(output / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(output / f"{name}.png", dpi=210, bbox_inches="tight")
    plt.close(fig)


def plots(output, tokens, categories, correlations, changes):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    teal, violet = "#126d72", "#6664a2"
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.3, 3.8), layout="constrained")
    for selected, color, label in [(True, teal, "Adaptive selected"), (False, violet, "Adaptive unselected")]:
        data = sorted(r["delta_base_minus_target"] for r in tokens if r["adaptive"] == selected)
        left.step(data, np.arange(1, len(data) + 1) / len(data), color=color, label=f"{label} (n={len(data):,})")
    left.axvline(0, color="#b5bac3", lw=1, ls=":")
    left.set(title="A   Discrepancy distribution", xlabel="log p(base) − log p(target), nats", ylabel="Cumulative token fraction", ylim=(0, 1.02))
    left.legend(loc="upper left", frameon=False, fontsize=8)
    labels = ["Author-name match", "Factual attribute", "Other content", "Function / control", "Punctuation", "Direct facts only"]
    shown = CATEGORIES + ["factual_attribute_direct_only"]
    lookup = {(r["category"], r["mask"]): r["coverage"] for r in categories}
    y = np.arange(len(labels))
    for mask, offset, color, label in [("adaptive", -.17, teal, "Adaptive"), ("random_adaptive", .17, violet, "Random-Adaptive")]:
        right.barh(y + offset, [100 * lookup[cat, mask] for cat in shown], height=.28, color=color, label=label)
    right.set(yticks=y, yticklabels=labels, xlabel="Selected share of category (%)", title="B   Coverage by category", xlim=(0, 70))
    right.axhline(4.5, color="#d1d5dc", lw=1)
    right.invert_yaxis()
    right.legend(frameon=False, fontsize=8)
    save_figure(fig, output, "discrepancy_and_coverage")

    fig, axes = plt.subplots(1, 3, figsize=(11.1, 3.25), layout="constrained")
    for ax, (objective, _, _) in zip(axes, OBJECTIVES):
        rows = [r for r in changes if r["objective"] == objective and r["step"] == 120 and r["metric"] == "nll_increase"]
        x = 100 * np.array([r["selection_ratio"] for r in rows[:200]])
        y = np.array([r["adaptive_forgetting"] for r in rows]).reshape(3, 200).mean(axis=0)
        stats = next(r for r in correlations if r["objective"] == objective and r["step"] == 120 and r["seed"] == "mean_of_3"
                     and r["metric"] == "nll_increase" and r["contrast"] == "adaptive_vs_target")
        ax.scatter(x, y, s=12, color=teal, alpha=.44, edgecolors="none")
        domain = np.array([x.min(), x.max()])
        ax.plot(domain, np.polyval(np.polyfit(x, y, 1), domain), color="#163640", lw=1.3)
        ax.set(title=f"{objective}\nPearson r = {stats['pearson_r']:+.2f}", xlabel="Adaptive selection ratio (%)")
        ax.grid(axis="y", color="#e7eaef", linewidth=.65)
    axes[0].set_ylabel("Increase in mean answer NLL (nats)")
    fig.suptitle("TOFU Forget05 · Step 120 · One point per QA, averaged over three training seeds", fontsize=11)
    save_figure(fig, output, "forgetting_correlations_step120")


def render_examples(output, tokens, samples):
    by_doc = {doc: [r for r in tokens if r["doc_id"] == doc] for doc in range(200)}
    cases = []
    for sample in samples:
        doc = sample["doc_id"]
        rows = by_doc[doc]
        factual = [r for r in rows if r["fact_scope"]]
        covered = sum(r["adaptive"] for r in factual)
        cases.append({**sample, "factual_tokens": len(factual), "covered": covered,
                      "fact_rate": covered / len(factual) if factual else 0.0,
                      "entity_overlap": sum(r["entity"] and r["adaptive"] for r in rows)})
    choices = [
        ("Direct fact fully selected", lambda r: r["scope"] == "D" and r["factual_tokens"] >= 3 and r["fact_rate"] == 1, lambda r: -r["factual_tokens"]),
        ("Direct fact entirely missed", lambda r: r["scope"] == "D" and r["factual_tokens"] >= 2 and r["fact_rate"] == 0, lambda r: -r["factual_tokens"]),
        ("Direct fact partially selected", lambda r: r["scope"] == "D" and .1 < r["fact_rate"] < .9, lambda r: abs(r["fact_rate"] - .5)),
        ("Adaptive and Entity agree on name pieces", lambda r: r["entity_overlap"] >= 2, lambda r: -r["entity_overlap"]),
        ("Descriptive answer content selected", lambda r: r["scope"] == "S" and r["factual_tokens"] >= 5 and r["fact_rate"] > .6, lambda r: -r["fact_rate"]),
        ("Descriptive answer content mostly missed", lambda r: r["scope"] == "S" and r["factual_tokens"] >= 5 and r["fact_rate"] < .3, lambda r: r["fact_rate"]),
        ("Low mask density", lambda r: r["scope"] == "D", lambda r: r["selection_ratio"]),
        ("High mask density", lambda r: r["scope"] == "D", lambda r: -r["selection_ratio"]),
    ]
    used_docs, used_authors, example_meta = set(), set(), []
    page = ['''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Adaptive token selection · TOFU Forget05</title>
<style>
:root{color-scheme:light;--text:#172833;--muted:#4e6172;--line:#dce4ea;--accent:#116a6b}
*{box-sizing:border-box}body{max-width:1000px;margin:38px auto 60px;padding:0 20px;background:#f7f9fb;color:var(--text);font:15px/1.55 system-ui,sans-serif}
header{padding-bottom:10px}h1{font-size:clamp(1.85rem,4vw,2.7rem);letter-spacing:-.035em;margin:.1em 0}
.kicker{color:var(--accent);font-weight:700;font-size:.8rem;letter-spacing:.1em;text-transform:uppercase}
p{margin:.6em 0}.muted{color:var(--muted)}.legend{display:flex;gap:12px;flex-wrap:wrap;padding:12px 0}
article{background:white;border:1px solid var(--line);border-radius:12px;margin:14px 0;padding:16px 20px;break-inside:avoid}
article h2{font-size:1.12rem;margin:0 0 5px}article .meta{font-size:.84rem;color:var(--muted)}
.answer{white-space:pre-wrap;line-height:2.1;font-size:1.03rem}.adaptive{background:#fae2a3;border-radius:2px}.entity{box-shadow:inset 0 -2px #6955ad}
.fact{text-decoration:underline dotted #126d72;text-underline-offset:4px}.special{font-size:.72rem;color:var(--muted);margin-left:.4em}
summary{color:var(--accent);cursor:pointer}.piece[tabindex]:focus{outline:1px solid #126d72}.label{font-weight:650}
@media print{body{background:white}article{break-inside:avoid}*{print-color-adjust:exact}}
</style>
<header><div class="kicker">Frozen-mask audit · TOFU Forget05</div><h1>Which answer tokens did Adaptive select?</h1>
<p class="muted">Eight illustrative cases chosen to show full coverage, misses, partial coverage, name overlap and different mask densities.</p>
<div class="legend"><span class="adaptive">Adaptive selected</span><span class="entity">Entity selected</span>
<span class="fact">Annotated answer attribute</span><span class="adaptive entity">Adaptive ∩ Entity</span></div>
<p class="muted">Hover over a token for its discrepancy, position and category. The question is context, and the token overlay covers eligible answer positions.</p></header>''']
    for title, predicate, rank in choices:
        pool = [r for r in cases if predicate(r) and r["doc_id"] not in used_docs and r["eligible"] <= 65]
        if not pool:
            continue
        sample = min(pool, key=lambda r: (r["author"] in used_authors, rank(r), r["eligible"], r["doc_id"]))
        doc = sample["doc_id"]
        used_docs.add(doc)
        used_authors.add(sample["author"])
        token_html = []
        cursor = 0
        for token in by_doc[doc]:
            start, end = token["char_start"], token["char_end"]
            classes = ["piece", *(name for name in ["adaptive", "entity"] if token[name])]
            if token["fact_scope"]:
                classes.append("fact")
            if token["special"]:
                classes.append("special")
            tip = escape(f"Position {token['position']} · Δ {token['delta_base_minus_target']:+.3f} nats · "
                         f"category {token['category'].replace('_', ' ')} · "
                         f"matched random {'yes' if token['random_adaptive'] else 'no'}", quote=True)
            text = sample["answer"][max(start, cursor):end]
            gap = sample["answer"][cursor:start]
            cursor = max(cursor, end)
            if token["special"]:
                text = "⟨EOT⟩"
            token_html.append(f'{escape(gap)}<span class="{" ".join(classes)}" title="{tip}">{escape(text)}</span>')
        token_html.append(escape(sample["answer"][cursor:]))
        summary = (f'{sample["adaptive_selected"]}/{sample["eligible"]} eligible tokens selected · '
                   f'{sample["covered"]}/{sample["factual_tokens"]} annotated attribute tokens selected · '
                   f'{sample["entity_overlap"]} Adaptive/Entity overlap')
        spans = " · ".join(escape(span) for span in sample["spans"]) or "No non-name answer span annotated"
        page.append(f'<article><div class="meta">Sample {doc} · {escape(sample["author"])} · {summary}</div>'
                    f'<h2>{escape(title)}</h2><p><span class="label">Question</span> {escape(sample["question"])}</p>'
                    f'<p class="answer">{"".join(token_html)}</p><details><summary>Annotated answer spans</summary><p>{spans}</p></details></article>')
        example_meta.append({"doc_id": doc, "author": sample["author"], "example": title})
    (output / "token_selection_examples.html").write_text("\n".join(page) + "\n")
    return example_meta


def report(output, summary, categories, coverage, correlations):
    dist = summary["discrepancy"]
    overlap_metrics = summary["overlap"]
    lines = [
        "Adaptive mask relevance on frozen TOFU Forget05",
        "",
        "Scope and verification",
        f"{summary['samples']} QA pairs from {summary['authors']} authors, {summary['eligible_tokens']:,} eligible answer targets.",
        f"The Adaptive mask selects {overlap_metrics['adaptive_tokens']:,} targets ({summary['selection_ratio']:.2%}).",
        "The archived training masks, exact answer token IDs, evaluation prompts and question/answer pairs were checked.",
        "The audit includes supervised EOT targets, counted as function/control tokens.",
        "Official pre-unlearning target-model metrics and the already evaluated Step 60 and Step 120 checkpoints were used.",
        f"Maximum absolute difference between archived target NLL and the target NLL reconstructed from the audit cache was {summary['max_target_nll_cache_difference']:.4f} nats.",
        "",
        "Discrepancy is log p(base) minus log p(target), in nats.",
        "Mask         Tokens    Mean      P05       Median    P95       Negative",
        *[f"{row['group']:<12} {row['n']:>6}  {row['mean']:>8.3f}  {row['p05']:>8.3f}  {row['p50']:>8.3f}  "
          f"{row['p95']:>8.3f}   {row['fraction_negative']:>8.2%}" for row in dist],
        "",
        f"Adaptive vs Entity  precision {overlap_metrics['micro_precision']:.2%}  recall {overlap_metrics['micro_recall']:.2%}  "
        f"micro Jaccard {overlap_metrics['micro_jaccard']:.2%}  macro Jaccard {overlap_metrics['macro_jaccard']:.2%}.",
        f"Intersection {overlap_metrics['intersection']} tokens, union {overlap_metrics['union']} tokens.",
        "Precision and recall refer to the Entity trainer's lexical author-name mask.",
        "",
        "Category coverage and composition",
        "Answer-bearing spans were annotated once at the character level on all 200 answers and projected onto the exact Llama tokens.",
        "The factual-attribute category includes concrete answers and broader descriptive content that directly addresses the question.",
        "Direct-only rows isolate concrete values such as titles, awards, places, professions and dates.",
        "Author names use the frozen Entity mask. Punctuation-only tokens are separate, and function tokens use the earlier lexical stopword audit.",
        "This is a single-rater annotation on fictional facts, with no independent agreement measurement.",
        "Category                           Adaptive coverage    Adaptive share    Random coverage",
    ]
    for category in CATEGORIES + ["factual_attribute_direct_only"]:
        a, b = [r for r in categories if r["category"] == category]
        lines.append(f"{category:<36}{a['coverage']:>8.2%}             {a['selected_composition']:>8.2%}          {b['coverage']:>8.2%}")
    lines += ["", "Fact coverage against the frozen density-matched random mask"]
    for row in coverage:
        lines.append(f"{row['scope']}   Adaptive {row['adaptive_coverage']:.2%}   random {row['random_coverage']:.2%}   "
                     f"random expectation {row['random_expected_coverage_given_counts']:.2%}   gap {100 * row['paired_gap']:+.2f} pp   "
                     f"author-bootstrap 95% interval [{100 * row['author_bootstrap_low']:+.2f}, "
                     f"{100 * row['author_bootstrap_high']:+.2f}] pp")
    lines += [
        "",
        "Selection density and per-sample forgetting",
        "Positive change means increasing NLL or decreasing probability, ROUGE-L recall or extraction strength.",
        "Correlations use the 200 paired examples. The mean-of-three result averages the outcome across seeds within each example.",
        "The adjusted Pearson column residualizes log eligible length and initial target NLL from density and the outcome.",
        "The interval resamples the 10 author blocks with replacement. All associations are descriptive.",
        "Step 120 is the main endpoint. Step 60 is separately available in correlations.csv.",
        "",
        "Step 120 Pearson r and Spearman rho",
        "Objective                 Metric                      Pearson r   Spearman rho  Adjusted r   Author-bootstrap 95% CI",
    ]
    for r in correlations:
        if r["step"] == 120 and r["seed"] == "mean_of_3" and r["contrast"] == "adaptive_vs_target":
            lines.append(f"{r['objective']:<25} {r['metric']:<27} {r['pearson_r']:>+8.3f} {r['spearman_rho']:>+12.3f} "
                         f"{r['partial_pearson_log_length_initial_nll']:>+11.3f}   "
                         f"[{r['pearson_author_bootstrap_low']:+.3f}, {r['pearson_author_bootstrap_high']:+.3f}]")
    lines += ["", "Step 120 matched-random NLL contrast",
              "Density is identical within each matched QA pair. The response is Adaptive forgetting minus Random-Adaptive forgetting."]
    for r in correlations:
        if r["step"] == 120 and r["seed"] == "mean_of_3" and r["contrast"] == "adaptive_minus_random" and r["metric"] == "nll_increase":
            lines.append(f"{r['objective']:<25} Pearson r {r['pearson_r']:+.3f}   "
                         f"author-bootstrap 95% interval [{r['pearson_author_bootstrap_low']:+.3f}, "
                         f"{r['pearson_author_bootstrap_high']:+.3f}]")
    lines += ["", "Reproduce from the repository root with python scripts/analyze_adaptive_relevance.py."]
    (output / "relevance_report.txt").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-root", type=Path, default=FORMAL)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--annotations", type=Path, default=OUTPUT / "answer_spans.tsv")
    parser.add_argument("--target-eval", type=Path, default=TARGET_EVAL)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    tokens, samples = token_data(args.formal_root, args.annotations)
    seed = 20261110
    draws = np.random.default_rng(seed).integers(0, len(AUTHORS), size=(2000, len(AUTHORS)))
    discrepancy = distribution(tokens)
    categories = by_category(tokens)
    alignment, sample_overlap = overlap(tokens)
    coverage = coverage_bootstrap(tokens, samples, draws)
    correlations, changes, max_difference = association(args.formal_root, samples, args.target_eval, draws)
    examples = render_examples(args.output, tokens, samples)
    if len(examples) != EXAMPLES:
        raise ValueError(f"Expected {EXAMPLES} distinct qualitative cases, got {len(examples)}")
    for name, rows in [
        ("tokens", tokens), ("discrepancy_quantiles", discrepancy), ("token_categories", categories),
        ("overlap_by_sample", sample_overlap), ("fact_coverage", coverage),
        ("correlations", correlations), ("forgetting_change_by_sample", changes), ("examples", examples),
    ]:
        write_csv(args.output / f"{name}.csv", rows)
    plots(args.output, tokens, categories, correlations, changes)
    lock = read(args.formal_root / "protocol_manifest.json")
    summary = {
        "protocol_sha256": lock["protocol_sha256"], "training_git_commit": lock["code_commit"],
        "analysis_script_sha256": sha(Path(__file__)), "annotations_sha256": sha(args.annotations),
        "archived_target_eval_sha256": sha(args.target_eval), "bootstrap_seed": seed, "bootstrap_replicates": len(draws),
        "samples": len(samples), "authors": len(AUTHORS), "eligible_tokens": len(tokens),
        "selection_ratio": alignment["adaptive_tokens"] / len(tokens),
        "max_target_nll_cache_difference": max_difference,
        "discrepancy": discrepancy, "overlap": alignment, "fact_coverage": coverage,
        "annotation_sample_counts": {scope: sum(r["scope"] == scope for r in samples) for scope in ["D", "S", "-"]},
        "examples": examples,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    report(args.output, summary, categories, coverage, correlations)
    print((args.output / "relevance_report.txt").read_text())


if __name__ == "__main__":
    main()
