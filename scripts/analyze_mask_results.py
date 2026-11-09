"""Analyze old RMU normalization sensitivity and paired forgetting correlations."""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from scipy.stats import pearsonr, spearmanr


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_review_summary(output, analysis, report):
    """Keep the human-readable review synchronized with the exported metrics."""
    lines = [
        "MASK IMPLEMENTATION AUDIT AND CONTROLLED COMPARISONS",
        "",
        "Scope: TOFU forget05/retain95, seed 0. Five mask variants; two NPO",
        "normalizations and one RMU objective. Focused QA metrics, not the full",
        "official benchmark suite. No cross-seed or cross-split robustness claim.",
        "",
        "IMPLEMENTATION",
        "Eligible target j: labels[j] != -100, j > 0, token j and j-1 attended.",
        "Real EOS/EOT attention comes from supplied masks or unpadded lengths.",
        "NPO predicts target j from logits[j-1]; RMU uses hidden[j].",
        "Entity intersects eligible targets with the entity mask. Adaptive uses",
        "finite delta = cached base/reference log p - original/current log p",
        "strictly below the per-sequence eligible mean. If no strict selection",
        "exists, use all finite eligible tokens; if none finite, all eligible.",
        "Missing stable sample index raises; first-batch mask reuse is removed.",
        "Controlled runs replay audited initialization masks throughout training.",
        "Random samples the exact reference count per example from eligible",
        "positions without replacement, with independent per-ID seed 1729.",
        "",
        "OBJECTIVES",
        "Let n_i = sum_j m_ij and N_theta_i = sum_j m_ij * (-log p_theta_ij).",
        "Summed masked NPO: mean_{n_i>0} [-2/beta * log sigmoid(",
        "    beta * (N_theta_i - N_frozen_initial_policy_i))].",
        "Token-normalized NPO replaces both N values by N/n_i, using the same",
        "mask and denominator for both policies. beta=0.1 in paired runs.",
        "RMU: mean_{n_i>0} [sum_j m_ij * mean_hidden((h_theta_ij-c)^2) / n_i].",
        "EntityRMU delegates to the identical activation-loss implementation.",
        "Empty samples are excluded from sample means. All-empty forget batches",
        "return graph-connected zero. DDP uses the global nonempty count.",
        "NPO retain: supervised target-token cross entropy. RMU retain:",
        "activation MSE against frozen initialization on eligible retain tokens.",
        "Total = gamma * forget + alpha * retain, gamma=alpha=1 in paired runs.",
        "",
        "AUDIT STATISTICS (fractions, not percentages)",
    ]
    for name, group in report.get("groups", {}).items():
        keys = ["sequence_mean_positive_rate", "sequence_mean_negative_rate",
                "sequence_mean_zero_rate", "selected_positive_fraction",
                "selected_negative_fraction", "selected_zero_fraction",
                "selection_ratio", "empty_mask_rate", "fallback_rate"]
        lines.append(name + ": " + json.dumps({k: group.get(k) for k in keys}))
        if group.get("overlap"):
            lines.append("  overlap: " + json.dumps({k: v for k, v in group["overlap"].items()
                                                    if not isinstance(v, (list, dict))}))
    lines += [
        "Legacy pre-fix TOFU ratios: forget01=31.97%, forget05=34.06%,",
        "forget10=34.22%. Corrected ratios above use refreshed TOFU logprobs",
        "and corrected attention/eligible positions; they are separate results.",
        "Token categories: see token_categories.csv. Lexical heuristics, not",
        "gold factual labels; validated character alignment covers TOFU only.",
        "Entity name locations and raw/effective masks: " + json.dumps(analysis.get("entity_alignment", {})),
        "",
        "COMPLETED COMPARISONS",
        "run,forget_probability,retain_probability,forget_ROUGE_L_recall",
    ]
    for row in analysis["experiments"]:
        metrics = row["metrics"]
        lines.append(f'{row["run"]},{metrics["forget05"]["answer_probability"]:.8f},'
                     f'{metrics["retain95"]["answer_probability"]:.8f},'
                     f'{metrics["forget05"]["rougeL_recall"]:.8f}')
    lines += [
        "",
        "Lower forget probability/ROUGE and higher retain probability are the",
        "directions used here. Loss-membership AUC is separately exported and",
        "must not be interpreted as monotonically better when lower.",
        "Normalization ordering preserved against matched Random (forget",
        "probability only): " + json.dumps(analysis.get("normalization_ordering_preserved_vs_random")),
        "Hyperparameter matching: " + json.dumps(analysis.get("hyperparameter_matching", {})),
        "",
        "CORRELATIONS",
        "See analysis.json per experiment: Pearson, Spearman and residual",
        "Pearson controlling log eligible length and initial NLL. TOFU author",
        "samples are dependent. These are exploratory, not causal estimates.",
        "",
        "HISTORICAL RMU NORMALIZATION",
        "Old broadcast: mean(s_i) * mean(1/n_i), from [B] / [B,1].",
        "Fixed: mean(s_i/n_i). Per-sample gradient ratio old/fixed is",
        "n_i * mean(1/n_i). For micro-batch 1 or identical nonempty n_i this",
        "specific bug has no effect. Gradient accumulation does not change B",
        "inside the faulty operation. Variable Adaptive counts matter even for",
        "uniform-length MUSE chunks. EntityRMU inherited the faulty retain loss.",
        "Observed-count simulations in analysis.json do not reconstruct a",
        "historical run or quantify changes to historical forgetting metrics.",
        "Historical configuration status: " + json.dumps(analysis.get("historical_config_status", {})),
        "Confirmed run sensitivity: " + json.dumps(analysis.get("historical_run_impact", {})),
        "",
        "PROVENANCE AND VALIDATION",
        "attention_refresh.json records refreshed TOFU logprobs, model revision",
        "checks and the MUSE attention-equivalence check. Frozen masks carry",
        "reference-cache hashes. Random runs export density_validation.json;",
        "RMU runs export control_vector.json. Validation logs are in validation/.",
        "Reproduce: python scripts/run_mask_experiments.py --objectives NPO RMU",
        "Reanalyze: python scripts/analyze_mask_results.py",
    ]
    (output / "review_summary.txt").write_text("\n".join(lines) + "\n")


def read_rows(path):
    return {int(r["doc_id"]): r for r in json.loads(path.read_text())["rows"]}


def correlations(x, y):
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return {"n": len(x), "pearson": None, "spearman": None, "reason": "insufficient variance or observations"}
    return {"n": len(x), "pearson": float(pearsonr(x, y).statistic), "spearman": float(spearmanr(x, y).statistic)}


def adjusted_pearson(x, y, lengths, initial_nll):
    """Exploratory OLS-residual correlation; no causal/significance claim."""
    controls = np.column_stack([np.ones(len(x)), np.log(lengths), initial_nll])
    rx = np.asarray(x) - controls @ np.linalg.lstsq(controls, x, rcond=None)[0]
    ry = np.asarray(y) - controls @ np.linalg.lstsq(controls, y, rcond=None)[0]
    if np.std(rx) < 1e-10 or np.std(ry) < 1e-10:
        return None
    return float(pearsonr(rx, ry).statistic)


def old_sensitivity(lengths, batch_size):
    rng = np.random.default_rng(0)
    lengths = np.asarray(lengths)[rng.permutation(len(lengths))]
    weights, inflation, empty_batches = [], [], 0
    for start in range(0, len(lengths), batch_size):
        counts = lengths[start:start + batch_size]
        if np.any(counts == 0):
            empty_batches += 1
            continue
        weights.extend((counts * np.mean(1.0 / counts)).tolist())
        inflation.append(float(np.mean(counts) * np.mean(1.0 / counts)))
    return {"nominal_micro_batch": batch_size, "shuffle_seed": 0, "zero_count_batches": empty_batches,
            "sample_gradient_weight_ratio_old_over_fixed_percentiles_0_5_50_95_100": np.percentile(weights, [0, 5, 50, 95, 100]).tolist() if weights else None,
            "equal_per_token_error_loss_inflation_mean": float(np.mean(inflation)) if inflation else None,
            "equal_per_token_error_loss_inflation_max": max(inflation) if inflation else None,
            "interpretation": "Sensitivity calculation using observed token counts and simulated seed-0 batches, not reconstruction of an unavailable historical run"}


def qualitative_figure(output, examples):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    chosen = []
    for dataset, index in [("TOFU_forget05", 0), ("TOFU_forget05", 1), ("MUSE_News", 0), ("MUSE_Books", 0)]:
        item = next((r for r in examples if r["dataset"] == dataset and r["doc_id"] == index and r.get("display_tokens")), None)
        if item is None and dataset.startswith("MUSE"):
            item = next((r for r in examples if r["dataset"] == dataset and r.get("display_tokens")), None)
        if item:
            chosen.append(item)
    if not chosen:
        return
    fig, axes = plt.subplots(len(chosen), 1, figsize=(13, 2.5 * len(chosen)), squeeze=False)
    font = FontProperties(size=10)
    fig.suptitle("Adaptive selection: yellow; entity: blue; overlap: green", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax, item in zip(axes[:, 0], chosen):
        ax.axis("off")
        ax.set_title(f'{item["dataset"]} / doc {item["doc_id"]}   mean delta={item["sequence_mean"]:.3f}', loc="left", fontsize=11)
        if item.get("question"):
            import textwrap
            ax.text(.01, .9, "\n".join(textwrap.wrap(item["question"], 125)), transform=ax.transAxes, fontsize=9, va="top", parse_math=False)
        x, y = .01, .57 if item.get("question") else .8
        width = ax.get_window_extent(renderer).width
        for token in item["display_tokens"]:
            if not token["eligible"]:
                continue
            text = token.get("display_token", token["token"]).replace("\n", " ")
            token_width = renderer.get_text_width_height_descent(text, font, False)[0] / width + .005
            if x + token_width > .98:
                x, y = .01, y - .2
            if y < .03:
                break
            color = "#b8e5c4" if token["selected"] and token["entity"] else "#ffdc80" if token["selected"] else "#bcd5f5" if token["entity"] else "white"
            ax.text(x, y, text, transform=ax.transAxes, fontproperties=font, va="top", parse_math=False,
                    bbox={"facecolor": color, "edgecolor": "none", "pad": 1})
            x += token_width
    fig.savefig(output / "qualitative.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root", type=Path, default=ROOT / "saves/audits/revision")
    parser.add_argument("--experiment-root", type=Path, default=ROOT / "saves/experiments/mask_controls")
    args = parser.parse_args()
    refresh = args.audit_root / "attention_refresh.json"
    report_file = args.audit_root / "audit_report.json"
    if refresh.exists() and report_file.exists():
        ledger = json.loads(refresh.read_text())
        report = json.loads(report_file.read_text())
        report["groups"].update(ledger["TOFU_groups"])
        report["attention_refresh"] = {k: v for k, v in ledger.items() if k != "TOFU_groups"}
        save(report_file, report)
    analysis = {"old_rmu_sensitivity": {}, "experiments": [], "correlation_limitations": "One split and seed; samples from the same author are dependent. Correlations are descriptive, with no causal or cross-seed robustness claim. Core metrics do not constitute the full official evaluation suite."}
    analysis["category_scope"] = "TOFU author-mask categories have verified character/token alignment. MUSE lexical labels have SentencePiece round-trip mismatches and must not be interpreted as validated semantic/factual labels. All categories remain heuristics, not gold annotations."
    historical = args.audit_root / "historical_config_status.json"
    if historical.exists():
        analysis["historical_config_status"] = json.loads(historical.read_text())
        history = analysis["historical_config_status"]
        if history.get("actual_micro_batch_verified") and report_file.exists():
            report = json.loads(report_file.read_text())
            report["old_rmu"].update({
                "actual_historical_run_micro_batch": history["confirmed_settings"]["per_device_train_batch_size"],
                "actual_historical_gradient_accumulation_steps": history["confirmed_settings"]["gradient_accumulation_steps"],
                "confirmation_scope": {k: history[k] for k in ["user_provided_path", "dataset", "forget_split", "confirmation_source"]},
                "reason": "The user explicitly confirmed these two settings for this TOFU run. Saved artifacts and the historical code revision have not been inspected; this confirmation does not establish settings of other TOFU or MUSE runs.",
            })
            save(report_file, report)
    alignment = args.audit_root / "entity_alignment_summary.json"
    if alignment.exists():
        analysis["entity_alignment"] = json.loads(alignment.read_text())
    for path in args.audit_root.glob("tokens_*.jsonl"):
        counts, adaptive_counts = [], []
        for line in path.open():
            record = json.loads(line)
            counts.append(sum(t["label"] != -100 for t in record["tokens"]))
            adaptive_counts.append(record["selected_count"])
        analysis["old_rmu_sensitivity"][path.stem] = {"sample_count": len(counts), "min_tokens": min(counts), "max_tokens": max(counts), "micro_batches": [old_sensitivity(counts, b) for b in [1, 2, 4, 16]]}
        analysis["old_rmu_sensitivity"][path.stem + "_adaptive"] = {"sample_count": len(adaptive_counts),
            "min_tokens": min(adaptive_counts), "max_tokens": max(adaptive_counts),
            "note": "AdaptiveRMU had variable selected counts even when raw MUSE chunk lengths were equal; uniform raw lengths do not establish immunity to the old bug. These are corrected initialization masks, not reconstructed historical training masks.",
            "micro_batches": [old_sensitivity(adaptive_counts, b) for b in [1, 2, 4, 16]]}
    # Retain also used the faulty inherited activation loss, including EntityRMU.
    if (args.audit_root / "tokens_TOFU_forget05.jsonl").exists():
        from data import get_datasets
        from model import get_tokenizer
        from omegaconf import OmegaConf
        model_cfg = OmegaConf.load(ROOT / "configs/model/Llama-3.2-1B-Instruct.yaml")
        tokenizer = get_tokenizer(model_cfg.tokenizer_args)
        retain_splits = {"retain95"}
        history = analysis.get("historical_config_status", {})
        historical_retain = {"forget01": "retain99", "forget05": "retain95", "forget10": "retain90"}.get(history.get("forget_split"))
        if history.get("actual_micro_batch_verified") and historical_retain:
            retain_splits.add(historical_retain)
        for split in sorted(retain_splits):
            cfg = OmegaConf.load(ROOT / "configs/data/datasets/TOFU_QA_retain.yaml")
            cfg.TOFU_QA_retain.args.hf_args.name = split
            dataset = get_datasets(cfg, tokenizer=tokenizer, template_args=model_cfg.template_args)
            counts = [int(dataset[i]["labels"].ne(-100).sum()) for i in range(len(dataset))]
            analysis["old_rmu_sensitivity"][f"TOFU_{split}"] = {"sample_count": len(counts), "min_tokens": min(counts), "max_tokens": max(counts), "micro_batches": [old_sensitivity(counts, b) for b in [1, 2, 4, 16]]}
        if history.get("actual_micro_batch_verified") and historical_retain:
            batch_size = history["confirmed_settings"]["per_device_train_batch_size"]
            impact = {"micro_batch": batch_size,
                      "gradient_accumulation_steps": history["confirmed_settings"]["gradient_accumulation_steps"],
                      "scope": history["user_provided_path"],
                      "code_revision_verified": False,
                      "interpretation": "If this run used the faulty [B]/[B,1] broadcast, variable token counts at micro-batch 4 affect both BaseRMU forget and retain losses. Accumulation does not remove this bug. Simulated seed-0 batches below are a sensitivity estimate, not reconstructed historical gradients or metric degradation."}
            for branch, key in [("forget", f'tokens_TOFU_{history["forget_split"]}'), ("retain", f"TOFU_{historical_retain}")]:
                group = analysis["old_rmu_sensitivity"].get(key)
                if group:
                    impact[branch] = next(item for item in group["micro_batches"] if item["nominal_micro_batch"] == batch_size)
            analysis["historical_run_impact"] = impact
    baseline_file = args.experiment_root / "baseline/forget05.json"
    plot_rows = []
    if baseline_file.exists():
        baseline = read_rows(baseline_file)
        initial = {int(r["doc_id"]): r for r in map(json.loads, (args.audit_root / "tokens_TOFU_forget05.jsonl").open())}
        masks = {source: {int(r["doc_id"]): r for r in map(json.loads, (args.audit_root / f"masks_TOFU_forget05_{source}.jsonl").open())} for source in ["entity", "adaptive"]}
        for path in sorted(args.experiment_root.glob("*/completed.json")):
            config = json.loads((path.parent / "resolved_config.json").read_text())
            execution = config["execution"]
            variant, norm = execution["variant"], execution["normalization"]
            after = read_rows(path.parent / "evals/forget05.json")
            metrics = json.loads((path.parent / "evals/metrics.json").read_text())
            xs, nll_change, prob_drop, rouge_drop, lengths, initial_nll = [], [], [], [], [], []
            selected_count = eligible_count = 0
            for index, row in after.items():
                eligible = initial[index]["eligible_count"]
                selected = eligible if variant == "base" else sum(masks["entity" if "entity" in variant else "adaptive"][index]["mask"])
                ratio = selected / eligible
                xs.append(ratio)
                lengths.append(eligible)
                initial_nll.append(baseline[index]["mean_nll"])
                selected_count += selected
                eligible_count += eligible
                nll_change.append(row["mean_nll"] - baseline[index]["mean_nll"])
                prob_drop.append(baseline[index]["answer_probability"] - row["answer_probability"])
                rouge_drop.append(baseline[index]["rougeL_recall"] - row["rougeL_recall"])
                if path.parent.name.startswith("NPO_"):
                    plot_rows.append((norm, variant, ratio, nll_change[-1]))
            item = {"run": path.parent.name, "variant": variant, "normalization": norm, "objective": "NPO" if path.parent.name.startswith("NPO_") else "RMU",
                    "aggregate_selection_ratio": selected_count / eligible_count, "mean_per_sequence_selection_ratio": float(np.mean(xs)),
                    "selection_vs_forget_nll_increase": correlations(xs, nll_change), "selection_vs_answer_probability_drop": correlations(xs, prob_drop),
                    "selection_vs_rouge_drop": correlations(xs, rouge_drop), "metrics": metrics}
            item["selection_vs_answer_length"] = correlations(xs, lengths)
            item["selection_vs_initial_nll"] = correlations(xs, initial_nll)
            item["selection_vs_nll_increase_adjusted_pearson"] = adjusted_pearson(xs, nll_change, lengths, initial_nll)
            item["adjustment_controls"] = ["log eligible target length", "initial model mean NLL"]
            item["training_hyperparameters"] = {key: config["trainer"]["args"].get(key) for key in [
                "per_device_train_batch_size", "gradient_accumulation_steps", "num_train_epochs",
                "learning_rate", "weight_decay", "warmup_epochs", "seed", "optim", "bf16",
                "gradient_checkpointing", "gradient_checkpointing_kwargs"]}
            item["objective_coefficients"] = {key: config["trainer"]["method_args"].get(key) for key in
                (["alpha", "gamma", "beta", "retain_loss_type"] if item["objective"] == "NPO" else
                 ["alpha", "gamma", "steering_coeff", "retain_loss_type", "module_regex", "trainable_params_regex"])}
            state_file = path.parent / "trainer_state.json"
            state = json.loads(state_file.read_text()) if state_file.exists() else {}
            item["actual_global_step"] = state.get("global_step")
            item["actual_epoch"] = state.get("epoch")
            control_file = path.parent / "control_vector.json"
            item["control_vector_sha256"] = json.loads(control_file.read_text())["sha256"] if control_file.exists() else None
            analysis["experiments"].append(item)
        conclusions = {}
        for objective in ["NPO", "RMU"]:
            for norm in ["summed", "token_mean"] if objective == "NPO" else ["summed"]:
                group = {row["variant"]: row for row in analysis["experiments"] if row["objective"] == objective and row["normalization"] == norm}
                comparisons = {}
                for source in ["entity", "adaptive"]:
                    if source in group and f"random_{source}" in group:
                        method, random = group[source], group[f"random_{source}"]
                        a, b = [row["metrics"]["forget05"]["answer_probability"] for row in [method, random]]
                        comparisons[source] = {"method_forget_answer_probability": a, "random_forget_answer_probability": b, "method_lower_than_matched_random": a < b,
                                               "method_retain_probability": method["metrics"]["retain95"]["answer_probability"], "random_retain_probability": random["metrics"]["retain95"]["answer_probability"]}
                        comparisons[source]["method_lower_forget_rouge"] = method["metrics"]["forget05"]["rougeL_recall"] < random["metrics"]["forget05"]["rougeL_recall"]
                        comparisons[source]["method_higher_retain_probability"] = method["metrics"]["retain95"]["answer_probability"] > random["metrics"]["retain95"]["answer_probability"]
                        comparisons[source]["method_pareto_dominates_random_on_probabilities"] = a <= b and method["metrics"]["retain95"]["answer_probability"] >= random["metrics"]["retain95"]["answer_probability"] and (a < b or method["metrics"]["retain95"]["answer_probability"] > random["metrics"]["retain95"]["answer_probability"])
                conclusions[f"{objective}_{norm}"] = {"matched_random_comparisons": comparisons,
                    "between_method_selection_vs_forget_probability": correlations([r["aggregate_selection_ratio"] for r in group.values()], [r["metrics"]["forget05"]["answer_probability"] for r in group.values()])}
        analysis["paired_conclusions"] = conclusions
        analysis["hyperparameter_matching"] = {}
        for objective in ["NPO", "RMU"]:
            group = [r for r in analysis["experiments"] if r["objective"] == objective]
            if group:
                reference = {key: group[0][key] for key in ["training_hyperparameters", "objective_coefficients"]}
                differences = [{"run": row["run"], "settings": {key: row[key] for key in reference}}
                               for row in group if any(row[key] != reference[key] for key in reference)]
                analysis["hyperparameter_matching"][objective] = {"matched": not differences,
                    "reference": reference, "differences": differences}
                if objective == "RMU":
                    hashes = {row["control_vector_sha256"] for row in group}
                    analysis["hyperparameter_matching"][objective]["control_vectors_identical_across_methods"] = len(hashes) == 1 and None not in hashes
                    analysis["hyperparameter_matching"][objective]["matched_including_control_vector"] = not differences and len(hashes) == 1 and None not in hashes
        analysis["normalization_ordering_preserved_vs_random"] = {source: (conclusions.get("NPO_summed", {}).get("matched_random_comparisons", {}).get(source, {}).get("method_lower_than_matched_random") == conclusions.get("NPO_token_mean", {}).get("matched_random_comparisons", {}).get(source, {}).get("method_lower_than_matched_random")) for source in ["entity", "adaptive"]} if len([r for r in analysis["experiments"] if r["objective"] == "NPO"]) == 10 else None
        analysis["ordering_definition"] = "The normalization ordering flag compares forget answer probability only. ROUGE, retain probability and Pareto comparisons are reported separately. It is not a cross-seed or full-benchmark robustness claim."
    else:
        analysis["experiment_status"] = "No completed paired comparisons yet; selection/forgetting correlations cannot be inferred from initialization ratios alone."
    args.audit_root.mkdir(parents=True, exist_ok=True)
    save(args.audit_root / "analysis.json", analysis)
    if analysis["experiments"]:
        with (args.audit_root / "experiment_comparison.csv").open("w") as stream:
            writer = csv.writer(stream)
            writer.writerow(["run", "objective", "variant", "normalization", "selection_ratio", "forget_probability", "retain_probability", "forget_rougeL_recall", "loss_membership_auc", "selection_vs_nll_increase_spearman", "actual_global_step", "actual_epoch"])
            for row in analysis["experiments"]:
                metrics = row["metrics"]
                writer.writerow([row["run"], row["objective"], row["variant"], row["normalization"], row["aggregate_selection_ratio"],
                    metrics["forget05"]["answer_probability"], metrics["retain95"]["answer_probability"], metrics["forget05"]["rougeL_recall"],
                    metrics["loss_membership_auc"], row["selection_vs_forget_nll_increase"]["spearman"], row["actual_global_step"], row["actual_epoch"]])
    if report_file.exists():
        report = json.loads(report_file.read_text())
        keys = ["sequences", "sequence_mean_positive_rate", "sequence_mean_negative_rate", "sequence_mean_zero_rate",
                "selected_positive_fraction", "selected_negative_fraction", "selected_zero_fraction", "selection_ratio", "empty_mask_rate", "fallback_rate"]
        with (args.audit_root / "mask_audit_summary.csv").open("w") as stream:
            writer = csv.writer(stream)
            writer.writerow(["dataset"] + keys + ["micro_jaccard", "entity_token_precision", "entity_token_recall", "legacy_selection_ratio"])
            for name, group in report["groups"].items():
                overlap = group.get("overlap", {})
                legacy = report["legacy_TOFU_selection_ratios"].get(name.removeprefix("TOFU_"))
                writer.writerow([name] + [group.get(k) for k in keys] + [overlap.get(k) for k in ["micro_jaccard", "entity_token_precision", "entity_token_recall"]] + [legacy])
        with (args.audit_root / "token_categories.csv").open("w") as stream:
            writer = csv.writer(stream)
            writer.writerow(["dataset", "heuristic_category", "eligible", "selected", "selection_ratio", "selected_composition"])
            for name, group in report["groups"].items():
                if name.startswith("TOFU_"):
                    for category, values in group["categories"].items():
                        writer.writerow([name, category] + [values[k] for k in ["eligible", "selected", "selection_ratio", "selected_composition"]])
    from audit_masks import qualitative
    examples = []
    for path in sorted(args.audit_root.glob("examples_*.json")):
        records = json.loads(path.read_text())
        examples.extend(records["representative"] + records["selected_positive_discrepancy"])
    if examples:
        qualitative(args.audit_root, examples)
        qualitative_figure(args.audit_root, examples)
    if plot_rows:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        for axis, norm in zip(axes, ["summed", "token_mean"]):
            for variant in ["base", "entity", "adaptive", "random_entity", "random_adaptive"]:
                points = [(x, y) for n, v, x, y in plot_rows if n == norm and v == variant]
                if points:
                    x, y = zip(*points)
                    axis.scatter(x, y, s=7, alpha=.3, label=variant)
            axis.set_title(norm)
            axis.set_xlabel("Per-sequence selection ratio")
            axis.axhline(0, color="gray", linewidth=.5)
        axes[0].set_ylabel("Forget mean NLL increase from initialization")
        axes[1].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(args.audit_root / "selection_forgetting.png", dpi=180)
        plt.close(fig)
    if report_file.exists():
        write_review_summary(args.audit_root, analysis, json.loads(report_file.read_text()))
    print("Saved", args.audit_root / "analysis.json", flush=True)


if __name__ == "__main__":
    main()
