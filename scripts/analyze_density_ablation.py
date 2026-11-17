"""Verify, analyze and export TOFU Forget05 density ablation Phase A."""

import argparse
import hashlib
import json
import math
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from prepare_density_ablation import DENSITIES, FORMAL, INITIAL, NAME, OUTPUT, STEPS
from prepare_low_confidence import read, require, rows, sha256, write_csv
from run_density_ablation import load_phase
from run_low_confidence import METRICS, TABLE_METRICS, digest, replay_metrics


CONTROL_LABELS = {"adaptive": "BelowMean Adaptive", "random_adaptive": "Random-Adaptive",
                  "base": "Base", "lowconfidence_matched": "LowConfidence-Matched"}
PARETO = {"fprob_vs_utility": ("utility", "forget_probability"),
          "fquality_vs_utility": ("utility", "forget_quality"), "privleak_vs_utility": ("utility", "privleak_abs"),
          "joint": ("utility", "forget_probability", "forget_quality", "privleak_abs")}
LOWER = {"forget_probability", "privleak_abs"}


def collect(root, phase):
    reference = read(root / "retain_only_reference_TOFU_EVAL.json")
    audit = read(root / "mask_audit/summary.json")
    records = []
    smoke_summary = {}
    for variant in ("Discrepancy-34", "Random-34"):
        smoke = root / "jobs" / variant / "smoke_test"
        proof = read(smoke / "smoke_validation.json")
        require(proof["passed"] and proof["initial_reference_bitwise_equal"]
                and proof["mask_sha256"] == phase["mask_hashes"][variant]
                and proof["unit_tests_sha256"] == sha256(root / "unit_tests.log")
                and set(proof["checkpoints"]) == {"2", "4"}, "Smoke proof differs")
        for rank in range(2):
            numeric = read(smoke / f"finite_checks_rank{rank}.json")
            require(len(numeric["loss"]) == 16 and len(numeric["gradient_l2_preclip"]) == 4
                    and all(math.isfinite(v) for v in numeric["loss"] + numeric["gradient_l2_preclip"]), "Smoke numeric checks differ")
        smoke_summary[variant] = proof
    for variant in FORMAL:
        job = root / "jobs" / variant
        collected = job / "collected_results/seed0"
        training = read(collected / "training_completed.json")
        require(training["global_step"] == 120 and set(training["checkpoints"]) == set(map(str, STEPS)),
                f"Checkpoint schedule differs: {variant}")
        require(not list((job / "runs").rglob("*.safetensors")), f"Formal weights remain for {variant}")
        manifest = read(job / "incremental_manifest.json")
        norms = read(collected / "optimizer_update_norms.json")
        for rank in range(2):
            numeric = read(collected / f"finite_checks_rank{rank}.json")
            require(len(numeric["loss"]) == 480 and len(numeric["gradient_l2_preclip"]) == 120
                    and all(math.isfinite(v) for v in numeric["loss"] + numeric["gradient_l2_preclip"]), "Formal numeric checks differ")
        for step in STEPS:
            cp = collected / f"checkpoint-{step}"
            item = manifest[f"seed0/checkpoint-{step}"]
            for name, checksum in item["files"].items():
                require(digest(cp / name) == checksum, f"Collected artifact changed: {variant}/{step}/{name}")
            folder = cp / "evals/official"
            check = read(folder / "validated.json")
            require(item["validated"] == check and check["global_step"] == step
                    and check["actual_epoch"] == phase["actual_epochs"][str(step)] and check["code_commit"] == phase["code_commit"]
                    and check["mask_sha256"] == phase["mask_hashes"][variant] and check["evaluator_sha256"] == phase["evaluator_sha256"],
                    f"Evaluation provenance differs: {variant}/{step}")
            counts = check["sample_counts"]
            require(all(counts[k] == n for k, n in [("forget_Q_A_Prob", 200), ("retain_Q_A_Prob", 400),
                    ("ra_Q_A_Prob", 100), ("wf_Q_A_Prob", 117), ("mia_min_k/holdout", 200)]), "Official sample count differs")
            provenance = read(folder / "evaluation_provenance.json")
            replay = replay_metrics(provenance["resolved_eval_config"], read(folder / "TOFU_EVAL.json"), reference)
            require(all(math.isclose(value, replay[k], rel_tol=1e-10, abs_tol=1e-14) for k, value in check["metrics"].items()),
                    f"Per-sample aggregate changed: {variant}/{step}")
            metrics = dict(check["metrics"])
            signed = metrics.pop("privleak")
            stats = check["training_diagnostics"]
            records.append({"variant": variant, "source": "phase_a_seed0", "seed": 0, "step": step,
                "actual_epoch": check["actual_epoch"], "selection_ratio": audit[variant]["aggregate_ratio"],
                **metrics, "privleak_signed": signed, "privleak_abs": abs(signed),
                "logged_training_loss": stats["logged_training_loss"], "gradient_l2_preclip": stats["gradient_l2_preclip"],
                "learning_rate_after_step": stats["learning_rate_after_step"], **norms[str(step)],
                "code_commit": phase["code_commit"], "mask_sha256": check["mask_sha256"],
                "resolved_config_sha256": digest(cp / "resolved_config.json"), "evaluator_sha256": phase["evaluator_sha256"]})
    require(len(records) == 24, "Expected 24 official checkpoints")
    return records, smoke_summary


def operating_points(root, formal):
    metrics = [*TABLE_METRICS]
    controls = []
    for row in rows(root / "existing_controls.csv"):
        require(row["seed"] == "0", "Phase A comparisons require seed-0 values")
        controls.append({"variant": CONTROL_LABELS[row["variant"]], "source": "archived_control_seed0", "seed": 0,
                         "step": int(row["step"]), "actual_epoch": float(row["actual_epoch"]),
                         "selection_ratio": float(row["selection_ratio"]), **{k: float(row[k]) for k in metrics}})
    data = [{k: row[k] for k in controls[0]} for row in formal] + controls
    baseline = read(root / "initial_target_TOFU_SUMMARY.json")
    initial = {key: baseline[official] for key, official in METRICS.items() if key not in {"forget_ratio", "privleak"}}
    initial["forget_ratio"] = read(INITIAL / "TOFU_EVAL.json")["forget_truth_ratio"]["agg_value"]
    initial["privleak_signed"] = baseline["privleak"]
    initial["privleak_abs"] = abs(baseline["privleak"])
    for row in data:
        row.update(utility_retained_fraction=row["utility"] / initial["utility"],
                   fprob_drop_vs_target=initial["forget_probability"] - row["forget_probability"],
                   frouge_drop_vs_target=initial["forget_rouge"] - row["forget_rouge"],
                   ES_drop_vs_target=initial["ES"] - row["ES"],
                   fquality_change_vs_target=row["forget_quality"] - initial["forget_quality"],
                   fratio_change_vs_target=row["forget_ratio"] - initial["forget_ratio"],
                   privleak_abs_drop_vs_target=initial["privleak_abs"] - row["privleak_abs"])
    write_csv(root / "operating_points.csv", data)
    return data, initial


def contrasts(root, data):
    index = {(row["variant"], row["step"]): row for row in data}
    names = [*TABLE_METRICS, "utility_retained_fraction", "fprob_drop_vs_target", "privleak_abs_drop_vs_target"]
    comparisons = []
    for step in STEPS:
        for rate in DENSITIES:
            lhs, rhs = [index[f"{name}-{rate}", step] for name in ("Discrepancy", "Random")]
            for metric in names:
                comparisons.append({"seed": 0, "step": step, "nominal_density": rate, "actual_density": lhs["selection_ratio"],
                    "metric": metric, "discrepancy": lhs[metric], "random": rhs[metric], "difference": lhs[metric] - rhs[metric]})
    write_csv(root / "paired_density_comparisons.csv", comparisons)
    control_comparisons = []
    for step in (60, 120):
        for variant in FORMAL:
            lhs = index[variant, step]
            for label in CONTROL_LABELS.values():
                rhs = index[label, step]
                for metric in TABLE_METRICS:
                    control_comparisons.append({"seed": 0, "step": step, "variant": variant, "control": label,
                        "metric": metric, "variant_value": lhs[metric], "control_value": rhs[metric],
                        "difference": lhs[metric] - rhs[metric]})
    write_csv(root / "control_comparisons.csv", control_comparisons)
    trends = []
    for step in STEPS:
        for metric in TABLE_METRICS:
            d = [next(row["difference"] for row in comparisons if row["step"] == step and row["metric"] == metric
                      and row["nominal_density"] == rate) for rate in DENSITIES]
            a, b, c = map(abs, d)
            trend = ("constant" if a == b == c else "increases" if a <= b <= c else
                     "decreases" if a >= b >= c else "nonmonotonic")
            trends.append({"step": step, "metric": metric, **{f"diff_{density}": value for density, value in zip(DENSITIES, d)},
                           "absolute_effect_trend": trend, "sign_reversal": min(d) < 0 < max(d)})
    write_csv(root / "identity_effect_by_density.csv", trends)
    return comparisons, trends


def pareto_flags(root, data):
    def beats(a, b, metrics):
        diffs = [(b[key] - a[key]) if key in LOWER else (a[key] - b[key]) for key in metrics]
        return all(x >= 0 for x in diffs) and any(x > 0 for x in diffs)

    result = []
    for scope in ("all_observed", *STEPS):
        cohort = [row for row in data if scope == "all_observed" or row["step"] == scope]
        for label, metrics in PARETO.items():
            for row in cohort:
                dominators = [f"{other['variant']}@{other['step']}" for other in cohort if beats(other, row, metrics)]
                result.append({"scope": scope, "objectives": label, "variant": row["variant"], "step": row["step"],
                               "seed": 0, "utility": row["utility"], "forget_probability": row["forget_probability"],
                               "forget_quality": row["forget_quality"], "privleak_abs": row["privleak_abs"],
                               "pareto": not dominators, "dominated_by": "|".join(dominators)})
    write_csv(root / "pareto_frontiers.csv", result)
    return result


def figures(root, data, frontiers):
    colors = {15: "#25747e", 34: "#bd573b", 50: "#674d9a"}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
    panels = [("forget_probability", "F-Prob ↓", "fprob_vs_utility"), ("forget_quality", "F-Quality ↑ (log scale)", "fquality_vs_utility"),
              ("privleak_abs", "|PrivLeak| ↓", "privleak_vs_utility")]
    floor = min(r["forget_quality"] for r in data if r["forget_quality"] > 0) / 10
    for ax, (metric, title, frontier) in zip(axes, panels):
        def val(row):
            return max(row[metric], floor) if metric == "forget_quality" else row[metric]
        for variant in FORMAL:
            density = int(variant.split("-")[1])
            points = [r for r in data if r["variant"] == variant]
            random = variant.startswith("Random")
            ax.plot([r["utility"] for r in points], [val(r) for r in points], color=colors[density],
                    linestyle="--" if random else "-", marker="s" if random else "o", markersize=4, label=variant)
            ax.scatter([points[-1]["utility"]], [val(points[-1])], marker="s" if random else "o", s=49,
                       color=colors[density], edgecolor="white", linewidth=.55, zorder=4)
        old = [r for r in data if r["source"] == "archived_control_seed0" and r["step"] == 120]
        ax.scatter([r["utility"] for r in old], [val(r) for r in old], marker="x", color="#626a73", s=22,
                   label="Archived controls, Step 120")
        points = sorted([r for r in frontiers if r["scope"] == 120 and r["objectives"] == frontier and r["pareto"]],
                        key=lambda r: r["utility"])
        ax.plot([r["utility"] for r in points], [val(r) for r in points], ls=":", color="#202830",
                label="Step 120 observed Pareto", linewidth=1.2)
        ax.set(xlabel="Utility ↑", ylabel=title)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.22)
        if metric == "forget_quality":
            ax.set_yscale("log")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.subplots_adjust(left=.065, right=.995, bottom=.14, top=.77, wspace=.30)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .92), ncol=4, frameon=False, fontsize=8)
    fig.suptitle("TOFU Forget05 · Seed 0 screening · trajectories at steps 20, 40, 60 and 120", y=.98, fontsize=11)
    fig.savefig(root / "utility_trajectories.pdf")
    fig.savefig(root / "utility_trajectories.png", dpi=195)
    plt.close(fig)


def report(root, data, initial, pairs, trends, frontiers):
    audit = read(root / "mask_audit/summary.json")
    labels = {"ES": "ES", "forget_probability": "F-Prob", "forget_rouge": "F-ROUGE", "forget_ratio": "F-Ratio",
              "forget_quality": "F-Quality", "utility": "Utility", "privleak_signed": "PrivLeak signed", "privleak_abs": "|PrivLeak|"}
    lines = [
        "TOFU Forget05 Adaptive discrepancy density ablation · Phase A",
        "",
        "Seed 0 screening only. Token-normalized NPO and the frozen v4 model, data and evaluator are shared.",
        "Discrepancy masks rank frozen base-minus-target log probabilities. Each random arm uses the same eligible positions and per-example count.",
        "The random-mask seed is 1729. The v4 per-sample sampler gives nested selections across density levels.",
        "",
        "Mask                    Selected / eligible    Realized density",
    ]
    for name, row in audit.items():
        lines.append(f"{name:<25} {row['selected']:>4}/{row['eligible']:<4}             {row['aggregate_ratio']:.4%}")
    lines += ["", "BelowMean regenerated byte-for-byte against the archived Adaptive mask.",
              "BelowMean and ZeroThreshold are mask-audit arms.", "",
              f"Archived initial target Utility {initial['utility']:.6f}, F-Prob {initial['forget_probability']:.6f}, "
              f"and |PrivLeak| {initial['privleak_abs']:.6f}.",
              "", "Step 120 operating points including seed-0 archived controls",
              "Variant                     Utility    Utility retained      F-Prob    F-Quality    |PrivLeak|"]
    for r in data:
        if r["step"] == 120:
            lines.append(f"{r['variant']:<27}{r['utility']:>8.5f}{r['utility_retained_fraction']:>16.2%} "
                         f"{r['forget_probability']:>13.6f}{r['forget_quality']:>13.5g}{r['privleak_abs']:>14.5f}")
    lines += ["", "Step 120 paired identity effects. Discrepancy minus Random at equal per-example density.",
              "Metric                    15% nominal     34% nominal     50% nominal    Absolute trend"]
    for r in trends:
        if r["step"] == 120:
            lines.append(f"{labels[r['metric']]:<23}{r['diff_15']:>14.5g}{r['diff_34']:>16.5g}{r['diff_50']:>16.5g}    {r['absolute_effect_trend']}")
    lines += ["", "All metrics and all four steps are in paired_density_comparisons.csv.",
              "The pairwise frontiers maximize Utility and use the stated direction of the second metric.",
              "The joint frontier uses Utility and F-Quality up, with F-Prob and absolute PrivLeak down.",
              "Frontiers compare observed seed-0 points, so they describe a finite screening grid.",
              "The figure uses Step 120 archived controls as gray crosses. Large markers denote the new runs' Step 120 endpoints.",
              "", "Phase B recommendation",
              "Replicate all six arms at seeds 1 and 2. Retain all three matched pairs so that the density-by-identity effects can be checked across seeds.",
              "At Step 120, Discrepancy-15 shows the largest matched F-Prob gap, Discrepancy-34 reaches the lowest F-Prob "
              "among the three discrepancy arms, and Discrepancy-50 reaches the lowest absolute PrivLeak.",
              "Phase A uses one seed at each density. The archived comparisons use seed 0 only.",
              "", f"Official evaluation and independent per-sample metric replay passed for all 24 checkpoints.",
              f"Result table contains {len(data)} operating points, {len(pairs)} within-density paired comparisons and "
              f"{sum(r['pareto'] for r in frontiers if r['scope'] == 120 and r['objectives'] == 'joint')} joint Step 120 frontier points."]
    (root / "phase_a_report.txt").write_text("\n".join(lines) + "\n")


def pack(root):
    files = {}
    for path in sorted(root.iterdir()):
        if path.is_file() and path.name not in {"queue.log", "queue.pid", "queue.lock", "status.json"}:
            files[path.name] = path
    for folder in ("masks", "mask_audit", "logs"):
        files.update({str(path.relative_to(root)): path for path in (root / folder).rglob("*") if path.is_file()})
    for variant in FORMAL:
        job = root / "jobs" / variant
        for name in ("protocol_manifest.json", "incremental_manifest.json"):
            files[f"runs/{variant}/{name}"] = job / name
        for source, prefix in [(job / "collected_results/seed0", f"runs/{variant}"),
                               (job / "smoke_test", f"smoke/{variant}")]:
            files.update({f"{prefix}/{path.relative_to(source)}": path for path in source.rglob("*")
                          if path.is_file() and not path.name.endswith((".safetensors", ".safetensors.index.json"))})
    archive = root.parents[1] / "exports" / f"{NAME}.zip"
    pending = archive.with_suffix(".zip.part")
    hashes = {name: digest(path) for name, path in files.items()}
    with ZipFile(pending, "w", ZIP_DEFLATED, compresslevel=4) as z:
        for name, path in sorted(files.items()):
            z.write(path, f"{NAME}/{name}")
        z.writestr(f"{NAME}/MANIFEST.json", json.dumps({"runs": 6, "checkpoints": 24, "files": hashes}, indent=2) + "\n")
    with ZipFile(pending) as z:
        require(len(z.namelist()) == len(hashes) + 1, "ZIP file count differs")
        for name, checksum in hashes.items():
            with z.open(f"{NAME}/{name}") as stream:
                require(hashlib.file_digest(stream, "sha256").hexdigest() == checksum, f"ZIP member differs: {name}")
    checksum = digest(pending)
    pending.replace(archive)
    archive.with_suffix(".zip.sha256").write_text(f"{checksum}  {archive.name}\n")
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=OUTPUT)
    args = parser.parse_args()
    root = args.root.resolve()
    status = read(root / "status.json")
    require(status["phase"] == "training_complete" and status["official_checkpoints"] == 24, "Phase A queue has not finished")
    phase = load_phase(root)
    data, smoke = collect(root, phase)
    write_csv(root / "official_metrics_long.csv", data)
    (root / "smoke_validation.json").write_text(json.dumps(smoke, indent=2, allow_nan=False) + "\n")
    points, initial = operating_points(root, data)
    pairs, trends = contrasts(root, points)
    frontier = pareto_flags(root, points)
    figures(root, points, frontier)
    report(root, points, initial, pairs, trends, frontier)
    print(pack(root))


if __name__ == "__main__":
    main()
