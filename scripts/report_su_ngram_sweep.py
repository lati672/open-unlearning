"""Verify and report the completed SU-NGram-Paper P0 sweep."""

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run_su_ngram_sweep import OUTPUT, RATES, STEPS, SWEEP, TABLE_METRICS, choose, csv_write, pack, read, require, rows_from


LABELS = {
    "ES": "ES", "forget_probability": "F-Prob", "forget_rouge": "F-ROUGE",
    "forget_ratio": "F-Ratio", "forget_quality": "F-Quality", "utility": "Utility",
    "privleak_signed": "PrivLeak signed", "privleak_abs": "|PrivLeak|",
    "gradient_l2_preclip": "Gradient L2 before clipping", "parameter_l2": "Parameter L2",
    "optimizer_step_update_l2": "Optimizer-step update L2",
}
COUNTS = {"forget_Q_A_Prob": 200, "retain_Q_A_Prob": 400, "ra_Q_A_Prob": 100,
          "wf_Q_A_Prob": 117, "mia_min_k/holdout": 200}


def make_figure(root, plan, scored, selected):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.35), layout="constrained")
    colors = ["#417d8c", "#8665a8", "#c05438"]
    for rate, color in zip(RATES, colors):
        rows = [r for r in scored if r["learning_rate_label"] == rate]
        for ax, key in zip(axes, ("utility", "forget_probability")):
            ax.plot(STEPS, [r[key] for r in rows], marker="o", label=f"LR {rate}", color=color)
    for ax, key, limit, title, baseline_key in [
        (axes[0], "utility", "utility_floor", "Utility ↑", "utility"),
        (axes[1], "forget_probability", "fprob_ceiling", "Forget Probability ↓", "forget_probability"),
    ]:
        ax.axhline(plan["baseline"][baseline_key], ls=":", color="#75818c", lw=1.2, label="Initial target")
        ax.axhline(plan["selection_rule"][limit], ls="--", color="#202c37", lw=1, label="Frozen gate")
        if selected:
            ax.scatter([selected["step"]], [selected[key]], marker="*", color="#1b2730", s=125, zorder=5, label="Selected")
        ax.set(xlabel="Optimizer step", ylabel=title, xticks=STEPS)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.2)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=len(labels), frameon=False, fontsize=8)
    fig.savefig(root / "sweep_utility_forgetting.pdf")
    fig.savefig(root / "sweep_utility_forgetting.png", dpi=190)
    plt.close(fig)


def summarize(root):
    state = read(root / "status.json")
    require(state["phase"] in {"complete", "no_eligible_checkpoint"}, "Sweep has not finished")
    plan = read(root / "run_plan.json")
    manifest = read(root / "incremental_manifest.json")
    results = rows_from(manifest)
    selected, scored = choose(plan, [row for row in results if row["seed"] == 0])
    expected = 17 if selected else 15
    require(len(manifest) == len(results) == expected and state["archived_checkpoints"] == expected, "Incomplete checkpoint matrix")
    require(read(root / "selection.json")["selected"] == selected, "Stored selection differs from independent scoring")
    for key, record in manifest.items():
        validation = record["validated"]
        require(all(validation["sample_counts"][metric] == count for metric, count in COUNTS.items()), f"Wrong sample counts: {key}")
    require(all(math.isfinite(row[metric]) for row in results for metric in [*TABLE_METRICS,
                "gradient_l2_preclip", "parameter_l2", "optimizer_step_update_l2"]), "Nonfinite metrics or optimizer diagnostics")
    require(not list((root / "runs").rglob("*.safetensors")), "Sweep weights still present after archive")
    baseline = plan["baseline"]
    gates = plan["selection_rule"]
    lines = [
        "SU-NGram-Paper P0 learning-rate sweep",
        "",
        "Seed 0 was used to choose one operating point. The fixed 120-step linear LR schedule uses six warmup steps.",
        "Sweep runs stop after step 60. Confirmation runs stop at the chosen step with the same scheduler prefix.",
        f"Initial target Utility {baseline['utility']:.8f}, F-Prob {baseline['forget_probability']:.8f}.",
        f"Frozen gates require Utility >= {gates['utility_floor']:.8f} and F-Prob <= {gates['fprob_ceiling']:.8f}.",
        "Eligible points are ranked by the harmonic mean of capped Utility retention and relative F-Prob reduction.",
        "",
        "Seed 0 results",
        "LR         Step   Utility      F-Prob       Relative drop  Gate",
    ]
    for row in scored:
        lines.append(f"{row['learning_rate_label']:<10} {row['step']:>4}  {row['utility']:>10.6f}  "
                     f"{row['forget_probability']:>10.6f}  {row['relative_fprob_drop']:>12.2%}  "
                     f"{'pass' if row['eligible'] else row['exclusions']}")
    with (OUTPUT / "official_metrics.csv").open() as stream:
        old = next(row for row in csv.DictReader(stream)
                   if row["variant"] == "SU-NGram-Paper" and row["seed"] == "0" and row["step"] == "60")
    new = next(row for row in scored if row["learning_rate_label"] == "1e-5" and row["step"] == 60)
    regression = [{"metric": metric, "prior_formal": float(old[metric]), "p0_sweep": new[metric],
                   "absolute_difference": abs(float(old[metric]) - new[metric])}
                  for metric in TABLE_METRICS if metric != "privleak_abs"]
    csv_write(root / "step60_regression.csv", regression)
    lines += ["", f"Seed 0, LR 1e-5, Step 60 against the prior formal run. "
              f"Maximum absolute difference across the seven official metrics {max(row['absolute_difference'] for row in regression):.6g}."]
    if selected:
        lines += ["", f"Selected {selected['learning_rate_label']} at step {selected['step']} ({selected['actual_epoch']:.1f} epochs).",
                  f"Seed 0 retains {selected['utility_fraction']:.2%} of initial Utility and reduces F-Prob by "
                  f"{selected['relative_fprob_drop']:.2%}. Seed 0 is reused from the sweep.",
                  "", "Confirmation at the fixed selected operating point",
                  "Metric                              Seed 0         Seed 1         Seed 2         Mean ± sample SD"]
        with (root / "selected_config_mean_std.csv").open() as stream:
            for row in csv.DictReader(stream):
                label = LABELS[row["metric"]]
                values = [float(row[key]) for key in ("seed_0", "seed_1", "seed_2", "mean", "std")]
                lines.append(f"{label:<34} {values[0]:>12.6g}   {values[1]:>12.6g}   {values[2]:>12.6g}   "
                             f"{values[3]:.6g} ± {values[4]:.6g}")
        confirmed = [row for row in results if row["learning_rate_label"] == selected["learning_rate_label"] and row["step"] == selected["step"]]
        passes = [row["seed"] for row in confirmed if row["utility"] >= gates["utility_floor"]
                  and row["forget_probability"] <= gates["fprob_ceiling"]]
        lines += ["", f"Seeds satisfying both frozen gates at the selected point {passes}."]
    else:
        lines += ["", "No checkpoint satisfied both frozen gates."]
    lines += [
        "", f"All {expected} checkpoints passed the seven-metric official TOFU and per-sample validations.",
        "Parameter L2 is computed immediately after the recorded optimizer step. Update L2 measures the actual change "
        "in stored trainable parameters across that single optimizer step.",
        "The seed 0 TOFU scores were used for tuning. Seeds 1 and 2 test training-seed stability on the same evaluation data.",
    ]
    (root / "p0_report.txt").write_text("\n".join(lines) + "\n")
    make_figure(root, plan, scored, selected)
    return pack(root, final=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=SWEEP)
    args = parser.parse_args()
    root = args.root.resolve()
    archive = summarize(root)
    print((root / "p0_report.txt").read_text())
    print(archive)


if __name__ == "__main__":
    main()
