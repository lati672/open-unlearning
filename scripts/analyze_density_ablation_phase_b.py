"""Validate the 72 official checkpoints, summarize the paired ablation, and export."""

import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev
from zipfile import ZIP_DEFLATED, ZipFile

from run_density_ablation_phase_b import A, B, SOURCE, WORKSPACE, compare_training
from prepare_density_ablation import DENSITIES, FORMAL, STEPS
from prepare_low_confidence import read, require, rows, write_csv
from run_density_ablation import load_phase
from run_low_confidence import ARTIFACTS, METRICS, TABLE_METRICS, digest, replay_metrics, validate_outputs

from analyze_density_ablation import CONTROL_LABELS, LOWER, PARETO
from matplotlib import pyplot as plt


SEEDS = (0, 1, 2)
EXPECTED_COUNTS = {"forget_Q_A_Prob": 200, "retain_Q_A_Prob": 400, "ra_Q_A_Prob": 100,
                   "wf_Q_A_Prob": 117, "mia_min_k/holdout": 200}
LABELS = {"ES": "ES", "forget_probability": "F-Prob", "forget_rouge": "F-ROUGE",
          "forget_ratio": "F-Ratio", "forget_quality": "F-Quality", "utility": "Utility",
          "privleak_signed": "PrivLeak signed", "privleak_abs": "|PrivLeak|"}
DIAGNOSTICS = ("logged_training_loss", "gradient_l2_preclip", "learning_rate_after_step",
               "optimizer_step_update_l2", "parameter_l2", "learning_rate_used")
ANALYSIS_FILES = ("official_metrics_long.csv", "mean_std.csv", "step120_mean_std.csv", "dynamics_mean_std.csv",
                  "paired_density_comparisons.csv", "density_identity_interactions.csv", "identity_effect_by_density.csv",
                  "trajectory_data.csv", "pareto_frontiers.csv", "phase_b_report.txt", "verification.json",
                  "utility_trajectories.pdf", "utility_trajectories.png")


def validate_masks(phase):
    audit = read(A / "mask_audit/summary.json")
    eligible = {}
    for row in rows(A / "mask_audit/eligible_tokens.csv"):
        eligible.setdefault(int(row["doc_id"]), {})[int(row["position"])] = int(row["token_id"])
    require(len(eligible) == 200 and sum(map(len, eligible.values())) == 7165, "Eligible token population differs")
    selected = {}
    for arm, checksum in phase["mask_hashes"].items():
        path = A / "masks" / f"{arm}.jsonl"
        require(digest(path) == checksum, f"Frozen mask hash differs: {arm}")
        chosen = {}
        for line in path.read_text().splitlines():
            row = json.loads(line)
            doc, mask, tokens = row["doc_id"], row["mask"], row["input_ids"]
            positions = {i for i, value in enumerate(mask) if value}
            require(doc not in chosen and len(mask) == len(tokens) and set(mask) <= {0, 1}
                    and positions <= eligible[doc].keys()
                    and all(tokens[i] == token_id for i, token_id in eligible[doc].items()), f"Invalid mask for {arm}/{doc}")
            chosen[doc] = len(positions)
        require(chosen.keys() == eligible.keys() and sum(chosen.values()) == audit[arm]["selected"]
                and math.isclose(sum(chosen.values()) / 7165, audit[arm]["aggregate_ratio"]), f"Mask totals differ: {arm}")
        selected[arm] = chosen
    for density in DENSITIES:
        require(selected[f"Discrepancy-{density}"] == selected[f"Random-{density}"],
                f"Per-example density or empty-mask pattern differs at {density}")
    return audit


def collect(phase, plan, audit, seeds=SEEDS):
    reference_path = A / "retain_only_reference_TOFU_EVAL.json"
    reference = read(reference_path)
    reference_hash = digest(reference_path)
    protocol = phase["protocol"]
    require(reference_hash == protocol["inputs"][protocol["retain_only_model"]["evaluation_logs"]],
            "Reference hash differs")
    signature = read(A / "v4_evaluation_signature.json")
    phase_a_rows = {(r["variant"], int(r["step"])): r for r in rows(A / "official_metrics_long.csv")}
    data = []
    checks = {}
    for seed in seeds:
        root = A if seed == 0 else B
        for variant in FORMAL:
            job = root / "jobs" / variant
            run = job / f"collected_results/seed{seed}"
            manifest_path = job / "incremental_manifest.json"
            manifest_hash = digest(manifest_path)
            manifest = read(manifest_path)
            expected_seeds = (0,) if seed == 0 else (1, 2)
            require(set(manifest) == {f"seed{s}/checkpoint-{step}" for s in expected_seeds for step in STEPS},
                    f"Archive coverage differs: {variant}/{seed}")
            if seed == 0:
                require(manifest_hash == plan["phase_a_archives_sha256"][variant], f"Phase A archive changed: {variant}")
            else:
                compare_training(B, variant, seed, phase)
            completed = read(run / "training_completed.json")
            require(completed["global_step"] == read(run / "trainer_state.json")["global_step"] == 120
                    and completed["seed"] == seed and completed["mask_sha256"] == phase["mask_hashes"][variant]
                    and completed["code_commit"] == phase["code_commit"]
                    and set(completed["checkpoints"]) == set(map(str, STEPS)), f"Training state differs: {variant}/{seed}")
            for rank in (0, 1):
                numeric = read(run / f"finite_checks_rank{rank}.json")
                require(len(numeric["loss"]) == 480 and len(numeric["gradient_l2_preclip"]) == 120
                        and all(math.isfinite(x) for x in numeric["loss"] + numeric["gradient_l2_preclip"]),
                        f"Nonfinite or incomplete training diagnostics: {variant}/{seed}/{rank}")
            require(not any((job / "runs" / f"seed{seed}").rglob("*.safetensors")), f"Formal weights remain: {variant}/{seed}")
            norms = read(run / "optimizer_update_norms.json")
            for step in STEPS:
                label = f"{variant}/seed{seed}/checkpoint-{step}"
                key = f"seed{seed}/checkpoint-{step}"
                cp = run / f"checkpoint-{step}"
                index = manifest[key]
                expected = index["files"]
                require({p.relative_to(cp).as_posix() for p in cp.rglob("*") if p.is_file()} == set(expected),
                        f"Archive file inventory differs: {label}")
                for name, checksum in expected.items():
                    require(digest(cp / name) == checksum, f"Archive file changed: {label}/{name}")
                folder = cp / "evals/official"
                validated = read(folder / "validated.json")
                provenance = read(folder / "evaluation_provenance.json")
                config = provenance["resolved_eval_config"]
                metadata = read(cp / "checkpoint_metadata.json")
                state = read(cp / "trainer_state.json")
                snapshot = read(cp / "resolved_config.json")
                require(metadata == provenance["checkpoint"] == completed["checkpoints"][str(step)]
                        and all(validated[k] == v for k, v in metadata.items())
                        and validated == index["validated"]
                        and validated["seed"] == seed and validated["global_step"] == state["global_step"] == step
                        and validated["actual_epoch"] == state["epoch"] == phase["actual_epochs"][str(step)]
                        and validated["code_commit"] == phase["code_commit"]
                        and validated["protocol_sha256"] == phase["v4_protocol_sha256"]
                        and validated["mask_sha256"] == phase["mask_hashes"][variant]
                        and validated["evaluator_sha256"] == provenance["evaluator_sha256"] == phase["evaluator_sha256"]
                        and provenance["retain_only_reference_sha256"] == reference_hash
                        and provenance["dataset_revision"] == protocol["dataset"]["revision"]
                        and provenance["model"] == snapshot["model"]
                        and snapshot["extension_manifest_sha256"] == digest(job / "protocol_manifest.json")
                        and validated["training_diagnostics"] == read(cp / "training_diagnostics.json")
                        and {k: v for k, v in config.items() if k not in {"output_dir", "retain_logs_path"}} == signature,
                        f"Checkpoint, training or evaluation provenance differs: {label}")
                require(set(provenance["weight_sha256"]) == set(metadata["saved_weight_files"])
                        and all(len(s) == 64 for s in provenance["weight_sha256"].values()), f"Weight hashes differ: {label}")
                require(set(validated["artifact_sha256"]) == set(ARTIFACTS) - {"validated.json"}
                        and all(expected[f"evals/official/{name}"] == checksum
                                for name, checksum in validated["artifact_sha256"].items()), f"Validation hashes differ: {label}")
                logs = read(folder / "TOFU_EVAL.json")
                metrics, counts = validate_outputs(config, protocol["official_evaluation"], logs,
                                                   read(folder / "TOFU_SUMMARY.json"), reference_path)
                replay = replay_metrics(config, logs, reference)
                require(counts == validated["sample_counts"] and all(counts[k] == v for k, v in EXPECTED_COUNTS.items())
                        and set(metrics) == set(METRICS)
                        and metrics == validated["metrics"] == read(folder / "official_metrics.json")
                        and all(math.isclose(value, replay[k], rel_tol=1e-10, abs_tol=1e-14)
                                for k, value in metrics.items()), f"Independent evaluation checks differ: {label}")
                stats = validated["training_diagnostics"]
                signed = metrics.pop("privleak")
                row = {"variant": variant, "source": f"phase_{'a' if seed == 0 else 'b'}_seed{seed}", "seed": seed,
                       "step": step, "actual_epoch": validated["actual_epoch"],
                       "nominal_density": int(variant.split("-")[-1]), "selection_ratio": audit[variant]["aggregate_ratio"],
                       **metrics, "privleak_signed": signed, "privleak_abs": abs(signed),
                       "logged_training_loss": stats["logged_training_loss"], "gradient_l2_preclip": stats["gradient_l2_preclip"],
                       "learning_rate_after_step": stats["learning_rate_after_step"], **norms[str(step)],
                       "code_commit": phase["code_commit"], "mask_sha256": validated["mask_sha256"],
                       "resolved_config_sha256": digest(cp / "resolved_config.json"),
                       "protocol_sha256": phase["v4_protocol_sha256"], "evaluator_sha256": phase["evaluator_sha256"]}
                require(all(math.isfinite(row[k]) for k in DIAGNOSTICS), f"Nonfinite checkpoint diagnostics: {label}")
                if seed == 0:
                    require(all(row[k] == float(phase_a_rows[variant, step][k])
                                for k in (*TABLE_METRICS, *DIAGNOSTICS, "selection_ratio")), f"Phase A table differs: {label}")
                data.append(row)
                checks[label] = {"archive_manifest_sha256": manifest_hash, "validated_sha256": digest(folder / "validated.json"),
                                 "sample_counts": counts, "all_metrics_replayed": True}
    require(len(data) == len(checks) == len(seeds) * len(FORMAL) * len(STEPS), "Checkpoint coverage differs")
    return data, checks


def summary(values):
    require(len(values) == len(SEEDS) and all(math.isfinite(x) for x in values), "Expected three finite seed values")
    return {**{f"seed{seed}": value for seed, value in zip(SEEDS, values)},
            "mean": mean(values), "sample_sd": stdev(values), "n": len(values)}


def paired_summary(values):
    signs = tuple("+" if x > 0 else "-" if x < 0 else "0" for x in values)
    return {**summary(values), "seed_signs": "".join(signs), "same_direction_all_seeds": len(set(signs)) == 1}


def summarize(data, control_rows):
    index = {(r["variant"], r["step"], r["seed"]): r for r in data}
    summaries, trajectory = [], []
    for variant in FORMAL:
        for step in STEPS:
            runs = [index[variant, step, seed] for seed in SEEDS]
            common = {k: runs[0][k] for k in ("variant", "step", "actual_epoch", "nominal_density", "selection_ratio")}
            metric_summaries = {k: summary([r[k] for r in runs]) for k in (*TABLE_METRICS, *DIAGNOSTICS)}
            summaries.extend({**common, "metric": k, **values} for k, values in metric_summaries.items())
            trajectory.append({"source": "density_ablation", **common,
                               **{k: metric_summaries[k]["mean"] for k in TABLE_METRICS},
                               **{k + "_sample_sd": metric_summaries[k]["sample_sd"] for k in TABLE_METRICS}, "n": 3})
    for name, label in CONTROL_LABELS.items():
        for step in (60, 120):
            runs = sorted([r for r in control_rows if r["variant"] == name and int(r["step"]) == step], key=lambda r: r["seed"])
            require([int(r["seed"]) for r in runs] == list(SEEDS), f"Archived control coverage differs: {label}/{step}")
            stats = {k: summary([float(r[k]) for r in runs]) for k in TABLE_METRICS}
            trajectory.append({"source": "archived_control_multiseed", "variant": label, "step": step,
                               "actual_epoch": float(runs[0]["actual_epoch"]), "nominal_density": "",
                               "selection_ratio": float(runs[0]["selection_ratio"]), **{k: stats[k]["mean"] for k in TABLE_METRICS},
                               **{k + "_sample_sd": stats[k]["sample_sd"] for k in TABLE_METRICS}, "n": 3})
    pairs, interactions, trends = [], [], []
    for step in STEPS:
        for metric in TABLE_METRICS:
            diffs = {}
            for density in DENSITIES:
                lhs, rhs = [[index[f"{arm}-{density}", step, seed] for seed in SEEDS] for arm in ("Discrepancy", "Random")]
                values = [a[metric] - b[metric] for a, b in zip(lhs, rhs)]
                diffs[density] = values
                pairs.append({"step": step, "nominal_density": density, "selection_ratio": lhs[0]["selection_ratio"],
                              "metric": metric, **paired_summary(values)})
            for lo, hi in ((15, 34), (34, 50), (15, 50)):
                values = [b - a for a, b in zip(diffs[lo], diffs[hi])]
                interactions.append({"step": step, "metric": metric, "lower_density": lo, "higher_density": hi,
                                     **paired_summary(values)})
            pooled = [mean(diffs[density]) for density in DENSITIES]
            magnitude = list(map(abs, pooled))
            trend = ("constant" if len(set(magnitude)) == 1 else "increases" if sorted(magnitude) == magnitude else
                     "decreases" if sorted(magnitude, reverse=True) == magnitude else "nonmonotonic")
            trends.append({"step": step, "metric": metric,
                           **{f"mean_diff_{density}": value for density, value in zip(DENSITIES, pooled)},
                           "absolute_effect_trend": trend, "mean_sign_reversal": min(pooled) < 0 < max(pooled)})
    return summaries, trajectory, pairs, interactions, trends


def frontiers(trajectory):
    def dominates(a, b, metrics):
        differences = [b[k] - a[k] if k in LOWER else a[k] - b[k] for k in metrics]
        return all(d >= 0 for d in differences) and any(d > 0 for d in differences)

    records = []
    for population in ("density_ablation", "with_archived_controls"):
        subset = [r for r in trajectory if population == "with_archived_controls" or r["source"] == "density_ablation"]
        for scope in (*STEPS, "all_observed"):
            cohort = [r for r in subset if scope == "all_observed" or r["step"] == scope]
            for label, metrics in PARETO.items():
                for row in cohort:
                    dominators = [f"{r['variant']}@{r['step']}" for r in cohort if dominates(r, row, metrics)]
                    records.append({"population": population, "scope": scope, "objectives": label, **row,
                                    "pareto": not dominators, "dominated_by": "|".join(dominators)})
    return records


def plot_trajectories(trajectory, pareto):
    panels = (("forget_probability", "F-Prob ↓", "fprob_vs_utility"),
              ("forget_quality", "F-Quality ↑ (log scale)", "fquality_vs_utility"),
              ("privleak_abs", "|PrivLeak| ↓", "privleak_vs_utility"))
    colors = {15: "#25747e", 34: "#bd573b", 50: "#674d9a"}
    positive = [r["forget_quality"] for r in trajectory if r["forget_quality"] > 0]
    floor = min(positive, default=1.0) / 10
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
    for ax, (metric, ylabel, objective) in zip(axes, panels):
        def values(points):
            return [max(r[metric], floor) if metric == "forget_quality" else r[metric] for r in points]

        for arm in FORMAL:
            points = [r for r in trajectory if r["variant"] == arm]
            color = colors[points[0]["nominal_density"]]
            is_random = arm.startswith("Random")
            marker = "s" if is_random else "o"
            ax.plot([r["utility"] for r in points], values(points), label=arm, color=color,
                    linestyle="--" if is_random else "-", marker=marker, markersize=4)
            ax.scatter([points[-1]["utility"]], values(points[-1:]), marker=marker, s=49,
                       color=color, edgecolor="white", linewidth=.55, zorder=4)
        controls = [r for r in trajectory if r["source"] == "archived_control_multiseed" and r["step"] == 120]
        ax.scatter([r["utility"] for r in controls], values(controls), marker="x", color="#626a73", s=22,
                   label="Archived controls at Step 120")
        frontier = sorted([r for r in pareto if r["population"] == "density_ablation" and r["scope"] == 120
                           and r["objectives"] == objective and r["pareto"]], key=lambda r: r["utility"])
        ax.plot([r["utility"] for r in frontier], values(frontier), ls=":", color="#202830",
                label="Step 120 Pareto frontier for density arms", linewidth=1.2)
        ax.set(xlabel="Utility ↑", ylabel=ylabel)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.22)
        if metric == "forget_quality":
            ax.set_yscale("log")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.subplots_adjust(left=.065, right=.995, bottom=.14, top=.77, wspace=.30)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .92), ncol=4, frameon=False, fontsize=8)
    fig.suptitle("TOFU Forget05 · Means across seeds 0, 1 and 2 · Steps 20, 40, 60 and 120", y=.98, fontsize=11)
    fig.savefig(B / "utility_trajectories.pdf")
    fig.savefig(B / "utility_trajectories.png", dpi=195)
    plt.close(fig)


def report(trajectory, pairs, interactions, trends, pareto):
    lines = [
        "TOFU Forget05 Adaptive discrepancy density ablation · Phase B",
        "",
        "Token-normalized NPO across seeds 0, 1 and 2. Seed 0 comes from Phase A.",
        "Phase A training source, optimizer, scheduler, target, NPO reference, evaluator and masks were reused.",
        "The primary endpoint is Step 120 (19.2 dataloader epochs). Steps 20, 40 and 60 describe training dynamics.",
        "The analysis reports all three matched densities. ZeroThreshold remains a mask-only diagnostic (94.61%).",
        "",
        "Step 120 arm means",
        "Arm                  Density       ES    F-Prob   F-ROUGE   F-Ratio    F-Quality   Utility    |PrivLeak|",
    ]
    for row in trajectory:
        if row["source"] == "density_ablation" and row["step"] == 120:
            lines.append(f"{row['variant']:<20}{row['selection_ratio']:>8.2%}"
                         f"{row['ES']:>9.4f}{row['forget_probability']:>10.4f}{row['forget_rouge']:>10.4f}"
                         f"{row['forget_ratio']:>10.4f}{row['forget_quality']:>13.4g}"
                         f"{row['utility']:>10.4f}{row['privleak_abs']:>13.4f}")
    lines += ["", "Step 120 paired effects (Discrepancy minus Random)",
              "Each entry shows mean ± sample SD and the signs for seeds 0, 1 and 2.",
              f"{'Metric':<17}{'15% nominal':>31}{'34% nominal':>31}{'50% nominal':>31}"]
    lookup = {(r["step"], r["metric"], r["nominal_density"]): r for r in pairs}
    for metric in TABLE_METRICS:
        values = [lookup[120, metric, density] for density in DENSITIES]
        text = [f"{r['mean']:+.4g} ± {r['sample_sd']:.3g} ({r['seed_signs']})" for r in values]
        lines.append(f"{LABELS[metric]:<17}" + "".join(f"{t:>31}" for t in text))
    lines += ["", "Density-by-identity interactions at Step 120",
              "Contrasts subtract the low-density paired effect from the high-density paired effect.",
              "The statistics are descriptive across three seeds.",
              f"{'Metric':<17}{'34 minus 15':>28}{'50 minus 34':>28}{'50 minus 15':>28}  Absolute mean trend"]
    lookup = {(r["metric"], r["lower_density"], r["higher_density"]): r for r in interactions if r["step"] == 120}
    for row in trends:
        if row["step"] == 120:
            values = [lookup[row["metric"], lo, hi] for lo, hi in ((15, 34), (34, 50), (15, 50))]
            text = [f"{r['mean']:+.4g} ± {r['sample_sd']:.3g} ({r['seed_signs']})" for r in values]
            lines.append(f"{LABELS[row['metric']]:<17}" + "".join(f"{t:>28}" for t in text)
                         + f"  {row['absolute_effect_trend']}")
    lines += ["", "Training dynamics",
              f"{'Step':>4}  {'Density':>7}  {'F-Prob mean (signs)':>22}  {'F-Quality mean (signs)':>22}"
              f"  {'Utility mean (signs)':>22}  {'|PrivLeak| mean (signs)':>22}"]
    for step in (20, 40, 60):
        for density in DENSITIES:
            values = [next(r for r in pairs if r["step"] == step and r["nominal_density"] == density and r["metric"] == metric)
                      for metric in ("forget_probability", "forget_quality", "utility", "privleak_abs")]
            text = [f"{r['mean']:+.4g} ({r['seed_signs']})" for r in values]
            lines.append(f"{step:>4}  {density:>6}%" + "".join(f"  {t:>22}" for t in text))
    lines += ["", "Observed Step 120 Pareto frontiers across the six density arms",
              "Frontiers compare mean operating points. Utility and F-Quality are maximized. F-Prob and |PrivLeak| are minimized."]
    for objective in PARETO:
        variants = [r["variant"] for r in pareto if r["population"] == "density_ablation" and r["scope"] == 120
                    and r["objectives"] == objective and r["pareto"]]
        lines.append(f"{objective:<24} {', '.join(variants)}")
    lines += ["", "The frontier CSV also includes all observed steps and a separate cohort with archived controls.",
              "Mean absolute PrivLeak averages seed-wise magnitudes. The signed OpenUnlearning value is retained.",
              "All 72 official checkpoints passed per-sample validation, metric replay and archive hash checks.",
              "See the CSVs for raw seed values, sample SDs, paired signs and complete training dynamics."]
    (B / "phase_b_report.txt").write_text("\n".join(lines) + "\n")


def pack(verification):
    files = {name: B / name for name in ANALYSIS_FILES}
    files.update({name: B / name for name in ("protocol_manifest.json", "phase_b_manifest.json", "status.json",
                  "existing_controls_multiseed.csv", "v4_evaluation_signature.json", "retain_only_reference_TOFU_EVAL.json",
                  "initial_target_TOFU_SUMMARY.json", "unit_tests.log")})

    def include(folder, prefix):
        for path in folder.rglob("*"):
            if path.is_file() and not path.is_symlink() and "__pycache__" not in path.parts:
                files[f"{prefix}/{path.relative_to(folder).as_posix()}"] = path

    include(A / "masks", "masks")
    include(A / "mask_audit", "mask_audit")
    for phase, root in (("phase_a", A), ("phase_b", B)):
        include(root / "logs", f"{phase}/logs")
        files[f"{phase}/queue.log"] = root / "queue.log"
        for arm in FORMAL:
            job = root / "jobs" / arm
            for name in ("protocol_manifest.json", "incremental_manifest.json"):
                files[f"{phase}/runs/{arm}/{name}"] = job / name
            include(job / "collected_results", f"{phase}/runs/{arm}")
    for name in ("phase_a_report.txt", "official_metrics_long.csv", "smoke_validation.json"):
        files[f"phase_a/{name}"] = A / name
    for folder in ("configs", "scripts", "src", "tests"):
        include(SOURCE / folder, f"frozen_source/{folder}")
    for name in ("LICENSE", "setup.py", "requirements.txt"):
        files[f"frozen_source/{name}"] = SOURCE / name
    for name in ("run_density_ablation_phase_b.py", Path(__file__).name):
        files[f"phase_b/scripts/{name}"] = WORKSPACE / "scripts" / name
    require(sum(name.endswith("/TOFU_EVAL.json") for name in files) == 72, "ZIP evaluation count differs")
    require(not any(name.endswith((".safetensors", ".pt", ".safetensors.index.json")) for name in files), "Unexpected weights in ZIP")
    hashes = {name: digest(path) for name, path in files.items()}
    export = B.parents[1] / "exports" / f"{B.name}.zip"
    pending = export.with_suffix(".zip.part")
    manifest = {**{k: verification[k] for k in ("runs", "official_checkpoints", "validated_checkpoints", "training_git_commit")},
                "files_sha256": hashes}
    with ZipFile(pending, "w", ZIP_DEFLATED, compresslevel=4) as zip_file:
        for name, path in sorted(files.items()):
            zip_file.write(path, f"{B.name}/{name}")
        zip_file.writestr(f"{B.name}/MANIFEST.json", json.dumps(manifest, indent=2) + "\n")
    with ZipFile(pending) as zip_file:
        require(len(zip_file.namelist()) == len(hashes) + 1 and zip_file.testzip() is None, "ZIP CRC or inventory differs")
        require(json.loads(zip_file.read(f"{B.name}/MANIFEST.json")) == manifest, "ZIP manifest differs")
        for name, checksum in hashes.items():
            with zip_file.open(f"{B.name}/{name}") as member:
                require(hashlib.file_digest(member, "sha256").hexdigest() == checksum, f"ZIP member differs: {name}")
    checksum = digest(pending)
    pending.replace(export)
    export.with_suffix(".zip.sha256").write_text(f"{checksum}  {export.name}\n")
    print(f"{export}\n{export.stat().st_size} bytes, SHA-256 {checksum}\nVerified {len(hashes)} members plus MANIFEST.json")


def main():
    status = read(B / "status.json")
    require(status["phase"] == "training_complete" and status["new_official_checkpoints"] == 48, "Phase B queue has not finished")
    phase, plan = load_phase(A), read(B / "phase_b_manifest.json")
    require(digest(A / "protocol_manifest.json") == digest(B / "protocol_manifest.json") == plan["phase_a_protocol_sha256"]
            and digest(A / "official_metrics_long.csv") == plan["phase_a_seed0_metrics_sha256"]
            and digest(B / "existing_controls_multiseed.csv") == plan["control_table_sha256"]
            and digest(WORKSPACE / "scripts/run_density_ablation_phase_b.py") == plan["phase_b_launcher_sha256"]
            and plan["masks_reused_sha256"] == phase["mask_hashes"]
            and plan["primary_endpoint"] == 120 and plan["secondary_training_dynamics_steps"] == [20, 40, 60],
            "Frozen Phase B analysis inputs differ")
    audit = validate_masks(phase)
    data, checks = collect(phase, plan, audit)
    summaries, trajectory, pairs, interactions, trends = summarize(data, rows(B / "existing_controls_multiseed.csv"))
    pareto = frontiers(trajectory)
    for name, records in (("official_metrics_long", data), ("mean_std", summaries),
                          ("step120_mean_std", [r for r in summaries if r["step"] == 120]),
                          ("dynamics_mean_std", [r for r in summaries if r["step"] != 120]),
                          ("trajectory_data", trajectory), ("paired_density_comparisons", pairs),
                          ("density_identity_interactions", interactions), ("identity_effect_by_density", trends),
                          ("pareto_frontiers", pareto)):
        write_csv(B / f"{name}.csv", records)
    plot_trajectories(trajectory, pareto)
    report(trajectory, pairs, interactions, trends, pareto)
    verification = {"runs": len({(r["variant"], r["seed"]) for r in data}), "official_checkpoints": len(data),
                    "validated_checkpoints": len(checks), "new_phase_b_checkpoints": 48, "phase_a_checkpoints": 24,
                    "training_git_commit": phase["code_commit"], "primary_step": 120, "secondary_steps": [20, 40, 60],
                    "phase_b_manifest_sha256": digest(B / "phase_b_manifest.json"), "mask_audit": audit,
                    "same_per_example_counts_and_empty_mask_pattern": True, "masks_reused": True,
                    "official_metrics_replayed": True, "formal_model_weights_remaining": 0,
                    "expected_sample_counts": EXPECTED_COUNTS, "checkpoints": checks}
    (B / "verification.json").write_text(json.dumps(verification, indent=2, allow_nan=False) + "\n")
    pack(verification)


if __name__ == "__main__":
    main()
