"""Analyze the TOFU Forget05 density ablation at matched observed strength."""

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
from statistics import mean, stdev
from zipfile import ZipFile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np


WORKSPACE = Path(__file__).resolve().parents[1]
EXPERIMENTS = WORKSPACE / "saves/experiments"
INPUT = EXPERIMENTS / "tofu_forget05_adaptive_density_phase_b"
OUTPUT = EXPERIMENTS / "tofu_forget05_matched_strength_v1"
ARCHIVE = WORKSPACE / "saves/exports/tofu_forget05_adaptive_density_phase_b.zip"
SEEDS = (0, 1, 2)
DENSITIES = (15, 34, 50)
STEPS = {20: 3.2, 40: 6.4, 60: 9.6, 120: 19.2}
FAMILIES = ("Discrepancy", "Random")
POSITIONS = (.25, .5, .75)
CALIPERS = (.005, .01, .02)
GRID_POINTS = 201
MODES = {"utility": ("Utility", "F-Prob"), "forgetting": ("F-Prob", "Utility")}
COLUMNS = {"Utility": "utility", "F-Prob": "forget_probability", "ES": "ES", "F-ROUGE": "forget_rouge",
           "F-Ratio": "forget_ratio", "F-Quality": "forget_quality", "absolute PrivLeak": "privleak_abs",
           "signed PrivLeak": "privleak_signed"}
SECONDARY = ("ES", "F-ROUGE", "F-Ratio", "absolute PrivLeak")
COLORS = {"Discrepancy": "#006d77", "Random": "#b14632"}
DASHES = ("-", "--", ":")
PAPER_CAUTION = ("After matching observed forgetting or utility levels, the relative advantage is operating-point "
                 "dependent; masking changes the attainable trade-off but does not uniformly dominate Random selection.")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows, fields=None):
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


def read_input(directory, archive_path):
    names = ("official_metrics_long.csv", "protocol_manifest.json")
    if not all((directory / name).is_file() for name in names):
        require(archive_path.is_file(), "The local CSV/protocol and the Phase B archive are both unavailable")
        source = str(archive_path)
        with ZipFile(archive_path) as archive:
            prefix = f"{archive_path.stem}/"
            manifest = json.loads(archive.read(prefix + "MANIFEST.json"))["files_sha256"]
            data = {name: archive.read(prefix + name) for name in names}
            require(all(sha(raw) == manifest[name] for name, raw in data.items()), "Input differs from the ZIP manifest")
        return source, data, True

    source = str(directory)
    data = {name: (directory / name).read_bytes() for name in names}
    checked = False
    if archive_path.is_file():
        with ZipFile(archive_path) as archive:
            prefix = f"{archive_path.stem}/"
            manifest = json.loads(archive.read(prefix + "MANIFEST.json"))["files_sha256"]
        require(all(sha(raw) == manifest[name] for name, raw in data.items()), "Local input differs from the Phase B export")
        checked = True
    return source, data, checked


def validate(data, source, archive_checked):
    phase = json.loads(data["protocol_manifest.json"])
    arms = {f"{family}-{density}" for family in FAMILIES for density in DENSITIES}
    expected = {(arm, seed, step) for arm in arms for seed in SEEDS for step in STEPS}
    require(set(phase["formal_variants"]) == arms and phase["actual_epochs"] == {str(k): v for k, v in STEPS.items()},
            "Protocol arm or epoch inventory differs")
    expected_hashes = {"code_commit": phase["code_commit"], "protocol_sha256": phase["v4_protocol_sha256"],
                       "evaluator_sha256": phase["evaluator_sha256"]}
    inputs = list(csv.DictReader(io.StringIO(data["official_metrics_long.csv"].decode())))
    points, seen, ratios = [], set(), {}
    for row in inputs:
        variant, seed, step = row["variant"], int(row["seed"]), int(row["step"])
        key = (variant, seed, step)
        require(key in expected and key not in seen, f"Unexpected or duplicate checkpoint {key}")
        seen.add(key)
        family, density = variant.split("-")
        density = int(density)
        epoch = float(row["actual_epoch"])
        require(int(row["nominal_density"]) == density and epoch == STEPS[step], f"Incorrect density or epoch {key}")
        require(all(row[k] == checksum for k, checksum in expected_hashes.items())
                and row["mask_sha256"] == phase["mask_hashes"][variant], f"Provenance differs for {key}")
        metrics = {name: float(row[column]) for name, column in COLUMNS.items()}
        require(all(map(math.isfinite, metrics.values())) and metrics["F-Ratio"] < 1
                and metrics["absolute PrivLeak"] == abs(metrics["signed PrivLeak"]), f"Invalid official metrics at {key}")
        ratio = float(row["selection_ratio"])
        require(math.isfinite(ratio) and 0 < ratio <= 1, f"Invalid selection ratio {key}")
        ratios.setdefault(density, set()).add(ratio)
        points.append({"seed": seed, "family": family, "variant": variant, "density": density, "step": step,
                       "actual_epoch": epoch, "selection_ratio": ratio, **metrics})
    require(seen == expected and len(points) == 72, "Expected every combination in the 72-checkpoint plan")
    require(all(len(values) == 1 for values in ratios.values()), "Selection ratios differ across families, seeds or steps")
    return points, {
        "passed": True, "source": source, "input_sha256": {name: sha(raw) for name, raw in data.items()},
        "matches_official_archive_manifest": archive_checked, "checkpoints": len(points), "arms": sorted(arms),
        "seeds": SEEDS, "steps_and_actual_epochs": STEPS, "selection_ratio_by_nominal_density": {
            density: next(iter(values)) for density, values in sorted(ratios.items())},
        **expected_hashes, "frozen_mask_hashes": {k: phase["mask_hashes"][k] for k in sorted(arms)},
        "official_metrics_finite": True, "strength_variable": "F-Prob", "retention_variable": "Utility",
        "signed_and_absolute_privleak_consistent": True,
    }


def frontier(points):
    ranked = sorted(points, key=lambda p: (-p["Utility"], p["F-Prob"], p["step"], p["density"]))
    kept = []
    for point in ranked:
        if not kept or point["F-Prob"] < kept[-1]["F-Prob"]:
            kept.append(point)
    return list(reversed(kept))


def bounds(points, key):
    values = [p[key] for p in points]
    return min(values), max(values)


def intersection(ranges):
    lower, upper = max(r[0] for r in ranges), min(r[1] for r in ranges)
    return (lower, upper) if lower <= upper else None


def interpolate(points, x, target, metrics):
    coordinates = np.array([p[x] for p in points])
    require(coordinates[0] <= target <= coordinates[-1], "Requested extrapolation")
    right = int(np.searchsorted(coordinates, target))
    if coordinates[right] == target:
        left = right
        weight = 0.
    else:
        left = right - 1
        weight = float((target - coordinates[left]) / (coordinates[right] - coordinates[left]))
    a, b = points[left], points[right]
    values = {k: a[k] + weight * (b[k] - a[k]) for k in metrics}
    values.update({"lower_variant": a["variant"], "lower_step": a["step"], "upper_variant": b["variant"],
                   "upper_step": b["step"], "upper_weight": weight, "observed": left == right,
                   "F-Quality_lower_observed": a["F-Quality"], "F-Quality_upper_observed": b["F-Quality"],
                   "F-Quality_at_target_if_observed": a["F-Quality"] if left == right else None})
    return values


def effect_grid(curves, x, y, support):
    targets = np.linspace(*support, GRID_POINTS)
    values = []
    for family in FAMILIES:
        xs, ys = ([p[k] for p in curves[family]] for k in (x, y))
        require(xs[0] <= targets[0] <= targets[-1] <= xs[-1], "Grid goes outside frontier support")
        values.append(np.interp(targets, xs, ys))
    return targets, values[0] - values[1]


def support_row(curves, x):
    ranges = {family: bounds(curves[family], x) for family in FAMILIES}
    common = intersection(ranges.values())
    return {"match_axis": x, **{f"{family}_{label}": value
                               for family, interval in ranges.items()
                               for label, value in zip(("min", "max"), interval)},
            "common_min": common[0] if common else None, "common_max": common[1] if common else None,
            "status": "point only" if common and common[0] == common[1] else "ok" if common else "no common support"}, common


def matched(curves, seed, mode, density=None):
    x, y = MODES[mode]
    support, common = support_row(curves, x)
    label = {"seed": seed, "analysis": mode, "density": density if density is not None else "pooled"}
    support = {**label, **support}
    if common is None:
        return support, [], None
    results = []
    for position in POSITIONS:
        target = common[0] + position * (common[1] - common[0])
        metrics = (y, *SECONDARY)
        values = {family: interpolate(curves[family], x, target, metrics) for family in FAMILIES}
        row = {**label, "normalized_position": position, "target": target, "matched_metric": x, "primary_outcome": y}
        row.update({f"{family}_{k}": v for family in FAMILIES for k, v in values[family].items()})
        row.update({f"delta_{k}": values[FAMILIES[0]][k] - values[FAMILIES[1]][k] for k in metrics})
        results.append(row)
    auc_name = "normalized_AUC_FProb" if mode == "utility" else "normalized_AUC_Utility"
    grid, deltas = effect_grid(curves, x, y, common)
    auc = {**label, "metric": auc_name, "value": float(deltas.mean()), "grid_points": len(grid),
           "common_min": common[0], "common_max": common[1],
           "effect_min_on_grid": float(deltas.min()), "effect_max_on_grid": float(deltas.max())}
    return support, results, auc


def numeric_summary(values):
    present = [v for v in values if v is not None]
    signs = "".join("+" if v > 0 else "-" if v < 0 else "0" for v in present)
    return {**{f"seed{s}": v for s, v in zip(SEEDS, values)}, "mean": mean(present) if present else None,
            "sample_sd": stdev(present) if len(present) >= 2 else None, "n": len(present), "signs": signs,
            "direction_consistent_all_seeds": len(signs) == len(SEEDS) and len(set(signs)) == 1}


def matched_summary(rows, auc, mode):
    _, y = MODES[mode]
    fields = ("target", *(f"{family}_{k}" for family in FAMILIES for k in (y, *SECONDARY)),
              *(f"delta_{k}" for k in (y, *SECONDARY)))
    items = []
    for pos in POSITIONS:
        indexed = {r["seed"]: r for r in rows if r["normalized_position"] == pos}
        for field in fields:
            items.append({"normalized_position": pos, "quantity": field,
                          **numeric_summary([indexed[s][field] if s in indexed else None for s in SEEDS])})
    for metric in sorted({r["metric"] for r in auc}):
        indexed = {r["seed"]: r["value"] for r in auc if r["metric"] == metric}
        items.append({"normalized_position": "grid_mean", "quantity": metric,
                      **numeric_summary([indexed.get(seed) for seed in SEEDS])})
    return items


def nearest(groups, mode):
    x, y = MODES[mode]
    output, summaries = [], []
    for seed in SEEDS:
        d, r = [groups[seed, family] for family in FAMILIES]
        common = intersection([bounds(d, x), bounds(r, x)])
        candidates = [p for p in r if common and common[0] <= p[x] <= common[1]]
        eligible = [p for p in d if common and common[0] <= p[x] <= common[1]]
        for caliper in CALIPERS:
            accepted = []
            for point in d:
                category = ("no common support" if common is None else
                            "outside common support" if point not in eligible else
                            "no Random point in common support" if not candidates else "candidate")
                row = {"seed": seed, "analysis": mode, "caliper": caliper, "match_axis": x, "outcome": y,
                       "discrepancy_variant": point["variant"], "discrepancy_step": point["step"],
                       "discrepancy_strength": point[x], "discrepancy_outcome": point[y],
                       "common_min": common[0] if common else None, "common_max": common[1] if common else None}
                if category == "candidate":
                    match = min(candidates, key=lambda p: (abs(point[x] - p[x]), p["step"], p["density"]))
                    gap = abs(point[x] - match[x])
                    category = "matched" if gap <= caliper else "outside caliper"
                    row.update(random_variant=match["variant"], random_step=match["step"], random_strength=match[x],
                               absolute_gap=gap, signed_gap=point[x] - match[x])
                    if category == "matched":
                        row.update(random_outcome=match[y], difference=point[y] - match[y])
                        accepted.append(row)
                output.append({**row, "status": category})
            summaries.append({"seed": seed, "analysis": mode, "caliper": caliper, "match_axis": x, "outcome": y,
                              "discrepancy_total": len(d), "discrepancy_in_support": len(eligible),
                              "random_in_support": len(candidates), "matched": len(accepted),
                              "mean_absolute_gap": mean(z["absolute_gap"] for z in accepted) if accepted else None,
                              "max_absolute_gap": max(z["absolute_gap"] for z in accepted) if accepted else None,
                              "mean_difference": mean(z["difference"] for z in accepted) if accepted else None,
                              "min_difference": min(z["difference"] for z in accepted) if accepted else None,
                              "max_difference": max(z["difference"] for z in accepted) if accepted else None,
                              "positive_pairs": sum(z["difference"] > 0 for z in accepted),
                              "negative_pairs": sum(z["difference"] < 0 for z in accepted),
                              "tied_pairs": sum(z["difference"] == 0 for z in accepted)})
    return output, summaries


def display_number(value):
    return "NA" if value is None else f"{value:+.5f}"


def directional_conclusion(auc_rows, matched_rows, nearest_summary):
    desired = {"utility": -1, "forgetting": 1}
    for mode, sign in desired.items():
        y = MODES[mode][1]
        ranked = [r for r in auc_rows if r["analysis"] == mode]
        matches = matched_rows[mode]
        observed = [r for r in nearest_summary if r["analysis"] == mode and r["caliper"] == .01]
        if (len(ranked) != 3 or len(matches) != 9 or len(observed) != 3
                or any(sign * r["value"] <= 0 for r in ranked)
                or any(sign * r[f"delta_{y}"] <= 0 for r in matches)
                or any(not r["matched"] or sign * r["mean_difference"] <= 0
                       or (r["negative_pairs"] if sign > 0 else r["positive_pairs"]) for r in observed)):
            return False
    return True


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=.17)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))


def save_figure(fig, output, name):
    fig.savefig(output / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(output / f"{name}.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_frontiers(curves, output):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.7), sharex=True, sharey=True, layout="constrained")
    # A mean line exists only at Utility values supported by all three seed curves in that family.
    for family, ax in zip(FAMILIES, axes):
        color = COLORS[family]
        for seed in SEEDS:
            points = curves[seed, family]
            ax.plot([p["Utility"] for p in points], [p["F-Prob"] for p in points],
                    marker="o", ms=4, color=color, lw=1.3, alpha=.4, ls=DASHES[seed])
        common = intersection([bounds(curves[s, family], "Utility") for s in SEEDS])
        if common:
            grid = np.linspace(*common, GRID_POINTS)
            means = [np.interp(grid, [p["Utility"] for p in curves[s, family]],
                               [p["F-Prob"] for p in curves[s, family]]) for s in SEEDS]
            ax.plot(grid, np.mean(means, axis=0), color=color, lw=2.8, zorder=2)
        labels = {}
        for seed in SEEDS:
            for p in curves[seed, family]:
                labels.setdefault((p["density"], p["step"]), []).append(p)
        for label, points in sorted(labels.items()):
            x = mean(p["Utility"] for p in points)
            y = mean(p["F-Prob"] for p in points)
            displacement = {(15, 20): (-8, 16), (34, 20): (-9, -15), (50, 20): (8, 5)}
            dx, dy = displacement.get(label, (5, 4)) if family == "Discrepancy" else (-5, 5)
            ax.annotate(f"{label[0]}% / {label[1]}", (x, y), (dx, dy), textcoords="offset points",
                        color=color, fontsize=7, ha="right" if dx < 0 else "left",
                        va="top" if dy < 0 else "bottom",
                        arrowprops={"arrowstyle": "-", "color": color, "alpha": .4, "lw": .6})
        ax.set(title=family, xlabel="Utility ↑")
        style(ax)
    axes[0].set_ylabel("F-Prob ↓")
    legend = [Line2D([0], [0], color="#4c6066", lw=1.3, ls=DASHES[s], alpha=.65, label=f"Seed {s}") for s in SEEDS]
    legend.append(Line2D([0], [0], color="#4c6066", lw=2.8, label="Mean on common seed support"))
    axes[0].legend(handles=legend, loc="upper left", fontsize=8, frameon=False)
    fig.suptitle("Observed forgetting–utility frontiers\nLabels show nominal density and optimizer step", fontsize=14)
    save_figure(fig, output, "matched_strength_fprob_vs_utility")


def plot_effects(curves, results, support, output, mode):
    x, y = MODES[mode]
    fig, ax = plt.subplots(figsize=(7.1, 4.4), layout="constrained")
    ax.axhline(0, color="#7b7b7b", lw=1)
    colors = ("#006d77", "#75518c", "#b14632")
    intervals = []
    for seed, color in zip(SEEDS, colors):
        interval = support[seed]
        if interval is None:
            continue
        intervals.append(interval)
        local = {family: curves[seed, family] for family in FAMILIES}
        grid, effects = effect_grid(local, x, y, interval)
        ax.plot(grid, effects, lw=1.5, color=color, ls=DASHES[seed], label=f"Seed {seed}")
        pts = [r for r in results if r["seed"] == seed]
        ax.scatter([r["target"] for r in pts], [r[f"delta_{y}"] for r in pts], color=color, s=24, zorder=3)
    if len(intervals) == len(SEEDS) and (common := intersection(intervals)):
        grid = np.linspace(*common, GRID_POINTS)
        values = np.stack([effect_grid({f: curves[s, f] for f in FAMILIES}, x, y, common)[1] for s in SEEDS])
        ax.plot(grid, values.mean(axis=0), lw=2.6, color="#27323b", label="Mean on common seed support")
    ax.set(xlabel=f"Matched {x}", ylabel=f"Δ {y} (Discrepancy − Random)",
           title="Matched utility" if mode == "utility" else "Matched forgetting")
    ax.legend(loc="best", frameon=False, fontsize=8)
    style(ax)
    save_figure(fig, output, f"matched_{mode}_effects")


def plot_privacy(results, output):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True, layout="constrained")
    for ax, (mode, (x, _)) in zip(axes, MODES.items()):
        for family in FAMILIES:
            for seed in SEEDS:
                pts = [r for r in results[mode] if r["seed"] == seed]
                ax.plot([p["target"] for p in pts], [p[f"{family}_absolute PrivLeak"] for p in pts],
                        color=COLORS[family], ls=DASHES[seed], marker="o", ms=4, lw=1.5,
                        label=family if seed == 0 else None)
        ax.set(xlabel=f"Matched {x}", title="Matched utility" if mode == "utility" else "Matched forgetting")
        style(ax)
    axes[0].set_ylabel("Absolute PrivLeak ↓")
    handles = [Line2D([0], [0], color=COLORS[f], label=f, lw=2) for f in FAMILIES]
    handles += [Line2D([0], [0], color="#4c6066", label=f"Seed {s}", ls=DASHES[s]) for s in SEEDS]
    axes[0].legend(handles=handles, frameon=False, fontsize=8, ncol=2)
    fig.suptitle("Privacy at the same matched points", fontsize=14)
    save_figure(fig, output, "privacy_at_matched_strength")


def report(output, validation, fronts, support, within, summaries, nn_summary, strong):
    lines = [
        "TOFU Forget05 matched-strength analysis",
        "",
        "This is a post-hoc descriptive analysis of 72 existing official checkpoints. No new training was run.",
        "The primary comparison uses the observed Pareto frontier within each seed and family across the three densities and four steps.",
        f"The inputs share commit {validation['code_commit']}, protocol SHA-256 and evaluator SHA-256.",
        "Paired selection ratios match exactly within each density across seeds and steps. See input_validation.json for hashes and coverage.",
        "",
        "Observed frontier coverage",
        "Seed  Discrepancy points  Random points  Common Utility        Common F-Prob",
    ]
    for seed in SEEDS:
        ranges = [next(s for s in support if s["seed"] == seed and s["analysis"] == mode) for mode in MODES]
        intervals = [f"[{r['common_min']:.6f}, {r['common_max']:.6f}]" if r["common_min"] is not None
                     else "no common support" for r in ranges]
        lines.append(f"{seed:>4}{len(fronts[seed, 'Discrepancy']):>20}{len(fronts[seed, 'Random']):>15}  "
                     f"{intervals[0]:<22}{intervals[1]}")
    lines += ["", "The effect is Discrepancy minus Random. Lower F-Prob at matched Utility and higher Utility at matched F-Prob favor Discrepancy.",
              f"The normalized AUC is the unweighted mean on a grid of {GRID_POINTS} equally spaced points including both endpoints.",
              "The 25%, 50% and 75% targets are normalized within each seed's own interval.",
              "ES, F-ROUGE, F-Ratio and absolute PrivLeak use the same segment and interpolation weights.",
              "F-Quality is reported for the observed segment endpoints, or at the target when it is an exact observed point.",
              ""]
    for mode, (_, outcome) in MODES.items():
        lines += ["Matched Utility · Δ F-Prob" if mode == "utility" else "Matched F-Prob · Δ Utility",
                  "Target        Seed 0      Seed 1      Seed 2       Mean  Sample SD  Signs"]
        for row in summaries[mode]:
            if row["quantity"] != f"delta_{outcome}" and row["normalized_position"] != "grid_mean":
                continue
            values = [display_number(row[key]) for key in ("seed0", "seed1", "seed2", "mean", "sample_sd")]
            position = row["normalized_position"]
            label = "Grid mean" if position == "grid_mean" else f"{position:.0%}"
            lines.append(f"{label:<10}" + "".join(f"{v:>12}" for v in values) + f"  {row['signs']}")
        lines.append("")
    lines += ["Observed-checkpoint sensitivity",
              "Nearest neighbors are selected with replacement within the intersection of the two families' observed ranges.",
              "The caliper is inclusive. Ties are resolved by earlier step and lower density.",
              "The mean gap is absolute. The mean effect uses the matched pairs within each seed.",
              "Match axis  Caliper   Seed  Matched / eligible       Mean gap    Mean effect   Pair signs (+ / − / 0)"]
    for row in nn_summary:
        lines.append(f"{row['match_axis']:<11}{row['caliper']:>8.3f}{row['seed']:>7}  "
                     f"{row['matched']:>5} / {row['discrepancy_in_support']:<14}"
                     f"{display_number(row['mean_absolute_gap']):>12}{display_number(row['mean_difference']):>15}   "
                     f"{row['positive_pairs']} / {row['negative_pairs']} / {row['tied_pairs']}")
    lines += ["", "Within-density supplementary results"]
    for density in DENSITIES:
        for mode, (axis, _) in MODES.items():
            checks = [r for r in within if r["analysis"] == mode and r["density"] == density]
            lines.append(f"{density}%, {axis} · {sum(r['observed_status'] == 'ok' for r in checks)}/3 seeds share an observed range"
                         f" and {sum(r['status'] == 'ok' for r in checks)}/3 share an undominated range.")
    primary_observed = [r for r in nn_summary if r["caliper"] == .01]
    lines += ["", "Interpretation"]
    for mode, (_, outcome) in MODES.items():
        observed = [r for r in primary_observed if r["analysis"] == mode]
        unfavorable = sum(r["positive_pairs"] if mode == "utility" else r["negative_pairs"] for r in observed)
        total = sum(r["matched"] for r in observed)
        primary = next(r for r in summaries[mode] if r["normalized_position"] == "grid_mean")
        lines.append(f"Matched {observed[0]['match_axis']}: mean grid Δ {outcome} {display_number(primary['mean'])} "
                     f"({primary['signs']}); {unfavorable}/{total} observed matches at the primary caliper favor Random.")
    lines += ["We did not extrapolate beyond common support.",
              "Interpolation cannot replace retraining at exactly matched strength. The observed-checkpoint sensitivity uses a tolerance and can have residual strength gaps.",
              "Three seeds provide descriptive variability and do not support strong claims about statistical significance.",
              "Matching the token budget does not match gradient or update magnitude.",
              "The primary comparison pools allowed settings across density and step and describes the observed achievable trade-off.",
              "", "Paper wording",
              ("Discrepancy provides a better achievable forgetting–utility trade-off than Random selection."
               if strong else PAPER_CAUTION)]
    (output / "matched_strength_report.txt").write_text("\n".join(lines) + "\n")


def run(input_dir, output_dir, archive):
    source, data, archive_checked = read_input(input_dir, archive)
    points, validation = validate(data, source, archive_checked)
    groups = {(seed, family): [p for p in points if p["seed"] == seed and p["family"] == family]
              for seed in SEEDS for family in FAMILIES}
    fronts = {key: frontier(value) for key, value in groups.items()}
    support, auc, within, within_matches, within_auc = [], [], [], [], []
    results, summaries, nearest_rows, nn_summary = {}, {}, [], []
    for mode, (x, _) in MODES.items():
        results[mode] = []
        for seed in SEEDS:
            curves = {family: fronts[seed, family] for family in FAMILIES}
            row, matches, grid_mean = matched(curves, seed, mode)
            support.append(row)
            results[mode].extend(matches)
            if grid_mean is not None:
                auc.append(grid_mean)
            for density in DENSITIES:
                raw = {family: [p for p in groups[seed, family] if p["density"] == density] for family in FAMILIES}
                observed, _ = support_row(raw, x)
                curves = {family: frontier(raw[family]) for family in FAMILIES}
                row, matches, grid_mean = matched(curves, seed, mode, density)
                within.append({**row, **{f"observed_{key}": value for key, value in observed.items()}})
                within_matches.extend(matches)
                if grid_mean is not None:
                    within_auc.append(grid_mean)
        summaries[mode] = matched_summary(results[mode], [r for r in auc if r["analysis"] == mode], mode)
        nn_rows, nn_data = nearest(groups, mode)
        nearest_rows.extend(nn_rows)
        nn_summary.extend(nn_data)
    strong = directional_conclusion(auc, results, nn_summary)

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, raw in data.items():
        (output_dir / name).write_bytes(raw)
    write_json(output_dir / "input_validation.json", validation)
    write_json(output_dir / "analysis_protocol.json", {
        "analysis": "post-hoc descriptive", "new_training": False, "extrapolation": False,
        "source_protocol": validation["protocol_sha256"], "targets": POSITIONS, "auc_grid_points": GRID_POINTS,
        "matching_axes": MODES, "secondary_metrics_interpolated": SECONDARY,
        "forget_quality": "observed bracket endpoints or exact observed matched points",
        "observed_matching": {"with_replacement": True, "calipers": CALIPERS, "primary_caliper": .01,
                              "support": "both points inside common observed coordinate range",
                              "tie_break": ["lower step", "lower density"]},
        "within_density": "Report raw four-step range and repeat matching on each density's observed nondominated points.",
        "primary_direction_gate": "Both analyses must favor Discrepancy at every target and grid mean in every seed. At caliper .01 each seed needs matched pairs, a favorable mean and zero opposing pairs.",
        "uniform_advantage_supported": strong,
        "source_script_sha256": digest(Path(__file__)),
    })
    tables = {
        "frontier_points": [p for key, pts in fronts.items() for p in pts],
        "common_support": support,
        "normalized_auc_by_seed": auc,
        "nearest_observed_matches": nearest_rows,
        "nearest_observed_summary": nn_summary,
        "within_density_support": within,
        "within_density_matches_by_seed": within_matches,
        "within_density_auc_by_seed": within_auc,
    }
    for mode in MODES:
        tables[f"matched_{mode}_by_seed"] = results[mode]
        tables[f"matched_{mode}_summary"] = summaries[mode]
    for name, rows in tables.items():
        write_csv(output_dir / f"{name}.csv", rows)
    plot_frontiers(fronts, output_dir)
    for mode in MODES:
        by_seed = {r["seed"]: (r["common_min"], r["common_max"]) if r["common_min"] is not None else None
                   for r in support if r["analysis"] == mode}
        plot_effects(fronts, results[mode], by_seed, output_dir, mode)
    plot_privacy(results, output_dir)
    report(output_dir, validation, fronts, support, within, summaries, nn_summary, strong)
    files = {path.name: digest(path) for path in sorted(output_dir.iterdir()) if path.is_file() and path.name != "MANIFEST.json"}
    write_json(output_dir / "MANIFEST.json", {"files_sha256": files, "analysis_script": str(Path(__file__)),
                                             "analysis_script_sha256": digest(Path(__file__))})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    args = parser.parse_args()
    run(args.input, args.output, args.archive)


if __name__ == "__main__":
    main()
