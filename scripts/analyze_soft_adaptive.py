"""Replay official results and compare the frozen Soft-Adaptive screen."""

import csv
import hashlib
import json
import math
import subprocess
from zipfile import ZIP_DEFLATED, ZipFile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from prepare_soft_adaptive import EXTENSION, SUPPORT, SOURCE, STEPS
from run_soft_adaptive import NAME, CONTROLS, ROOT, WORKSPACE, control_archive, digest, read, require, save_json, verify_lock
from prepare_influence_adaptive import DIAGNOSTICS, rows
from run_low_confidence import ARTIFACTS, METRICS, TABLE_METRICS, replay_metrics, validate_outputs

ARMS = (NAME, *CONTROLS)
AXES = {"utility": "forget_probability", "forget_probability": "utility"}
LABELS = ("ES", "F-Prob", "F-ROUGE", "F-Ratio", "F-Quality", "Utility", "Signed PrivLeak", "Absolute PrivLeak")


def table(name, records):
    columns = list(dict.fromkeys(k for row in records for k in row))
    with (ROOT / f"{name}.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)


def verify_evaluation(run, step, lock, smoke=False):
    cp = run / f"checkpoint-{step}"
    ev = cp / "evals/official"
    proof = read(ev / "validated.json")
    provenance = read(ev / "evaluation_provenance.json")
    meta = read(cp / "checkpoint_metadata.json")
    state = read(cp / "trainer_state.json")
    config = provenance["resolved_eval_config"]
    protocol = lock["protocol"]
    epoch = protocol["checkpoint_validation" if smoke else "training"]["checkpoints"][str(step)]
    snapshot = read(cp / "resolved_config.json")
    require(provenance["checkpoint"] == meta and all(proof[k] == value for k, value in meta.items())
            and proof["variant"] == NAME and meta["global_step"] == state["global_step"] == step
            and meta["actual_epoch"] == state["epoch"] == epoch and meta["seed"] == 0
            and meta["mask_sha256"] == lock["weights_sha256"] and meta["code_commit"] == lock["code_commit"]
            and meta["protocol_sha256"] == lock["v4_protocol_sha256"]
            and proof["evaluator_sha256"] == provenance["evaluator_sha256"] == lock["evaluator_sha256"]
            and provenance["dataset_revision"] == protocol["dataset"]["revision"]
            and provenance["model"] == snapshot["model"]
            and snapshot == read(run / "resolved_config.json")
            and snapshot["extension_manifest_sha256"] == digest(ROOT / "protocol_manifest.json")
            and {k: v for k, v in config.items() if k not in {"output_dir", "retain_logs_path"}}
            == read(ROOT / "v4_evaluation_signature.json"), f"Provenance differs at {cp}")
    require(set(proof["artifact_sha256"]) == set(ARTIFACTS) - {"validated.json"}
            and all(digest(ev / n) == h for n, h in proof["artifact_sha256"].items()), "Evaluation file hash differs")
    weights = cp / "model.safetensors"
    require(weights.is_file() and not weights.is_symlink() and meta["saved_weight_files"] == [weights.name]
            and provenance["weight_sha256"] == {weights.name: digest(weights)}, "Checkpoint weight hash differs")
    reference = ROOT / "retain_only_reference_TOFU_EVAL.json"
    require(digest(reference) == provenance["retain_only_reference_sha256"], "Retain-only reference hash differs")
    logs = read(ev / "TOFU_EVAL.json")
    metrics, counts = validate_outputs(config, protocol["official_evaluation"], logs, read(ev / "TOFU_SUMMARY.json"), reference)
    replay = replay_metrics(config, logs, read(reference))
    require(set(metrics) == set(METRICS) and metrics == proof["metrics"] == read(ev / "official_metrics.json")
            and counts == proof["sample_counts"] and all(math.isclose(v, replay[k], rel_tol=1e-10, abs_tol=1e-14)
                                                         for k, v in metrics.items()), "Official metric replay differs")
    signed = metrics.pop("privleak")
    return {**metrics, "privleak_signed": signed, "privleak_abs": abs(signed)}, {
        "step": step, "epoch": epoch, "sample_counts": counts, "validated_sha256": digest(ev / "validated.json"),
        "weight_sha256": provenance["weight_sha256"][weights.name]}


def frontier(points):
    chosen = [point for point in points if not any(
        q["utility"] >= point["utility"] and q["forget_probability"] <= point["forget_probability"]
        and (q["utility"], q["forget_probability"]) != (point["utility"], point["forget_probability"]) for q in points)]
    return sorted({(p["utility"], p["forget_probability"]): p for p in reversed(chosen)}.values(), key=lambda p: p["utility"])


def bracket(points, axis, value):
    order = sorted(points, key=lambda point: point[axis])
    if not order[0][axis] <= value <= order[-1][axis]:
        return None
    for point in order:
        if point[axis] == value:
            return point, point, 0.
    for left, right in zip(order, order[1:]):
        if left[axis] < value < right[axis]:
            return left, right, (value - left[axis]) / (right[axis] - left[axis])
    raise AssertionError("Missing interpolation bracket")


def matched(records, gate):
    observed = {arm: [p for p in records if p["variant"] == arm] for arm in ARMS}
    fronts = {arm: frontier(points) for arm, points in observed.items()}
    interpolated, nearest, comparisons = [], [], []
    for axis, outcome in AXES.items():
        direction = -1 if outcome == "forget_probability" else 1
        for soft in observed[NAME]:
            baseline = {}
            value = soft[axis]
            soft_frontier = soft in fronts[NAME]
            for arm in CONTROLS:
                pts = observed[arm]
                lower = max(min(p[axis] for p in pts), min(p[axis] for p in observed[NAME]))
                upper = min(max(p[axis] for p in pts), max(p[axis] for p in observed[NAME]))
                for eps in gate["nearest_calipers"]:
                    row = {"axis": axis, "outcome": outcome, "soft_step": soft["step"], "soft_axis": value,
                           "control": arm, "caliper": eps, "support_lower": lower, "support_upper": upper}
                    candidates = [p for p in pts if lower <= p[axis] <= upper]
                    if lower <= value <= upper and candidates:
                        p = min(candidates, key=lambda q: (abs(q[axis] - value), q["step"]))
                        row.update(control_step=p["step"], control_axis=p[axis], axis_difference=value - p[axis],
                                   outcome_difference=soft[outcome] - p[outcome],
                                   abs_privleak_difference=soft["privleak_abs"] - p["privleak_abs"],
                                   update_ratio=soft["optimizer_step_update_l2"] / p["optimizer_step_update_l2"],
                                   status="matched" if abs(p[axis] - value) <= eps else "outside caliper")
                    else:
                        row["status"] = "outside common support" if not lower <= value <= upper else "no peer inside support"
                    nearest.append(row)
                match = bracket(fronts[arm], axis, value)
                row = {"axis": axis, "outcome": outcome, "soft_step": soft["step"], "soft_axis": value,
                       "soft_on_own_frontier": soft_frontier, "control": arm, "status": "no shared support",
                       "soft_outcome": soft[outcome], "soft_absolute_privleak": soft["privleak_abs"]}
                if match:
                    a, b, weight = match
                    control_outcome = a[outcome] + weight * (b[outcome] - a[outcome])
                    control_privleak = a["privleak_abs"] + weight * (b["privleak_abs"] - a["privleak_abs"])
                    update_ratios = [soft["optimizer_step_update_l2"] / p["optimizer_step_update_l2"] for p in (a, b)]
                    row.update(status="observed exact" if a is b else "interpolated", lower_step=a["step"],
                               upper_step=b["step"], upper_weight=weight, control_outcome=control_outcome,
                               outcome_difference=soft[outcome] - control_outcome, control_absolute_privleak=control_privleak,
                               abs_privleak_difference=soft["privleak_abs"] - control_privleak,
                               lower_update_ratio=update_ratios[0], upper_update_ratio=update_ratios[1],
                               conservative_update_ratio=max(update_ratios))
                    baseline[arm] = row
                interpolated.append(row)
            if len(baseline) == 2:
                d34, r34 = [baseline[arm] for arm in CONTROLS]
                gain = direction * d34["outcome_difference"]
                original_gap = direction * (r34["control_outcome"] - d34["control_outcome"])
                new_gap = direction * (r34["control_outcome"] - soft[outcome])
                conservative_ratio = d34["conservative_update_ratio"]
                comparisons.append({"axis": axis, "soft_step": soft["step"], "coordinate": value,
                    "outcome": outcome, "preferred_direction_gain_over_discrepancy": gain,
                    "random_gap_relative_to_discrepancy": original_gap, "random_gap_relative_to_soft": new_gap,
                    "disc_update_ratio": conservative_ratio, "random_update_ratio": r34["conservative_update_ratio"],
                    "soft_on_own_frontier": soft_frontier,
                    "passes": soft_frontier and gain > 0 and original_gap > 0 and new_gap < original_gap
                    and conservative_ratio <= gate["max_update_ratio"]})
    return interpolated, nearest, comparisons


def figures(records, pooled):
    palette = ("#73549c", "#00786f", "#a94026")
    for x, label, file in (("utility", "Utility ↑", "utility"), ("privleak_abs", "Absolute PrivLeak ↓", "abs_privleak")):
        fig, ax = plt.subplots(figsize=(7.2, 4.8), layout="constrained")
        for i, (variant, color) in enumerate(zip(ARMS, palette, strict=True)):
            points = [p for p in records if p["variant"] == variant]
            ax.plot([p[x] for p in points], [p["forget_probability"] for p in points], "o-", color=color,
                    markersize=4.5, linewidth=1.4, label=variant)
            for p in points:
                ax.annotate(p["step"], (p[x], p["forget_probability"]),
                            xytext=(6, (8, -13, 1)[i]), textcoords="offset points", fontsize=8, color=color)
        if x == "utility":
            ax.scatter([p[x] for p in pooled], [p["forget_probability"] for p in pooled], s=75, color="#34464b",
                       facecolor="none", linewidth=1, label="Observed Pareto frontier")
        ax.set(xlabel=label, ylabel="F-Prob ↓", title="TOFU Forget05 · Seed 0")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=.18)
        ax.legend(frameon=False, fontsize=8)
        for extension in ("pdf", "png"):
            fig.savefig(ROOT / f"fprob_vs_{file}_trajectory.{extension}", dpi=180)
        plt.close(fig)


def report(records, pooled, interp, NN, decision, audit):
    notes = ["TOFU Forget05 · Soft-Adaptive-ESS34 seed-0 screen", "",
             "The formal weighted run and both archived binary controls use the same optimizer, scheduler, retain objective, data, model and evaluation.",
             "Discrepancy and Random assign uniform 1/k weights over their binary selections.",
             f"The weights preserve {audit['exact_top_k_samples']}/200 frozen Discrepancy-34 top-k sets with a "
             f"{audit['aggregate_nominal_ratio']:.4%} nominal ESS ratio. All {audit['positive_weight_tokens']} eligible tokens receive positive weight.",
             f"Tau mean {audit['tau']['mean']:.5f}, median {audit['tau']['p50']:.5f}. "
             f"FP32 ESS error mean {audit['fp32_ess_absolute_error']['mean']:.3g}, median {audit['fp32_ess_absolute_error']['p50']:.3g}, "
             f"max {audit['fp32_ess_absolute_error']['p100']:.3g}. Boundary samples {len(audit['boundary_samples'])}.",
             f"Per-example entropy median {audit['entropy_nats']['p50']:.4f} nats. Maximum token weight median "
             f"{audit['maximum_weight']['p50']:.2%}, max {audit['maximum_weight']['p100']:.2%}.",
             "Global top 1%, 5%, 10% weight mass " + ", ".join(f"{audit['top_token_weight_mass'][str(p)]:.2%}" for p in (1, 5, 10)) + ".",
             f"Average top-k mass {audit['top_k_mass']['mean']:.2%} versus 100% for the binary mask. "
             f"Factual weight mass {audit['factual_weight_mass']:.2%} versus {audit['binary_factual_weight_mass']:.2%}.",
             "", "Official metrics, sorted by method and optimizer step",
             f"{'Method':24}{'Step':>5}" + "".join(f"{k:>16}" for k in LABELS)]
    for r in records:
        notes.append(f"{r['variant']:24}{r['step']:>5}" + "".join(f"{r[k]:>16.6g}" for k in TABLE_METRICS))
    notes += ["", "Common-support comparisons", "Linear interpolation follows the observed control frontiers and uses no extrapolation.",
              "Interpolation provides descriptive estimates for these runs. Exact matched-strength retraining would require its own experiments.",
              "The update threshold uses the larger ratio of the two bracket checkpoint update norms against Discrepancy-34.",
              "Update norm measures only the checkpoint's optimizer step."]
    for row in interp:
        if row["status"] != "no shared support":
            notes.append(f"{row['axis']}, Step {row['soft_step']} vs {row['control']} "
                         f"[{row['lower_step']}, {row['upper_step']}] · Δ {row['outcome']} {row['outcome_difference']:+.5f}, "
                         f"Δ absolute PrivLeak {row['abs_privleak_difference']:+.3f}, update ratio {row['conservative_update_ratio']:.3f}×.")
    observed = [r for r in NN if r["caliper"] == .01 and r["status"] == "matched"]
    notes += ["", f"Observed nearest neighbors at a 0.01 caliper · {len(observed)} pairs"]
    for row in observed:
        notes.append(f"{row['axis']}, Step {row['soft_step']} vs {row['control']} Step {row['control_step']} · "
                     f"axis gap {row['axis_difference']:+.5f}, Δ {row['outcome']} {row['outcome_difference']:+.5f}, "
                     f"Δ absolute PrivLeak {row['abs_privleak_difference']:+.3f}, update ratio {row['update_ratio']:.3f}×.")
    notes += ["", "Observed pooled F-Prob–Utility Pareto frontier",
              ", ".join(f"{p['variant']} Step {p['step']}" for p in pooled), "",
              "Seed expansion " + ("warrants follow-up" if decision["worth_expanding"] else "criteria were not met") + ".",
              f"{len(decision['passing_soft_checkpoints'])} distinct Soft-Adaptive checkpoints passed the joint comparison gate.",
              "At least two distinct checkpoints and no extreme concentration were required."]
    if not decision["worth_expanding"]:
        notes.append("The exact ESS and preserved top-k establish that the experiment tests redistributing weight within the same discrepancy ranking. "
                     "The weight outside top-k and the factual mass shift describe this weighting. These seed-0 runs isolate neither "
                     "temperature-family effects nor the effect of changing the discrepancy ranking.")
    (ROOT / "concise_report.txt").write_text("\n".join(notes) + "\n")


def export(lock, proofs):
    ignored = {"queue.pid", "queue.lock", "status.json", "MANIFEST.json", "export_verification.json"}
    files = {str(p.relative_to(ROOT)): p for p in ROOT.rglob("*") if p.is_file() and p.name not in ignored
             and p.suffix not in {".safetensors", ".pt", ".tmp", ".part"} and "events.out.tfevents" not in p.name}
    for name, checksum in lock["control_artifact_hashes"].items():
        arm, relative = name.split("/", 1)
        p = control_archive(arm) / relative
        require(digest(p) == checksum, f"Control file changed: {name}")
        files[f"archived_controls/{name}"] = p
    for arm in CONTROLS:
        p = control_archive(arm).parents[3] / "logs" / f"{arm}.train.log"
        require(p.is_file(), f"Missing training log for {arm}")
        files[f"archived_controls/{arm}/training.log"] = p
    diffs = []
    for name in EXTENSION:
        files[f"implementation/{name}"] = WORKSPACE / name
        done = subprocess.run(["git", "diff", "--no-index", "--", "/dev/null", name], cwd=WORKSPACE, capture_output=True, check=False)
        require(done.returncode == 1 and done.stdout, f"Diff failed for {name}")
        diffs.append(done.stdout)
    for name in SUPPORT:
        p = SOURCE / name
        files[f"implementation/support/{name}"] = p if p.is_file() else WORKSPACE / name
    diff = ROOT / "implementation.diff"
    diff.write_bytes(b"".join(diffs))
    files[diff.name] = diff
    require(sum(n.endswith("/evals/official/TOFU_EVAL.json") for n in files) == 13, "Incomplete per-sample output bundle")
    hashes = {n: digest(p) for n, p in sorted(files.items())}
    manifest = {"code_commit": lock["code_commit"], "weights_sha256": lock["weights_sha256"], "formal_checkpoints": 4,
                "archived_control_checkpoints": 8, "smoke_checkpoints": 1, "files_sha256": hashes}
    save_json(ROOT / "MANIFEST.json", manifest)
    archive = WORKSPACE / "saves/exports" / f"{ROOT.name}.zip"
    pending = archive.with_suffix(".zip.part")
    with ZipFile(pending, "w", ZIP_DEFLATED, compresslevel=4) as out:
        for name, p in sorted({**files, "MANIFEST.json": ROOT / "MANIFEST.json"}.items()):
            out.write(p, f"{ROOT.name}/{name}")
    with ZipFile(pending) as zip_file:
        require(zip_file.testzip() is None and len(zip_file.namelist()) == len(hashes) + 1
                and json.loads(zip_file.read(f"{ROOT.name}/MANIFEST.json")) == manifest, "Archive integrity failed")
        for name, checksum in hashes.items():
            with zip_file.open(f"{ROOT.name}/{name}") as stream:
                require(hashlib.file_digest(stream, "sha256").hexdigest() == checksum, f"Exported {name} differs")
    checksum = digest(pending)
    pending.replace(archive)
    sidecar = archive.with_suffix(".zip.sha256")
    sidecar.write_text(f"{checksum}  {archive.name}\n")
    subprocess.run(["sha256sum", "-c", sidecar.name], cwd=archive.parent, check=True)
    targets = {ROOT / name / "model.safetensors": p["weight_sha256"] for name, p in proofs.items()}
    require(all(not p.is_symlink() and digest(p) == h for p, h in targets.items()), "Checkpoint differs before cleanup")
    for p in targets:
        p.unlink()
    save_json(ROOT / "export_verification.json", {"sha256": checksum, "bytes": archive.stat().st_size,
              "files_verified": len(hashes), "model_weights_deleted": len(targets)})


def main():
    lock = verify_lock()
    audit = read(ROOT / "weight_audit.json")
    run = ROOT / "runs/seed0"
    smoke = ROOT / "smoke_test"
    smoke_validation = read(ROOT / "smoke_validation.json")
    require(smoke_validation["passed"] and smoke_validation["two_rank_replay"], "Smoke verification missing")
    proofs = {"smoke_test/checkpoint-4": verify_evaluation(smoke, 4, lock, smoke=True)[1]}
    require(proofs["smoke_test/checkpoint-4"] == smoke_validation["official_replay"], "Smoke replay differs")
    records = []
    updates = read(run / "optimizer_update_norms.json")
    for step in STEPS:
        metrics, proof = verify_evaluation(run, step, lock)
        proofs[f"runs/seed0/checkpoint-{step}"] = proof
        diag = read(run / f"checkpoint-{step}/training_diagnostics.json") | updates[str(step)]
        require(all(math.isfinite(diag[k]) for k in DIAGNOSTICS), "Nonfinite training diagnostics")
        records.append({"variant": NAME, "seed": 0, "step": step, "actual_epoch": proof["epoch"],
                        "selection_ratio": audit["aggregate_nominal_ratio"], **metrics, **{k: diag[k] for k in DIAGNOSTICS},
                        "mask_sha256": lock["weights_sha256"], "code_commit": lock["code_commit"]})
    records.extend({"variant": r["variant"], "seed": int(r["seed"]), "step": int(r["step"]),
                    **{k: float(r[k]) for k in ("actual_epoch", "selection_ratio", *TABLE_METRICS, *DIAGNOSTICS)},
                    **{k: r[k] for k in ("mask_sha256", "code_commit")}} for r in rows(ROOT / "existing_controls.csv"))
    require(len(records) == len({(r["variant"], r["step"]) for r in records}) == 12, "Invalid control schedule")
    by_step = {r["step"]: r for r in records if r["variant"] == NAME}
    require(all(r["selection_ratio"] == by_step[r["step"]]["selection_ratio"] and
                r["learning_rate_used"] == by_step[r["step"]]["learning_rate_used"] and
                r["learning_rate_after_step"] == by_step[r["step"]]["learning_rate_after_step"] for r in records),
            "Selection budget or learning rate differs")
    diffs = []
    for ctrl in records[len(STEPS):]:
        soft = by_step[ctrl["step"]]
        diffs.append({"control": ctrl["variant"], "step": ctrl["step"],
                      **{f"delta_{k}": soft[k] - ctrl[k] for k in (*TABLE_METRICS, *DIAGNOSTICS)},
                      "update_norm_ratio": soft["optimizer_step_update_l2"] / ctrl["optimizer_step_update_l2"]})
    gate = lock["screening"]
    interp, nearest, comparisons = matched(records, gate)
    pooled = frontier(records)
    passes = sorted({r["soft_step"] for r in comparisons if r["passes"]})
    concentrated = bool(audit["samples_with_single_token_mass_above_half"] or audit["top_token_weight_mass"]["1"] > .25)
    decision = {"worth_expanding": len(passes) >= gate["require_distinct_soft_checkpoints_passing"] and not concentrated,
                "passing_soft_checkpoints": passes, "numerical_concentration_gate_failed": concentrated,
                "comparisons": comparisons, "frozen_rules": gate, "scope": "descriptive seed-0 feasibility screen"}
    save_json(ROOT / "screening_decision.json", decision)
    save_json(ROOT / "verification.json", {"replay": proofs, "archived_controls_replayed": 8, "training_completed": 120,
                                          "protocol_sha256": digest(ROOT / "protocol_manifest.json")})
    for name, values in (("official_metrics_long", records), ("checkpointwise_differences", diffs),
                         ("observed_pareto_frontier", pooled), ("matched_strength", interp),
                         ("nearest_observed_sensitivity", nearest)):
        table(name, values)
    figures(records, pooled)
    report(records, pooled, interp, nearest, decision, audit)
    export(lock, proofs)


if __name__ == "__main__":
    main()
