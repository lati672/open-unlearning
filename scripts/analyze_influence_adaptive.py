"""Validate, compare, export and clean up the Influence-Adaptive seed-0 run."""

import json
import math
import subprocess
from zipfile import ZIP_DEFLATED, ZipFile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from prepare_influence_adaptive import CONTROLS, ROOT, STEPS, VARIANT, WORKSPACE, DIAGNOSTICS, digest, read, require, rows, save_json, write_csv
from run_influence_adaptive import SCRIPTS, verify_lock
from run_low_confidence import METRICS, TABLE_METRICS, replay_metrics, validate_outputs


ARMS = (VARIANT, *CONTROLS)
AXES = (("utility", "forget_probability", -1), ("forget_probability", "utility", 1))
LABELS = {"ES": "ES", "forget_probability": "F-Prob", "forget_rouge": "F-ROUGE", "forget_ratio": "F-Ratio",
          "forget_quality": "F-Quality", "utility": "Utility", "privleak_abs": "|PrivLeak|", "privleak_signed": "signed PrivLeak"}


def verify_evaluation(run, step, lock, smoke=False):
    checkpoint = run / f"checkpoint-{step}"
    ev = checkpoint / "evals/official"
    proof = read(ev / "validated.json")
    provenance = read(ev / "evaluation_provenance.json")
    config = provenance["resolved_eval_config"]
    state = read(checkpoint / "trainer_state.json")
    settings = lock["protocol"]
    epoch = .64 if smoke else settings["training"]["checkpoints"][str(step)]
    require(proof["variant"] == VARIANT and proof["global_step"] == state["global_step"] == step
            and proof["actual_epoch"] == state["epoch"] == epoch and proof["seed"] == 0
            and proof["mask_sha256"] == lock["mask_sha256"] and proof["code_commit"] == lock["code_commit"]
            and proof["protocol_sha256"] == lock["v4_protocol_sha256"]
            and proof["evaluator_sha256"] == provenance["evaluator_sha256"] == lock["evaluator_sha256"]
            and {k: v for k, v in config.items() if k not in {"output_dir", "retain_logs_path"}}
            == read(ROOT / "v4_evaluation_signature.json"), f"Evaluation protocol differs: {checkpoint}")
    meta = read(checkpoint / "checkpoint_metadata.json")
    require(provenance["checkpoint"] == meta and all(proof[k] == v for k, v in meta.items()),
            "Checkpoint provenance differs")
    require(all(digest(ev / name) == checksum for name, checksum in proof["artifact_sha256"].items()),
            "Official file hash differs")
    require(set(provenance["weight_sha256"]) == set(meta["saved_weight_files"]) == {"model.safetensors"} and
            digest(checkpoint / "model.safetensors") == provenance["weight_sha256"]["model.safetensors"],
            "Checkpoint weight hash differs")
    snapshot = read(checkpoint / "resolved_config.json")
    require(snapshot == read(run / "resolved_config.json") and snapshot["mask_sha256"] == lock["mask_sha256"]
            and snapshot["extension_manifest_sha256"] == digest(ROOT / "protocol_manifest.json")
            and provenance["model"] == snapshot["model"]
            and provenance["dataset_revision"] == settings["dataset"]["revision"], "Resolved snapshot differs")
    ref = ROOT / "retain_only_reference_TOFU_EVAL.json"
    require(digest(ref) == provenance["retain_only_reference_sha256"], "Frozen evaluation reference changed")
    logs = read(ev / "TOFU_EVAL.json")
    metrics, counts = validate_outputs(config, settings["official_evaluation"], logs, read(ev / "TOFU_SUMMARY.json"), ref)
    replay = replay_metrics(config, logs, read(ref))
    require(set(metrics) == set(METRICS) and metrics == read(ev / "official_metrics.json") == proof["metrics"]
            and counts == proof["sample_counts"] and all(math.isclose(v, replay[k], rel_tol=1e-10, abs_tol=1e-14)
                                                        for k, v in metrics.items()), "Official metric replay differs")
    signed = metrics.pop("privleak")
    return {**metrics, "privleak_signed": signed, "privleak_abs": abs(signed)}, {
        "actual_epoch": epoch, "sample_counts": counts, "validated_sha256": digest(ev / "validated.json"),
        "weight_sha256": provenance["weight_sha256"]["model.safetensors"]}


def front(points):
    return [p for p in points if not any(q["utility"] >= p["utility"] and q["forget_probability"] <= p["forget_probability"]
                                        and (q["utility"] > p["utility"] or q["forget_probability"] < p["forget_probability"])
                                        for q in points)]


def nearest(records, gate):
    groups = {arm: [r for r in records if r["variant"] == arm] for arm in ARMS}
    pairs, triplets, ranges = [], [], []
    for axis, outcome, desired in AXES:
        bounds = {arm: (min(r[axis] for r in values), max(r[axis] for r in values)) for arm, values in groups.items()}
        for control in CONTROLS:
            lo = max(bounds[VARIANT][0], bounds[control][0])
            hi = min(bounds[VARIANT][1], bounds[control][1])
            ranges.append({"match_axis": axis, "control": control, "common_min": lo, "common_max": hi,
                           "has_common_support": lo <= hi})
            candidates = [r for r in groups[control] if lo <= r[axis] <= hi]
            for caliper in (*gate["sensitivity_calipers"], gate["nearest_caliper"]):
                for p in groups[VARIANT]:
                    row = {"match_axis": axis, "outcome": outcome, "control": control, "caliper": caliper,
                           "influence_step": p["step"], "influence_axis_value": p[axis], "influence_outcome": p[outcome],
                           "common_min": lo, "common_max": hi}
                    if lo > hi:
                        status = "no common support"
                    elif not lo <= p[axis] <= hi:
                        status = "outside common support"
                    elif not candidates:
                        status = "no control checkpoint inside support"
                    else:
                        q = min(candidates, key=lambda r: (abs(r[axis] - p[axis]), r["step"]))
                        gap = p[axis] - q[axis]
                        status = "matched" if abs(gap) <= caliper else "outside caliper"
                        row.update(control_step=q["step"], control_axis_value=q[axis], axis_gap=gap,
                                   control_outcome=q[outcome], outcome_difference=p[outcome] - q[outcome],
                                   update_ratio=p["optimizer_step_update_l2"] / q["optimizer_step_update_l2"])
                    pairs.append({**row, "status": status})
        # Both reference matches must describe the same Influence checkpoint,
        # and the two reference points must satisfy the caliper with each other.
        primary = [r for r in pairs if r["match_axis"] == axis and r["caliper"] == gate["nearest_caliper"]
                   and r["status"] == "matched"]
        for step in STEPS:
            matched = {r["control"]: r for r in primary if r["influence_step"] == step}
            if len(matched) != len(CONTROLS):
                continue
            d, r = [matched[arm] for arm in CONTROLS]
            ref_gap = abs(d["control_axis_value"] - r["control_axis_value"])
            if ref_gap > gate["nearest_caliper"]:
                continue
            original_gap = desired * (r["control_outcome"] - d["control_outcome"])
            influence_gap = desired * (r["control_outcome"] - d["influence_outcome"])
            improves = desired * d["outcome_difference"] > 0
            # Signed gaps are shown alongside the gate. A negative remaining
            # gap means Influence exceeded Random on that matched metric.
            closes_gap = original_gap > 0 and influence_gap < original_gap
            modest_update = d["update_ratio"] <= gate["max_update_ratio_to_matched_discrepancy"]
            triplets.append({"match_axis": axis, "influence_step": step, "discrepancy_step": d["control_step"],
                             "random_step": r["control_step"], "max_axis_gap": max(ref_gap, abs(d["axis_gap"]), abs(r["axis_gap"])),
                             "random_advantage_over_discrepancy": original_gap, "random_advantage_over_influence": influence_gap,
                             "influence_improves_over_discrepancy": improves, "narrows_random_gap": closes_gap,
                             "update_ratio_vs_matched_discrepancy": d["update_ratio"], "passes_update_gate": modest_update,
                             "passes_screen": improves and closes_gap and modest_update})
    return pairs, triplets, ranges


def figures(records, frontier):
    colors = {VARIANT: "#75629b", CONTROLS[0]: "#007b7b", CONTROLS[1]: "#b64e33"}
    for metric in ("utility", "privleak_abs"):
        fig, ax = plt.subplots(figsize=(7, 4.7), layout="constrained")
        for arm in ARMS:
            points = [r for r in records if r["variant"] == arm]
            color = colors[arm]
            ax.plot([r[metric] for r in points], [r["forget_probability"] for r in points],
                    color=color, marker="o", lw=1.7, ms=5, label=arm)
            for r in points:
                direction = (5, (-10 if arm == CONTROLS[1] else 4))
                ax.annotate(r["step"], (r[metric], r["forget_probability"]), direction, textcoords="offset points",
                            fontsize=8, color=color)
        if metric == "utility":
            pts = sorted(frontier, key=lambda r: r["utility"])
            ax.scatter([r[metric] for r in pts], [r["forget_probability"] for r in pts],
                       facecolors="none", edgecolors="#33434d", s=93, linewidth=1, label="Observed frontier")
        ax.set(xlabel=LABELS[metric] + (" ↑" if metric == "utility" else " ↓"), ylabel="F-Prob ↓",
               title="TOFU Forget05 · Seed 0 · Steps 20, 60 and 120")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=.2)
        ax.legend(frameon=False, fontsize=8)
        filename = ROOT / f"fprob_vs_{metric}_trajectory"
        fig.savefig(filename.with_suffix(".pdf"))
        fig.savefig(filename.with_suffix(".png"), dpi=180)
        plt.close(fig)


def report(records, differences, frontier, ranges, pairs, triplets, audit, gate):
    passes = any(r["passes_screen"] for r in triplets)
    lines = ["TOFU Forget05 Influence-Adaptive · Seed 0 screening", "",
             "All three methods use the frozen token-normalized NPO configuration and identical per-example mask counts.",
             "Influence-Adaptive completed one 120-step run. Both archived controls were independently replayed.",
             f"Selected {audit['selected_tokens']}/{audit['eligible_tokens']} eligible tokens ({audit['selection_ratio']:.4%}). "
             f"Empty-mask rate {audit['empty_mask_rate']:.2%}.",
             "", "Official metrics and optimizer-step update norm",
             "Method                       Step       ES     F-Prob    F-ROUGE    F-Ratio   F-Quality     Utility   |PrivLeak|    Update L2"]
    for row in records:
        lines.append(f"{row['variant']:<28}{row['step']:>5}" +
                     "".join(f"{row[k]:>11.5g}" for k in (*[k for k in TABLE_METRICS if k != "privleak_signed"], "optimizer_step_update_l2")))
    lines += ["", "Mask audit"]
    for name, stat in audit["micro_overlap"].items():
        lines.append(f"Jaccard versus {name} {stat['jaccard']:.4f}.")
    correlations = {"delta and gradient": "delta_vs_gradient", "delta and score": "delta_vs_score",
                    "gradient and factual token indicator": "gradient_norm_vs_factual_indicator"}
    for label, key in correlations.items():
        result = audit[key]
        lines.append(f"{label} · Pearson {result['pearson']:+.4f}, Spearman {result['spearman']:+.4f}.")
    gradients = audit["gradient"]
    lines.append(f"Gradient norm mean {gradients['mean']:.4g}, sample SD {gradients['sample_sd']:.4g}, "
                 f"median {gradients['p50']:.4g}, p95 {gradients['p95']:.4g}, p99 {gradients['p99']:.4g}.")
    diagnostic = audit["recomputed_delta_vs_original"]["jaccard"]
    lines.append(f"Recomputed delta-only Jaccard with the archived Discrepancy-34 was {diagnostic:.4f}. "
                 "The untrained delta-only comparison shows how much the scoring pass changed the ranking.")
    lines.append(f"The top 1% of tokens by absolute score contribute {audit['top_1pct_absolute_score_mass_fraction']:.2%} of absolute score mass. "
                 "The selector uses scores within each example.")
    coverage = rows(ROOT / "token_category_factual_coverage.csv")
    for name in (VARIANT, CONTROLS[0]):
        c = audit["concentration"][name]
        punctuation = next(float(r["selected_composition"]) for r in coverage if r["method"] == name and r["group"] == "category:punctuation")
        factual = next(float(r["token_coverage"]) for r in coverage if r["method"] == name and r["group"] == "factual_all")
        lines.append(f"{name} selects {c['lowest_confidence_quartile_fraction']:.2%} from the lowest-confidence quartile, "
                     f"{c['top_1pct_gradient_fraction']:.2%} from the highest-gradient 1%, and {punctuation:.2%} punctuation. "
                     f"It covers {factual:.2%} of audited factual tokens.")
        span = audit["full_factual_span_coverage"].get(name, {}).get("all")
        if span:
            lines.append(f"{name} fully covers {span['full']}/{span['total']} annotated factual spans.")
    lines += ["", "Observed matching",
              "Both checkpoints must fall inside their common observed range. Distances are absolute and matching is with replacement.",
              f"The primary caliper is {gate['nearest_caliper']:.3f}. The gate uses a provisional "
              f"{gate['max_update_ratio_to_matched_discrepancy']:.2f}× maximum update norm relative to the matched Discrepancy checkpoint.",
              "A qualifying comparison also has to narrow a positive Random advantage at the same matched operating point.",
              "Update norm measures the optimizer step at the checkpoint and serves as a local diagnostic."]
    for row in ranges:
        axis, control = row["match_axis"], row["control"]
        matched = [r for r in pairs if r["match_axis"] == axis and r["control"] == control
                   and r["status"] == "matched" and r["caliper"] == gate["nearest_caliper"]]
        support = f"[{row['common_min']:.5f}, {row['common_max']:.5f}]" if row["has_common_support"] else "no common support"
        lines.append(f"{LABELS[axis]} versus {control} · {support}, {len(matched)} observed pairs.")
        for p in matched:
            lines.append(f"  Steps {p['influence_step']} vs {p['control_step']} · axis gap {p['axis_gap']:+.5f}, "
                         f"Δ {LABELS[p['outcome']]} {p['outcome_difference']:+.5f}, update ratio {p['update_ratio']:.3f}×.")
    lines += ["", "Observed pooled frontier",
              ", ".join(f"{p['variant']} at Step {p['step']}" for p in frontier), "",
              "Decision · " + ("worth expanding to seeds 1 and 2" if passes else "screening criteria for seed expansion were not met")]
    if not triplets:
        lines.append("The sampled checkpoints supply no three-way match within the primary caliper, so the Random gap criterion could not be established.")
    lines += ["This exploratory seed-0 result supports descriptive comparisons. Additional training seeds were outside this screen.",
              "The local gradient uses a detached FP32 LM head and cross-entropy on BF16 target hidden states. "
              "The scoring precision and the earlier cached discrepancy can differ at ranking boundaries."]
    (ROOT / "concise report.txt").write_text("\n".join(lines) + "\n")


def export(run, smoke, lock, proofs):
    files = {p.relative_to(ROOT).as_posix(): p for p in ROOT.iterdir() if p.is_file()
             and p.name not in {"MANIFEST.json", "status.json", "queue.pid", "queue.lock"}}
    for folder in (ROOT / "logs", run, smoke):
        for p in folder.rglob("*"):
            if p.is_file() and not p.name.endswith((".safetensors", ".pt", ".tmp")) and "events.out.tfevents" not in p.name:
                files[p.relative_to(ROOT).as_posix()] = p
    from prepare_influence_adaptive import control_archive
    for name, checksum in lock["control_artifact_hashes"].items():
        arm, relative = name.split("/", 1)
        path = control_archive(arm) / relative
        require(digest(path) == checksum, f"Control artifact changed: {name}")
        files[f"archived_controls/{name}"] = path
    for name in SCRIPTS:
        files[f"implementation/{name}"] = WORKSPACE / "scripts" / name
    diff = ROOT / "implementation.diff"
    parts = []
    for name in SCRIPTS:
        result = subprocess.run(["git", "diff", "--no-index", "--", "/dev/null", f"scripts/{name}"], cwd=WORKSPACE,
                                capture_output=True, text=True, check=False)
        require(result.returncode == 1, f"Cannot generate implementation diff for {name}")
        parts.append(result.stdout)
    diff.write_text("".join(parts))
    files[diff.name] = diff
    require(sum(n.endswith("/evals/official/TOFU_EVAL.json") for n in files) == 10, "Missing official sample outputs")
    hashes = {name: digest(p) for name, p in sorted(files.items())}
    manifest = {"code_commit": lock["code_commit"], "new_checkpoints": 3, "control_checkpoints": 6,
                "smoke_checkpoints": 1, "mask_sha256": lock["mask_sha256"], "files_sha256": hashes}
    save_json(ROOT / "MANIFEST.json", manifest)
    path = WORKSPACE / "saves/exports" / f"{ROOT.name}.zip"
    pending = path.with_suffix(".zip.part")
    with ZipFile(pending, "w", ZIP_DEFLATED, compresslevel=4) as bundle:
        for name, local in sorted({**files, "MANIFEST.json": ROOT / "MANIFEST.json"}.items()):
            bundle.write(local, f"{ROOT.name}/{name}")
    import hashlib
    with ZipFile(pending) as bundle:
        require(bundle.testzip() is None and len(bundle.namelist()) == len(files) + 1
                and json.loads(bundle.read(f"{ROOT.name}/MANIFEST.json")) == manifest, "ZIP failed integrity or manifest check")
        for name, checksum in hashes.items():
            with bundle.open(f"{ROOT.name}/{name}") as stream:
                require(hashlib.file_digest(stream, "sha256").hexdigest() == checksum, f"ZIP member differs: {name}")
    expected = digest(pending)
    pending.replace(path)
    sidecar = path.with_suffix(".zip.sha256")
    sidecar.write_text(f"{expected}  {path.name}\n")
    subprocess.run(["sha256sum", "-c", sidecar.name], cwd=path.parent, check=True)
    # Release weights only after ZIP, its manifest and sidecar all verify.
    targets = [(run / f"checkpoint-{step}/model.safetensors", proofs[str(step)]["weight_sha256"]) for step in STEPS]
    targets += [(smoke / "checkpoint-4/model.safetensors", proofs["smoke"]["weight_sha256"])]
    require(all(not p.is_symlink() and digest(p) == checksum for p, checksum in targets), "Weight changed before cleanup")
    for p, _ in targets:
        p.unlink()
    save_json(ROOT / "export_verification.json", {"archive": str(path), "archive_sha256": expected,
              "archive_bytes": path.stat().st_size, "verified_members": len(hashes), "checkpoint_weights_deleted": len(targets)})
    print(f"Verified {len(hashes)} members and removed {len(targets)} checkpoint weights after ZIP validation", flush=True)


def main():
    lock = verify_lock()
    run = ROOT / "formal/runs/seed0"
    smoke = ROOT / "smoke/smoke_test"
    audit = read(ROOT / "mask_audit.json")
    smoke_record = read(ROOT / "smoke_verification.json")
    require(smoke_record["official_metric_replay_passed"] and smoke_record["token_mean_and_same_mask_tests_passed"]
            and smoke_record["mask_sha256"] == lock["mask_sha256"], "Smoke gate differs")
    records = [{"variant": r["variant"], "source": "archived_density", "seed": int(r["seed"]), "step": int(r["step"]),
                **{k: float(r[k]) for k in ("actual_epoch", "selection_ratio", *TABLE_METRICS, *DIAGNOSTICS)},
                **{k: r[k] for k in ("code_commit", "mask_sha256", "protocol_sha256", "evaluator_sha256")}}
               for r in rows(ROOT / "existing_controls.csv")]
    proofs = {"smoke": verify_evaluation(smoke, 4, lock, smoke=True)[1]}
    update = read(run / "optimizer_update_norms.json")
    for step in STEPS:
        metrics, proof = verify_evaluation(run, step, lock)
        proofs[str(step)] = proof
        diagnostics = {**read(run / f"checkpoint-{step}/training_diagnostics.json"), **update[str(step)]}
        require(all(math.isfinite(diagnostics[k]) for k in DIAGNOSTICS), "Nonfinite training diagnostics")
        records.append({"variant": VARIANT, "source": "new_seed0", "seed": 0, "step": step, "actual_epoch": proof["actual_epoch"],
                        "selection_ratio": audit["selection_ratio"], **metrics,
                        **{k: diagnostics[k] for k in DIAGNOSTICS}, "code_commit": lock["code_commit"],
                        "mask_sha256": lock["mask_sha256"], "protocol_sha256": lock["v4_protocol_sha256"],
                        "evaluator_sha256": lock["evaluator_sha256"]})
    index = {(r["variant"], r["step"]): r for r in records}
    require(len(records) == len(index) == len(ARMS) * len(STEPS)
            and all(r["seed"] == 0 and r["selection_ratio"] == audit["selection_ratio"] and
                    r["learning_rate_used"] == index[VARIANT, r["step"]]["learning_rate_used"] and
                    r["learning_rate_after_step"] == index[VARIANT, r["step"]]["learning_rate_after_step"]
                    for r in records), "Screening comparison coverage or schedule differs")
    ordered = [index[arm, step] for arm in ARMS for step in STEPS]
    diffs = []
    for arm in CONTROLS:
        for step in STEPS:
            a, b = index[VARIANT, step], index[arm, step]
            diffs.append({"control": arm, "step": step, **{f"delta_{k}": a[k] - b[k] for k in (*TABLE_METRICS, *DIAGNOSTICS)},
                          "update_norm_ratio": a["optimizer_step_update_l2"] / b["optimizer_step_update_l2"]})
    frontier = front(ordered)
    gate = lock["screening_gate"]
    pairs, triplets, ranges = nearest(ordered, gate)
    for name, table in (("official_metrics_long", ordered), ("comparison_summary", diffs),
                        ("observed_pareto_frontier", frontier), ("nearest_observed_matches", pairs),
                        ("matched_support", ranges)):
        write_csv(ROOT / f"{name}.csv", table)
    save_json(ROOT / "screening_decision.json", {"worth_expanding": any(t["passes_screen"] for t in triplets),
              "three_way_matches": triplets, "gate": gate, "scope": "seed 0, observed checkpoints only"})
    save_json(ROOT / "verification.json", {"new_checkpoints": proofs, "official_metric_replay_passed": True,
              "controls_reused": 6, "control_protocol_matched": True, "mask_sha256": lock["mask_sha256"]})
    figures(ordered, frontier)
    report(ordered, diffs, frontier, ranges, pairs, triplets, audit, gate)
    export(run, smoke, lock, proofs)


if __name__ == "__main__":
    main()
