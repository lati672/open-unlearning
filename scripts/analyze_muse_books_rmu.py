"""Replay and summarize all corrected MUSE Books RMU results, then export."""

import argparse
import ast
import csv
import hashlib
import json
import math
from statistics import mean, stdev
from zipfile import ZIP_DEFLATED, ZipFile

from prepare_muse_books_rmu import ARMS, COMMIT, ROOT, SEEDS, SOURCE, WORKSPACE, digest, read, require, save, table
from run_muse_books_rmu import EVAL_COUNTS, METRICS, NORMALIZATION_TEST, PAPER_DIRECTIONS, SCRIPTS, locked, validate_eval


CONTRASTS = (("adaptive", "base"), ("random_adaptive", "base"), ("adaptive", "random_adaptive"))


def validate_archive(record, arm, seed, step, manifest, artifact_root=ROOT):
    folder = artifact_root / record["archive"]
    hashes = read(folder / "archive_hashes.json")
    inventory = {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()}
    require(inventory == set(hashes) | {"archive_hashes.json"}, f"Archive inventory differs: {folder}")
    require(digest(folder / "archive_hashes.json") == record["archive_hashes_sha256"]
            and all(digest(folder / name) == sha for name, sha in hashes.items()), f"Archive hash differs: {folder}")
    proof_path = folder / "evals/validated.json"
    proof = read(proof_path)
    require(digest(proof_path) == record["validated_sha256"] and proof["checkpoint_reload_passed"] and proof["metric_replay_passed"]
            and proof["code_commit"] == COMMIT and proof["dataset_revision"] == manifest["dataset"]["revision"]
            and proof["target_revision"] == manifest["models"]["target"]["revision"]
            and proof["control_sha256"] == manifest["control_vector"]["float32_sha256"]
            and proof["mask_sha256"] == manifest["mask_hashes"][arm]
            and proof["protocol_sha256"] == digest(artifact_root / "protocol_manifest.json")
            and proof["evaluator_config_sha256"] == manifest["config_hashes"]["official_evaluator.json"]
            and proof["arm"] == arm and proof["seed"] == seed and proof["optimizer_step"] == step
            and proof["weights_sha256"] == record["weights_sha256"] and proof["weights_sha256"], f"Provenance differs: {folder}")
    state = read(folder / "trainer_state.json")
    require(state["global_step"] == step and state["epoch"] == proof["actual_epoch"] == record["actual_epoch"]
            and record["optimizer_step"] == step, f"Checkpoint state differs: {folder}")
    metrics, counts = validate_eval(folder / "evals")
    require(metrics == proof["metrics"] == record["metrics"] == read(folder / "evals/official_metrics.json")
            and counts == proof["sample_counts"] == record["sample_counts"] == EVAL_COUNTS,
            f"Archived official evaluation changed: {folder}")
    require(proof["resolved_training_config_sha256"] == digest(folder / "run_resolved_config.json")
            and all(digest(folder / "evals" / name) == sha for name, sha in proof["files_sha256"].items()),
            f"Eval/config hash differs: {folder}")
    resolved = read(folder / "run_resolved_config.json")
    expected = read(artifact_root / "frozen_configs" / f"{arm}.json")
    exceptions = {"output_dir", "logging_dir", "seed"}
    args = resolved["trainer"]["args"]
    if folder.parts[-3] == "smoke_collected":
        exceptions |= {"max_steps", "save_steps", "logging_steps"}
    require(all(resolved[k] == expected[k] for k in ("model", "data", "collator", "distributed"))
            and resolved["trainer"]["handler"] == expected["trainer"]["handler"]
            and resolved["trainer"]["method_args"] == expected["trainer"]["method_args"]
            and {k: v for k, v in args.items() if k not in exceptions} ==
                {k: v for k, v in expected["trainer"]["args"].items() if k not in exceptions}
            and args["seed"] == seed, f"Frozen training configuration differs: {folder}")
    return folder, proof


def validate_sampling(folders, steps, accumulation, smoke=False):
    result = {}
    for rank in (0, 1):
        hashes = set()
        for folder in folders:
            check = read(folder / f"run_checks_rank{rank}.json")
            require(check["no_skipped_steps"] and len(check["microbatch_losses"]) == steps * accumulation
                    and len(check["optimizer_grad_norms"]) == steps and check["gpu_peak_bytes"] > 0
                    and all(math.isfinite(x) for x in check["microbatch_losses"] + check["optimizer_grad_norms"])
                    and all(x > 0 for x in check["optimizer_grad_norms"])
                    and (not smoke or check["reference_probes_equal"]), f"Invalid rank checks: {folder}/{rank}")
            hashes.add(check["sampling_order_sha256"])
        require(len(hashes) == 1, f"Paired sampling order differs at rank {rank}")
        result[str(rank)] = hashes.pop()
    return result


def verify_prior_smoke(manifest, plan):
    smoke_root = ROOT / "provisional_smoke"
    previous = read(smoke_root / "protocol_manifest.json")
    proof = read(smoke_root / "verification.json")
    smoke_index = read(smoke_root / "smoke_manifest.json")
    require(proof["provisional_smoke"] and proof["checks_passed"] == len(ARMS)
            and proof["official_evaluation_and_metric_replay_passed"] and proof["checkpoint_index"] == smoke_index,
            "Smoke coverage or original validation differs")
    for name in ("code_commit", "source_sha256", "models", "dataset", "packages", "torch_cuda", "mask_hashes",
                 "control_vector", "mask_audit_sha256", "normalization_test_sha256", "normalization_test_log_sha256"):
        require(previous[name] == manifest[name], f"Smoke provenance differs in {name}")
    require(previous["target_snapshot_files"] == manifest["target_snapshot_files"], "Smoke target weights differ")
    for script, checksum in previous["experiment_script_hashes"].items():
        require(digest(smoke_root / "experiment_code" / script) == checksum, f"Smoke script changed: {script}")
    runner = "run_muse_books_rmu.py"
    original_code = ast.parse((smoke_root / "experiment_code" / runner).read_text())
    current_code = ast.parse((WORKSPACE / "scripts" / runner).read_text())
    operations = {"TrainingChecks", "train", "eval_checkpoint", "validate_eval", "replay_auc"}
    def definitions(code):
        return {node.name: ast.dump(node) for node in code.body
                if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in operations}
    require(definitions(original_code) == definitions(current_code), "Smoke training or evaluation implementation differs")
    for name, sha in previous["config_hashes"].items():
        require(digest(smoke_root / "frozen_configs" / name) == sha, f"Smoke config changed: {name}")
        prior_text = (smoke_root / "frozen_configs" / name).read_text().replace(str(smoke_root), str(ROOT))
        require(json.loads(prior_text) == read(ROOT / "frozen_configs" / name), f"Smoke and formal settings differ: {name}")
    steps = plan["smoke_steps"]
    accumulation = manifest["training"]["accumulation"]
    folders = []
    for arm in ARMS:
        key = f"{arm}_seed0/checkpoint-{steps}"
        folder, checked = validate_archive(smoke_index[key], arm, 0, steps, previous, smoke_root)
        batches = round(steps * accumulation / checked["actual_epoch"])
        require(batches == manifest["training"]["dataloader_batches_per_rank_per_epoch"],
                f"Smoke dataloader length differs for {arm}")
        folders.append(folder)
    order = validate_sampling(folders, steps, accumulation, smoke=True)
    require(order == proof["rank_sampling_order_sha256"], "Smoke sampling hash differs")
    return {"passed": True, "checkpoints": len(folders), "smoke_optimizer_steps": steps,
            "final_protocol_sha256": digest(ROOT / "protocol_manifest.json"),
            "smoke_protocol_sha256": digest(smoke_root / "protocol_manifest.json"),
            "smoke_manifest_sha256": digest(smoke_root / "smoke_manifest.json"),
            "sampling_order_sha256": order,
            "dataloader_batches_per_rank_per_epoch": manifest["training"]["dataloader_batches_per_rank_per_epoch"],
            "difference": "The four-step smoke changes max_steps, save_steps and logging_steps. Paths differ."}


def summarize(values):
    require(len(values) == 3 and all(math.isfinite(x) for x in values), "Expected three finite seed values")
    return {**{f"seed{seed}": value for seed, value in zip(SEEDS, values)},
            "mean": mean(values), "sample_sd": stdev(values), "n": len(values)}


def make_tables(data, plan):
    index = {(r["arm"], r["seed"], r["optimizer_step"]): r for r in data}
    summaries, contrasts = [], []
    for step in (plan["midpoint_step"], plan["primary_step"]):
        for metric in METRICS:
            common = {"optimizer_step": step, "endpoint": "primary" if step == plan["primary_step"] else "midpoint",
                      "metric": metric, "direction": PAPER_DIRECTIONS[metric]}
            for arm in ARMS:
                values = [index[arm, seed, step][metric] for seed in SEEDS]
                summaries.append({**common, "arm": arm, **summarize(values)})
            for lhs, rhs in CONTRASTS:
                values = [index[lhs, seed, step][metric] - index[rhs, seed, step][metric] for seed in SEEDS]
                signs = "".join("+" if value > 0 else "-" if value < 0 else "0" for value in values)
                contrasts.append({**common, "left_arm": lhs, "right_arm": rhs, **summarize(values),
                                  "seed_signs": signs, "same_direction": len(set(signs)) == 1})
    return summaries, contrasts


def report(manifest, plan, verification, contrasts):
    audit = read(ROOT / "mask_audit.json")
    final = plan["primary_step"]
    mid = plan["midpoint_step"]
    lines = [
        manifest["protocol_name"],
        "",
        "All 9 runs and 18 official checkpoints completed. The 3 smoke checkpoints also passed.",
        "Every checkpoint reloaded from disk and passed full metric replay before its weight files were deleted.",
        "Corrected RMU passed both manual-value/gradient tests for unequal masks and all-empty batches.",
        f"Git commit {COMMIT}. Both ranks used the same sampling order across the three arms within each seed.",
        f"The primary endpoint was fixed at optimizer step {final} ({plan['planned_epochs'][str(final)]:.6f} dataloader epochs).",
        f"The midpoint at step {mid} ({plan['planned_epochs'][str(mid)]:.6f} dataloader epochs) describes training dynamics.",
        f"Training budget source — {manifest['budget_source']}",
        "Table 6 is a historical comparison of metric directions.",
        "",
        "Mask alignment and density",
    ]
    for arm, summary in audit["arms"].items():
        lines.append(f"{arm:>16}  {summary['selected_tokens']}/{summary['eligible_tokens']} tokens"
                     f"  {summary['aggregate_selection_ratio']:.4%}  {summary['empty_masks']} empty masks")
    lines += ["Adaptive and Random–Adaptive had identical per-sample selected counts and empty-mask patterns.",
              "The same masks and steering vector were used across seeds.",
              "", "Primary endpoint · Adaptive minus Random–Adaptive",
              f"{'Metric':<18}{'Seed 0':>12}{'Seed 1':>12}{'Seed 2':>12}{'Mean':>12}{'Sample SD':>12}  Signs"]
    primary = [r for r in contrasts if r["optimizer_step"] == final and r["left_arm"] == "adaptive"
               and r["right_arm"] == "random_adaptive"]
    for row in primary:
        lines.append(f"{row['metric']:<18}" + "".join(f"{row[key]:>+12.5g}" for key in ("seed0", "seed1", "seed2", "mean"))
                     + f"{row['sample_sd']:>12.5g}  {row['seed_signs']}")
    direction = {"lower": "-", "higher": "+"}
    better, worse, mixed = [], [], []
    for row in primary:
        metric = row["metric"]
        if row["direction"] not in direction:
            continue
        if row["same_direction"] and row["seed_signs"][0] == direction[row["direction"]]:
            better.append(metric)
        elif row["same_direction"] and row["seed_signs"][0] in "+-":
            worse.append(metric)
        else:
            mixed.append(metric)
    lines += ["", "Interpretation",
              "The matched pair isolates token selection at the same per-example density within this experiment.",
              f"Consistent improvements across seeds — {', '.join(better) or 'none'}.",
              f"Consistent decreases in performance across seeds — {', '.join(worse) or 'none'}.",
              f"Mixed directions or ties — {', '.join(mixed) or 'none'}.",
              "Use the primary endpoint's metric-specific paired effects for the main text. Mixed or tied directions,",
              "the midpoint, and descriptive three-seed variability belong in the exploratory analysis.",
              "PrivLeak preserves its raw signed value and uses magnitude for distance from zero.",
              "", "Comparison with the old Table 6"]
    legacy = ROOT / "legacy_table6.csv"
    if legacy.exists():
        with legacy.open(newline="") as file:
            old = {r["arm"]: r for r in csv.DictReader(file)}
        lines.append("Table 6 supplies three-decimal values and reports nonnegative PrivLeak.")
        for row in contrasts:
            metric = row["metric"]
            if row["optimizer_step"] != final or (row["left_arm"], row["right_arm"]) != ("adaptive", "base"):
                continue
            if not (old["adaptive"].get(metric) and old["base"].get(metric)):
                continue
            delta = float(old["adaptive"][metric]) - float(old["base"][metric])
            legacy_sign = "+" if delta > 0 else "-" if delta < 0 else "0"
            if legacy_sign == "0":
                verdict = "Table 6 is tied at its reported precision"
            elif row["seed_signs"] == legacy_sign * 3:
                verdict = "same direction in all three seeds"
            elif row["seed_signs"] == ("-" if legacy_sign == "+" else "+") * 3:
                verdict = "opposite direction in all three seeds"
            else:
                verdict = "corrected effects vary across seeds or include ties"
            lines.append(f"{metric} — legacy Adaptive minus Base {delta:+.5g}, corrected mean {row['mean']:+.5g}"
                         f" ({row['seed_signs']}), {verdict}.")
    else:
        lines.append("The old Table 6 metric cells were unavailable. Direct agreement with the old observations remains unverified.")
    lines += ["", "The CSVs include the raw seed values, signed PrivLeak and Min-40% AUC, both endpoints and all paired contrasts."]
    (ROOT / "concise_report.txt").write_text("\n".join(lines) + "\n")


def export(verification):
    files = {}
    def include(folder):
        for path in folder.rglob("*"):
            if path.is_file() and not path.is_symlink() and "__pycache__" not in path.parts:
                files[path.relative_to(ROOT).as_posix()] = path

    for folder in ("collected_results", "frozen_configs", "frozen_inputs", "logs"):
        include(ROOT / folder)
    smoke_root = ROOT / "provisional_smoke"
    for folder in ("smoke_collected", "frozen_configs", "logs", "experiment_code"):
        include(smoke_root / folder)
    for name in ("protocol_manifest.json", "run_plan.json", "smoke_manifest.json", "verification.json"):
        files[f"provisional_smoke/{name}"] = smoke_root / name
    for folder in ("src", "scripts", "configs", "tests"):
        include(SOURCE / folder)
    for name in ("protocol_manifest.json", "run_plan.json", "incremental_manifest.json", "smoke_compatibility.json", "mask_audit.json",
                 "mask_per_example.csv", "official_metrics_long.csv", "final_mean_std.csv", "midpoint_mean_std.csv",
                 "paired_comparisons.csv", "verification.json", "concise_report.txt"):
        files[name] = ROOT / name
    for name in ("legacy_table6.csv", "legacy_table6.tex"):
        if (ROOT / name).exists():
            files[name] = ROOT / name
    for name in SCRIPTS:
        files[f"experiment_code/{name}"] = WORKSPACE / "scripts" / name
    files[f"experiment_code/{NORMALIZATION_TEST.name}"] = NORMALIZATION_TEST
    require(not any(name.endswith((".safetensors", ".incomplete", ".tmp")) for name in files), "Unexpected export member")
    require(sum(name.startswith("collected_results/") and name.endswith("/MUSE_EVAL.json") for name in files) == 18,
            "Export checkpoint coverage differs")
    hashes = {name: digest(path) for name, path in files.items()}
    export_path = WORKSPACE / "saves/exports" / f"{ROOT.name}.zip"
    pending = export_path.with_suffix(".zip.part")
    manifest = {**{k: verification[k] for k in ("runs", "checkpoints", "smoke_checkpoints", "code_commit")}, "files_sha256": hashes}
    with ZipFile(pending, "w", ZIP_DEFLATED, compresslevel=4) as z:
        for name, path in sorted(files.items()):
            z.write(path, f"{ROOT.name}/{name}")
        z.writestr(f"{ROOT.name}/MANIFEST.json", json.dumps(manifest, indent=2) + "\n")
    with ZipFile(pending) as z:
        require(z.testzip() is None and len(z.namelist()) == len(hashes) + 1, "ZIP CRC or inventory differs")
        require(json.loads(z.read(f"{ROOT.name}/MANIFEST.json")) == manifest, "ZIP manifest differs")
        for name, sha in hashes.items():
            with z.open(f"{ROOT.name}/{name}") as handle:
                require(hashlib.file_digest(handle, "sha256").hexdigest() == sha, f"ZIP hash differs: {name}")
    sha = digest(pending)
    pending.replace(export_path)
    export_path.with_suffix(".zip.sha256").write_text(f"{sha}  {export_path.name}\n")
    print(f"Verified {len(hashes)} ZIP members. {export_path} ({export_path.stat().st_size} bytes), SHA-256 {sha}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-smoke", action="store_true")
    args = parser.parse_args()
    manifest, plan = locked()
    smoke_check = verify_prior_smoke(manifest, plan)
    if args.verify_smoke:
        save(ROOT / "smoke_compatibility.json", smoke_check)
        print(json.dumps(smoke_check, indent=2), flush=True)
        return
    require(read(ROOT / "smoke_compatibility.json") == smoke_check, "Smoke compatibility record differs")
    index = read(ROOT / "incremental_manifest.json")
    require(len(plan["runs"]) == 9 and len(index) == 18, "Missing official checkpoints")
    audit = read(ROOT / "mask_audit.json")
    require(audit["per_example_count_match"] and audit["empty_pattern_match"] and audit["mask_hashes"] == manifest["mask_hashes"],
            "Frozen matched density differs")
    accumulation = manifest["training"]["accumulation"]
    smoke_order = smoke_check["sampling_order_sha256"]
    rows, final_folders = [], {seed: [] for seed in SEEDS}
    for run in plan["runs"]:
        arm, seed = run["arm"], run["seed"]
        for step in run["checkpoint_steps"]:
            key = f"{arm}_seed{seed}/checkpoint-{step}"
            folder, proof = validate_archive(index[key], arm, seed, step, manifest)
            if step == plan["primary_step"]:
                final_folders[seed].append(folder)
            rows.append({"arm": arm, "seed": seed, "optimizer_step": step, "actual_epoch": proof["actual_epoch"],
                         "endpoint": "primary" if step == plan["primary_step"] else "midpoint",
                         "selection_ratio": audit["arms"][arm]["aggregate_selection_ratio"], **proof["metrics"],
                         **{k: proof[k] for k in ("code_commit", "target_revision", "dataset_revision", "mask_sha256",
                                                  "control_sha256", "protocol_sha256", "evaluator_config_sha256")}})
    orders = {str(seed): validate_sampling(folders, plan["max_steps"], accumulation) for seed, folders in final_folders.items()}
    weights = list((ROOT / "runs").rglob("*.safetensors")) + list((ROOT / "smoke").rglob("*.safetensors"))
    require(not weights, "Validated weights remain outside archive")
    summaries, contrasts = make_tables(rows, plan)
    table(ROOT / "official_metrics_long.csv", rows)
    table(ROOT / "final_mean_std.csv", [r for r in summaries if r["endpoint"] == "primary"])
    table(ROOT / "midpoint_mean_std.csv", [r for r in summaries if r["endpoint"] == "midpoint"])
    table(ROOT / "paired_comparisons.csv", contrasts)
    verification = {"runs": 9, "checkpoints": 18, "smoke_checkpoints": 3, "code_commit": COMMIT, "weights_remaining": 0,
                    "normalization_tests_passed": True, "matched_density": True, "sample_counts": EVAL_COUNTS,
                    "metric_replay_passed": True, "checkpoints_reloaded": True, "sampling_order_sha256": orders,
                    "smoke_sampling_order_sha256": smoke_order, "primary_step": plan["primary_step"],
                    "midpoint_step": plan["midpoint_step"], "checkpoint_index": index}
    save(ROOT / "verification.json", verification)
    report(manifest, plan, verification, contrasts)
    export(verification)


if __name__ == "__main__":
    main()
