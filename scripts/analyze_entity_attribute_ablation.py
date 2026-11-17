"""Recheck and package the paired Entity+Attribute Forget05 ablation."""

import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev
from zipfile import ZIP_DEFLATED, ZipFile

from prepare_entity_attribute_ablation import (
    ARMS, CONTROLS, EXPORTS, FORMAL, INPUTS, ROOT, SEEDS, SOURCE, STEPS, WORKSPACE, digest, read, require, save_json, write_csv,
)
from prepare_low_confidence import rows
from run_entity_attribute_ablation import (
    COPIED, EXECUTION_SOURCES, SMOKE_STEPS, WRAPPERS, execution_manifest, job_path, paired_order, protocol_signature, root_manifest,
)
from run_low_confidence import METRICS, TABLE_METRICS, replay_metrics, validate_outputs


DISPLAY = {"base": "Base", "entity": "Entity", "random_entity": "RandomEntity", **{arm: arm for arm in ARMS}}
CONTRASTS = ((ARMS[0], "entity"), (ARMS[0], ARMS[1]), (ARMS[0], "base"), ("entity", "random_entity"))
LABELS = {"ES": "ES", "forget_probability": "F-Prob", "forget_rouge": "F-ROUGE", "forget_ratio": "F-Ratio",
          "forget_quality": "F-Quality", "utility": "Utility", "privleak_abs": "|PrivLeak|", "privleak_signed": "PrivLeak signed"}


def verify_new_archive(arm, seed, step, lock, smoke=False):
    job = job_path(arm, smoke)
    key = f"seed{seed}/checkpoint-{step}"
    checkpoint = job / "collected_results" / key
    archived = read(job / "incremental_manifest.json")[key]
    files = archived["files"]
    require({p.relative_to(checkpoint).as_posix() for p in checkpoint.rglob("*") if p.is_file()} == set(files)
            and all(digest(checkpoint / name) == checksum for name, checksum in files.items()), f"Archive changed: {arm}/{key}")
    folder = checkpoint / "evals/official"
    valid, proof = [read(folder / name) for name in ("validated.json", "evaluation_provenance.json")]
    config = proof["resolved_eval_config"]
    state = read(checkpoint / "trainer_state.json")
    epoch = .64 if smoke else lock["protocol"]["training"]["checkpoints"][str(step)]
    require(valid == archived["validated"] and valid["variant"] == arm and valid["seed"] == seed
            and valid["global_step"] == state["global_step"] == step
            and valid["actual_epoch"] == state["epoch"] == epoch
            and valid["mask_sha256"] == lock["mask_hashes"][arm] and valid["code_commit"] == lock["code_commit"]
            and valid["protocol_sha256"] == lock["v4_protocol_sha256"]
            and valid["evaluator_sha256"] == proof["evaluator_sha256"] == lock["evaluator_sha256"],
            f"Checkpoint provenance differs: {arm}/{key}")
    signature = {k: v for k, v in config.items() if k not in {"output_dir", "retain_logs_path"}}
    ref = INPUTS / "TOFU_EVAL.json"
    require(signature == read(ROOT / "v4_evaluation_signature.json") and digest(ref) == proof["retain_only_reference_sha256"],
            "Evaluator config differs from archived v4")
    require(proof["weight_sha256"] and set(proof["weight_sha256"]) == set(valid["saved_weight_files"]),
            "Checkpoint reload proof has missing weight hashes")
    require(all(files[f"evals/official/{name}"] == sha for name, sha in valid["artifact_sha256"].items()),
            "Archive evaluation hashes differ")
    logs = read(folder / "TOFU_EVAL.json")
    metrics, counts = validate_outputs(config, lock["protocol"]["official_evaluation"], logs, read(folder / "TOFU_SUMMARY.json"), ref)
    replay = replay_metrics(config, logs, read(ref))
    require(set(metrics) == set(METRICS) and metrics == read(folder / "official_metrics.json") == valid["metrics"]
            and counts == valid["sample_counts"] and all(math.isclose(value, replay[k], rel_tol=1e-10, abs_tol=1e-14)
                                                        for k, value in metrics.items()), f"Metric replay differs: {arm}/{key}")
    run = checkpoint.parent
    run_cfg = read(run / "resolved_config.json")
    require(run_cfg == read(checkpoint / "resolved_config.json")
            and run_cfg["execution_manifest_sha256"] == digest(ROOT / "execution_manifest.json")
            and run_cfg["extension_manifest_sha256"] == digest(ROOT / "protocol_manifest.json"),
            f"Resolved protocol proof differs: {arm}/{key}")
    if not smoke:
        require(protocol_signature(run_cfg, read(run / "training_args.json")) == read(ROOT / "training_signature.json"),
                f"v4 training signature differs: {arm}/{key}")
    for rank in (0, 1):
        checks = read(run / f"finite_checks_rank{rank}.json")
        max_steps = 4 if smoke else 120
        require(len(checks["loss"]) == 4 * max_steps and len(checks["gradient_l2_preclip"]) == max_steps
                and all(map(math.isfinite, checks["loss"] + checks["gradient_l2_preclip"]))
                and checks["actual_mask_sha256"] == valid["mask_sha256"]
                and (not smoke or checks["initial_reference_bitwise_equal"]), f"Training checks differ: {arm}/{seed}/{rank}")
    require(not list((job / f"runs/seed{seed}" / checkpoint.name).glob("*.safetensors")),
            f"Validated weights were not cleaned: {arm}/{key}")
    signed = metrics.pop("privleak")
    return {**metrics, "privleak_signed": signed, "privleak_abs": abs(signed)}, {
        "validated_sha256": digest(folder / "validated.json"), "sample_counts": counts, "actual_epoch": epoch}


def summarize(values):
    return {"seed0": values[0], "seed1": values[1], "seed2": values[2],
            "mean": mean(values), "sample_sd": stdev(values), "n": len(values)}


def make_tables(records):
    lookup = {(row["variant"], row["seed"], row["step"]): row for row in records}
    summaries, pairs = [], []
    for step in STEPS:
        for metric in ("selection_ratio", *TABLE_METRICS):
            for arm in (*CONTROLS, *ARMS):
                summaries.append({"step": step, "variant": arm, "metric": metric,
                                  **summarize([lookup[arm, seed, step][metric] for seed in SEEDS])})
            for arm, control in CONTRASTS:
                values = [lookup[arm, seed, step][metric] - lookup[control, seed, step][metric] for seed in SEEDS]
                signs = "".join("+" if val > 0 else "-" if val < 0 else "0" for val in values)
                pairs.append({"step": step, "variant": arm, "control": control, "metric": metric, **summarize(values),
                              "seed_signs": signs, "same_direction": len(set(signs)) == 1})
    return summaries, pairs


def comparison_table(summaries):
    primary = {(r["variant"], r["metric"]): r for r in summaries if r["step"] == 120}
    names = ("selection_ratio", "ES", "forget_probability", "forget_rouge",
             "forget_ratio", "forget_quality", "utility", "privleak_abs")
    table = []
    for arm in (*CONTROLS, *ARMS):
        cells = {}
        for metric in names:
            r = primary[arm, metric]
            label = LABELS.get(metric, "Selection ratio")
            cells[label] = (f"{r['mean']:.2%} ± {r['sample_sd']:.2%}" if metric == "selection_ratio"
                            else f"{r['mean']:.6g} ± {r['sample_sd']:.3g}")
        table.append({"Mask": DISPLAY[arm], **cells})
    return table


def report(table, comparisons, audit):
    lines = ["TOFU Forget05 Entity+Attribute mask ablation", "",
             "Six new summed masked NPO runs and twelve official checkpoints passed. Three frozen v4 controls were imported.",
             "Step 120 is the primary endpoint. Step 60 describes training dynamics.",
             "All 200 samples had identical selection counts for EntityAttribute and its Random control.",
             f"The union selected {audit['totals']['entity_attribute']}/{audit['totals']['eligible']} eligible tokens "
             f"({audit['selection_ratios'][ARMS[0]]:.2%}). Both arms had {audit['empty_masks'][ARMS[0]]} empty masks.",
             "", "Step 120 mean ± sample SD", " | ".join(table[0])]
    lines.extend(" | ".join(row.values()) for row in table)
    lines += ["", "Step 120 paired differences by seed", "Mask minus control | Metric | Seed 0 | Seed 1 | Seed 2 | Mean ± sample SD | Signs"]
    for row in comparisons:
        if row["step"] != 120 or row["metric"] == "selection_ratio":
            continue
        lhs, rhs, metric = row["variant"], row["control"], row["metric"]
        lines.append(f"{DISPLAY[lhs]} − {DISPLAY[rhs]} | {LABELS[metric]} | "
                     + " | ".join(f"{row[f'seed{s}']:+.6g}" for s in SEEDS)
                     + f" | {row['mean']:+.6g} ± {row['sample_sd']:.3g} | {row['seed_signs']}")
    lines += ["", "Negative ES, F-Prob, F-ROUGE and absolute PrivLeak differences favor the first arm.",
              "Positive F-Ratio, F-Quality and Utility differences favor the first arm.",
              "Signed PrivLeak is retained as a diagnostic. Comparisons use distance from zero."]
    (ROOT / "report.txt").write_text("\n".join(lines) + "\n")


def pack(verification, lock):
    files = {}
    for name in ("protocol_manifest.json", "execution_manifest.json", "run_plan.json", "archived_controls.csv",
                 "v4_evaluation_signature.json", "training_signature.json", "mask_audit.json", "mask_per_sample.csv",
                 "annotation_alignment.json", "decoded_examples.json", "smoke_verification.json", "verification.json",
                 "official_metrics_long.csv", "step120_mean_std.csv", "step60_mean_std.csv", "paired_comparisons.csv",
                 "step120_comparison_table.csv", "report.txt"):
        files[name] = ROOT / name
    for folder in ("frozen_inputs", "logs"):
        files.update({p.relative_to(ROOT).as_posix(): p for p in (ROOT / folder).rglob("*") if p.is_file()})
    for stage in ("jobs", "smoke"):
        for arm in ARMS:
            job = ROOT / stage / arm
            files[f"{stage}/{arm}/incremental_manifest.json"] = job / "incremental_manifest.json"
            files.update({p.relative_to(ROOT).as_posix(): p for p in (job / "collected_results").rglob("*") if p.is_file()})
    files["frozen_inputs/retain_only_TOFU_EVAL.json"] = INPUTS / "TOFU_EVAL.json"
    files["provenance/v4_environment.json"] = FORMAL / "environment.json"
    files["provenance/v4_protocol_manifest.json"] = FORMAL / "protocol_manifest.json"
    for name, checksum in lock["control_artifacts"].items():
        path = FORMAL / "collected_results" / name.removeprefix("runs/")
        require(digest(path) == checksum, f"Archived control changed: {name}")
        files[f"v4_controls/{name}"] = path
    for name in WRAPPERS:
        files[f"experiment_code/{name}"] = WORKSPACE / name
    for name in (*lock["source_sha256"], *EXECUTION_SOURCES):
        files[f"v4_source/{name}"] = SOURCE / name
    require(not any(name.endswith((".safetensors", ".part")) for name in files), "Unexpected weights in export")
    hashes = {name: digest(path) for name, path in files.items()}
    archive = EXPORTS / f"{ROOT.name}.zip"
    partial = archive.with_suffix(".zip.part")
    manifest = {**{k: verification[k] for k in ("new_runs", "new_checkpoints", "smoke_checkpoints", "imported_checkpoints",
                                                "code_commit", "primary_step")}, "files_sha256": hashes}
    with ZipFile(partial, "w", ZIP_DEFLATED, compresslevel=4) as out:
        for name, path in sorted(files.items()):
            out.write(path, f"{ROOT.name}/{name}")
        out.writestr(f"{ROOT.name}/MANIFEST.json", json.dumps(manifest, indent=2) + "\n")
    with ZipFile(partial) as out:
        require(out.testzip() is None and len(out.namelist()) == len(hashes) + 1, "ZIP CRC or inventory differs")
        require(json.loads(out.read(f"{ROOT.name}/MANIFEST.json")) == manifest, "ZIP manifest differs")
        for name, sha in hashes.items():
            with out.open(f"{ROOT.name}/{name}") as stream:
                require(hashlib.file_digest(stream, "sha256").hexdigest() == sha, f"ZIP member changed: {name}")
    sha = digest(partial)
    partial.replace(archive)
    archive.with_suffix(".zip.sha256").write_text(f"{sha}  {archive.name}\n")
    print(f"Verified {len(hashes)} members. {archive} ({archive.stat().st_size} bytes) SHA-256 {sha}", flush=True)


def main():
    lock = root_manifest()
    execution_manifest()
    audit = read(ROOT / "mask_audit.json")
    smoke = read(ROOT / "smoke_verification.json")
    require(smoke["passed"] and smoke["validated_checkpoints"] == len(ARMS), "Missing smoke proof")
    smoke_proofs = {arm: verify_new_archive(arm, 0, SMOKE_STEPS[-1], lock, smoke=True)[1] for arm in ARMS}
    records, new_proofs = [], {}
    for row in rows(ROOT / "archived_controls.csv"):
        records.append({"variant": row["variant"], "seed": int(row["seed"]), "step": int(row["step"]),
                        "source": "validated_v4_export", "actual_epoch": float(row["actual_epoch"]),
                        "selection_ratio": float(row["selection_ratio"]), **{k: float(row[k]) for k in TABLE_METRICS}})
    orders = {}
    for seed in SEEDS:
        orders[str(seed)] = paired_order(seed, False, lock)
        for arm in ARMS:
            for step in STEPS:
                metrics, proof = verify_new_archive(arm, seed, step, lock)
                records.append({"variant": arm, "seed": seed, "step": step, "source": "entity_attribute_ablation",
                                "actual_epoch": proof["actual_epoch"], "selection_ratio": audit["selection_ratios"][arm], **metrics})
                new_proofs[f"{arm}/seed{seed}/checkpoint-{step}"] = proof
    require(len(records) == 30 and len(new_proofs) == 12, "Combined checkpoint coverage differs")
    summary, comparisons = make_tables(records)
    table = comparison_table(summary)
    for name, data in (("official_metrics_long", records), ("paired_comparisons", comparisons),
                       ("step120_mean_std", [r for r in summary if r["step"] == 120]),
                       ("step60_mean_std", [r for r in summary if r["step"] == 60]), ("step120_comparison_table", table)):
        write_csv(ROOT / f"{name}.csv", data)
    report(table, comparisons, audit)
    verification = {"new_runs": 6, "new_checkpoints": 12, "smoke_checkpoints": 2, "imported_checkpoints": 18,
                    "code_commit": lock["code_commit"], "primary_step": 120, "all_validated": True,
                    "weights_remaining": 0, "seed_sample_order_sha256": orders, "mask_hashes": lock["mask_hashes"],
                    "smoke": smoke_proofs, "checkpoints": new_proofs}
    save_json(ROOT / "verification.json", verification)
    pack(verification, lock)


if __name__ == "__main__":
    main()
