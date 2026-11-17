"""Independently verify and summarize the LowConfidence-Matched experiment."""

import argparse
import math
from pathlib import Path

from prepare_low_confidence import CONTROLS, OUTPUT, controls, read, require, rows, write_csv
from run_low_confidence import ARTIFACTS, FORMAL_OUTPUT, SEEDS, STEPS, TABLE_METRICS, VARIANT, digest, load_lock, pack, replay_metrics


LABELS = {
    "ES": "ES", "forget_probability": "F-Prob", "forget_rouge": "F-ROUGE", "forget_ratio": "F-Ratio",
    "forget_quality": "F-Quality", "utility": "Utility", "privleak_signed": "PrivLeak signed", "privleak_abs": "|PrivLeak|",
}


def make_report(root):
    state = read(root / "status.json")
    require(state["phase"] == "complete" and state["archived_official_checkpoints"] == 6, "Formal runs are incomplete")
    lock = load_lock(root)
    old = read(FORMAL_OUTPUT / "protocol_manifest.json")
    _, records, _ = controls(lock["protocol"], old)
    require(records == lock["control_artifacts"], "Existing controls changed during the experiment")
    audit = read(root / "mask_audit/summary.json")
    smoke = read(root / "smoke_test/smoke_validation.json")
    require(smoke["passed"] and smoke["initial_reference_bitwise_equal"] and smoke["mask_sha256"] == lock["mask_sha256"],
            "Smoke validation differs")
    require(smoke["unit_tests_sha256"] == digest(root / "smoke_test/unit_tests.log"), "Original smoke unit-test log differs")
    manifest = read(root / "incremental_manifest.json")
    require(set(manifest) == {f"seed{seed}/checkpoint-{step}" for seed in SEEDS for step in STEPS}, "Checkpoint matrix differs")
    reference = read(root / "retain_only_reference_TOFU_EVAL.json")
    diagnostics = []
    for key, record in sorted(manifest.items()):
        checkpoint = root / "collected_results" / key
        require(all(digest(checkpoint / path) == checksum for path, checksum in record["files"].items()),
                f"Collected checksum differs: {key}")
        folder = checkpoint / "evals/official"
        require(all((folder / name).is_file() for name in ARTIFACTS), f"Missing official file: {key}")
        validated = read(folder / "validated.json")
        require(validated == record["validated"] and validated["mask_sha256"] == lock["mask_sha256"]
                and validated["evaluator_sha256"] == lock["evaluator_sha256"] and validated["code_commit"] == lock["code_commit"],
                f"Evaluation provenance differs: {key}")
        counts = validated["sample_counts"]
        require(all(counts[name] == n for name, n in {
            "forget_Q_A_Prob": 200, "retain_Q_A_Prob": 400, "ra_Q_A_Prob": 100,
            "wf_Q_A_Prob": 117, "mia_min_k/holdout": 200,
        }.items()), f"Incorrect official sample counts: {key}")
        provenance = read(folder / "evaluation_provenance.json")
        replay = replay_metrics(provenance["resolved_eval_config"], read(folder / "TOFU_EVAL.json"), reference)
        require(all(math.isclose(value, replay[name], rel_tol=1e-10, abs_tol=1e-14)
                    for name, value in validated["metrics"].items()), f"Metrics did not reproduce: {key}")
        result = dict(validated["training_diagnostics"])
        require(all(math.isfinite(result[name]) for name in ("logged_training_loss", "learning_rate_after_step", "gradient_l2_preclip")),
                f"Nonfinite training diagnostic: {key}")
        result.pop("gradient_l2_preclip_by_rank")
        diagnostics.append({"seed": validated["seed"], "step": validated["step"], **result,
                            "mask_sha256": lock["mask_sha256"], "code_commit": lock["code_commit"],
                            "evaluator_sha256": lock["evaluator_sha256"]})
    write_csv(root / "training_diagnostics.csv", diagnostics)
    require(not list((root / "runs").rglob("*.safetensors")), "Formal checkpoint weights remain")
    index = {(r["variant"], r["metric"]): r for r in rows(root / "step120_mean_std.csv")}
    pairs = rows(root / "paired_comparisons.csv")
    require(len(rows(root / "official_metrics_long.csv")) == 24 and len(index) == 32 and len(pairs) == 48,
            "Missing aggregate rows or paired comparisons")
    paired = {(r["rhs"], r["metric"]): r for r in pairs if r["step"] == "120"}
    lines = [
        "TOFU Forget05 LowConfidence-Matched",
        "",
        "A target-only absolute-confidence selection-criterion baseline using the v4 token-normalized NPO trainer.",
        "The frozen target supplied shifted log-probabilities and Adaptive supplied the per-example counts.",
        "The v4 archive controls were verified by configuration, provenance and file hashes.",
        "",
        f"Mask selects {audit['selected_tokens']:,} of {audit['eligible_tokens']:,} eligible tokens ({audit['selection_ratio']:.6%}).",
        "All 200 samples match Adaptive's selected count. Independent target forward passes reproduced the same mask hash.",
        f"Selected target log-probability median {audit['target_logprob_distribution'][0]['median']:.6f} nats. "
        f"Unselected median {audit['target_logprob_distribution'][1]['median']:.6f} nats.",
        "",
        "Overlap reference          Intersection    Precision    Recall    Micro Jaccard    Macro Jaccard",
    ]
    for variant, r in audit["overlap"].items():
        lines.append(f"{variant:<24} {r['intersection']:>10} {r['precision']:>13.2%} {r['recall']:>9.2%} "
                     f"{r['micro_jaccard']:>15.2%} {r['macro_jaccard']:>16.2%}")
    lines += ["", "Factual-span coverage      LowConfidence     Adaptive    Random-Adaptive"]
    coverage = {(r["category"], r["mask"]): r["coverage"] for r in audit["factual_coverage"]}
    for scope in ("all_answer_bearing", "direct", "descriptive"):
        lines.append(f"{scope:<26} " + " ".join(f"{coverage[scope, variant]:>13.2%}" for variant in (VARIANT, "adaptive", "random_adaptive")))
    lines += [
        "", "Primary Step 120 official results across three seeds",
        "Metric           LowConfidence mean ± SD    Adaptive mean ± SD         Random mean ± SD           Base mean ± SD",
    ]
    for metric in TABLE_METRICS:
        cells = [f"{float(index[variant, metric]['mean']):.5g} ± {float(index[variant, metric]['sample_std']):.4g}"
                 for variant in (VARIANT, *CONTROLS)]
        lines.append(f"{LABELS[metric]:<16} " + " ".join(f"{cell:<26}" for cell in cells))
    lines += [
        "", "Step 120 paired differences from Adaptive (LowConfidence minus Adaptive)",
        "Metric              Seed 0        Seed 1        Seed 2          Mean          SD       Direction",
    ]
    for metric in TABLE_METRICS:
        row = paired["adaptive", metric]
        values = [float(row[key]) for key in ("seed_0", "seed_1", "seed_2", "mean_difference", "sample_std_difference")]
        lines.append(f"{LABELS[metric]:<16}" + " ".join(f"{value:>12.5g}" for value in values) + f"   {row['direction']}")
    lines += [
        "", "All 48 paired rows, including Random-Adaptive and Base comparisons, are in paired_comparisons.csv.",
        "Step 60 is the secondary training-dynamics endpoint. The signed PrivLeak value is retained alongside its magnitude.",
        "The equal-density masks overlap only partially. The training differences are metric-specific and summarize three paired seeds.",
        "The factual spans are the existing single-rater Forget05 annotations.",
        "", "The first evaluation attempt stopped because a nested retain-only reference path differed from v4.",
        "The resumed evaluator used the original path after verifying the reference copy had identical bytes and the resolved v4 signature matched.",
        "Seed 0 training was reused. No control runs were repeated.",
        "All six new checkpoint metrics were independently aggregated from saved per-sample scores, and formal weights were cleaned after verification.",
    ]
    (root / "comparison_report.txt").write_text("\n".join(lines) + "\n")
    return pack(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=OUTPUT)
    args = parser.parse_args()
    archive = make_report(args.root.resolve())
    print(archive)


if __name__ == "__main__":
    main()
