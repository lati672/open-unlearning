"""Freeze rank-softmax weights with the Phase A per-sample ESS budget."""

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.stats import rankdata

# Importing the Phase B entry puts the pinned Phase A modules first.
from run_density_ablation_phase_b import A, B, SOURCE, WORKSPACE, verify_git_tree
from prepare_influence_adaptive import (CONTROLS, DIAGNOSTICS, control_archive, digest, distribution, frozen_jsonl,
                                        read, require, rows, save_json, write_csv)
from freeze_mask_protocol import check_environment, save_jsonl
from run_density_ablation import load_phase
from run_low_confidence import FORMAL_OUTPUT, replay_metrics, validate_outputs

NAME = "Soft-Adaptive-ESS34"
ROOT = WORKSPACE / "saves/experiments/tofu_forget05_soft_adaptive_ess34_seed0_v1"
STEPS = (20, 40, 60, 120)
EXTENSION = ("scripts/prepare_soft_adaptive.py", "scripts/run_soft_adaptive.py", "scripts/analyze_soft_adaptive.py",
             "src/data/frozen_token_weights.py", "src/trainer/unlearn/weighted_npo.py", "tests/test_soft_adaptive.py")
SUPPORT = ("scripts/prepare_influence_adaptive.py",
           "scripts/run_density_ablation_phase_b.py", "scripts/run_density_ablation.py",
           "scripts/run_low_confidence.py", "scripts/resume_low_confidence.py")


def ess_weights(delta, budget):
    n = len(delta)
    if not 1 <= budget <= n or not np.isfinite(delta).all():
        raise ValueError("ESS requires finite discrepancies and a budget between one and the sequence length.")
    if n == 1:
        return np.ones(1), np.ones(1), 1.0, 1.0, None
    rank = (rankdata(delta, method="average") - 1) / (n - 1)
    shifted = rank - rank.max()

    def at(inverse_tau):
        w = np.exp(shifted * inverse_tau)
        w /= w.sum()
        return w, float(1 / (w @ w))

    # Positive finite tau and positive normal FP32 weight at the cold limit.
    lo, hi = 1e-10, 80.0 / max(float(np.ptp(rank)), 1e-10)
    cold, cold_ess = at(hi)
    warm, warm_ess = at(lo)
    if cold_ess > budget:
        return rank, cold, 1 / hi, cold_ess, "cold limit or tied highest ranks"
    if warm_ess <= budget:
        return rank, warm, 1 / lo, warm_ess, "uniform limit"
    for _ in range(100):
        mid = (lo + hi) / 2
        _, ess = at(mid)
        if ess > budget:
            lo = mid
        else:
            hi = mid
    inv = min((lo, hi), key=lambda inverse: abs(at(inverse)[1] - budget))
    w, ess = at(inv)
    return rank, w, 1 / inv, ess, None


def verify_controls(phase):
    """Replay all eight archived control evaluations and record their file hashes."""
    b = read(B / "phase_b_manifest.json")
    source = A / "protocol_manifest.json"
    require(digest(source) == digest(B / source.name) == b["phase_a_protocol_sha256"], "Phase A protocol differs")
    table = {(r["variant"], int(r["step"])): r for r in rows(B / "official_metrics_long.csv") if r["seed"] == "0"}
    f = A / "retain_only_reference_TOFU_EVAL.json"
    protocol = phase["protocol"]
    require(digest(f) == protocol["inputs"][protocol["retain_only_model"]["evaluation_logs"]], "Retain-only logs differ")
    ref = read(f)
    signature = read(A / "v4_evaluation_signature.json")
    control_rows, hashes = [], {}
    for arm in CONTROLS:
        archive = control_archive(arm)
        index_path = archive.parents[1] / "incremental_manifest.json"
        require(digest(index_path) == b["phase_a_archives_sha256"][arm], f"Archive index differs: {arm}")
        index = read(index_path)
        for p in archive.glob("*.json"):
            hashes[f"{arm}/{p.name}"] = digest(p)
        snapshot, args = read(archive / "resolved_config.json"), read(archive / "training_args.json")
        require(snapshot["trainer"]["handler"] == "TokenNormalizedNPO" and
                snapshot["trainer"]["method_args"]["normalization"] == "token_mean" and
                snapshot["mask_sha256"] == phase["mask_hashes"][arm] and args["seed"] == 0, f"Control config differs: {arm}")
        for step in STEPS:
            cp = archive / f"checkpoint-{step}"
            archived = index[f"seed0/checkpoint-{step}"]
            for name, checksum in archived["files"].items():
                require(digest(cp / name) == checksum, f"Control changed: {arm}/{step}/{name}")
                hashes[f"{arm}/checkpoint-{step}/{name}"] = checksum
            ev = cp / "evals/official"
            proof = read(ev / "validated.json")
            provenance = read(ev / "evaluation_provenance.json")
            config = provenance["resolved_eval_config"]
            require(proof == archived["validated"] and proof["seed"] == 0 and proof["global_step"] == step
                    and proof["actual_epoch"] == phase["actual_epochs"][str(step)]
                    and proof["mask_sha256"] == phase["mask_hashes"][arm] and proof["code_commit"] == phase["code_commit"]
                    and proof["evaluator_sha256"] == phase["evaluator_sha256"]
                    and proof["protocol_sha256"] == phase["v4_protocol_sha256"]
                    and {k: v for k, v in config.items() if k not in {"output_dir", "retain_logs_path"}} == signature,
                    f"Control provenance differs: {arm}/{step}")
            logs = read(ev / "TOFU_EVAL.json")
            metrics, counts = validate_outputs(config, protocol["official_evaluation"], logs, read(ev / "TOFU_SUMMARY.json"), f)
            replay = replay_metrics(config, logs, ref)
            row = table[arm, step]
            require(metrics == proof["metrics"] and counts == proof["sample_counts"] and all(
                np.isclose(replay[k], v, rtol=1e-10, atol=1e-14) and
                v == float(row[k if k != "privleak" else "privleak_signed"]) for k, v in metrics.items()),
                f"Control replay differs: {arm}/{step}")
            update = read(archive / "optimizer_update_norms.json")[str(step)]
            require(all(row[k] and np.isfinite(float(row[k])) for k in DIAGNOSTICS)
                    and all(float(row[k]) == v for k, v in update.items()), "Control diagnostics differ")
            control_rows.append(row)
    return control_rows, hashes


def audit_weights(phase):
    frozen = FORMAL_OUTPUT / "frozen_inputs"
    target_delta = frozen / "tokens_TOFU_forget05.jsonl"
    base_mask = frozen / "masks_TOFU_forget05_base.jsonl"
    base_logp = frozen / "logprobs.json"
    d34 = A / "masks/Discrepancy-34.jsonl"
    r34 = A / "masks/Random-34.jsonl"
    a = A / "mask_audit/eligible_tokens.csv"
    expected = read(FORMAL_OUTPUT / "protocol_manifest.json")["derived_inputs"]
    inputs = {Path(p).name: h for p, h in phase["protocol"]["inputs"].items()}
    require(digest(target_delta) == phase["source_discrepancy_sha256"]
            and digest(d34) == phase["mask_hashes"]["Discrepancy-34"]
            and digest(r34) == phase["mask_hashes"]["Random-34"]
            and digest(base_mask) == expected[base_mask.name] and digest(base_logp) == inputs[base_logp.name],
            "Frozen rank or mask source differs")
    meta = base_logp.with_suffix(".metadata.json")
    base = phase["protocol"]["base_model"]
    base_metadata = read(meta)
    target_meta = frozen / "masks_TOFU_forget05_adaptive.metadata.json"
    target = phase["protocol"]["target_model"]
    target_metadata = read(target_meta)
    require(base_metadata["model"] == base["id"] and base_metadata["model_commit"] == base["revision"]
            and target_metadata["logprob_sha256"] == digest(base_logp)
            and digest(target_meta) == inputs[target_meta.name]
            and target_metadata.get("target_model", target_metadata.get("original_model")) == target["id"]
            and target_metadata.get("target_model_commit", target_metadata.get("original_model_commit")) == target["revision"],
            "Discrepancy model revision differs")
    annotation = {(int(row["doc_id"]), int(row["position"])): row for row in rows(a)}
    records, stats, details, tokens, examples = [], [], [], [], []
    for doc, eligibility, discrepancy, random, logp in zip(*(frozen_jsonl(p) for p in
            (target_delta, base_mask, d34, r34, base_logp)), strict=True):
        index, ids = doc["doc_id"], eligibility["input_ids"]
        eligible = [p for p, t in enumerate(eligibility["mask"]) if t]
        chosen = [p for p, t in enumerate(discrepancy["mask"]) if t]
        k = len(chosen)
        labels = [t["label"] for t in doc["tokens"]]
        require(discrepancy["input_ids"] == random["input_ids"] == ids and sum(random["mask"]) == k
                and len(ids) == len(labels) and eligible == [p for p, t in enumerate(doc["tokens"]) if t["eligible"]]
                and (doc["question"], doc["answer"]) == (logp["question"], logp["answer"])
                and all(p > 0 and ids[p] == labels[p] and np.isfinite(logp["token_logprobs"][p]) for p in eligible),
                f"Discrepancy/label eligibility differs for sample {index}")
        delta = np.asarray([-doc["tokens"][p]["delta"] for p in eligible])
        rank, weight, tau, ess, boundary = ess_weights(delta, k)
        full = np.zeros(len(ids), dtype=np.float32)
        full[eligible] = weight
        order = sorted(eligible, key=lambda p: (-full[p], p))
        require(set(order[:k]) == set(chosen) and np.all(full[eligible] > 0) and np.isfinite(tau) and tau > 0
                and abs(np.sum(full, dtype=np.float64) - 1) < 1e-7, f"Top-k or FP32 normalization differs for sample {index}")
        runtime_ess = 1 / np.square(full.astype(np.float64)).sum()
        summary = {"doc_id": index, "eligible_count": len(eligible), "binary_count": k, "tau": tau,
                   "ESS": ess, "ess_absolute_error": abs(ess - k), "fp32_ESS": runtime_ess,
                   "fp32_ess_absolute_error": abs(runtime_ess - k), "entropy": float(-np.sum(weight * np.log(weight))),
                   "normalized_entropy": float(-np.sum(weight * np.log(weight)) / np.log(len(eligible))) if len(eligible) > 1 else 1.,
                   "maximum_weight": float(weight.max()), "top_k_mass": float(np.sum(full[chosen], dtype=np.float64)),
                   "boundary": boundary}
        per_token = []
        for pos, d, r in zip(eligible, delta, rank, strict=True):
            t = annotation[index, pos]
            require(int(t["token_id"]) == ids[pos] and int(t["logit_position"]) == pos - 1 and
                    float(t["delta"]) == -d, "Frozen annotation alignment differs")
            per_token.append({"doc_id": index, "label_position": pos, "logit_position": pos - 1, "token_id": ids[pos],
                              "decoded_token": t["token"], "delta": d, "rank": r, "weight": float(full[pos]),
                              "binary_top_k": int(pos in chosen), "category": t["category"], "fact_scope": t["fact_scope"]})
        records.append({"doc_id": index, "input_ids": ids, "labels": labels, "weights": full.tolist()})
        details.append({**summary, "tokens": per_token})
        stats.append(summary)
        tokens.extend(per_token)
        examples.append({**summary, "question": doc["question"], "answer": doc["answer"], "tokens": per_token})
    require(len(stats) == 200 and len(tokens) == len(annotation) == 7165 and sum(r["binary_count"] for r in stats) == 2448,
            "Weight audit inventory differs")
    per_doc_k = {r["doc_id"]: r["binary_count"] for r in stats}
    mass_categories, binary_categories = Counter(), Counter()
    factual = binary_factual = 0.
    for t in tokens:
        binary = t["binary_top_k"] / per_doc_k[t["doc_id"]]
        mass_categories[t["category"]] += t["weight"]
        binary_categories[t["category"]] += binary
        if t["fact_scope"]:
            factual += t["weight"]
            binary_factual += binary
    total = sum(mass_categories.values())
    top = sorted([t["weight"] for t in tokens], reverse=True)
    concentrated = [r["doc_id"] for r in stats if r["maximum_weight"] > .5]
    save_jsonl(ROOT / "weights.jsonl", records)
    save_jsonl(ROOT / "per_example_weights.jsonl", details)
    write_csv(ROOT / "per_example_weight_statistics.csv", stats)
    write_csv(ROOT / "token_weights.csv", tokens)
    # Include three high-mass cases and seven evenly spaced examples.
    picked = sorted(examples, key=lambda r: (-r["maximum_weight"], r["doc_id"]))[:3]
    remaining = [r for r in examples if r["doc_id"] not in {p["doc_id"] for p in picked}]
    picked += [remaining[i] for i in np.linspace(0, len(remaining) - 1, 7, dtype=int)]
    save_json(ROOT / "decoded_examples.json", picked)
    result = {"examples": len(stats), "eligible_tokens": len(tokens), "binary_budget": 2448,
              "aggregate_nominal_ratio": 2448 / len(tokens), "exact_top_k_samples": 200, "positive_weight_tokens": len(tokens),
              "tau": distribution([r["tau"] for r in stats]),
              "ess_absolute_error": distribution([r["ess_absolute_error"] for r in stats]),
              "fp32_ess_absolute_error": distribution([r["fp32_ess_absolute_error"] for r in stats]),
              "entropy_nats": distribution([r["entropy"] for r in stats]),
              "normalized_entropy": distribution([r["normalized_entropy"] for r in stats]),
              "maximum_weight": distribution([r["maximum_weight"] for r in stats]),
              "top_k_mass": distribution([r["top_k_mass"] for r in stats]),
              "top_token_weight_mass": {str(p): sum(top[:int(np.ceil(len(top) * p / 100))]) / total for p in (1, 5, 10)},
              "category_weight_mass": {k: v / total for k, v in sorted(mass_categories.items())},
              "binary_category_weight_mass": {k: v / len(stats) for k, v in sorted(binary_categories.items())},
              "factual_weight_mass": factual / total, "binary_factual_weight_mass": binary_factual / len(stats),
              "samples_with_single_token_mass_above_half": concentrated,
              "boundary_samples": [r for r in stats if r["boundary"] is not None]}
    save_json(ROOT / "weight_audit.json", result)
    return [target_delta, base_mask, base_logp, meta, target_meta, d34, r34, a, a.parent / "answer_spans.tsv"]


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    require(not (ROOT / "protocol_manifest.json").exists(), "Soft-Adaptive inputs are already frozen")
    require((ROOT / "unit_tests.log").read_text().endswith("\nOK\n"), "Weighted NPO unit tests did not pass")
    phase = load_phase(A)
    check_environment(phase["protocol"])
    count = verify_git_tree(phase["code_commit"])
    control_rows, archives = verify_controls(phase)
    sources = audit_weights(phase)
    write_csv(ROOT / "existing_controls.csv", control_rows)
    for name in ("retain_only_reference_TOFU_EVAL.json", "v4_evaluation_signature.json"):
        shutil.copyfile(A / name, ROOT / name)
    source_dir = ROOT / "frozen_sources"
    source_dir.mkdir(exist_ok=True)
    for path in sources:
        shutil.copyfile(path, source_dir / path.name)
    weight_path = ROOT / "weights.jsonl"
    weight_hash = digest(weight_path)
    (ROOT / "weights.jsonl.sha256").write_text(f"{weight_hash}  {weight_path.name}\n")
    protocol = deepcopy(phase["protocol"])
    protocol["training"]["checkpoints"] = phase["actual_epochs"]
    protocol["checkpoint_validation"] = {"max_steps": 4, "save_steps": 4, "checkpoints": {"4": .64}}
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in ROOT.rglob("*") if p.is_file() and
              p.parent in {ROOT, source_dir}}
    lock = {"method": NAME, "seed": 0, "steps": STEPS, "max_steps": 120, "protocol": protocol,
            "code_commit": phase["code_commit"], "verified_git_blobs": count, "v4_protocol_sha256": phase["v4_protocol_sha256"],
            "evaluator_sha256": phase["evaluator_sha256"], "phase_a_sha256": digest(A / "protocol_manifest.json"),
            "phase_b_manifest_sha256": digest(B / "phase_b_manifest.json"),
            "mask_sha256": weight_hash, "weights_sha256": weight_hash, "input_hashes": hashes,
            "extension_hashes": {p: digest(WORKSPACE / p) for p in EXTENSION},
            "support_hashes": {p: digest(SOURCE / p if (SOURCE / p).is_file() else WORKSPACE / p) for p in SUPPORT},
            "control_artifact_hashes": archives,
            "delta": "Negate the hash-verified frozen base-minus-target log-probability difference for the pinned models",
            "ranking": "sample-local ascending average fractional rank; lexical label position breaks top-k ties",
            "ess_solver": "bisect inverse temperature in FP64; positive runtime FP32 weights; record numerical limits",
            "screening": {"max_update_ratio": 1.10, "nearest_calipers": [.005, .01, .02],
                          "require_distinct_soft_checkpoints_passing": 2,
                          "extreme_concentration": "any sample above 50% weight on a single token or global top 1% above 25% of mass",
                          "gap": "Both outcomes improve relative to D34 and close a positive R34 advantage on one matched axis",
                          "interpolation": "Observed Soft Pareto checkpoints against control Pareto frontier linear interpolation within support; use maximum of two bracket update ratios",
                          "nearest": "Observed checkpoints inside common support; higher Utility and lower F-Prob count as improvements"}}
    save_json(ROOT / "protocol_manifest.json", lock)
    print(f"Frozen {weight_hash}; verified {len(control_rows)} archival control checkpoints", flush=True)


if __name__ == "__main__":
    main()
