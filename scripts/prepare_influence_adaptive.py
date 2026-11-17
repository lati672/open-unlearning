"""Score frozen Forget05 label tokens and prepare a density-matched influence mask."""

import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

import numpy as np
from omegaconf import OmegaConf
from scipy.stats import pearsonr, spearmanr
import torch

from run_density_ablation_phase_b import A, B, SOURCE, WORKSPACE, verify_git_tree

from data import get_data
from data.collators import DataCollatorForSupervisedDataset, FrozenMaskCollator
from freeze_mask_protocol import check_environment, save_jsonl, verify_dataset_revision
from model import get_model
from prepare_low_confidence import read, require, rows, save_json, write_csv
from run_density_ablation import load_phase
from run_low_confidence import FORMAL_OUTPUT, digest, replay_metrics, validate_outputs
from trainer.masking import eligible_positions

ROOT = WORKSPACE / "saves/experiments/tofu_forget05_influence_adaptive_seed0_v1"
VARIANT = "Influence-Adaptive"
CONTROLS = ("Discrepancy-34", "Random-34")
STEPS = (20, 60, 120)
DIAGNOSTICS = ("logged_training_loss", "gradient_l2_preclip", "optimizer_step_update_l2",
               "parameter_l2", "learning_rate_used", "learning_rate_after_step")
EPS = 1e-8


def control_archive(arm):
    return A / "jobs" / arm / "collected_results/seed0"


def frozen_jsonl(path):
    data = [json.loads(line) for line in path.read_text().splitlines()]
    require([r["doc_id"] for r in data] == list(range(200)), f"Unstable IDs: {path}")
    return data


def verify_controls(phase):
    b = read(B / "phase_b_manifest.json")
    require(read(B / "status.json")["new_official_checkpoints"] == 48 and
            digest(A / "protocol_manifest.json") == digest(B / "protocol_manifest.json") == b["phase_a_protocol_sha256"],
            "Phase B source protocol differs")
    table = {(r["variant"], int(r["step"])): r for r in rows(B / "official_metrics_long.csv") if r["seed"] == "0"}
    ref_path = A / "retain_only_reference_TOFU_EVAL.json"
    reference = read(ref_path)
    settings = phase["protocol"]["official_evaluation"]
    reference_key = phase["protocol"]["retain_only_model"]["evaluation_logs"]
    require(digest(ref_path) == phase["protocol"]["inputs"][reference_key], "Retain-only logs differ")
    signature = read(A / "v4_evaluation_signature.json")
    artifacts, controls = {}, []
    common = None
    for arm in CONTROLS:
        folder = control_archive(arm)
        index_path = folder.parents[1] / "incremental_manifest.json"
        require(digest(index_path) == b["phase_a_archives_sha256"][arm], f"Phase A archive index changed: {arm}")
        index = read(index_path)
        config = read(folder / "resolved_config.json")
        args = read(folder / "training_args.json")
        require(config["trainer"]["handler"] == "TokenNormalizedNPO" and
                config["trainer"]["method_args"]["normalization"] == "token_mean" and
                config["execution"]["seed"] == args["seed"] == 0 and
                config["model"]["model_args"]["revision"] == phase["protocol"]["target_model"]["revision"] and
                config["mask_sha256"] == phase["mask_hashes"][arm], f"Control config differs: {arm}")
        method = {k: v for k, v in config["trainer"]["method_args"].items() if k != "mask"}
        train_args = {k: v for k, v in args.items() if k not in {"output_dir", "logging_dir", "run_name"}}
        match = [config["model"], config["data"], method, train_args, config["frozen_protocol"]]
        require(common is None or match == common, f"Control training signature differs: {arm}")
        common = match
        for name in ("resolved_config.json", "training_args.json", "optimizer_update_norms.json"):
            artifacts[f"{arm}/{name}"] = digest(folder / name)
        for step in STEPS:
            cp = folder / f"checkpoint-{step}"
            item = index[f"seed0/checkpoint-{step}"]
            for name, expected in item["files"].items():
                require(digest(cp / name) == expected, f"Control artifact differs: {arm}/{step}/{name}")
                artifacts[f"{arm}/checkpoint-{step}/{name}"] = expected
            ev = cp / "evals/official"
            proof = read(ev / "validated.json")
            provenance = read(ev / "evaluation_provenance.json")
            cfg = provenance["resolved_eval_config"]
            require(proof == item["validated"] and proof["seed"] == 0 and proof["global_step"] == step
                    and proof["mask_sha256"] == phase["mask_hashes"][arm]
                    and proof["code_commit"] == phase["code_commit"]
                    and proof["evaluator_sha256"] == phase["evaluator_sha256"]
                    and proof["protocol_sha256"] == phase["v4_protocol_sha256"]
                    and {k: v for k, v in cfg.items() if k not in {"output_dir", "retain_logs_path"}} == signature,
                    f"Control provenance differs: {arm}/{step}")
            logs = read(ev / "TOFU_EVAL.json")
            metrics, counts = validate_outputs(cfg, settings, logs, read(ev / "TOFU_SUMMARY.json"), ref_path)
            replay = replay_metrics(cfg, logs, reference)
            row = table[arm, step]
            require(metrics == proof["metrics"] and counts == proof["sample_counts"] and all(
                    math.isclose(v, replay[k], abs_tol=1e-14, rel_tol=1e-10) and
                    v == float(row[k if k != "privleak" else "privleak_signed"]) for k, v in metrics.items()),
                    f"Control replay differs: {arm}/{step}")
            norms = read(folder / "optimizer_update_norms.json")[str(step)]
            require(all(float(row[k]) == v for k, v in norms.items()) and
                    all(math.isfinite(float(row[k])) for k in DIAGNOSTICS) and
                    float(row["actual_epoch"]) == phase["actual_epochs"][str(step)]
                    and float(row["selection_ratio"]) == 2448 / 7165, "Control diagnostics or density differs")
            controls.append(row)
    return controls, artifacts


def local_head_scores(head, hidden, labels):
    # The summed backward gives independent gradients for each row because
    # the LM head and cross entropy do not mix positions.
    with torch.enable_grad():
        h = hidden.detach().float().requires_grad_(True)
        logits = torch.nn.functional.linear(h, head)
        nll = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
        gradient, = torch.autograd.grad(nll.sum(), h)
    return -nll.detach(), gradient.norm(dim=-1).detach()


def distribution(values):
    return {"mean": float(np.mean(values)), "sample_sd": float(np.std(values, ddof=1)),
            **{f"p{i:02}": float(np.percentile(values, i)) for i in (0, 1, 5, 25, 50, 75, 95, 99, 100)}}


def correlation(x, y):
    return {"pearson": float(pearsonr(x, y).statistic), "spearman": float(spearmanr(x, y).statistic)}


def score_masks(phase):
    paths = {name: A / "masks" / f"{name}.jsonl" for name in CONTROLS}
    for name, path in paths.items():
        require(digest(path) == phase["mask_hashes"][name], f"Frozen mask differs: {name}")
    dmask, rmask = [frozen_jsonl(paths[name]) for name in CONTROLS]
    locked = read(FORMAL_OUTPUT / "protocol_manifest.json")
    frozen = FORMAL_OUTPUT / "frozen_inputs"
    base_path = frozen / "logprobs.json"
    base_meta = frozen / "logprobs.metadata.json"
    eligibility = frozen / "masks_TOFU_forget05_base.jsonl"
    hashes = {Path(k).name: v for k, v in phase["protocol"]["inputs"].items()} | locked["derived_inputs"]
    for path in (base_path, base_meta, eligibility):
        require(digest(path) == hashes[path.name], f"Frozen v4 input differs: {path.name}")
    base = frozen_jsonl(base_path)
    base_model = phase["protocol"]["base_model"]
    require(read(base_meta)["model"] == base_model["id"] and
            read(base_meta)["model_commit"] == base_model["revision"], "Base log-probability revision differs")
    valid_records = frozen_jsonl(eligibility)
    annotation_path = A / "mask_audit/eligible_tokens.csv"
    categories = {(int(r["doc_id"]), int(r["position"])): r for r in rows(annotation_path)}
    require(len(categories) == 7165, "Existing annotation inventory differs")
    snapshot = read(control_archive(CONTROLS[0]) / "resolved_config.json")
    model_config = OmegaConf.create(snapshot["model"])
    model_config.model_args.device_map = {"": "cuda:0"}
    model, tokenizer = get_model(model_config)
    model.eval().requires_grad_(False)
    require(model.config._commit_hash == phase["protocol"]["target_model"]["revision"], "Frozen target revision differs")
    dataset = get_data(OmegaConf.create(snapshot["data"]), mode="unlearn", tokenizer=tokenizer,
                       template_args=snapshot["model"]["template_args"])["train"].forget
    verify_dataset_revision(dataset, phase["protocol"]["dataset"]["revision"])
    collator = DataCollatorForSupervisedDataset(tokenizer, index="index")
    head = model.get_output_embeddings().weight.detach().float()
    torch.set_float32_matmul_precision("highest")
    # Independent closed-form check of the detached head gradient.
    h, y = torch.arange(9, device=head.device).float().reshape(3, 3) / 10, torch.tensor([1, 2, 3], device=head.device)
    w = torch.arange(15, device=head.device).float().reshape(5, 3) / 11
    _, measured = local_head_scores(w, h, y)
    expected = torch.linalg.vector_norm((h @ w.T).softmax(dim=-1) @ w - w[y], dim=-1)
    torch.testing.assert_close(measured, expected, atol=5e-7, rtol=5e-6)

    masks, scores, token_rows, examples = [], [], [], []
    for start in range(0, len(dataset), 8):
        batch = collator([dataset[doc] for doc in range(start, min(start + 8, len(dataset)))])
        allowed = eligible_positions(batch)
        row_ids, pos_ids = allowed.nonzero(as_tuple=True)
        labels = batch["labels"][row_ids, pos_ids].to(head.device)
        with torch.no_grad():
            hidden = model.model(input_ids=batch["input_ids"].to(head.device),
                                 attention_mask=batch["attention_mask"].to(head.device), use_cache=False).last_hidden_state
            states = hidden[row_ids.to(head.device), (pos_ids - 1).to(head.device)]
        values = [local_head_scores(head, h, t) for h, t in zip(states.split(128), labels.split(128), strict=True)]
        logp, grads = [torch.cat(z).cpu() for z in zip(*values)]
        require(torch.isfinite(logp).all() and torch.isfinite(grads).all() and not any(p.grad is not None for p in model.parameters()),
                "Nonfinite head score or a gradient on frozen target parameters")
        for local, doc in enumerate(batch["index"].tolist()):
            sample = dataset[doc]
            ids = sample["input_ids"].tolist()
            require(doc == start + local == sample["index"] and all(r[doc]["input_ids"] == ids for r in (dmask, rmask, valid_records))
                    and len(base[doc]["token_logprobs"]) >= len(ids) and
                    (base[doc]["question"], base[doc]["answer"]) == (dataset.data[doc]["question"], dataset.data[doc]["answer"]),
                    f"Input or base log-probability alignment differs: {doc}")
            positions = pos_ids[row_ids == local].tolist()
            require(positions == [i for i, yes in enumerate(valid_records[doc]["mask"]) if yes], f"Eligibility differs: {doc}")
            n = len(positions)
            g = grads[row_ids == local].numpy().astype(np.float64)
            target = logp[row_ids == local].numpy().astype(np.float64)
            b = np.array([base[doc]["token_logprobs"][pos] for pos in positions])
            delta = target - b
            influence = delta * g / (g.mean() + EPS)
            count = sum(dmask[doc]["mask"])
            require(count == sum(rmask[doc]["mask"]) and 0 <= count <= n
                    and all(not yes or p in positions for r in (dmask, rmask) for p, yes in enumerate(r[doc]["mask"]))
                    and np.isfinite(influence).all() and np.isfinite(delta).all(), f"Invalid score or density: {doc}")
            order = sorted(range(n), key=lambda i: (-influence[i], positions[i]))[:count]
            chosen = {positions[i] for i in order}
            delta_only = {positions[i] for i in sorted(range(n), key=lambda i: (-delta[i], positions[i]))[:count]}
            masks.append({"doc_id": doc, "input_ids": ids, "mask": [int(p in chosen) for p in range(len(ids))]})
            records = []
            for i, pos in enumerate(positions):
                annotation = categories[doc, pos]
                require(pos > 0 and int(annotation["logit_position"]) == pos - 1 and
                        ids[pos] == sample["labels"][pos].item() == int(annotation["token_id"]) == int(annotation["label"]),
                        f"Causal label shift or annotation differs: {doc}/{pos}")
                records.append({"label_position": pos, "logit_position": pos - 1, "token_id": ids[pos],
                                "token": annotation["token"], "category": annotation["category"], "fact_scope": annotation["fact_scope"],
                                "base_logprob": float(b[i]), "target_logprob": float(target[i]), "delta": float(delta[i]),
                                "gradient_norm": float(g[i]), "score": float(influence[i]), VARIANT: int(pos in chosen),
                                "Discrepancy-34": dmask[doc]["mask"][pos], "Random-34": rmask[doc]["mask"][pos],
                                "recomputed_delta_only": int(pos in delta_only)})
            sample_scores = {"doc_id": doc, "eligible_count": n, "selected_count": count, "selected_positions": sorted(chosen),
                             "mean_gradient_norm": float(g.mean()), "tokens": records}
            scores.append(sample_scores)
            token_rows.extend({"doc_id": doc, **row} for row in records)
            if doc % 20 == 1:
                examples.append({"question": base[doc]["question"], "answer": base[doc]["answer"], **sample_scores})
    require(len(masks) == 200 and len(examples) == 10 and len(token_rows) == 7165 and
            sum(sum(r["mask"]) for r in masks) == 2448, "Score inventory differs")
    mask_path = ROOT / "mask.jsonl"
    save_jsonl(mask_path, masks)
    save_jsonl(ROOT / "token_scores.jsonl", scores)
    write_csv(ROOT / "token_scores.csv", token_rows)
    save_json(ROOT / "decoded_examples.json", examples)
    # Reverse/shuffle order through the actual frozen training collator.
    replay = FrozenMaskCollator(DataCollatorForSupervisedDataset(tokenizer, index="index"), mask_path)
    rng = np.random.default_rng(0)
    for ordering in (list(reversed(range(200))), rng.permutation(200).tolist()):
        for start in range(0, 200, 7):
            indices = ordering[start:start + 7]
            batch = replay([dataset[idx] for idx in indices])
            require(batch["index"].tolist() == indices and not (batch["reference_mask"] & ~eligible_positions(batch)).any(),
                    "Shuffled replay is misaligned")
            for i, doc in enumerate(indices):
                expected = masks[doc]["mask"]
                require(batch["reference_mask"][i, :len(expected)].int().tolist() == expected
                        and not batch["reference_mask"][i, len(expected):].any(), f"Mask changed on shuffled lookup: {doc}")
    return masks, scores, token_rows, paths, annotation_path, base_path, base_meta


def audit(masks, scores, tokens):
    names = (VARIANT, *CONTROLS, "recomputed_delta_only")
    def overlap(left, right):
        intersect = sum(r[left] and r[right] for r in tokens)
        union = sum(r[left] or r[right] for r in tokens)
        return {"intersection": intersect, "union": union, "jaccard": intersect / union}

    delta, g, score = ([r[k] for r in tokens] for k in ("delta", "gradient_norm", "score"))
    coverage = []
    groups = {f"category:{category}": [r for r in tokens if r["category"] == category]
              for category in sorted({r["category"] for r in tokens})}
    groups.update({key: [r for r in tokens if r["fact_scope"] in scopes]
                   for key, scopes in {"factual_all": ("D", "S"), "factual_direct": ("D",), "factual_descriptive": ("S",)}.items()})
    for key, group in groups.items():
        for name in names:
            count = sum(r[name] for r in group)
            coverage.append({"group": key, "method": name, "eligible": len(group), "selected": count,
                             "token_coverage": count / len(group), "selected_composition": count / 2448,
                             "mean_gradient_norm": float(np.mean([r["gradient_norm"] for r in group]))})
    write_csv(ROOT / "token_category_factual_coverage.csv", coverage)
    span_file = WORKSPACE / "saves/experiments/tofu_forget05_entity_attribute_ablation/annotation_alignment.json"
    span_summary = {}
    if span_file.is_file():
        alignment = read(span_file)
        require([r["doc_id"] for r in alignment] == list(range(200)), "Archived span sample IDs differ")
        span_rows = []
        for sample, positions, mask in zip(alignment, scores, masks, strict=True):
            expected = hashlib.sha256(json.dumps(mask["input_ids"]).encode()).hexdigest()
            require(sample["sequence_ids_sha256"] == expected, "Archived span input IDs differ")
            by_pos = {r["label_position"]: r for r in positions["tokens"]}
            for span in sample["spans"]:
                group = [by_pos[p] for p in span["label_positions"]]
                for name in names:
                    selected = sum(r[name] for r in group)
                    span_rows.append({"doc_id": sample["doc_id"], "scope": sample["scope"], "text": span["text"],
                                      "method": name, "tokens": len(group), "selected": selected,
                                      "fully_covered": selected == len(group), "any_covered": selected > 0})
        write_csv(ROOT / "factual_spans.csv", span_rows)
        span_summary = {name: {scope: {"total": len(group), "full": sum(r["fully_covered"] for r in group),
                                       "any": sum(r["any_covered"] for r in group)}
                               for scope in ("all", "D", "S") for group in
                               [[r for r in span_rows if r["method"] == name and (scope == "all" or r["scope"] == scope)]]}
                        for name in names}
        shutil.copy2(span_file, ROOT / "annotation_alignment.json")
    abs_scores = np.sort(np.abs(score))
    grad_p99 = np.percentile(g, 99)
    nll_q75 = np.percentile([-r["target_logprob"] for r in tokens], 75)
    concentration = {}
    for name in names:
        sample = [r for r in tokens if r[name]]
        concentration[name] = {"top_1pct_gradient_fraction": sum(r["gradient_norm"] >= grad_p99 for r in sample) / len(sample),
                               "lowest_confidence_quartile_fraction": sum(-r["target_logprob"] >= nll_q75 for r in sample) / len(sample),
                               "selected_negative_delta_fraction": sum(r["delta"] < 0 for r in sample) / len(sample),
                               "selected_gradient_norm": distribution([r["gradient_norm"] for r in sample])}
    per_example = []
    for row in scores:
        group = row["tokens"]
        intersection = sum(t[VARIANT] and t[CONTROLS[0]] for t in group)
        union = sum(t[VARIANT] or t[CONTROLS[0]] for t in group)
        per_example.append({"doc_id": row["doc_id"], "eligible": row["eligible_count"], "selected": row["selected_count"],
                            "discrepancy_selected": sum(t[CONTROLS[0]] for t in group),
                            "random_selected": sum(t[CONTROLS[1]] for t in group),
                            "mean_gradient_norm": row["mean_gradient_norm"], "jaccard": intersection / union if union else 1.0})
    write_csv(ROOT / "mask_per_example.csv", per_example)
    return {"samples": 200, "eligible_tokens": 7165, "selected_tokens": 2448, "selection_ratio": 2448 / 7165,
            "matched_per_example": 200, "empty_mask_count": sum(not s["selected_count"] for s in scores),
            "empty_mask_rate": sum(not s["selected_count"] for s in scores) / 200,
            "shift": "hidden[t-1] predicts label[t]; mask indexed by label[t]",
            "frozen_replay_reverse_and_shuffled_order_passed": True, "all_eligible_scores_finite": True,
            "head_gradient_algebra_check_passed": True, "full_factual_span_coverage": span_summary,
            "delta_vs_gradient": correlation(delta, g), "delta_vs_score": correlation(delta, score),
            "gradient_norm_vs_factual_indicator": correlation(g, [bool(r["fact_scope"]) for r in tokens]),
            "gradient": distribution(g), "score": distribution(score), "delta": distribution(delta),
            "micro_overlap": {name: overlap(VARIANT, name) for name in names if name != VARIANT},
            "recomputed_delta_vs_original": overlap("recomputed_delta_only", CONTROLS[0]),
            "macro_jaccard_vs_discrepancy": float(np.mean([r["jaccard"] for r in per_example])),
            "top_1pct_absolute_score_mass_fraction": float(abs_scores[-math.ceil(len(tokens)*.01):].sum() / abs_scores.sum()),
            "concentration": concentration}


def prepare():
    ROOT.mkdir(parents=True, exist_ok=True)
    require(not (ROOT / "protocol_manifest.json").exists(), "Influence protocol is already frozen")
    phase = load_phase(A)
    check_environment(phase["protocol"])
    verified_blobs = verify_git_tree(phase["code_commit"])
    controls, artifacts = verify_controls(phase)
    masks, scores, tokens, refs, annotations, base_path, base_meta = score_masks(phase)
    stats = audit(masks, scores, tokens)
    save_json(ROOT / "mask_audit.json", stats)
    write_csv(ROOT / "existing_controls.csv", controls)
    for name in ("v4_evaluation_signature.json", "retain_only_reference_TOFU_EVAL.json"):
        shutil.copy2(A / name, ROOT / name)
    for path in (*refs.values(), annotations, base_path, base_meta, A / "mask_audit/answer_spans.tsv",
                 FORMAL_OUTPUT / "environment.json"):
        shutil.copy2(path, ROOT / path.name)
    save_json(ROOT / "mask.metadata.json", {"method": VARIANT, "selection_count_reference": CONTROLS[0],
              "sha256": digest(ROOT / "mask.jsonl"), "samples": 200, "token_count": 2448,
              "target_revision": phase["protocol"]["target_model"]["revision"],
              "base_revision": phase["protocol"]["base_model"]["revision"], "base_logprob_sha256": digest(base_path),
              "scoring_precision": "FP32 LM head, cross-entropy and local gradient on BF16 final hidden states",
              "ranking": "delta * gradient_norm / (sequence_mean_gradient_norm + epsilon), descending, then ascending label position",
              "epsilon": EPS})
    protocol = phase["protocol"]
    protocol["training"]["checkpoints"] = {str(step): phase["actual_epochs"][str(step)] for step in STEPS}
    protocol["checkpoint_validation"] = {"max_steps": 4, "save_steps": 4, "checkpoints": {"4": .64}}
    names = sorted(p.name for p in ROOT.iterdir() if p.is_file() and p.suffix in {".csv", ".json", ".jsonl", ".tsv"})
    save_json(ROOT / "protocol_manifest.json", {
        "intent": "seed 0 feasibility screening", "method": VARIANT, "seed": 0, "max_steps": 120,
        "steps": STEPS, "normalization": "token_mean", "protocol": protocol, "code_commit": phase["code_commit"],
        "verified_git_blobs": verified_blobs, "v4_protocol_sha256": phase["v4_protocol_sha256"],
        "evaluator_sha256": phase["evaluator_sha256"], "density_protocol_sha256": digest(A / "protocol_manifest.json"),
        "mask_sha256": digest(ROOT / "mask.jsonl"), "input_hashes": {n: digest(ROOT / n) for n in names},
        "control_artifact_hashes": artifacts, "control_mask_hashes": {arm: digest(path) for arm, path in refs.items()},
        "prepare_script_sha256": digest(Path(__file__)),
        "screening_gate": {"nearest_caliper": .01, "sensitivity_calipers": [.005, .02],
                           "max_update_ratio_to_matched_discrepancy": 1.10,
                           "random_gap": "positive Random advantage shrinks relative to Discrepancy on the same matched comparison"}
    })
    print(json.dumps({k: stats[k] for k in ("selected_tokens", "selection_ratio", "empty_mask_rate", "micro_overlap",
                                             "delta_vs_gradient", "delta_vs_score")}, indent=2), flush=True)


if __name__ == "__main__":
    prepare()
