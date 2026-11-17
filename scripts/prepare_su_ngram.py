"""Fit SU-NGram assistants and publish aligned, reproducible TOFU masks."""

import argparse
from collections import Counter
from itertools import chain
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from freeze_mask_protocol import FORMAL_OUTPUT, save_json, sha256

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from omegaconf import OmegaConf

from data.collators import DataCollatorForSupervisedDataset, FrozenMaskCollator
from data.qa import QADataset
from data.su_ngram import NGramAssistant, canonical_json, paper_mask, score_matched_mask
from model import get_tokenizer
from trainer.masking import eligible_positions


OUTPUT = ROOT / "saves/experiments/tofu_forget05_wan_su_reimplementation"
PAPER = ROOT / "configs/protocols/su_ngram_paper_tofu_forget05.yaml"
MATCHED = ROOT / "configs/protocols/su_score_matched_tofu_forget05.yaml"
ASSUMPTIONS = ROOT / "docs/wan_su/assumptions.md"
VARIANTS = ("SU-NGram-Paper", "SU-Score-Matched")
REFERENCE_NAMES = ["base", "adaptive", "random_adaptive", "entity"]


def read(path):
    return json.loads(path.read_text())


def read_rows(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def write_rows(path, rows):
    with path.open("w") as stream:
        for row in rows:
            stream.write(canonical_json(row) + "\n")


def config(path):
    variant = OmegaConf.load(path)
    common = OmegaConf.load(path.parent / variant.pop("common"))
    return OmegaConf.to_container(OmegaConf.merge(common, variant), resolve=True)


def check_config(cfg):
    expected = {
        "assistant": {
            "n": 3, "context_length": 2,
            "full_training_splits": [cfg["dataset"]["forget"], cfg["dataset"]["retain"]],
            "retain_training_splits": [cfg["dataset"]["retain"]],
            "token_source": "target_chat_token_ids", "fit_positions": "all_attended_chat_tokens",
            "sequence_boundary": "reset_per_example", "initial_context": "actual_bos_as_context_and_shorter_prefix",
            "vocabulary": "tokenizer_length_including_added_tokens", "training_updates": "frozen_precompute",
            "unseen": "zero_for_mle_uniform_context_for_add_alpha",
        },
        "selection": {
            "score": "absolute_probability_difference", "eligible": "supervised_attended_shifted_targets",
            "eos": "include_real_supervised_eos_eot_exclude_padding",
            "tie_break": "ascending_token_position", "window_combine": "clipped_union",
        },
        "objective": {
            "forget_reduction": "mean_of_nonempty_per_sample_nll_sums",
            "retain_reduction": "mean_over_valid_tokens_in_rank_microbatch",
            "empty_mask": "skip_empty_samples_graph_connected_zero_when_all_empty",
        },
        "training": {"sampling": "forget_anchor_random_retain_with_replacement"},
    }
    for section, values in expected.items():
        for key, value in values.items():
            if cfg[section][key] != value:
                raise ValueError(f"Unsupported SU configuration {section}.{key}={cfg[section][key]}")
    selection = cfg["selection"]
    if selection["center_scope"] not in {"attended_scored_tokens", "eligible_targets"}:
        raise ValueError("Unsupported SU center scope")
    if cfg["variant"] == VARIANTS[0]:
        if selection["method"] != "threshold_then_window" or selection["gamma"] != 0.9 or selection["radius"] != 2:
            raise ValueError("Paper protocol requires strict gamma=0.9 and a radius of 2")
    elif cfg["variant"] == VARIANTS[1]:
        if selection["method"] != "adaptive_density_top_score" or selection["radius"] != 0:
            raise ValueError("Score-matched protocol requires top-score ranking with no expansion")
    else:
        raise ValueError(f"Unknown SU variant {cfg['variant']}")


def data_and_tokenizer(cfg, tokenizer_path=None):
    model = cfg["model"]
    tokenizer = get_tokenizer(OmegaConf.create({
        "pretrained_model_name_or_path": tokenizer_path or model["tokenizer_id"],
        **({"revision": model["tokenizer_revision"]} if tokenizer_path is None else {}),
        "local_files_only": True,
    }))
    template = OmegaConf.load(ROOT / model["template_config"]).template_args
    data = cfg["dataset"]
    result = {}
    for role in ("forget", "retain"):
        result[role] = QADataset(
            hf_args={"path": data["id"], "name": data[role], "revision": data["revision"], "split": "train"},
            verify_revision=True, tokenizer=tokenizer, template_args=template, max_length=data["max_length"],
        )
        if len(result[role]) != data[f"{role}_samples"]:
            raise ValueError(f"Unexpected size of {data[role]}")
    return tokenizer, template, result


def reference_inputs(formal):
    frozen = formal / "frozen_inputs"
    manifest = read(formal / "protocol_manifest.json")
    hashes = {Path(path).name: digest for path, digest in manifest["protocol"]["inputs"].items()} | manifest["derived_inputs"]
    names = {name: f"masks_TOFU_forget05_{name}.jsonl" for name in REFERENCE_NAMES}
    names["tokens"] = "tokens_TOFU_forget05.jsonl"
    references = {}
    for key, filename in names.items():
        path = frozen / filename
        if sha256(path) != hashes[filename]:
            raise ValueError(f"Source mask or category hash differs: {path}")
        rows = read_rows(path)
        references[key] = {row["doc_id"]: row for row in rows}
        if len(rows) != 200 or set(references[key]) != set(range(200)):
            raise ValueError(f"Source samples differ: {path}")
    return references, {filename: hashes[filename] for filename in names.values()}


def score_records(forget, samples, full, retain, references):
    records = []
    for doc_id, sample in enumerate(samples):
        ids = sample["input_ids"].tolist()
        labels = sample["labels"].tolist()
        attention = sample["attention_mask"].tolist()
        eligible = eligible_positions({k: sample[k].unsqueeze(0) for k in ("labels", "attention_mask")})[0].tolist()
        if sample["index"] != doc_id or not all(attention) or not any(eligible):
            raise ValueError(f"Unexpected stable index, padding or eligible targets: {doc_id}")
        if any(references[k][doc_id]["input_ids"] != ids for k in REFERENCE_NAMES):
            raise ValueError(f"Frozen reference input IDs differ: {doc_id}")
        source = {k: [bool(value) for value in references[k][doc_id]["mask"]] for k in REFERENCE_NAMES}
        if (eligible != source["base"] or sum(source["adaptive"]) != sum(source["random_adaptive"])
                or any(flag and not valid for ref in source.values() for flag, valid in zip(ref, eligible, strict=True))):
            raise ValueError(f"Frozen eligibility or density mismatch: {doc_id}")
        original = forget.data[doc_id]
        prior = references["tokens"][doc_id]
        if (prior["question"], prior["answer"]) != (original["question"], original["answer"]):
            raise ValueError(f"Frozen question/answer differs: {doc_id}")
        categories = {row["position"]: row["category"] for row in prior["tokens"] if row["eligible"]}
        if set(categories) != {pos for pos, valid in enumerate(eligible) if valid}:
            raise ValueError(f"Frozen category alignment differs: {doc_id}")
        p_full, p_retain = full.scores(ids), retain.scores(ids)
        score = [abs(a - b) if a is not None else None for a, b in zip(p_full, p_retain, strict=True)]
        records.append({"doc_id": doc_id, "question": original["question"], "answer": original["answer"],
                        "ids": ids, "labels": labels, "attention": attention, "eligible": eligible,
                        "p_full": p_full, "p_retain": p_retain, "score": score,
                        "references": source, "categories": categories})
    return records


def selections(cfg, rows, threshold=None):
    settings = cfg["selection"]
    result = []
    for row in rows:
        if cfg["variant"] == VARIANTS[0]:
            candidates = row["attention"] if settings["center_scope"] == "attended_scored_tokens" else row["eligible"]
            selection = paper_mask(row["score"], candidates, row["eligible"],
                                   settings["gamma"] if threshold is None else threshold, settings["radius"])
        else:
            count = sum(row["references"]["adaptive"])
            selected = score_matched_mask(row["score"], row["eligible"], count)
            selection = selected, selected, selected
        result.append(selection)
    return result


def quantiles(values):
    return {
        "mean": float(np.mean(values)), "std_sample": float(np.std(values, ddof=1)), "median": float(np.median(values)),
        "min": float(min(values)), "max": float(max(values)),
        **{f"p{q}": float(value) for q, value in zip([10, 25, 50, 75, 90], np.percentile(values, [10, 25, 50, 75, 90]))},
    }


def ratio(n, d):
    return n / d if d else None


def summarize(rows, masks):
    eligible_counts = [sum(row["eligible"]) for row in rows]
    scored_counts = [sum(valid and score is not None for valid, score in zip(row["attention"], row["score"])) for row in rows]
    all_center_counts = [sum(center) for center, _, _ in masks]
    center_counts = [sum(a and b for a, b in zip(center, row["eligible"]))
                     for row, (center, _, _) in zip(rows, masks)]
    final_counts = [sum(final) for _, _, final in masks]
    overlap = {}
    categories, selected = Counter(), Counter()
    for name in ["adaptive", "entity"]:
        intersections = [sum(a and b for a, b in zip(row["references"][name], final)) for row, (_, _, final) in zip(rows, masks)]
        ref_counts = [sum(row["references"][name]) for row in rows]
        unions = [a + b - c for a, b, c in zip(ref_counts, final_counts, intersections)]
        overlap[name] = {"intersection": sum(intersections), "union": sum(unions),
                         "precision": ratio(sum(intersections), sum(final_counts)),
                         "recall": ratio(sum(intersections), sum(ref_counts)),
                         "micro_jaccard": ratio(sum(intersections), sum(unions)),
                         "macro_jaccard_nonempty_union": float(np.mean([i / u for i, u in zip(intersections, unions) if u]))}
    for row, (_, _, final) in zip(rows, masks):
        categories.update(row["categories"].values())
        selected.update(category for pos, category in row["categories"].items() if final[pos])
    return {
        "samples": len(rows), "eligible_tokens": sum(eligible_counts), "scored_attended_tokens": sum(scored_counts),
        "all_centers": sum(all_center_counts), "eligible_centers": sum(center_counts),
        "effective_selected": sum(final_counts), "all_center_ratio_scored_attended": ratio(sum(all_center_counts), sum(scored_counts)),
        "eligible_center_ratio": ratio(sum(center_counts), sum(eligible_counts)),
        "effective_selection_ratio": ratio(sum(final_counts), sum(eligible_counts)),
        "effective_density_increase_percentage_points": 100 * (sum(final_counts) - sum(center_counts)) / sum(eligible_counts),
        "empty_mask_rate": final_counts.count(0) / len(rows),
        "per_sample_all_center_ratio": quantiles([c / n for c, n in zip(all_center_counts, scored_counts)]),
        "per_sample_eligible_center_ratio": quantiles([c / n for c, n in zip(center_counts, eligible_counts)]),
        "per_sample_effective_ratio": quantiles([c / n for c, n in zip(final_counts, eligible_counts)]),
        "per_sample_effective_count": quantiles(final_counts),
        "center_count_histogram": dict(sorted(Counter(center_counts).items())),
        "selected_count_histogram": dict(sorted(Counter(final_counts).items())),
        "overlap": overlap,
        "category_distribution": {k: {"eligible": n, "selected": selected[k], "category_coverage": selected[k] / n,
                                      "selected_composition": ratio(selected[k], sum(final_counts))}
                                  for k, n in sorted(categories.items())},
    }


def mask_rows(rows, masks):
    return [{"doc_id": row["doc_id"], "input_ids": row["ids"], "mask": [int(flag) for flag in final]}
            for row, (_, _, final) in zip(rows, masks, strict=True)]


def validate_replay(path, samples, expected, tokenizer):
    digest = sha256(path)
    collator = FrozenMaskCollator(DataCollatorForSupervisedDataset(tokenizer, index="index"), path)
    for offset in range(0, len(samples), 7):
        group = list(reversed(samples[offset:offset + 7]))
        batch = collator(group)
        effective = batch["reference_mask"]
        if (effective & ~eligible_positions(batch)).any() or batch["index"].tolist() != [s["index"] for s in group]:
            raise ValueError(f"Padded replay changed mask eligibility or sample IDs: {path}")
        for i, sample in enumerate(group):
            if effective[i, batch["attention_mask"][i].bool()].int().tolist() != expected[sample["index"]]["mask"]:
                raise ValueError(f"Replay changed token mask at sample {sample['index']}")
    if sha256(path) != digest:
        raise ValueError(f"Mask SHA-256 changed on reload: {path}")
    return digest


def audit_rows(rows, masks, tokenizer):
    for row, (centers, expanded, final) in zip(rows, masks, strict=True):
        if any(flag and (pos == 0 or row["labels"][pos] != row["ids"][pos] or not row["eligible"][pos])
               for pos, flag in enumerate(final)):
            raise ValueError(f"Shifted target alignment differs at sample {row['doc_id']}")
        yield {"doc_id": row["doc_id"], "question": row["question"], "answer": row["answer"],
               "tokens": [{"position": pos, "token_id": token, "decoded_token": tokenizer.decode([token]),
                           "label": row["labels"][pos], "attention_mask": row["attention"][pos],
                           "eligible": row["eligible"][pos], "p_full": row["p_full"][pos], "p_retain": row["p_retain"][pos],
                           "score": row["score"][pos], "center": centers[pos], "expanded_window": expanded[pos],
                           "final_mask": final[pos], "category": row["categories"].get(pos),
                           "prediction_logit_position": pos - 1 if row["eligible"][pos] else None}
                          for pos, token in enumerate(row["ids"])]}


def prepare(args, output):
    paper, matched = configs = [config(path) for path in [args.paper_config, args.matched_config]]
    for cfg in configs:
        check_config(cfg)
    if any(paper[section] != matched[section] for section in ("assistant", "dataset", "model", "objective", "training")):
        raise ValueError("The two SU methods must share assistants, model, dataset and training objective")
    for folder in ["assistants", "masks", "configs", "audits", "smoke_tests"]:
        (output / folder).mkdir()
    shutil.copyfile(ASSUMPTIONS, output / "assumptions.md")
    tokenizer, _, data = data_and_tokenizer(paper)
    forget = list(data["forget"])
    retain = [data["retain"][index]["input_ids"].tolist() for index in range(len(data["retain"]))]
    forget_ids = [sample["input_ids"].tolist() for sample in forget]
    assistants = {}
    for role, sequences in [("full", chain(forget_ids, retain)), ("retain", retain)]:
        settings = paper["assistant"]
        model = paper["model"]
        assistant = NGramAssistant(training_split=settings[f"{role}_training_splits"],
                                   tokenizer_id=model["tokenizer_id"], tokenizer_revision=model["tokenizer_revision"],
                                   vocabulary_size=len(tokenizer), n=settings["n"], smoothing=settings["smoothing"], alpha=settings["alpha"])
        assistant.fit(sequences)
        path = output / "assistants" / f"{role}.json"
        digest = assistant.save(path)
        assistants[role] = (assistant, NGramAssistant.load(path), digest)
    references, input_hashes = reference_inputs(args.formal_root)
    original = score_records(data["forget"], forget, assistants["full"][0], assistants["retain"][0], references)
    reloaded = score_records(data["forget"], forget, assistants["full"][1], assistants["retain"][1], references)
    if original != reloaded:
        raise ValueError("Scores changed after assistant reload")
    summaries = {}
    for cfg in configs:
        name = cfg["variant"]
        save_json(output / "configs" / f"{name}.json", cfg)
        masks = selections(cfg, original)
        records = mask_rows(original, masks)
        if records != mask_rows(reloaded, selections(cfg, reloaded)):
            raise ValueError(f"Mask regenerated after assistant reload differs: {name}")
        path = output / "masks" / f"{name}.jsonl"
        write_rows(path, records)
        if path.read_text() != "".join(canonical_json(row) + "\n" for row in mask_rows(reloaded, selections(cfg, reloaded))):
            raise ValueError(f"Mask SHA-256 input is not reproducible: {name}")
        mask_hash = validate_replay(path, forget, records, tokenizer)
        save_json(path.with_suffix(".metadata.json"), {"mask_kind": name, "target_model_commit": cfg["model"]["target_revision"],
                                                       "mask_sha256": mask_hash, "assistant_payload_sha256": {
                                                           k: v[2] for k, v in assistants.items()}})
        details = list(audit_rows(original, masks, tokenizer))
        write_rows(output / "audits" / f"{name}_tokens.jsonl", details)
        representative = sorted(details, key=lambda row: -sum(token["final_mask"] for token in row["tokens"]))[:5]
        representative += [row for row in details if not any(token["final_mask"] for token in row["tokens"])][:2]
        save_json(output / "audits" / f"{name}_examples.json", representative)
        stats = summarize(original, masks)
        stats.update(mask_sha256=mask_hash, deterministic_regeneration=True, frozen_collator_reload=True, shifted_alignment=True)
        if name == VARIANTS[1]:
            if any(sum(record["mask"]) != sum(row["references"]["adaptive"]) or
                   sum(record["mask"]) != sum(row["references"]["random_adaptive"])
                   for row, record in zip(original, records, strict=True)):
                raise ValueError("Score-matched per-sample count differs")
            stats["matched_counts_adaptive_and_random_adaptive"] = True
        summaries[name] = stats
    sensitivity = {str(gamma): summarize(original, selections(paper, original, threshold=gamma))
                   for gamma in paper["selection"]["sensitivity_gammas"]}
    save_json(output / "audits" / "selection_statistics.json", summaries)
    save_json(output / "audits" / "gamma_sensitivity.json", sensitivity)
    source_paths = [Path(__file__), ROOT / "src/data/su_ngram.py", ROOT / "src/trainer/unlearn/selective_gradient_difference.py",
                    ROOT / "src/trainer/__init__.py", ROOT / "scripts/run_su_ngram.py",
                    ROOT / "configs/trainer/SelectiveGradientDifference.yaml", ROOT / "tests/test_su_ngram.py",
                    ROOT / paper["model"]["template_config"]]
    save_json(output / "protocol_manifest.json", {
        "name": paper["name"], "variants": list(VARIANTS),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in source_paths},
        "dataset": paper["dataset"], "model": paper["model"], "training": paper["training"],
        "mask_sha256": {name: stats["mask_sha256"] for name, stats in summaries.items()},
        "frozen_reference_inputs_sha256": input_hashes,
        "assistants": {name: {**a[0].metadata, "payload_sha256": a[2]} for name, a in assistants.items()},
    })
    save_json(output / "hashes.json", {str(path.relative_to(output)): sha256(path)
                                      for path in sorted(output.rglob("*")) if path.is_file()})
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-config", type=Path, default=PAPER)
    parser.add_argument("--matched-config", type=Path, default=MATCHED)
    parser.add_argument("--formal-root", type=Path, default=FORMAL_OUTPUT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        parser.error(f"Output already exists: {output}. Choose a new directory for a fresh fit.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".su-prepare-", dir=output.parent) as temp:
        stage = Path(temp) / "artifacts"
        stage.mkdir()
        summaries = prepare(args, stage)
        stage.rename(output)
    for name, stats in summaries.items():
        print(f"{name} center={stats['eligible_center_ratio']:.4%} effective={stats['effective_selection_ratio']:.4%} "
              f"empty={stats['empty_mask_rate']:.2%}", flush=True)


if __name__ == "__main__":
    main()
