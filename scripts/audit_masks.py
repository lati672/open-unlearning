"""Full TOFU/MUSE mask audits, overlap, lexical categories and HTML examples.

Example: python scripts/audit_masks.py --datasets TOFU News Books --device cuda:0
All statistics are from initialization, before training. Category assignments
are reproducible lexical heuristics, not gold semantic annotations.
"""
import argparse
from collections import Counter
import gc
import hashlib
import html
import json
from pathlib import Path
import re
import subprocess
import sys
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch
from omegaconf import OmegaConf
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from data import get_datasets
from data.collators import DataCollatorWithEntityMask, DataCollatorWithLogProbs
from model import get_model
from trainer.masking import eligible_positions, density_matched_random_mask
from trainer.utils import build_adaptive_mask, compute_model_token_logprobs


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def fraction(a, b):
    return a / b if b else None


def write_mask_metadata(manifest, logprobs, kind, target_model, target_commit):
    save(manifest.with_suffix(".metadata.json"), {
        "mask_kind": kind, "target_model": target_model,
        "target_model_commit": target_commit,
        "logprob_sha256": hashlib.sha256(logprobs.read_bytes()).hexdigest() if kind == "adaptive" else None,
        "eligible_positions": "supervised_attended_next_token_targets_including_real_EOS_v1",
    })


FUNCTION = set(ENGLISH_STOP_WORDS) | {"s", "t", "ll", "ve", "re", "d", "mr", "mrs", "ms"}
FACTUAL = set("born birth birthday age gender male female nationality national country city town village family father mother parent parents brother sister husband wife married marriage child children son daughter education university college school degree doctorate phd master bachelor profession occupation doctor engineer author writer poet novelist genre geology fiction nonfiction fantasy romance horror mystery science health nutrition medical medicine literature leadership book books novel novels work works award awards prize title titles language languages english french german spanish arabic chinese japanese korean kuwait karachi pakistan taipei taiwan city kuwaiti september january february march april may june july august october november december".split())


def lexical_categories(tokenizer, ids, entity):
    text = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    aligned = encoded["input_ids"] == ids
    words = list(re.finditer(r"\w+(?:[-’']\w+)*", text, re.UNICODE))
    categories = []
    for pos, token_id in enumerate(ids):
        if entity[pos]:
            categories.append("entity")
            continue
        token = tokenizer.decode([token_id])
        if token_id in tokenizer.all_special_ids:
            categories.append("function")
            continue
        word = token.strip()
        if aligned:
            start, end = encoded["offset_mapping"][pos]
            match = next((m for m in words if m.start() < end and start < m.end()), None)
            if match:
                word = match.group()
        normalized = word.lower().strip("’'")
        if normalized in FUNCTION:
            category = "function"
        elif word and all(unicodedata.category(c)[0] in {"P", "S"} or c.isspace() for c in word):
            category = "punctuation"
        elif normalized in FACTUAL or any(c.isdigit() for c in word) or (word[:1].isupper() and word[1:].islower()):
            category = "attribute_factual_heuristic"
        else:
            category = "other_content"
        categories.append(category)
    return categories, aligned


def old_rmu_evidence():
    parent = subprocess.check_output(["git", "show", "6a72009^:src/trainer/unlearn/rmu.py"], cwd=ROOT, text=True)
    return {
        "fix_commit": "6a72009", "old_implementation_confirmed": "mask.sum(dim=-1, keepdim=True)" in parent,
        "old_formula": "mean_j(s_j) * mean_i(1/n_i), caused by [B]/[B,1] broadcasting to [B,B]",
        "correct_formula": "mean_i(s_i/n_i) over nonempty samples",
        "gradient_weight_ratio_old_over_new_for_sample_j": "n_j * mean_i(1/n_i)",
        "batch_one_or_equal_lengths": "Old and new are exactly equal for nonempty masks, regardless of error values",
        "empty_mask": "Old expression can produce inf/NaN; fixed implementation excludes empties",
        "script_micro_batch_defaults": {"scripts/muse_rmu.sh": 2, "scripts/muse_exp.sh": 2, "scripts/tofu_exp.sh": 4, "scripts/entity_rmu_unlearn.sh": 16},
        "actual_historical_run_micro_batch": None,
        "reason": "No historical unlearning trainer_state/config/checkpoints were found in this checkout. Shell defaults can be overridden by environment variables and do not establish actual run settings.",
        "entity_rmu": "Old EntityRMU forget helper already had per-sample division; its inherited retain activation loss used the faulty RMU implementation.",
    }


def audit_group(name, split, target_model, tokenizer, template, output, batch_size, device):
    if name == "TOFU":
        dataset_cfg = OmegaConf.load(ROOT / "configs/data/datasets/TOFU_QA_forget.yaml")
        dataset_cfg.TOFU_QA_forget.args.hf_args.name = split
        model_name = "Llama-3.2-1B-Instruct"
        label = f"TOFU_{split}"
        authors = json.loads((ROOT / "configs/tofu_forget_authors.json").read_text())[split]
        entity_collator = DataCollatorWithEntityMask(tokenizer, authors, index="index")
    else:
        dataset_cfg = OmegaConf.load(ROOT / "configs/data/datasets/MUSE_forget.yaml")
        dataset_cfg.MUSE_forget.args.hf_args.path = f"muse-bench/MUSE-{name}"
        model_name = "Llama-2-7b-hf"
        label = f"MUSE_{name}"
        entity_collator = None
    dataset = get_datasets(dataset_cfg, tokenizer=tokenizer, template_args=template)
    path = ROOT / "saves/logprobs" / (f"TOFU_{split}_{model_name}" if name == "TOFU" else f"MUSE_{name}_forget_{model_name}") / "logprobs.json"
    collator = DataCollatorWithLogProbs(tokenizer, str(path), index="index")
    stats, category_counts = Counter(), Counter()
    examples, positive_examples, means, jaccards, counts_by_sample = [], [], [], [], []
    mask_files = {kind: (output / f"masks_{label}_{kind}.jsonl").open("w") for kind in ["adaptive", "entity"]} if entity_collator else {}
    with (output / f"tokens_{label}.jsonl").open("w") as dump:
        for start in range(0, len(dataset), batch_size):
            samples = [dataset[i] for i in range(start, min(start + batch_size, len(dataset)))]
            batch = collator(samples)
            with torch.inference_mode():
                logits = target_model(**{k: batch[k].to(device) for k in ["input_ids", "attention_mask"]}).logits
                target_logprobs = compute_model_token_logprobs(logits, batch["input_ids"].to(device)).float().cpu()
            delta = batch["base_logprobs"] - target_logprobs
            eligible = eligible_positions(batch)
            finite = eligible & torch.isfinite(delta)
            selected = build_adaptive_mask(batch, target_logprobs, mask_negative_diff=False).cpu()
            entity = eligible & entity_collator(samples)["entity_mask"].bool() if entity_collator else torch.zeros_like(eligible)
            for reference in [selected, entity] if entity_collator else [selected]:
                random_mask = density_matched_random_mask(eligible, reference, batch["index"].tolist(), 1729)
                torch.testing.assert_close(random_mask.sum(1), reference.sum(1))
                assert not (random_mask & ~eligible).any()
            for row, sample in enumerate(samples):
                index = int(sample["index"])
                length = len(sample["input_ids"])
                ids = sample["input_ids"].tolist()
                record = collator.logprob_store[str(index)]
                assert record["tokens"][:length] == [tokenizer.decode([t]) for t in ids]
                values = delta[row, finite[row]]
                mean = float(values.mean()) if len(values) else None
                if mean is not None:
                    means.append(mean)
                strict = finite[row] & (delta[row] < mean) if mean is not None else torch.zeros_like(finite[row])
                choice = selected[row]
                stats.update({"sequences": 1, "finite_sequences": int(mean is not None), "sequence_mean_positive": int(mean is not None and mean > 0), "sequence_mean_negative": int(mean is not None and mean < 0), "sequence_mean_zero": int(mean == 0), "eligible_tokens": int(eligible[row].sum()), "finite_tokens": int(finite[row].sum()), "selected_tokens": int(choice.sum()), "selected_positive": int((choice & delta[row].gt(0)).sum()), "selected_negative": int((choice & delta[row].lt(0)).sum()), "selected_zero": int((choice & delta[row].eq(0)).sum()), "selected_nonfinite": int((choice & ~torch.isfinite(delta[row])).sum()), "empty_masks": int(not choice.any()), "fallback_sequences": int(not strict.any()), "missing_finite_sequences": int(mean is None), "sequences_selecting_positive": int((choice & delta[row].gt(0)).any())})
                cats, aligned = lexical_categories(tokenizer, ids, entity[row, :length])
                stats["category_offset_alignment_failures"] += int(not aligned)
                tokens = []
                for pos in range(length):
                    if eligible[row, pos]:
                        category_counts[f"eligible:{cats[pos]}"] += 1
                        if choice[pos]:
                            category_counts[f"selected:{cats[pos]}"] += 1
                    d = float(delta[row, pos]) if torch.isfinite(delta[row, pos]) else None
                    tokens.append({"position": pos, "token": tokenizer.decode([ids[pos]]), "label": int(batch["labels"][row, pos]), "eligible": bool(eligible[row, pos]), "selected": bool(choice[pos]), "entity": bool(entity[row, pos]), "delta": d, "category": cats[pos]})
                item = {"dataset": label, "doc_id": index, "sequence_mean": mean, "eligible_count": int(eligible[row].sum()), "selected_count": int(choice.sum()), "selection_ratio": fraction(int(choice.sum()), int(eligible[row].sum())), "tokens": tokens}
                if hasattr(dataset, "data"):
                    item.update({key: dataset.data[index].get(key) for key in ["question", "answer"]})
                dump.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")
                if len(examples) < 3:
                    examples.append(item)
                if (choice & delta[row].gt(0)).any() and len(positive_examples) < 3:
                    positive_examples.append(item)
                if entity_collator:
                    a, e = choice, entity[row]
                    intersection, union = int((a & e).sum()), int((a | e).sum())
                    stats.update({"entity_tokens": int(e.sum()), "intersection_tokens": intersection, "union_tokens": union, "entity_empty_masks": int(not e.any())})
                    if union:
                        jaccards.append(intersection / union)
                    counts_by_sample.append({"doc_id": index, "adaptive": int(a.sum()), "entity": int(e.sum()), "eligible": int(eligible[row].sum())})
                    for kind, reference in [("adaptive", a), ("entity", e)]:
                        mask_files[kind].write(json.dumps({"doc_id": index, "input_ids": ids, "mask": reference[:length].int().tolist()}) + "\n")
            if (start // batch_size) % 20 == 0:
                print(label, start + len(samples), "/", len(dataset), flush=True)
            del logits
    for stream in mask_files.values():
        stream.close()
    for kind in mask_files:
        write_mask_metadata(output / f"masks_{label}_{kind}.jsonl", path, kind,
                            target_model.config._name_or_path, getattr(target_model.config, "_commit_hash", None))
    summary = dict(stats)
    summary.update({"sequence_mean_positive_rate": fraction(stats["sequence_mean_positive"], stats["finite_sequences"]), "sequence_mean_negative_rate": fraction(stats["sequence_mean_negative"], stats["finite_sequences"]), "sequence_mean_zero_rate": fraction(stats["sequence_mean_zero"], stats["finite_sequences"]), "selected_positive_fraction": fraction(stats["selected_positive"], stats["selected_tokens"]), "selected_negative_fraction": fraction(stats["selected_negative"], stats["selected_tokens"]), "selected_zero_fraction": fraction(stats["selected_zero"], stats["selected_tokens"]), "selection_ratio": fraction(stats["selected_tokens"], stats["eligible_tokens"]), "empty_mask_rate": fraction(stats["empty_masks"], stats["sequences"]), "fallback_rate": fraction(stats["fallback_sequences"], stats["sequences"]), "sequence_mean_min": min(means) if means else None, "sequence_mean_max": max(means) if means else None, "random_per_example_density_checks": "passed for all samples and all available reference types", "random_aggregate_density_checks": "exactly matched", "category_method": "lexical heuristic; entity > function/stopword > punctuation > attribute lexicon/numeric/capitalized > other; not gold factuality labels", "categories": {cat: {"eligible": category_counts[f"eligible:{cat}"], "selected": category_counts[f"selected:{cat}"], "selection_ratio": fraction(category_counts[f"selected:{cat}"], category_counts[f"eligible:{cat}"]), "selected_composition": fraction(category_counts[f"selected:{cat}"], stats["selected_tokens"])} for cat in ["entity", "attribute_factual_heuristic", "other_content", "function", "punctuation"]}})
    if entity_collator:
        summary["overlap"] = {"micro_jaccard": fraction(stats["intersection_tokens"], stats["union_tokens"]), "macro_jaccard_nonempty_union": sum(jaccards) / len(jaccards) if jaccards else None, "entity_token_precision": fraction(stats["intersection_tokens"], stats["selected_tokens"]), "entity_token_recall": fraction(stats["intersection_tokens"], stats["entity_tokens"]), "per_sample_counts": counts_by_sample}
    save(output / f"summary_{label}.json", summary)
    save(output / f"examples_{label}.json", {"representative": examples, "selected_positive_discrepancy": positive_examples})
    print(label, "DONE", json.dumps({k: v for k, v in summary.items() if k not in {"categories", "overlap"}}), flush=True)
    return summary, examples + positive_examples


def qualitative(output, examples):
    muse_tokenizer = None
    if any(item["dataset"].startswith("MUSE") for item in examples):
        from model import get_tokenizer
        muse_cfg = OmegaConf.load(ROOT / "configs/model/Llama-2-7b-hf.yaml")
        muse_tokenizer = get_tokenizer(muse_cfg.tokenizer_args)
    muse_choice = {}
    for item in examples:
        if item["dataset"].startswith("MUSE"):
            positive = any(t["selected"] and t["delta"] is not None and t["delta"] > 0 for t in item["tokens"])
            if item["dataset"] not in muse_choice or positive:
                muse_choice[item["dataset"]] = item
    parts = ['<!doctype html><meta charset="utf-8"><title>Mask audit examples</title><style>body{max-width:1100px;margin:40px auto;font:16px/1.8 system-ui}article{margin:30px 0;padding:20px;border:1px solid #ccc}mark{background:#ffdc80;border-radius:3px}span.entity{border-bottom:3px solid #679be7}.ignored{opacity:.4}summary{cursor:pointer}</style><h1>Selected-token examples</h1><p>Yellow: Adaptive selection. Blue underline: entity match. Tooltip: discrepancy/category. Gray: ineligible. Categories are lexical heuristics. MUSE shows one short excerpt per corpus.</p>']
    for item in examples:
        is_muse = item["dataset"].startswith("MUSE")
        if is_muse and muse_choice[item["dataset"]] is not item:
            continue
        parts.append(f'<article><h2>{html.escape(item["dataset"])} / {item["doc_id"]}</h2><p>mean={item["sequence_mean"]}; ratio={item["selection_ratio"]:.4f}</p>')
        if item.get("question"):
            parts.append(f'<p><b>Question:</b> {html.escape(item["question"])}</p><p><b>Answer:</b> {html.escape(item["answer"])}</p>')
        display = item["tokens"]
        if is_muse:
            positives = [i for i, t in enumerate(display) if t["selected"] and t["delta"] is not None and t["delta"] > 0]
            choices = positives or [i for i, t in enumerate(display) if t["selected"]]
            display = display[max(0, (choices[0] if choices else 0) - 2):]
            parts.append("… ")
        budget = 40 if is_muse else float("inf")
        item["display_tokens"] = []
        for token in display:
            tag = "mark" if token["selected"] else "span"
            cls = "entity" if token["entity"] else ""
            if not token["eligible"]:
                cls += " ignored"
            title = html.escape(f'position={token["position"]}, delta={token["delta"]}, {token["category"]}')
            text = token["token"]
            if muse_tokenizer is not None and item["dataset"].startswith("MUSE") and token["label"] >= 0:
                # SentencePiece's single-token decode strips word-leading spaces.
                raw = muse_tokenizer.convert_ids_to_tokens(token["label"])
                if raw.startswith("▁") and not text.startswith(" "):
                    text = " " + text
            if budget <= 0:
                break
            if len(text) > budget:
                text = text[:int(budget)]
            budget -= len(text)
            token["display_token"] = text
            item["display_tokens"].append(token)
            parts.append(f'<{tag} class="{cls}" title="{title}">{html.escape(text)}</{tag}>')
        if is_muse:
            parts.append(" …")
        parts.append('</article>')
    (output / "qualitative.html").write_text("".join(parts))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["TOFU", "News", "Books"], choices=["TOFU", "News", "Books"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "saves/audits/revision")
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    report = {"scope": "Full initialization audit; EOS-corrected eligible targets. No optimizer steps.", "legacy_TOFU_selection_ratios": {"forget01": .3197, "forget05": .3406, "forget10": .3422}, "legacy_ratios_note": "Historical pre-EOS-fix ratios; corrected ratios are reported separately, not silently substituted.", "old_rmu": old_rmu_evidence(), "groups": {}}
    all_examples = []
    for name in args.datasets:
        cfg = OmegaConf.load(ROOT / f'configs/model/{"Llama-3.2-1B-Instruct" if name == "TOFU" else "Llama-2-7b-hf"}.yaml')
        template = OmegaConf.to_container(cfg.template_args)
        target_model_id = "open-unlearning/tofu_Llama-3.2-1B-Instruct_full" if name == "TOFU" else f"muse-bench/MUSE-{name}_target"
        cfg.model_args.pretrained_model_name_or_path = target_model_id
        cfg.model_args.device_map = {"": args.device}
        print("Loading target model", target_model_id, flush=True)
        target_model, tokenizer = get_model(cfg)
        target_model.eval()
        for split in ["forget01", "forget05", "forget10"] if name == "TOFU" else ["forget"]:
            label = f"TOFU_{split}" if name == "TOFU" else f"MUSE_{name}"
            summary, examples = audit_group(name, split, target_model, tokenizer, template, output, args.batch_size, args.device)
            summary["target_model"] = target_model_id
            summary["target_model_commit"] = getattr(target_model.config, "_commit_hash", None)
            report["groups"][label] = summary
            all_examples.extend(examples)
            save(output / "audit_report.json", report)
            qualitative(output, all_examples)
        del target_model
        gc.collect()
        torch.cuda.empty_cache()
    print("All audits completed", output, flush=True)


if __name__ == "__main__":
    main()
