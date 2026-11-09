"""Export raw Entity masks, supervised positions, and TOFU name locations.

CPU-only companion to audit_masks.py; it does not recompute model logprobs.
python scripts/audit_entity_alignment.py
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from omegaconf import OmegaConf
from data import get_datasets
from data.collators import DataCollatorWithEntityMask, _nfkc_lower
from model import get_tokenizer
from trainer.masking import eligible_positions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "saves/audits/revision")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    model_cfg = OmegaConf.load(ROOT / "configs/model/Llama-3.2-1B-Instruct.yaml")
    tokenizer = get_tokenizer(model_cfg.tokenizer_args)
    names = json.loads((ROOT / "configs/tofu_forget_authors.json").read_text())
    summary = {"matching_note": "Name-location counts use the collator's normalized full-name regex. The token mask also matches individual subnames; these are lexical matches, not semantic entity annotations.", "groups": {}}
    for split in ["forget01", "forget05", "forget10"]:
        cfg = OmegaConf.load(ROOT / "configs/data/datasets/TOFU_QA_forget.yaml")
        cfg.TOFU_QA_forget.args.hf_args.name = split
        dataset = get_datasets(cfg, tokenizer=tokenizer, template_args=model_cfg.template_args)
        collator = DataCollatorWithEntityMask(tokenizer, names[split], index="index")
        manifest = args.output / f"masks_TOFU_{split}_entity.jsonl"
        audited = {int(r["doc_id"]): r["mask"] for r in map(json.loads, manifest.open())} if manifest.exists() else None
        counts = Counter()
        with (args.output / f"entity_alignment_{split}.jsonl").open("w") as stream:
            for i in range(len(dataset)):
                sample = dataset[i]
                batch = collator([sample])
                raw = batch["entity_mask"][0].bool()
                eligible = eligible_positions(batch)[0]
                effective = raw & eligible
                if audited is not None and effective.int().tolist() != audited[int(sample["index"])]:
                    raise ValueError(f"Entity mask differs from audited manifest: {split}, sample {i}")
                source = dataset.data[i]
                locations = {}
                for field in ["question", "answer"]:
                    text = _nfkc_lower(source[field])
                    locations[field] = bool(collator.rx_full and collator.rx_full.search(text))
                location = "both" if all(locations.values()) else "question_only" if locations["question"] else "answer_only" if locations["answer"] else "neither_full_name"
                counts[location] += 1
                counts["samples"] += 1
                counts["empty_effective_masks"] += int(not effective.any())
                counts["raw_entity_tokens"] += int(raw.sum())
                counts["ignored_entity_tokens"] += int((raw & ~eligible).sum())
                counts["effective_entity_tokens"] += int(effective.sum())
                ids = batch["input_ids"][0].tolist()
                labels = batch["labels"][0].tolist()
                record = {"doc_id": int(sample["index"]), "question": source["question"],
                          "answer": source["answer"], "full_name_location": location,
                          "input_ids": ids, "tokens": [tokenizer.decode([t]) for t in ids],
                          "labels": labels, "ignored_positions": [j for j, v in enumerate(labels) if v == -100],
                          "attention_mask": batch["attention_mask"][0].tolist(),
                          "raw_entity_mask": raw.int().tolist(),
                          "eligible_positions": eligible.int().tolist(),
                          "effective_entity_mask": effective.int().tolist(),
                          "effective_count": int(effective.sum())}
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        counts["empty_mask_rate"] = counts["empty_effective_masks"] / counts["samples"]
        summary["groups"][split] = dict(counts)
    (args.output / "entity_alignment_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
