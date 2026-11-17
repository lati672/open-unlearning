"""Verify the v4 controls and freeze annotated Entity+Attribute TOFU masks."""

import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sys
from zipfile import ZipFile

WORKSPACE = Path(__file__).resolve().parents[1]
EXPERIMENTS = WORKSPACE / "saves/experiments"
ROOT = EXPERIMENTS / "tofu_forget05_entity_attribute_ablation"
SOURCE = EXPERIMENTS / "tofu_forget05_adaptive_density_phase_a/source_v4"
FORMAL = EXPERIMENTS / "tofu_forget05_multiseed_v4_official"
INPUTS = FORMAL / "frozen_inputs"
RELEVANCE = WORKSPACE / "saves/audits/adaptive_relevance"
EXPORTS = WORKSPACE / "saves/exports"
COMMIT = "629363802f79b4ccb7b1d4e38bedc2980f6f47df"
SEEDS = (0, 1, 2)
STEPS = (60, 120)
ARMS = ("EntityAttribute", "RandomEntityAttribute")
CONTROLS = ("base", "entity", "random_entity")
ARG_PATHS = {"output_dir", "logging_dir", "run_name"}
MASK_ARGS = {"mask", "random_reference", "mask_seed"}

sys.path[:0] = [str(SOURCE / "scripts"), str(SOURCE / "src")]

from freeze_mask_protocol import check_environment, save_json, verify_dataset_revision
from prepare_low_confidence import read, require, write_csv
from run_low_confidence import ARTIFACTS, METRICS, digest, replay_metrics, validate_outputs

import torch
from omegaconf import OmegaConf

from data import get_datasets
from data.collators import DataCollatorForSupervisedDataset, FrozenMaskCollator
from model import get_tokenizer
from trainer.masking import density_matched_random_mask, eligible_positions


def run_name(variant, seed):
    return f"NPO_{variant}_summed_forget05_seed{seed}_steps120"


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def source_hashes():
    return {p.relative_to(SOURCE).as_posix(): digest(p)
            for folder in ("src", "configs") for p in sorted((SOURCE / folder).rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts}


def protocol_signature(config, args):
    return {"training_args": {k: v for k, v in args.items() if k not in ARG_PATHS | {"seed"}},
            "objective_args": {k: v for k, v in config["trainer"]["method_args"].items() if k not in MASK_ARGS},
            "model": config["model"], "data": config["data"], "protocol": config["frozen_protocol"],
            "backend": config["execution"]["backend"], "world_size": config["execution"]["world_size"]}


def verified_member(archive, prefix, files, name):
    value = archive.read(f"{prefix}/{name}")
    require(sha_bytes(value) == files[name], f"Export member hash differs: {name}")
    return value


def verify_controls(lock):
    name = "tofu_forget05_multiseed_official_v4"
    source = EXPORTS / f"{name}.zip"
    zip_hash = digest(source)
    require(zip_hash == source.with_suffix(".zip.sha256").read_text().split()[0], "v4 export SHA-256 differs")
    reference = INPUTS / "TOFU_EVAL.json"
    settings = lock["protocol"]
    reference_hash = settings["inputs"][settings["retain_only_model"]["evaluation_logs"]]
    require(digest(reference) == reference_hash, "Retain-only evaluator reference differs")
    reference_values = read(reference)
    input_hashes = {Path(path).name: value for path, value in settings["inputs"].items()} | lock["derived_inputs"]
    signature = evaluation = None
    controls, records = [], {}
    with ZipFile(source) as archive:
        manifest = json.loads(archive.read(f"{name}/MANIFEST.json"))
        require(manifest["code_commit"] == lock["code_commit"] == COMMIT
                and manifest["protocol_sha256"] == lock["protocol_sha256"], "v4 archive provenance differs")
        files = {row["path"]: row["sha256"] for row in manifest["files"]}
        table = list(csv.DictReader(io.StringIO(verified_member(archive, name, files,
                                    "tables/multiseed_official_long.csv").decode())))
        require(verified_member(archive, name, files, "provenance/protocol_manifest.json")
                == (FORMAL / "protocol_manifest.json").read_bytes(), "Local protocol differs from official export")
        for variant in CONTROLS:
            for seed in SEEDS:
                run = run_name(variant, seed)
                prefix = f"runs/{run}/"
                evidence = {}
                for relative in ("resolved_config.json", "training_args.json",
                                 *(f"checkpoint-{step}/trainer_state.json" for step in STEPS),
                                 *(f"checkpoint-{step}/evals/official/{a}" for step in STEPS for a in ARTIFACTS)):
                    key = prefix + relative
                    raw = verified_member(archive, name, files, key)
                    require(sha_bytes(raw) == digest(FORMAL / "collected_results" / run / relative),
                            f"Local control differs from official export: {key}")
                    evidence[relative] = json.loads(raw)
                    records[key] = files[key]
                cfg, args = (evidence[f"{k}.json"] for k in ("resolved_config", "training_args"))
                require(cfg["trainer"]["handler"] == ("RandomNPO" if variant == "random_entity" else "NPO")
                        and cfg["execution"]["normalization"] == "summed" and args["seed"] == cfg["execution"]["seed"] == seed
                        and cfg["execution"]["variant"] == variant
                        and cfg["mask_sha256"] == input_hashes[f"masks_TOFU_forget05_{variant}.jsonl"],
                        f"Wrong v4 control: {run}")
                found = protocol_signature(cfg, args)
                require(signature is None or signature == found, f"Training protocol differs for {run}")
                signature = found
                for step in STEPS:
                    checkpoint = f"checkpoint-{step}"
                    local_prefix = f"{checkpoint}/evals/official/"
                    val, summary, detail, logs, metrics = [evidence[local_prefix + n + ".json"] for n in
                        ("validated", "TOFU_SUMMARY", "evaluation_provenance", "TOFU_EVAL", "official_metrics")]
                    state = evidence[f"{checkpoint}/trainer_state.json"]
                    require(val["seed"] == seed and val["step"] == val["global_step"] == state["global_step"] == step
                            and val["actual_epoch"] == state["epoch"] == settings["training"]["checkpoints"][str(step)]
                            and val["mask_sha256"] == cfg["mask_sha256"] and val["code_commit"] == COMMIT
                            and val["protocol_sha256"] == lock["protocol_sha256"], f"Checkpoint provenance differs: {run}/{step}")
                    context = detail["context"]
                    eval_cfg = context["resolved_eval_config"]
                    found = {k: v for k, v in eval_cfg.items() if k not in {"output_dir", "retain_logs_path"}}
                    require(evaluation is None or evaluation == found, f"Evaluator config differs for {run}/{step}")
                    require(context["retain_only_logs_sha256"] == reference_hash, "Evaluation reference differs")
                    evaluation = found
                    official, counts = validate_outputs(eval_cfg, settings["official_evaluation"], logs, summary, reference)
                    replay = replay_metrics(eval_cfg, logs, reference_values)
                    require(official == metrics == val["metrics"] and counts == val["sample_counts"]
                            and all(math.isclose(v, replay[k], rel_tol=1e-10, abs_tol=1e-14) for k, v in metrics.items()),
                            f"Archived metric replay differs: {run}/{step}")
                    row, = [r for r in table if r["run"] == run and int(r["step"]) == step]
                    require(all(float(row[k if k != "privleak" else "privleak_signed"]) == v for k, v in metrics.items())
                            and float(row["privleak_abs"]) == abs(metrics["privleak"]), f"Archived table differs: {run}/{step}")
                    controls.append(row)
    require(len(controls) == 18, "Expected 18 v4 control checkpoints")
    return signature, evaluation, controls, records, zip_hash


def annotation_sources(lock):
    annotation_paths = [RELEVANCE / "answer_spans.tsv"] + [
        EXPERIMENTS / run / "mask_audit/answer_spans.tsv" for run in
        ("tofu_forget05_adaptive_density_phase_a", "tofu_forget05_adaptive_density_phase_b",
         "tofu_forget05_lowconfidence_matched")]
    present = [p for p in annotation_paths if p.is_file()]
    require(present, "Upload the original answer_spans.tsv before running the ablation")
    checksums = {p.relative_to(WORKSPACE).as_posix(): digest(p) for p in present}
    source = EXPORTS / "tofu_forget05_adaptive_mask_relevance.zip"
    with ZipFile(source) as archive:
        prefix = source.stem
        manifest = json.loads(archive.read(f"{prefix}/MANIFEST.json"))
        require(manifest["protocol_sha256"] == lock["protocol_sha256"], "Annotation export references a different protocol")
        files = manifest["files"]
        original, answers, tokens = [verified_member(archive, prefix, files, f"data/token_audit/{name}")
                                     for name in ("answer_spans.tsv", "samples.csv", "tokens.csv")]
    require(set(checksums.values()) == {sha_bytes(original)}, "Annotation copies differ")
    require(digest(RELEVANCE / "tokens.csv") == sha_bytes(tokens), "Archived token offset table differs")
    return original, answers, tokens, checksums


def parse_annotations(raw):
    rows = {}
    for line in raw.decode().splitlines():
        doc, scope, *spans = line.split("\t")
        doc = int(doc)
        require(doc not in rows and scope in {"D", "S", "-"} and (scope != "-") == bool(spans) and all(spans),
                f"Malformed annotation at {doc}")
        rows[doc] = (scope, spans)
    require(list(rows) == list(range(200)), "Annotation sample IDs or order differ")
    return rows


def annotation_interval(answer, text):
    # The original TSV encodes a literal substring within a fixed, ID-keyed answer.
    # Resolve it only if its exact character coordinates are unique in that answer.
    starts = [m.start() for m in re.finditer(f"(?={re.escape(text)})", answer)]
    require(len(starts) == 1, f"Missing or ambiguous original annotation {text!r}")
    return starts[0], starts[0] + len(text)


def span_labels(offsets, start, end, valid):
    return {pos for pos, (left, right) in enumerate(offsets)
            if pos in valid and left < right and max(start, left) < min(end, right)}


def union_positions(entity, attribute, valid):
    return (entity | attribute) & valid


def render_offsets(tokenizer, template, question, raw_answer, training_tokens):
    messages = [{"role": "system", "content": template["system_prompt"]},
                {"role": "user", "content": question}]
    options = {"date_string": template["date_string"]}
    prefix = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **options)
    text = tokenizer.apply_chat_template([*messages, {"role": "assistant", "content": raw_answer}],
                                         tokenize=False, add_generation_prompt=False, **options)
    answer = raw_answer.strip()
    require(text == prefix + answer + template["asst_end_tag"], "Assistant content boundary differs from frozen template")
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
    if ids[-1] != tokenizer.eos_token_id:
        ids.append(tokenizer.eos_token_id)
        offsets.append((len(text), len(text)))
    require(ids == training_tokens, "Rendered conversation tokens differ from actual training input IDs")
    return len(prefix), offsets


def indexed_jsonl(path):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    require([row["doc_id"] for row in rows] == list(range(200)), f"Sample IDs or order differ in {path.name}")
    return rows


def audit_masks(root, lock, original, archived_answers, archived_tokens, checksums):
    settings = lock["protocol"]
    model = read(FORMAL / "collected_results" / run_name("entity", 0) / "resolved_config.json")["model"]
    tokenizer = get_tokenizer(OmegaConf.create(model["tokenizer_args"]))
    require(tokenizer.is_fast and model["template_args"]["apply_chat_template"], "Expected frozen fast tokenizer and chat template")
    config = read(FORMAL / "collected_results" / run_name("entity", 0) / "resolved_config.json")["data"]["forget"]
    dataset = get_datasets(OmegaConf.create(config), tokenizer=tokenizer, template_args=model["template_args"])
    verify_dataset_revision(dataset, settings["dataset"]["revision"])
    require(len(dataset) == 200 and dataset.fs_data is None and dataset.max_length == 512, "Unexpected Forget05 training dataset")
    collator = DataCollatorForSupervisedDataset(tokenizer, index="index")
    hashes = {Path(path).name: value for path, value in settings["inputs"].items()} | lock["derived_inputs"]
    def load_verified(name):
        path = INPUTS / name
        require(digest(path) == hashes[name], f"Frozen input differs: {name}")
        return indexed_jsonl(path)

    base, entity, original_random = [load_verified(f"masks_TOFU_forget05_{arm}.jsonl")
                                     for arm in ("base", "entity", "random_entity")]
    tokens = load_verified("tokens_TOFU_forget05.jsonl")
    logprobs = load_verified("logprobs.json")
    answers = list(csv.DictReader(io.StringIO(archived_answers.decode())))
    require([int(row["doc_id"]) for row in answers] == list(range(200)), "Archived answer IDs or order differ")
    old_offsets = {}
    for row in csv.DictReader(io.StringIO(archived_tokens.decode())):
        key = (int(row["doc_id"]), int(row["position"]))
        require(key not in old_offsets, "Duplicate archived token offset")
        old_offsets[key] = row
    annotations = parse_annotations(original)
    counts, alignments, examples = [], [], []
    masks = {arm: [] for arm in ARMS}
    scopes = {name: {"samples": 0, "spans": 0, "attribute_tokens": 0,
                     **{arm: {"selected_tokens": 0, "fully_covered_spans": 0, "partly_or_fully_covered_spans": 0}
                        for arm in ("Entity", *ARMS)}} for name in ("direct", "descriptive")}
    consumed_offsets = set()
    for doc in range(200):
        raw, item = dataset.data[doc], dataset[doc]
        question, answer = raw[dataset.question_key], raw[dataset.answer_key]
        require(raw["index"] == item["index"] == doc and question == answers[doc]["question"]
                and answer.strip() == answers[doc]["answer"]
                and all((record["question"], record["answer"]) == (question, answer)
                        for record in (tokens[doc], logprobs[doc])),
                f"Archived annotation answers or sample order differ at doc_id {doc}")
        ids = item["input_ids"].tolist()
        batch = collator([item])
        eligible = eligible_positions(batch)
        valid = set(eligible[0].nonzero().flatten().tolist())
        base_set, entity_set, old_random = [
            {i for i, flag in enumerate(row[doc]["mask"]) if flag} for row in (base, entity, original_random)]
        require(all(row[doc]["input_ids"] == ids for row in (base, entity, original_random))
                and valid == base_set and entity_set <= valid, f"Frozen positions differ for doc_id {doc}")
        frozen_audit = {t["position"]: t for t in tokens[doc]["tokens"] if t["eligible"]}
        require(valid == frozen_audit.keys() and
                {p for p, t in frozen_audit.items() if t["entity"]} == entity_set, f"Audited Entity differs at {doc}")
        for pos in valid:
            base_lp = logprobs[doc]["token_logprobs"][pos]
            delta = frozen_audit[pos]["delta"]
            require(base_lp is not None and delta is not None and all(map(math.isfinite, (base_lp, delta, base_lp - delta)))
                    and item["labels"][pos].item() == ids[pos] and item["attention_mask"][pos].item()
                    and pos > 0 and item["attention_mask"][pos-1].item(),
                    f"Ineligible or nonfinite score at {doc}/{pos}")
        prefix, offsets = render_offsets(tokenizer, model["template_args"], question, answer, ids)
        answer = answer.strip()
        for pos in sorted(valid):
            old = old_offsets[doc, pos]
            left, right = offsets[pos]
            old_left, old_right = int(old["char_start"]), int(old["char_end"])
            if old_left < old_right:
                require((left - prefix, right - prefix) == (old_left, old_right)
                        and 0 <= old_left < old_right <= len(answer), f"Archived offset differs at {doc}/{pos}")
            else:
                require(ids[pos] in tokenizer.all_special_ids, f"Empty offset on a content token at {doc}/{pos}")
            require(ids[pos] == int(old["token_id"]), f"Archived token ID differs at {doc}/{pos}")
            consumed_offsets.add((doc, pos))
        scope, spans = annotations[doc]
        intervals, attribute = [], set()
        for span in spans:
            start, end = annotation_interval(answer, span)
            selected = span_labels(offsets, prefix + start, prefix + end, valid)
            require(selected and answer[start:end] == span, f"Annotation failed at doc_id {doc}: {span!r}")
            attribute.update(selected)
            intervals.append({"text": span, "answer_char_start": start, "answer_char_end": end,
                              "chat_char_start": prefix + start, "chat_char_end": prefix + end,
                              "label_positions": sorted(selected)})
        union = union_positions(entity_set, attribute, valid)
        to_tensor = lambda positions: torch.tensor([[p in positions for p in range(len(ids))]], dtype=torch.bool)
        seed = settings["random_mask_seed"]
        verify_random = density_matched_random_mask(eligible, to_tensor(entity_set), [doc], seed)
        require(set(verify_random[0].nonzero().flatten().tolist()) == old_random, f"Random sampler differs from v4 at {doc}")
        random = density_matched_random_mask(eligible, to_tensor(union), [doc], seed)
        matched = set(random[0].nonzero().flatten().tolist())
        require(len(matched) == len(union) and matched <= valid and union <= valid, f"Invalid matched count or eligibility at {doc}")
        text_positions = {p for p in valid if ids[p] not in tokenizer.all_special_ids}
        arms = {"Entity": entity_set, ARMS[0]: union, ARMS[1]: matched}
        if scope != "-":
            group = scopes[{"D": "direct", "S": "descriptive"}[scope]]
            group["samples"] += 1
            group["spans"] += len(intervals)
            group["attribute_tokens"] += len(attribute)
            for arm, positions in arms.items():
                group[arm]["selected_tokens"] += len(attribute & positions)
                for interval in intervals:
                    covered = set(interval["label_positions"])
                    group[arm]["fully_covered_spans"] += int(covered <= positions)
                    group[arm]["partly_or_fully_covered_spans"] += int(bool(covered & positions))
        counts.append({"doc_id": doc, "scope": scope, "eligible": len(valid), "answer_text_tokens": len(text_positions),
                       "entity": len(entity_set), "attribute": len(attribute), "overlap": len(entity_set & attribute),
                       "entity_attribute": len(union), "random_entity_attribute": len(matched),
                       "selection_ratio": len(union) / len(valid),
                       "all_eligible_selected": union == valid, "all_answer_text_selected": text_positions <= union})
        alignments.append({"doc_id": doc, "scope": scope, "rendered_answer_char_start": prefix, "spans": intervals,
                           "sequence_ids_sha256": sha_bytes(json.dumps(ids).encode())})
        for arm in ARMS:
            masks[arm].append({"doc_id": doc, "input_ids": ids, "mask": to_tensor(arms[arm])[0].int().tolist()})
        if doc % 20 in (1, 7):
            examples.append({"doc_id": doc, "question": question, "answer": answer, "scope": scope, "annotations": intervals,
                             "eligible_tokens": [{"label_index": p, "logit_index": p-1, "token_id": ids[p],
                                 "decoded": tokenizer.decode([ids[p]], clean_up_tokenization_spaces=False),
                                 "entity": p in entity_set, "attribute": p in attribute,
                                 "union": p in union, "matched_random": p in matched} for p in sorted(valid)]})

    require(consumed_offsets == old_offsets.keys() and len(examples) >= 10, "Incomplete audited token inventory")
    overall = {k: sum(group[k] for group in scopes.values()) for k in ("samples", "spans", "attribute_tokens")}
    for arm in ("Entity", *ARMS):
        overall[arm] = {k: sum(group[arm][k] for group in scopes.values()) for k in scopes["direct"][arm]}
    scopes["overall"] = overall
    for group in scopes.values():
        for arm in ("Entity", *ARMS):
            group[arm]["attribute_token_coverage"] = group[arm]["selected_tokens"] / group["attribute_tokens"]
            group[arm]["full_span_coverage"] = group[arm]["fully_covered_spans"] / group["spans"]
    totals = {k: sum(row[k] for row in counts) for k in ("eligible", "entity", "attribute", "overlap",
                                                        "entity_attribute", "random_entity_attribute")}
    require(0 < totals["entity_attribute"] == totals["random_entity_attribute"] < totals["eligible"],
            "Masks have invalid density")
    frozen = root / "frozen_inputs"
    frozen.mkdir(exist_ok=True)
    (frozen / "answer_spans.tsv").write_bytes(original)
    mask_hashes = {}
    for arm, records in masks.items():
        path = frozen / f"{arm}.jsonl"
        with path.open("w") as stream:
            for row in records:
                stream.write(json.dumps(row, separators=(",", ":")) + "\n")
        mask_hashes[arm] = digest(path)
        save_json(path.with_suffix(".metadata.json"), {"arm": arm, "sha256": mask_hashes[arm], "samples": 200,
            "annotation_sha256": sha_bytes(original), "random_mask_seed": seed if arm == ARMS[1] else None,
            "source_mask_sha256": mask_hashes.get(ARMS[0]), "target_revision": settings["target_model"]["revision"],
            "dataset_revision": settings["dataset"]["revision"], "sample_index": "doc_id",
            "position_convention": "input-aligned target label position t; prediction at logit t-1"})
    write_csv(root / "mask_per_sample.csv", counts)
    save_json(root / "annotation_alignment.json", alignments)
    save_json(root / "decoded_examples.json", examples)
    summary = {"samples": 200, "matched_samples": 200, "totals": totals,
               "selection_ratios": {"Entity": totals["entity"] / totals["eligible"],
                   **{arm: totals["entity_attribute"] / totals["eligible"] for arm in ARMS}},
               "empty_masks": {arm: sum(row[key] == 0 for row in counts) for arm, key in
                               (("Entity", "entity"), (ARMS[0], "entity_attribute"), (ARMS[1], "random_entity_attribute"))},
               "fully_selected_eligible_samples": sum(row["all_eligible_selected"] for row in counts),
               "fully_selected_answer_text_samples": [row["doc_id"] for row in counts if row["all_answer_text_selected"]],
               "factual_coverage": scopes, "annotation_sources": checksums, "annotation_sha256": sha_bytes(original),
               "mask_hashes": mask_hashes, "random_mask_seed": seed,
               "alignment": "Original literal spans require one exact location in an ID-matched archived answer. Full training chat offsets must agree with audited token offsets.",
               "selected_noneligible": 0, "selected_prompt_only": 0, "selected_ignore_index": 0,
               "selected_nonfinite": 0, "selected_out_of_range": 0, "empty_pattern_match": True,
               "original_random_sampler_reproduced_samples": 200, "decoded_examples": len(examples)}
    save_json(root / "mask_audit.json", summary)
    return summary


def prepare(root):
    root.mkdir(parents=True, exist_ok=True)
    require(not (root / "protocol_manifest.json").exists(), "A frozen manifest already exists")
    lock = read(FORMAL / "protocol_manifest.json")
    require(lock["code_commit"] == COMMIT
            and digest(SOURCE / "configs/protocols/tofu_forget05_multiseed.json") == lock["protocol_sha256"],
            "v4 source protocol differs")
    check_environment(lock["protocol"])
    original, answers, tokens, checksums = annotation_sources(lock)
    signature, eval_signature, controls, archived, zip_hash = verify_controls(lock)
    audit = audit_masks(root, lock, original, answers, tokens, checksums)
    write_csv(root / "archived_controls.csv", controls)
    save_json(root / "v4_evaluation_signature.json", eval_signature)
    save_json(root / "training_signature.json", signature)
    save_json(root / "run_plan.json", {"new_runs": [{"arm": arm, "seed": seed} for seed in SEEDS for arm in ARMS],
              "control_arms": CONTROLS, "steps": STEPS, "primary_step": 120, "smoke_steps": 4, "smoke_seed": 0})
    paths = ["archived_controls.csv", "v4_evaluation_signature.json", "training_signature.json", "run_plan.json",
             "mask_audit.json", "mask_per_sample.csv", "annotation_alignment.json", "decoded_examples.json",
             "frozen_inputs/answer_spans.tsv", *(f"frozen_inputs/{arm}.{ext}" for arm in ARMS for ext in ("jsonl", "metadata.json"))]
    save_json(root / "protocol_manifest.json", {
        "experiment": root.name, "protocol": lock["protocol"], "code_commit": COMMIT,
        "v4_protocol_sha256": lock["protocol_sha256"], "control_zip_sha256": zip_hash, "control_artifacts": archived,
        "source_sha256": source_hashes(), "evaluator_sha256": digest(SOURCE / "scripts/evaluate_tofu_checkpoint.py"),
        "inputs_sha256": {name: digest(root / name) for name in paths}, "mask_hashes": audit["mask_hashes"],
        "annotation_sha256": audit["annotation_sha256"], "mask_seed": audit["random_mask_seed"],
        "checkpoints": list(STEPS), "primary_step": 120, "seeds": list(SEEDS)})
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    prepare(ROOT)
