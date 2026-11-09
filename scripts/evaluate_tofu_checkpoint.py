"""Run and verify the official TOFU suite before cleaning checkpoint weights."""
import argparse
import hashlib
import json
import logging
import math
from pathlib import Path
import sys

from cleanup_model_weights import remove_model_weights
from freeze_mask_protocol import DEFAULT_PROTOCOL, FORMAL_OUTPUT, ROOT, check_frozen_run, save_json, sha256

sys.path.insert(0, str(ROOT / "src"))
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from scipy.stats import hmean, ks_2samp
from evals import get_evaluators
from model import get_model

METRICS = {
    "ES": "extraction_strength",
    "forget_probability": "forget_Q_A_Prob",
    "forget_rouge": "forget_Q_A_ROUGE",
    "forget_ratio": "forget_truth_ratio",
    "forget_quality": "forget_quality",
    "utility": "model_utility",
    "privleak": "privleak",
}


def read(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def metric_configs(metrics):
    for name, cfg in metrics.items():
        yield name, cfg
        yield from metric_configs(cfg.get("pre_compute", {}))


def configure_eval(protocol, output, reference):
    dataset = protocol["dataset"]
    settings = protocol["official_evaluation"]
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        cfg = compose(config_name="eval", overrides=["experiment=eval/tofu/default",
                      f"forget_split={dataset['forget']}", f"holdout_split={dataset['holdout']}"])
    cfg.eval.tofu.output_dir = str(output)
    cfg.eval.tofu.retain_logs_path = str(reference)
    cfg.eval.tofu.batch_size = settings["batch_size"]
    cfg.eval.tofu.overwrite = False
    resolved = OmegaConf.to_container(cfg.eval.tofu, resolve=True)
    resolved["strict_outputs"] = True
    for _, metric in metric_configs(resolved["metrics"]):
        for spec in metric.get("datasets", {}).values():
            args = spec["args"]
            hf_args = args["hf_args"]
            require(hf_args["path"] == dataset["id"] and hf_args["name"] in settings["dataset_counts"],
                    f"Unexpected official evaluation dataset: {hf_args}")
            hf_args["revision"] = dataset["revision"]
            args["verify_revision"] = True
        if metric["handler"] in {"probability", "extraction_strength"}:
            metric["record_details"] = True
        if metric["handler"] == "mia_min_k":
            require(metric["k"] == settings["min_k"], "Official Min-K configuration differs from the protocol")
    return resolved


def sample_rows(data, expected, name):
    rows = data["value_by_index"]
    require(set(rows) == {str(i) for i in range(expected)}, f"Incomplete sample IDs for {name}")
    return rows.values()


def verify_probability(row, name, candidates):
    fields = ("prob", "avg_loss", "sum_logprob", "mean_logprob", "token_count", "target_token_ids", "token_logprobs")
    entries = [row]
    if candidates > 1:
        require(all(isinstance(row.get(key), list) and len(row[key]) == candidates for key in fields),
                f"Incomplete answer candidates for {name}")
        entries = [dict(zip(fields, values)) for values in zip(*(row[key] for key in fields))]
    for entry in entries:
        require(all(number(entry.get(key)) for key in fields[:4]), f"Invalid probability/log probability for {name}")
        count = entry["token_count"]
        require(count > 0 and count == len(entry["target_token_ids"]) == len(entry["token_logprobs"])
                and all(number(value) for value in entry["token_logprobs"]), f"Incomplete token log probabilities for {name}")


def expected_count(spec, counts):
    if spec.get("datasets"):
        dataset = next(iter(spec["datasets"].values()))
        return counts[dataset["args"]["hf_args"]["name"]]
    return expected_count(next(iter(spec["pre_compute"].values())), counts)


def validate_outputs(config, settings, logs, summary, reference):
    counts = settings["dataset_counts"]
    reference_logs = read(reference)
    reference_truth = [row["score"] for row in sample_rows(reference_logs["forget_truth_ratio"],
                       counts["forget05_perturbed"], "retain-only truth ratio")]
    require(all(number(value) for value in reference_truth), "Invalid retain-only truth ratio")
    reference_auc = reference_logs["mia_min_k"]["agg_value"]
    require(number(reference_auc), "Missing retain-only Min-K AUC")
    validated = {}
    for name, spec in dict(metric_configs(config["metrics"])).items():
        data = logs.get(name, {})
        require(number(data.get("agg_value")), f"Missing or nonfinite aggregate: {name}")
        handler = spec["handler"]
        if handler == "mia_min_k":
            require(number(data.get("auc")) and data["auc"] == data["agg_value"], "Invalid Min-K AUC")
            for ds in spec["datasets"].values():
                split = ds["access_key"]
                size = counts[ds["args"]["hf_args"]["name"]]
                rows = sample_rows(data[split], size, f"{name}/{split}")
                require(all(number(row.get("score")) for row in rows), f"Invalid Min-K score for {split}")
                validated[f"{name}/{split}"] = size
        elif handler in {"probability", "rouge", "extraction_strength", "truth_ratio", "probability_w_options"}:
            size = expected_count(spec, counts)
            for row in sample_rows(data, size, name):
                if handler == "probability":
                    dataset = next(iter(spec["datasets"].values()))
                    candidates = (settings["perturbed_answer_counts"][dataset["args"]["hf_args"]["name"]]
                                  if dataset["args"]["answer_key"] == "perturbed_answer" else 1)
                    verify_probability(row, name, candidates)
                elif handler == "rouge":
                    require(number(row.get(spec["rouge_type"])) and all(isinstance(row.get(k), str)
                            for k in ("input", "ground_truth", "generation")), f"Incomplete generation for {name}")
                elif handler == "extraction_strength":
                    require(number(row.get("score")) and "matching_suffix_start" in row
                            and len(row["target_token_ids"]) == len(row["predicted_token_ids"]), "Incomplete extraction output")
                else:
                    require(number(row.get("score" if handler == "truth_ratio" else "prob")), f"Invalid sample value for {name}")
            validated[name] = size
    aggregate = {key: summary.get(name) for key, name in METRICS.items()}
    require(all(number(v) for v in aggregate.values()), "Missing official metric in TOFU_SUMMARY.json")
    require(all(aggregate[key] == logs[name]["agg_value"] for key, name in METRICS.items()), "Official summary differs from detail logs")
    auc = logs["mia_min_k"]["auc"]
    expected = ((1 - auc) - (1 - reference_auc)) / (1 - reference_auc + 1e-10) * 100
    require(math.isclose(aggregate["privleak"], expected, rel_tol=1e-12, abs_tol=1e-12), "PrivLeak did not use the pinned retain-only reference")
    target_truth = [row["score"] for row in logs["forget_truth_ratio"]["value_by_index"].values()]
    require(aggregate["forget_quality"] == ks_2samp(target_truth, reference_truth).pvalue,
            "Forget Quality did not use the pinned retain-only reference")
    utility_inputs = config["metrics"]["model_utility"]["pre_compute"]
    require(len(utility_inputs) == 9 and math.isclose(aggregate["utility"],
            hmean([logs[name]["agg_value"] for name in utility_inputs]), rel_tol=1e-12), "Invalid nine-metric Model Utility")
    return aggregate, validated


def weight_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluate_checkpoint(checkpoint, output_root, protocol, lock):
    metadata = read(checkpoint / "checkpoint_metadata.json")
    config = read(checkpoint / "resolved_config.json")
    state = read(checkpoint / "trainer_state.json")
    for key, value in {"code_commit": lock["code_commit"], "protocol_sha256": lock["protocol_sha256"],
                       "seed": config["execution"]["seed"], "mask_sha256": config["mask_sha256"],
                       "global_step": state["global_step"]}.items():
        require(metadata[key] == value, f"Checkpoint {key} differs: {checkpoint}")
    reference = output_root / "frozen_inputs" / Path(protocol["retain_only_model"]["evaluation_logs"]).name
    output = checkpoint / "evals" / "official"
    output.mkdir(parents=True, exist_ok=True)
    eval_cfg = configure_eval(protocol, output, reference)
    context = {"checkpoint": metadata, "retain_only_logs_sha256": sha256(reference), "resolved_eval_config": eval_cfg}
    context_path = output / "evaluation_provenance.json"
    validated_path = output / "validated.json"
    weights = [checkpoint / name for name in metadata["saved_weight_files"]]
    if validated_path.exists():
        prior = read(context_path)
        require(prior["context"] == context, "Evaluation provenance differs from the requested checkpoint")
        require(all(weight_hash(path) == prior["weight_sha256"][path.name] for path in weights if path.exists()), "Checkpoint weights changed")
    else:
        require(all(path.is_file() for path in weights), "Checkpoint weights missing before complete evaluation")
        provenance = {"context": context, "weight_sha256": {p.name: weight_hash(p) for p in weights}}
        if context_path.exists():
            require(read(context_path) == provenance, "Partial evaluation belongs to a different checkpoint")
        save_json(context_path, provenance)
        model_cfg = OmegaConf.create(config["model"])
        for args in (model_cfg.model_args, model_cfg.tokenizer_args):
            args.pop("revision", None)
            args.pretrained_model_name_or_path = str(checkpoint)
            args.local_files_only = True
        model_cfg.model_args.device_map = {"": "cuda:0"}
        model, tokenizer = get_model(model_cfg)
        evaluator = get_evaluators(OmegaConf.create({"tofu": eval_cfg}))["tofu"]
        evaluator.evaluate(model=model, tokenizer=tokenizer, template_args=model_cfg.template_args)
    log_path, summary_path = output / "TOFU_EVAL.json", output / "TOFU_SUMMARY.json"
    aggregate, counts = validate_outputs(eval_cfg, protocol["official_evaluation"],
                                         read(log_path), read(summary_path), reference)
    result = {**metadata, "evaluation_profile": "official_tofu", "metrics": aggregate, "sample_counts": counts,
              "artifact_sha256": {path.name: sha256(path) for path in (log_path, summary_path, context_path)}}
    save_json(output / "official_metrics.json", aggregate)
    save_json(validated_path, result)
    remove_model_weights(checkpoint)
    save_json(checkpoint / "evaluated.json", {**result, "weights_removed_after_eval": True})
    print(f"Validated 7 official metrics at step {state['global_step']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-root", type=Path, default=FORMAL_OUTPUT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    protocol, lock = check_frozen_run(args.protocol, args.output_root.resolve())
    evaluate_checkpoint(args.checkpoint.resolve(), args.output_root.resolve(), protocol, lock)


if __name__ == "__main__":
    main()
