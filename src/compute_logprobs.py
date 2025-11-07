# Example: compute TOFU forget01 logprobs
# python src/compute_logprobs.py dataset.TOFU_QA_forget.args.hf_args.name=forget01 +dataset_split=forget01

import json
import logging
import os
from pathlib import Path

import hydra
import torch
from data import get_datasets
from data.collators import DataCollatorForSupervisedDataset
from model import (
    _add_or_replace_eos_token,
    get_model as load_hf_model,
    get_tokenizer as load_hf_tokenizer,
)
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf, open_dict
from torch.utils.data import DataLoader
from transformers import set_seed
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def _resolve_dataset_metadata(dataset_cfg, fallback_split=None):
    if dataset_cfg is None:
        raise ValueError("dataset_cfg is required to determine logprob output path.")

    if isinstance(dataset_cfg, DictConfig):
        dataset_items = list(dataset_cfg.items())
    else:
        dataset_cfg = dict(dataset_cfg)
        dataset_items = list(dataset_cfg.items())

    if len(dataset_items) != 1:
        raise ValueError(
            f"Expected exactly one dataset entry, found {len(dataset_items)} in {dataset_cfg}."
        )

    dataset_name, dataset_spec = dataset_items[0]
    if isinstance(dataset_spec, DictConfig):
        args_cfg = dataset_spec.get("args", {})
    else:
        args_cfg = dataset_spec.get("args", {})

    args = (
        OmegaConf.to_container(args_cfg, resolve=True)
        if isinstance(args_cfg, DictConfig)
        else args_cfg or {}
    )
    hf_args = args.get("hf_args", {}) if isinstance(args, dict) else {}

    dataset_split = (
        hf_args.get("name")
        if isinstance(hf_args, dict)
        else getattr(hf_args, "name", None)
    )
    if dataset_split is None and isinstance(hf_args, dict):
        dataset_split = hf_args.get("split") or hf_args.get("subset")
    if dataset_split is None and not isinstance(hf_args, dict):
        dataset_split = getattr(hf_args, "split", None) or getattr(
            hf_args, "subset", None
        )
    if dataset_split is None:
        dataset_split = fallback_split

    if dataset_split is None:
        raise ValueError(
            "Unable to determine dataset split from configuration. Provide `forget_split` or specify hf_args.name."
        )

    return dataset_name, dataset_split


def _sanitize_model_name(model_cfg: DictConfig):
    tokenizer_args = getattr(model_cfg, "tokenizer_args", None)
    if tokenizer_args  is None:
        raise ValueError("Model configuration missing `tokenizer_args`.")

    if isinstance(tokenizer_args, DictConfig):
        pretrained_path = tokenizer_args.get("pretrained_model_name_or_path", None)
    else:
        pretrained_path = tokenizer_args.get("pretrained_model_name_or_path", None)

    if pretrained_path is None:
        raise ValueError(
            "Model configuration must include `pretrained_model_name_or_path`."
        )
    model_name = os.path.basename(pretrained_path.rstrip("/"))
    return model_name.replace("/", "_")


def construct_logprob_path(
    dataset_cfg,
    model_cfg: DictConfig,
    forget_split=None,
    project_root=None,
    base_dir="saves/logprobs",
    filename="logprobs.json",
):
    dataset_name, dataset_split = _resolve_dataset_metadata(
        dataset_cfg, fallback_split=forget_split
    )
    model_name = _sanitize_model_name(model_cfg)

    project_root_path = (
        Path(project_root) if project_root is not None else Path(get_original_cwd())
    )
    base_dir_path = Path(base_dir)
    if not base_dir_path.is_absolute():
        base_dir_path = project_root_path / base_dir_path

    output_dir = base_dir_path / f"{dataset_name}_{dataset_split}_{model_name}"
    return output_dir / filename, dataset_name, dataset_split, model_name


def configure_logprob_collator(
    collator_cfg,
    data_cfg,
    model_cfg: DictConfig,
    forget_split=None,
    project_root=None,
    base_dir="saves/logprobs",
    filename="logprobs.json",
):
    if "DataCollatorWithLogProbs" not in collator_cfg:
        return None

    dataset_cfg_for_path = None
    if data_cfg is None:
        raise ValueError("data_cfg is required to configure logprob collator.")

    priority_splits = ("forget", "train")
    for split_key in priority_splits:
        section_cfg = data_cfg.get(split_key)
        if section_cfg:
            dataset_cfg_for_path = section_cfg
            break

    if dataset_cfg_for_path is None:
        raise ValueError(
            "Unable to locate dataset configuration (forget/train) to determine logprob path."
        )

    logprob_path, dataset_name, split_name, model_name = construct_logprob_path(
        dataset_cfg_for_path,
        model_cfg,
        forget_split=forget_split,
        project_root=project_root,
        base_dir=base_dir,
        filename=filename,
    )

    logger.info(
        "Using logprob file %s (dataset=%s, split=%s, model=%s)",
        logprob_path,
        dataset_name,
        split_name,
        model_name,
    )

    collator_section = collator_cfg.get("DataCollatorWithLogProbs")
    if collator_section is None:
        raise ValueError(
            "Collator configuration expected to contain 'DataCollatorWithLogProbs'."
        )

    with open_dict(collator_section["args"]):
        collator_section["args"]["logprob_path"] = str(logprob_path)

    return logprob_path


def _compute_sequence_logprobs(log_probs_row, sequence_ids):
    """
    Align log probabilities with sequence_ids, using None where logits are undefined
    (e.g., the first token which has no previous context).
    """
    row = log_probs_row.detach().cpu()
    seq_len = row.shape[0]
    seq_logprobs = []
    for idx, token_id in enumerate(sequence_ids):
        logit_idx = idx - 1
        if 0 <= logit_idx < seq_len:
            seq_logprobs.append(row[logit_idx, token_id].item())
        else:
            seq_logprobs.append(None)
    return seq_logprobs


def compute_token_logprobs(model, tokenizer, input_ids, attention_mask, questions=None):
    """
    Compute log probabilities for each token in the answer portion.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        input_ids: Input token ids [batch_size, seq_len]
        attention_mask: Attention mask [batch_size, seq_len]
        questions: Optional list of question strings (unused; kept for compatibility)
    
    Returns:
        List of dictionaries containing token-level log probabilities
    """
    with torch.no_grad():
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        logits = outputs.logits  # [batch_size, seq_len, vocab_size]
        log_probs = torch.log_softmax(logits, dim=-1)  # [batch_size, seq_len, vocab_size]
        
        results = []

        if questions is not None:
            if len(questions) != input_ids.shape[0]:
                raise ValueError(
                    "Length of `questions` must match batch size when provided."
                )

        for i in range(input_ids.shape[0]):
            sequence_ids = input_ids[i].detach().cpu().tolist()
            tokens_text = [tokenizer.decode([tid]) for tid in sequence_ids]
            token_logprobs = _compute_sequence_logprobs(log_probs[i], sequence_ids)

            results.append({
                'token_logprobs': token_logprobs,
                'tokens': tokens_text,
                'num_tokens': len(sequence_ids)
            })
        return results


@hydra.main(version_base=None, config_path="../configs", config_name="compute_logprobs")
def main(config):
    logger.info("="*80)
    logger.info("Starting Log Probability Computation")
    logger.info("="*80)

    dataset_cfg = config.get("dataset", None)
    if dataset_cfg is None:
        raise ValueError(
            "Missing `dataset` configuration. Please include a dataset entry in compute_logprobs.yaml."
        )

    model_cfg: DictConfig = config.get("model")
    if model_cfg is None:
        raise ValueError(
            "Missing `model` configuration. Please provide a model defaults entry (e.g., model: Llama-3.2-1B-Instruct)."
        )

    base_dir_setting = (
        config.get("output_dir_base", None)
        or config.get("output_dir", None)
        or "outdir"
    )
    logprob_file_path, dataset_label, dataset_split, model_name = construct_logprob_path(
        dataset_cfg,
        model_cfg,
        forget_split=config.get("dataset_split", None),
        base_dir=base_dir_setting,
    )

    outdir_override = config.get("outdir", None)
    if outdir_override:
        run_dir = os.path.abspath(outdir_override)
        output_filename = os.path.basename(logprob_file_path)
    else:
        run_dir = str(logprob_file_path.parent)
        output_filename = logprob_file_path.name

    os.makedirs(run_dir, exist_ok=True)
    logger.info(f"Using output directory: {run_dir}")

    output_dir = run_dir

    log_file = os.path.join(output_dir, "compute_logprobs.log")
    existing_file_handlers = [
        handler for handler in logger.handlers
        if isinstance(handler, logging.FileHandler)
        and getattr(handler, "baseFilename", None) == os.path.abspath(log_file)
    ]
    if not existing_file_handlers:
        file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setFormatter(
            logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        )
        logger.addHandler(file_handler)
    logger.info(f"Logging to: {log_file}")

    output_json_path = os.path.join(output_dir, output_filename)

    config_snapshot = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    config_snapshot.output_dir = output_dir
    config_snapshot.output_path = output_json_path
    config_path = os.path.join(output_dir, "config.yaml")
    OmegaConf.save(config=config_snapshot, f=config_path, resolve=True)
    logger.info(f"Saved run configuration to: {config_path}")
    
    seed_value = config.get('seed', 0)
    set_seed(seed_value)
    logger.info(f"Random seed set to: {seed_value}")
    
    template_args_cfg = model_cfg.get('template_args', {})
    template_args = OmegaConf.to_container(template_args_cfg, resolve=True) or {}
    max_length = template_args.get('max_length', 512)
    logger.info(f"Using chat template with max_length: {max_length}")

    with open_dict(model_cfg.model_args):
        model_cfg.model_args.setdefault("device_map", "auto")

    # Load model and tokenizer using shared utilities
    model, _ = load_hf_model(model_cfg)
    tokenizer = load_hf_tokenizer(model_cfg.tokenizer_args)

    desired_pad_token = template_args.get('pad_token')
    if desired_pad_token:
        _add_or_replace_eos_token(tokenizer, desired_pad_token)
        tokenizer.pad_token = desired_pad_token
    elif tokenizer.eos_token is None:
        _add_or_replace_eos_token(tokenizer, "<|endoftext|>")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    model.resize_token_embeddings(len(tokenizer))
    model.eval()  # Set to evaluation mode
    logger.info("Model set to evaluation mode")
    
    # Determine the device for inputs based on model's device
    # If model uses device_map, get the first device
    if hasattr(model, 'hf_device_map'):
        # Model is distributed across devices
        first_device = list(model.hf_device_map.values())[0]
        logger.info(f"Model is distributed. First device: {first_device}")
        logger.info(f"Device map: {model.hf_device_map}")
    else:
        # Model is on a single device
        first_device = next(model.parameters()).device
        logger.info(f"Model is on single device: {first_device}")

    logger.info(f"Selected dataset: {dataset_label} (split: {dataset_split})")

    dataset = get_datasets(
        dataset_cfg,
        tokenizer=tokenizer,
        template_args=template_args,
        predict_with_generate=False,
    )

    if isinstance(dataset, dict):
        raise ValueError(
            "compute_logprobs expects a single dataset configuration, but multiple were provided."
        )

    logger.info(f"Dataset loaded: {len(dataset)} examples")

    collator = DataCollatorForSupervisedDataset(
        tokenizer=tokenizer,
        padding_side=tokenizer.padding_side,
        index="index",
    )

    data_loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=collator,
    )
    logger.info(f"Batch size: {config['batch_size']}")
    
    results = []
    
    logger.info("="*80)
    logger.info("Processing batches...")
    logger.info("="*80)
    
    # Process each batch
    for batch_idx, batch in enumerate(tqdm(data_loader, desc="Computing log probabilities")):
        input_ids = batch["input_ids"].to(first_device)
        attention_mask = batch["attention_mask"].to(first_device)

        indices = batch.get("index", None)
        if indices is None:
            raise ValueError(
                "DataCollatorForSupervisedDataset must be configured with index='index' to track sample ids."
            )
        doc_indices = indices.detach().cpu().tolist()

        # Compute log probabilities
        logprob_results = compute_token_logprobs(
            model,
            tokenizer,
            input_ids,
            attention_mask,
        )

        question_key = getattr(dataset, "question_key", "question")
        answer_key = getattr(dataset, "answer_key", "answer")

        # Combine with batch data
        for i, doc_id in enumerate(doc_indices):
            raw_record = {}
            if hasattr(dataset, "data"):
                record_obj = dataset.data[int(doc_id)]
                if isinstance(record_obj, dict):
                    raw_record = record_obj
                else:
                    raw_record = dict(record_obj)

            question = raw_record.get(question_key) if raw_record else None
            answer = raw_record.get(answer_key) if raw_record else None

            result = {
                'doc_id': int(doc_id),
                'question': question,
                'answer': answer,
                'num_tokens': logprob_results[i]['num_tokens'],
                'token_logprobs': logprob_results[i]['token_logprobs'],
                'tokens': logprob_results[i]['tokens']
            }

            results.append(result)
    
    logger.info("="*80)
    logger.info("Saving results...")
    logger.info("="*80)
    
    # Save results
    with open(output_json_path, 'w') as f:
        for result in results:
            json.dump(result, f)
            f.write('\n')
    
    logger.info(f"Results saved to: {output_json_path}")
    logger.info(f"Log file saved to: {log_file}")
    logger.info(f"Config snapshot saved to: {config_path}")
    logger.info(f"Total examples processed: {len(results)}")
    
    logger.info("="*80)
    logger.info("Summary Statistics")
    logger.info("="*80)
    logger.info("Computation completed successfully!")
    logger.info("="*80)


if __name__ == '__main__':
    main()
