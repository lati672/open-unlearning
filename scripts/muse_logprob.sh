#!/bin/bash

set -euo pipefail

# Compute log probabilities for the MUSE News and Books splits using Llama-2-7b-hf.

MODEL=${MODEL:-Llama-2-7b-hf}
DATA_SPLITS=("News" "Books")
FORGET_SPLIT=${FORGET_SPLIT:-forget}
RETAIN_SPLIT=${RETAIN_SPLIT:-retain1}
BATCH_SIZE=${BATCH_SIZE:-4}

declare -a DATASETS=(
  "MUSE_forget:${FORGET_SPLIT}"
  "MUSE_retain:${RETAIN_SPLIT}"
)

for muse_split in "${DATA_SPLITS[@]}"; do
    dataset_path="muse-bench/MUSE-${muse_split}"

    for dataset_cfg in "${DATASETS[@]}"; do
        IFS=":" read -r dataset_key split_name <<< "${dataset_cfg}"
        echo "Computing logprobs for ${dataset_key} (${split_name}) on ${muse_split} with ${MODEL}"

        python src/compute_logprobs.py \
            model="${MODEL}" \
            batch_size="${BATCH_SIZE}" \
            dataset="${dataset_key}" \
            dataset_split="${split_name}" \
            dataset."${dataset_key}".args.hf_args.path="${dataset_path}" \
            dataset."${dataset_key}".args.hf_args.split="${split_name}" \
            dataset."${dataset_key}".args.hf_args.name=raw
    done
done
