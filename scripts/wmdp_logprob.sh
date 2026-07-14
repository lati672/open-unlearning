#!/bin/bash

set -euo pipefail

# Compute base-model token log probabilities for the WMDP forget corpora.
# These files are consumed by AdaptiveNPO and AdaptiveRMU in wmdp_exp.sh.

MODEL=${MODEL:-Llama-2-7b-hf}
BATCH_SIZE=${BATCH_SIZE:-4}

data_splits=(
    "cyber"
    "bio"
)

for data_split in "${data_splits[@]}"; do
    data_file="data/wmdp/wmdp-corpora/${data_split}-forget-corpus.jsonl"
    if [[ ! -f "${data_file}" ]]; then
        echo "Missing WMDP corpus: ${data_file}" >&2
        exit 1
    fi

    echo "Computing WMDP ${data_split} forget logprobs with ${MODEL}"

    python src/compute_logprobs.py \
        model="${MODEL}" \
        batch_size="${BATCH_SIZE}" \
        data/datasets@dataset=WMDP_forget \
        dataset.WMDP_forget.args.hf_args.data_files="${data_file}" \
        dataset.WMDP_forget.args.hf_args.name="${data_split}"
done
