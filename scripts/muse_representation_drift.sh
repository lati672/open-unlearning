#!/bin/bash

set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python}
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    PYTHON_BIN=/venv/unlearning/bin/python
fi

model=${MODEL:-Llama-2-7b-hf}
data_split=${DATA_SPLIT:-Books}
run_tag=${RUN_TAG:-epochs${NUM_TRAIN_EPOCHS:-10}}

reference_model=${REFERENCE_MODEL:-muse-bench/MUSE-${data_split}_target}
checkpoint_root=${CHECKPOINT_ROOT:-saves/unlearn}
# Use the tokenizer saved during training; the reference model may omit tokenizer files.
tokenizer=${TOKENIZER:-${checkpoint_root}/muse_${model}_${data_split}_RMU_layers5_7_${run_tag}}
output_dir=${OUTPUT_DIR:-saves/analysis/muse_${model}_${data_split}_representation_drift_${run_tag}}

chunks_per_book=${CHUNKS_PER_BOOK:-24}
num_prompts=${NUM_PROMPTS:-0}
max_length=${MAX_LENGTH:-512}
batch_size=${BATCH_SIZE:-2}
seed=${SEED:-0}
pooling=${POOLING:-mean}
reference_device=${REFERENCE_DEVICE:-auto}
comparison_device=${COMPARISON_DEVICE:-auto}
dtype=${DTYPE:-auto}
bootstrap_samples=${BOOTSTRAP_SAMPLES:-2000}
bootstrap_unit=${BOOTSTRAP_UNIT:-book}

model_args=()
for layer_name in all layers5_7 layers15_17 layers25_27; do
    checkpoint="${checkpoint_root}/muse_${model}_${data_split}_RMU_${layer_name}_${run_tag}"
    if [[ ! -d "${checkpoint}" ]]; then
        echo "Missing checkpoint: ${checkpoint}" >&2
        echo "Run scripts/muse_rep.sh first, or override CHECKPOINT_ROOT/RUN_TAG." >&2
        exit 1
    fi
    if [[ "${layer_name}" == "all" ]]; then
        label=default-all-layers
    else
        label=conservative-${layer_name}
    fi
    model_args+=(--model "${label}=${checkpoint}")
done

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1} \
"${PYTHON_BIN}" src/representation_drift.py \
    --reference-model "${reference_model}" \
    --tokenizer "${tokenizer}" \
    "${model_args[@]}" \
    --dataset-path "muse-bench/MUSE-${data_split}" \
    --dataset-name raw \
    --dataset-split retain1 \
    --chunks-per-book "${chunks_per_book}" \
    --num-prompts "${num_prompts}" \
    --max-length "${max_length}" \
    --batch-size "${batch_size}" \
    --seed "${seed}" \
    --pooling "${pooling}" \
    --reference-device "${reference_device}" \
    --comparison-device "${comparison_device}" \
    --dtype "${dtype}" \
    --bootstrap-samples "${bootstrap_samples}" \
    --bootstrap-unit "${bootstrap_unit}" \
    --output-dir "${output_dir}" \
    "$@"
