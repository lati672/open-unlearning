#!/bin/bash

set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python}
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    PYTHON_BIN=/venv/unlearning/bin/python
fi

ACCELERATE_BIN=${ACCELERATE_BIN:-accelerate}
if ! command -v "${ACCELERATE_BIN}" >/dev/null 2>&1; then
    ACCELERATE_BIN=/venv/unlearning/bin/accelerate
fi

export MASTER_PORT=$(${PYTHON_BIN} -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: ${MASTER_PORT}"

model=${MODEL:-Llama-2-7b-hf}
data_split=${DATA_SPLIT:-Books}
run_eval=${RUN_EVAL:-1}

if [[ "${data_split}" != "Books" ]]; then
    echo "Unsupported MUSE data split: ${data_split}. This launcher is for MUSE Books." >&2
    exit 1
fi

per_device_train_batch_size=${PER_DEVICE_TRAIN_BATCH_SIZE:-2}
gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS:-4}
learning_rate=${LEARNING_RATE:-5e-6}

use_flash_attention=${USE_FLASH_ATTENTION:-1}
flash_attn_override=()
if [[ "${use_flash_attention}" != "0" ]]; then
    if "${PYTHON_BIN}" -c "import flash_attn" >/dev/null 2>&1; then
        flash_attn_override+=("model.model_args.attn_implementation=flash_attention_2")
    else
        echo "flash_attn is not installed; using the model config attention implementation."
    fi
fi

# Both methods use the NPO trainer. Adaptive NPO additionally loads the
# precomputed reference log probabilities and enables adaptive token masking.
trainer_entries=(
    "NPO|"
    "AdaptiveNPO|collator=DataCollatorWithLogProbs trainer.method_args.mask=adaptive"
)

for entry in "${trainer_entries[@]}"; do
    IFS='|' read -r variant extra_overrides <<< "${entry}"
    task_name=muse_${model}_${data_split}_${variant}${RUN_TAG:+_${RUN_TAG}}

    if [[ "${variant}" == "AdaptiveNPO" ]]; then
        logprob_path="saves/logprobs/MUSE_${data_split}_forget_${model}/logprobs.json"
        if [[ ! -f "${logprob_path}" ]]; then
            echo "Missing logprobs at ${logprob_path}; run scripts/muse_logprob.sh first." >&2
            exit 1
        fi
    fi

    train_overrides=()
    if [[ -n "${extra_overrides}" ]]; then
        read -r -a train_overrides <<< "${extra_overrides}"
    fi

    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1} "${ACCELERATE_BIN}" launch \
        --config_file configs/accelerate/default_config.yaml \
        --main_process_port "${MASTER_PORT}" \
        src/train.py --config-name=unlearn.yaml \
        experiment=unlearn/muse/default.yaml \
        model="${model}" \
        "${flash_attn_override[@]}" \
        data_split="${data_split}" \
        trainer=NPO \
        task_name="${task_name}" \
        retain_logs_path="saves/eval/muse_${model}_${data_split}_retrain/MUSE_EVAL.json" \
        trainer.args.per_device_train_batch_size="${per_device_train_batch_size}" \
        trainer.args.gradient_accumulation_steps="${gradient_accumulation_steps}" \
        trainer.args.learning_rate="${learning_rate}" \
        trainer.args.ddp_find_unused_parameters=true \
        trainer.args.gradient_checkpointing=true \
        "${train_overrides[@]}"

    if [[ "${run_eval}" == "1" ]]; then
        CUDA_VISIBLE_DEVICES=${EVAL_CUDA_VISIBLE_DEVICES:-0} "${PYTHON_BIN}" src/eval.py \
            experiment=eval/muse/default.yaml \
            data_split="${data_split}" \
            task_name="${task_name}" \
            model="${model}" \
            model.model_args.pretrained_model_name_or_path="saves/unlearn/${task_name}" \
            paths.output_dir="saves/unlearn/${task_name}/evals" \
            retain_logs_path="saves/eval/muse_${model}_${data_split}_retrain/MUSE_EVAL.json"
    fi
done
