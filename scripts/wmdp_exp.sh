#!/bin/bash

set -euo pipefail

export MASTER_PORT=${MASTER_PORT:-$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")}
echo "Master Port: ${MASTER_PORT}"

MODEL=${MODEL:-Llama-2-7b-hf}
GPU_IDS=${GPU_IDS:-0,1}
EVAL_GPU_ID=${EVAL_GPU_ID:-0}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-1}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-16}

# Allow disabling flash attention, for example with USE_FLASH_ATTENTION=0.
USE_FLASH_ATTENTION=${USE_FLASH_ATTENTION:-1}
flash_attn_override=()
if [[ "${USE_FLASH_ATTENTION}" != "0" ]]; then
    flash_attn_override+=("model.model_args.attn_implementation=flash_attention_2")
fi

data_splits=(
    "cyber"
    "bio"
)

# Entries are variant|trainer|additional Hydra overrides.
trainer_entries=(
    "GradAscent|GradAscent|"
    "GradDiff|GradDiff|"
    "NPO|NPO|"
    "RMU|RMU|"
    "AdaptiveNPO|NPO|collator=DataCollatorWithLogProbs trainer.method_args.mask=adaptive"
    "AdaptiveRMU|AdaptiveRMU|collator=DataCollatorWithLogProbs"
)

for data_split in "${data_splits[@]}"; do
    forget_file="data/wmdp/wmdp-corpora/${data_split}-forget-corpus.jsonl"
    retain_file="data/wmdp/wmdp-corpora/${data_split}-retain-corpus.jsonl"

    if [[ ! -f "${forget_file}" || ! -f "${retain_file}" ]]; then
        echo "Missing WMDP ${data_split} corpora; expected ${forget_file} and ${retain_file}." >&2
        exit 1
    fi

    for entry in "${trainer_entries[@]}"; do
        IFS='|' read -r variant trainer extra_overrides <<< "${entry}"
        task_name="wmdp_${MODEL}_${data_split}_${variant}"

        # `name` distinguishes cyber and bio in the generated logprob path. The
        # underlying JSONL datasets still use their normal `train` split.
        common_data_overrides=(
            "data.forget.WMDP_forget.args.hf_args.name=${data_split}"
            "data.retain.WMDP_retain.args.hf_args.name=${data_split}"
        )

        extra_args=()
        if [[ -n "${extra_overrides}" ]]; then
            read -r -a extra_args <<< "${extra_overrides}"
        fi

        CUDA_VISIBLE_DEVICES="${GPU_IDS}" accelerate launch \
            --config_file configs/accelerate/default_config.yaml \
            --main_process_port "${MASTER_PORT}" \
            src/train.py --config-name=unlearn.yaml \
            experiment=unlearn/wmdp/default.yaml \
            model="${MODEL}" \
            "${flash_attn_override[@]}" \
            data_split="${data_split}" \
            trainer="${trainer}" \
            task_name="${task_name}" \
            "${common_data_overrides[@]}" \
            trainer.args.per_device_train_batch_size="${PER_DEVICE_TRAIN_BATCH_SIZE}" \
            trainer.args.gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS}" \
            trainer.args.ddp_find_unused_parameters=true \
            trainer.args.gradient_checkpointing=true \
            "${extra_args[@]}"

        CUDA_VISIBLE_DEVICES="${EVAL_GPU_ID}" python src/eval.py \
            experiment=eval/wmdp/default.yaml \
            data_split="${data_split}" \
            task_name="${task_name}" \
            model="${MODEL}" \
            model.model_args.pretrained_model_name_or_path="saves/unlearn/${task_name}" \
            paths.output_dir="saves/unlearn/${task_name}/evals"

        if [[ "${variant}" == *"RMU"* ]]; then
            find "saves/unlearn/${task_name}" -maxdepth 1 -type f \
                \( -name "*.safetensors" -o -name "model.safetensors.index.json" \) \
                -print -delete
        fi
    done
done
