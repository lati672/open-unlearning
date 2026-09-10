#!/bin/bash

set -euo pipefail

# ============================================================
# Environment
# ============================================================

PYTHON_BIN=${PYTHON_BIN:-python}
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    PYTHON_BIN=/venv/unlearning/bin/python
fi

ACCELERATE_BIN=${ACCELERATE_BIN:-accelerate}
if ! command -v "${ACCELERATE_BIN}" >/dev/null 2>&1; then
    ACCELERATE_BIN=/venv/unlearning/bin/accelerate
fi

export MASTER_PORT=$(
    "${PYTHON_BIN}" -c \
    "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()"
)

echo "Master Port: ${MASTER_PORT}"


# ============================================================
# Basic experiment settings
# ============================================================

model=${MODEL:-Llama-2-7b-hf}
run_eval=${RUN_EVAL:-1}

# Dataset
if [[ -n "${DATA_SPLITS:-}" ]]; then
    read -r -a data_splits <<< "${DATA_SPLITS}"
else
    data_splits=("${DATA_SPLIT:-Books}")
fi

for data_split in "${data_splits[@]}"; do
    case "${data_split}" in
        Books|News) ;;
        *)
            echo "Unsupported MUSE data split: ${data_split}. Expected Books or News." >&2
            exit 1
            ;;
    esac
done


# ============================================================
# Training epochs
#
# Default: 10 epochs
# Can be overridden with:
# NUM_TRAIN_EPOCHS=8 bash script.sh
# ============================================================

num_train_epochs=${NUM_TRAIN_EPOCHS:-10}


# ============================================================
# Shared RMU hyperparameters
#
# IMPORTANT:
# These are kept identical across all layer configurations.
# The only experimental variable is the layer configuration.
# ============================================================

per_device_train_batch_size=${PER_DEVICE_TRAIN_BATCH_SIZE:-2}
gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS:-4}

learning_rate=${LEARNING_RATE:-1e-5}
retain_alpha=${RETAIN_ALPHA:-3}
steering_coeff=${STEERING_COEFF:-1}


# ============================================================
# Layer ablation
#
# Format:
#   name | RMU representation layer | trainable transformer layers
#
# 5-7:
#   representation loss is applied at layer 7
#   layers 5, 6, 7 are trainable
#
# 15-17:
#   representation loss is applied at layer 17
#   layers 15, 16, 17 are trainable
#
# 25-27:
#   representation loss is applied at layer 27
#   layers 25, 26, 27 are trainable
#
# all:
#   broad layer-wise baseline
#   RMU representation target remains at layer 7
# ============================================================

layer_entries=(
    "layers5_7|model\\.layers\\.7|model\\.layers\\.(5|6|7)\\..*"
    "layers15_17|model\\.layers\\.17|model\\.layers\\.(15|16|17)\\..*"
    "layers25_27|model\\.layers\\.27|model\\.layers\\.(25|26|27)\\..*"
    "all|model\\.layers\\.7|model\\.layers\\..*"
)


# ============================================================
# Flash Attention
# ============================================================

use_flash_attention=${USE_FLASH_ATTENTION:-1}
flash_attn_override=()

if [[ "${use_flash_attention}" != "0" ]]; then
    if "${PYTHON_BIN}" -c "import flash_attn" >/dev/null 2>&1; then
        flash_attn_override+=(
            "model.model_args.attn_implementation=flash_attention_2"
        )
    else
        echo "flash_attn is not installed; using the model config attention implementation."
    fi
fi


# ============================================================
# Training
# ============================================================

for data_split in "${data_splits[@]}"; do

    run_tag=${RUN_TAG:-epochs${num_train_epochs}}

    for entry in "${layer_entries[@]}"; do

        IFS='|' read -r \
            layer_name \
            module_regex \
            trainable_layers_regex \
            <<< "${entry}"

        task_name="muse_${model}_${data_split}_RMU_${layer_name}_${run_tag}"

        echo
        echo "============================================================"
        echo "Running RMU layer ablation"
        echo "Dataset:          ${data_split}"
        echo "Epochs:           ${num_train_epochs}"
        echo "Layer setting:    ${layer_name}"
        echo "Module regex:     ${module_regex}"
        echo "Trainable regex:  ${trainable_layers_regex}"
        echo "Learning rate:    ${learning_rate}"
        echo "Task name:        ${task_name}"
        echo "============================================================"
        echo

        CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1} \
        "${ACCELERATE_BIN}" launch \
            --config_file configs/accelerate/default_config.yaml \
            --main_process_port "${MASTER_PORT}" \
            src/train.py \
            --config-name=unlearn.yaml \
            experiment=unlearn/muse/default.yaml \
            model="${model}" \
            "${flash_attn_override[@]}" \
            data_split="${data_split}" \
            trainer=RMU \
            task_name="${task_name}" \
            retain_logs_path="saves/eval/muse_${model}_${data_split}_retrain/MUSE_EVAL.json" \
            trainer.args.per_device_train_batch_size="${per_device_train_batch_size}" \
            trainer.args.gradient_accumulation_steps="${gradient_accumulation_steps}" \
            trainer.args.learning_rate="${learning_rate}" \
            trainer.args.num_train_epochs="${num_train_epochs}" \
            trainer.args.ddp_find_unused_parameters=true \
            trainer.args.gradient_checkpointing=true \
            trainer.method_args.alpha="${retain_alpha}" \
            trainer.method_args.steering_coeff="${steering_coeff}" \
            trainer.method_args.module_regex="${module_regex}" \
            "trainer.method_args.trainable_params_regex=['${trainable_layers_regex}']"

        # ====================================================
        # Evaluation
        # ====================================================

        if [[ "${run_eval}" == "1" ]]; then

            echo
            echo "Evaluating ${task_name} ..."
            echo

            CUDA_VISIBLE_DEVICES=${EVAL_CUDA_VISIBLE_DEVICES:-0} \
            "${PYTHON_BIN}" src/eval.py \
                experiment=eval/muse/default.yaml \
                data_split="${data_split}" \
                task_name="${task_name}" \
                model="${model}" \
                model.model_args.pretrained_model_name_or_path="saves/unlearn/${task_name}" \
                paths.output_dir="saves/unlearn/${task_name}/evals" \
                retain_logs_path="saves/eval/muse_${model}_${data_split}_retrain/MUSE_EVAL.json"
        fi

    done
done
