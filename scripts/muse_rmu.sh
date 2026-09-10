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

export MASTER_PORT=$("${PYTHON_BIN}" -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: ${MASTER_PORT}"

model=${MODEL:-Llama-2-7b-hf}
run_eval=${RUN_EVAL:-1}
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

if [[ -n "${NUM_TRAIN_EPOCHS:-}" ]]; then
    epoch_options=("${NUM_TRAIN_EPOCHS}")
else
    epoch_options=(${EPOCH_OPTIONS:-10})
fi

per_device_train_batch_size=${PER_DEVICE_TRAIN_BATCH_SIZE:-2}
gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS:-4}

# MUSE settings. Lower forget metrics are better, privleak should move
# toward 0, and retain should stay high. Keep updates restricted to layers 5/6/7.
learning_rate=${LEARNING_RATE:-1e-5}
retain_alpha=${RETAIN_ALPHA:-3}
steering_coeff=${STEERING_COEFF:-1}
module_regex=${MODULE_REGEX:-model\\.layers\\.7}
trainable_layers_regex=${TRAINABLE_LAYERS_REGEX:-model\\.layers\\.(5|6|7)\\..*}

use_flash_attention=${USE_FLASH_ATTENTION:-1}
flash_attn_override=()
if [[ "${use_flash_attention}" != "0" ]]; then
    if "${PYTHON_BIN}" -c "import flash_attn" >/dev/null 2>&1; then
        flash_attn_override+=("model.model_args.attn_implementation=flash_attention_2")
    else
        echo "flash_attn is not installed; using the model config attention implementation."
    fi
fi

# trainer entries: variant|trainer|extra_overrides
trainer_entries=(
    "RMU|RMU|"
    "AdaptiveRMU|AdaptiveRMU|collator=DataCollatorWithLogProbs"
)

for data_split in "${data_splits[@]}"; do
    for num_train_epochs in "${epoch_options[@]}"; do
        run_tag=${RUN_TAG:-epochs${num_train_epochs}}

        for entry in "${trainer_entries[@]}"; do
            IFS='|' read -r variant trainer extra_overrides <<< "${entry}"

            task_name=muse_${model}_${data_split}_${variant}_${run_tag}

            if [[ "${trainer}" == "AdaptiveRMU" ]]; then
                logprob_path="saves/logprobs/MUSE_${data_split}_forget_${model}/logprobs.json"
                if [[ ! -f "${logprob_path}" ]]; then
                    echo "Missing logprobs at ${logprob_path}; run scripts/muse_logprob.sh first."
                    exit 1
                fi
            fi

            CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1} "${ACCELERATE_BIN}" launch \
                --config_file configs/accelerate/default_config.yaml \
                --main_process_port "${MASTER_PORT}" \
                src/train.py --config-name=unlearn.yaml \
                experiment=unlearn/muse/default.yaml \
                model="${model}" \
                "${flash_attn_override[@]}" \
                data_split="${data_split}" \
                trainer="${trainer}" \
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
                "trainer.method_args.trainable_params_regex=['${trainable_layers_regex}']" \
                ${extra_overrides}

            if [[ "${run_eval}" == "1" ]]; then
                CUDA_VISIBLE_DEVICES=${EVAL_CUDA_VISIBLE_DEVICES:-0} "${PYTHON_BIN}" src/eval.py \
                    experiment=eval/muse/default.yaml \
                    data_split="${data_split}" \
                    task_name="${task_name}" \
                    model="${model}" \
                    model.model_args.pretrained_model_name_or_path="saves/unlearn/${task_name}" \
                    paths.output_dir="saves/unlearn/${task_name}/evals" \
                    retain_logs_path="saves/eval/muse_${model}_${data_split}_retrain/MUSE_EVAL.json"

                find "saves/unlearn/${task_name}" -maxdepth 1 -type f \( -name "*.safetensors" -o -name "model.safetensors.index.json" \) -print -delete
            fi
        done
    done
done
