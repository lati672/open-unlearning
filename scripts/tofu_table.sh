#!/bin/bash

set -euo pipefail

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

models=(
    "Llama-3.2-1B-Instruct"
)

trainer_configs=(
    #"BaseNPO|NPO|unlearn/tofu/default.yaml|DataCollatorForSupervisedDataset|trainer.method_args.mask=null"
    #"BaseRMU|RMU|unlearn/tofu/default.yaml|DataCollatorForSupervisedDataset|"
    #"EntityRMU|EntityRMU|unlearn/tofu/default.yaml|DataCollatorWithEntityMask|"
    #"EntityNPO|NPO|unlearn/tofu/default.yaml|DataCollatorWithEntityMask|trainer.method_args.mask=entity"
    #"AdaptiveRMU|AdaptiveRMU|unlearn/tofu/default.yaml|DataCollatorWithLogProbs|"
    #"AdaptiveNPO|NPO|unlearn/tofu/default.yaml|DataCollatorWithLogProbs|trainer.method_args.mask=adaptive"
    "GradAscent|GradAscent|unlearn/tofu/default.yaml|DataCollatorForSupervisedDataset|"
    "GradDiff|GradDiff|unlearn/tofu/default.yaml|DataCollatorForSupervisedDataset|"
    "DPO|DPO|unlearn/tofu/idk.yaml|DataCollatorForSupervisedDataset|"
)

splits=(
    "forget01 holdout01 retain99"
    "forget05 holdout05 retain95"
    "forget10 holdout10 retain90"
)

per_device_train_batch_size=4
gradient_accumulation_steps=4
epoch_options=(10)

for split in "${splits[@]}"; do
    forget_split=$(echo "$split" | cut -d' ' -f1)
    holdout_split=$(echo "$split" | cut -d' ' -f2)
    retain_split=$(echo "$split" | cut -d' ' -f3)

    for model in "${models[@]}"; do
        for trainer_entry in "${trainer_configs[@]}"; do
            IFS='|' read -r variant trainer experiment collator extra_overrides <<< "${trainer_entry}"

            for num_epochs in "${epoch_options[@]}"; do

                task_name="tofu_${model}_${forget_split}_${variant}_${trainer}_E${num_epochs}"
                model_path="open-unlearning/tofu_${model}_full"

                echo "${task_name}: Unlearning ${model_path} with ${trainer} (${experiment}), variant=${variant}, collator=${collator}, epochs=${num_epochs}, splits ${forget_split}/${holdout_split}/${retain_split}"

                train_overrides=(
                    "experiment=${experiment}"
                    "trainer=${trainer}"
                    "collator=${collator}"
                    "task_name=${task_name}"
                    "model=${model}"
                    "forget_split=${forget_split}"
                    "retain_split=${retain_split}"
                    "model.model_args.pretrained_model_name_or_path=${model_path}"
                    "model.model_args.attn_implementation=flash_attention_2"
                    "retain_logs_path=saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json"
                    "trainer.args.per_device_train_batch_size=${per_device_train_batch_size}"
                    "trainer.args.gradient_accumulation_steps=${gradient_accumulation_steps}"
                    "trainer.args.ddp_find_unused_parameters=true"
                    "trainer.args.gradient_checkpointing=true"
                    "trainer.args.num_train_epochs=${num_epochs}"
                )

                if [[ -n "${extra_overrides}" ]]; then
                    IFS=';' read -ra extra_array <<< "${extra_overrides}"
                    for override in "${extra_array[@]}"; do
                        if [[ -n "${override}" ]]; then
                            train_overrides+=("${override}")
                        fi
                    done
                fi

                CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file configs/accelerate/default_config.yaml --main_process_port "$MASTER_PORT" \
                    src/train.py --config-name=unlearn.yaml \
                    "${train_overrides[@]}"

                CUDA_VISIBLE_DEVICES=0 python src/eval.py \
                    experiment=eval/tofu/default.yaml \
                    forget_split="${forget_split}" \
                    holdout_split="${holdout_split}" \
                    model="${model}" \
                    task_name="${task_name}" \
                    model.model_args.pretrained_model_name_or_path="saves/unlearn/${task_name}" \
                    paths.output_dir="saves/unlearn/${task_name}/evals" \
                    retain_logs_path="saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json"

                # Remove checkpoint weights to save disk after evaluation
                if [[ -d "saves/unlearn/${task_name}" ]]; then
                    find "saves/unlearn/${task_name}" -maxdepth 1 -type f -name "*.safetensors" -print -delete
                fi
            done
        done
    done
done
