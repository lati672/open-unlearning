#!/bin/bash

shset -euo pipefail

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
    "AdaptiveNPO|NPO|unlearn/tofu/default.yaml|DataCollatorWithLogProbs|trainer.method_args.mask=adaptive"
)

splits=(
    "forget01 holdout01 retain99"
    "forget05 holdout05 retain95"
    "forget10 holdout10 retain90"
)

# Hyperparameter grid for AdaptiveNPO (3 beta values x 4 alpha/gamma pairs)
beta_grid=(0.05 0.1 0.2)
alpha_gamma_grid=(
    "alpha=0.5;gamma=0.5"
    "alpha=1.0;gamma=1.0"
    "alpha=2.0;gamma=2.0"
    "alpha=4.0;gamma=4.0"
)

per_device_train_batch_size=4
gradient_accumulation_steps=4
epoch_options=(20)

for split in "${splits[@]}"; do
    forget_split=$(echo "$split" | cut -d' ' -f1)
    holdout_split=$(echo "$split" | cut -d' ' -f2)
    retain_split=$(echo "$split" | cut -d' ' -f3)

    for model in "${models[@]}"; do
        for trainer_entry in "${trainer_configs[@]}"; do
            IFS='|' read -r variant trainer experiment collator extra_overrides <<< "${trainer_entry}"

            hyperparameter_settings=("")
            if [[ "${trainer}" == "NPO" && "${variant}" == "AdaptiveNPO" ]]; then
                hyperparameter_settings=()
                for beta_value in "${beta_grid[@]}"; do
                    for alpha_gamma in "${alpha_gamma_grid[@]}"; do
                        hyperparameter_settings+=("beta=${beta_value};${alpha_gamma}")
                    done
                done
            fi

            for hyperparams in "${hyperparameter_settings[@]}"; do
                beta_value=""
                alpha_value=""
                gamma_value=""
                hp_tag=""

                if [[ -n "${hyperparams}" ]]; then
                    IFS=';' read -r beta_override alpha_override gamma_override <<< "${hyperparams}"
                    beta_value="${beta_override#beta=}"
                    alpha_value="${alpha_override#alpha=}"
                    gamma_value="${gamma_override#gamma=}"

                    beta_tag="${beta_value//./p}"
                    alpha_tag="${alpha_value//./p}"
                    gamma_tag="${gamma_value//./p}"
                    hp_tag="_B${beta_tag}_A${alpha_tag}_G${gamma_tag}"
                fi

                for num_epochs in "${epoch_options[@]}"; do
                    task_name="tofu_${model}_${forget_split}_${variant}_${trainer}${hp_tag}_E${num_epochs}"
                    model_path="open-unlearning/tofu_${model}_full"

                    hyper_desc=""
                    if [[ -n "${beta_value}" ]]; then
                        hyper_desc=", beta=${beta_value}, alpha=${alpha_value}, gamma=${gamma_value}"
                    fi

                    echo "${task_name}: Unlearning ${model_path} with ${trainer} (${experiment}), variant=${variant}, collator=${collator}, epochs=${num_epochs}, splits ${forget_split}/${holdout_split}/${retain_split}${hyper_desc}"

                    train_overrides=(
                        "experiment=${experiment}"
                        "trainer=${trainer}"
                        "collator=${collator}"
                        "task_name=${task_name}"
                        "model=${model}"
                        "forget_split=${forget_split}"
                        "retain_split=${retain_split}"
                        "model.model_args.pretrained_model_name_or_path=${model_path}"
                        "model.model_args.attn_implementation=flashattention2"
                        "retain_logs_path=saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json"
                        "trainer.args.per_device_train_batch_size=${per_device_train_batch_size}"
                        "trainer.args.gradient_accumulation_steps=${gradient_accumulation_steps}"
                        "trainer.args.ddp_find_unused_parameters=true"
                        "trainer.args.gradient_checkpointing=true"
                        "trainer.args.num_train_epochs=${num_epochs}"
                    )

                    if [[ -n "${beta_value}" ]]; then
                        train_overrides+=(
                            "trainer.method_args.beta=${beta_value}"
                            "trainer.method_args.alpha=${alpha_value}"
                            "trainer.method_args.gamma=${gamma_value}"
                        )
                    fi

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
done
