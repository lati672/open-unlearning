#!/bin/bash
set -euo pipefail

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

log() { echo -e "\033[1;34m[INFO]\033[0m $1"; }

models=(
    "Llama-3.2-1B-Instruct"
)

trainers_experiments=(
    "GradAscent unlearn/tofu/default.yaml"
    "GradDiff unlearn/tofu/default.yaml NLL"
    "GradDiff unlearn/tofu/default.yaml KL"
    "NPO unlearn/tofu/default.yaml"
    "DPO unlearn/tofu/idk.yaml"
)

forget_retain_splits=(
    "forget01 retain99"
    "forget05 retain95"
    "forget10 retain90"
)

per_device_train_batch_size=8
gradient_accumulation_steps=4

BASE_DIR=saves
EVAL_DIR=$BASE_DIR/eval
UNLEARN_DIR=$BASE_DIR/unlearn

for split in "${forget_retain_splits[@]}"; do
    forget_split=$(echo $split | cut -d' ' -f1)
    retain_split=$(echo $split | cut -d' ' -f2)
    
    for model in "${models[@]}"; do
        for trainer_experiment in "${trainers_experiments[@]}"; do
            read -r trainer experiment method <<< "$trainer_experiment"
            
            if [ -n "${method:-}" ]; then
                task_name=tofu_${model}_${forget_split}_${trainer}_${method}
            else
                task_name=tofu_${model}_${forget_split}_${trainer}
            fi

            model_path=open-unlearning/tofu_${model}_full
            retain_logs_path=$EVAL_DIR/tofu_${model}_${retain_split}/TOFU_EVAL.json

            # Print important variables
            echo "==== DEBUG INFO ===="
            echo "task_name: $task_name"
            echo "trainer: $trainer"
            echo "experiment: $experiment"
            echo "method: ${method:-None}"
            echo "model: $model"
            echo "forget_split: $forget_split"
            echo "retain_split: $retain_split"
            echo "model_path: $model_path"
            echo "retain_logs_path: $retain_logs_path"
            echo "====================="

            exit 0  # Stop script here after printing
            

#            # Unlearn
#            CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/accelerate/default_config.yaml --num_processes=2 --main_process_port $MASTER_PORT \
#            src/train.py --config-name=unlearn.yaml \
#            experiment=${experiment} \
#            trainer=${trainer} \
#            task_name=${task_name} \
#            model=${model} \
#            forget_split=${forget_split} \
#            retain_split=${retain_split} \
#            model.model_args.pretrained_model_name_or_path=${model_path} \
#            retain_logs_path=${retain_logs_path} \
#            trainer.args.per_device_train_batch_size=$per_device_train_batch_size \
#            trainer.args.gradient_accumulation_steps=$gradient_accumulation_steps \
#            trainer.args.ddp_find_unused_parameters=true \
#            trainer.args.gradient_checkpointing=true \
#            ${method:+trainer.method.args=${method}}

#            # Eval
#            if [ -d "$UNLEARN_DIR/${task_name}" ]; then
#                CUDA_VISIBLE_DEVICES=0 python src/eval.py \
#                experiment=eval/tofu/default.yaml \
#                forget_split=${forget_split} \
#                model=${model} \
#                task_name=${task_name} \
#                model.model_args.pretrained_model_name_or_path=$UNLEARN_DIR/${task_name} \
#                paths.output_dir=$UNLEARN_DIR/${task_name}/evals \
#                retain_logs_path=${retain_logs_path}
#            else
#                log "Skipping eval for ${task_name} — model output not found."
#            fi
        done
    done
done
