#!/bin/bash

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

models=(
    #"phi-1_5"
    "Llama-3.2-1B-Instruct"
    #"Llama-3.2-3B-Instruct"
    #"Llama-3.1-8B-Instruct"
)

trainers_experiments=(
    "GradAscent unlearn/tofu/default.yaml"
    "GradDiff unlearn/tofu/default.yaml trainer.method_args.retain_loss_type=KL"
    "GradDiff unlearn/tofu/default.yaml trainer.method_args.retain_loss_type=NLL"
    "NPO unlearn/tofu/default.yaml"
    "DPO unlearn/tofu/idk.yaml"
    "GradDiffRev unlearn/tofu/default.yaml"
    "GradSeqDiff unlearn/tofu/default.yaml"
)

forget_retain_splits=(
    #"forget01 retain99"
    #"forget05 retain95"
    "forget10 retain90"
)

per_device_train_batch_size=32
gradient_accumulation_steps=8


########################################################################################################################
########################################### Unlearn TOFU models ########################################################
########################################################################################################################

for split in "${forget_retain_splits[@]}"; do
    forget_split=$(echo $split | cut -d' ' -f1)
    retain_split=$(echo $split | cut -d' ' -f2)
    for model in "${models[@]}"; do
        for trainer_experiment in "${trainers_experiments[@]}"; do
            IFS=' ' read -r trainer experiment optional_arg <<< "$trainer_experiment"
            
            if [ -n "$optional_arg" ]; then
                task_name=new_tofu_${model}_${forget_split}_${trainer}_$(echo $optional_arg | tr '.=' '__')
            else
                task_name=new_tofu_${model}_${forget_split}_${trainer}
            fi

            model_path=open-unlearning/tofu_${model}_full
            echo "${task_name}: Unlearning ${model_path} using ${trainer} ${optional_arg}"

            args=(
                experiment=${experiment}
                trainer=${trainer}
                task_name=${task_name}
                model=${model}
                forget_split=${forget_split}
                retain_split=${retain_split}
                model.model_args.pretrained_model_name_or_path=${model_path}
                retain_logs_path=saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json
                trainer.args.per_device_train_batch_size=$per_device_train_batch_size
                trainer.args.gradient_accumulation_steps=$gradient_accumulation_steps
                trainer.args.ddp_find_unused_parameters=true
                trainer.args.gradient_checkpointing=true
            )

            if [ -n "$optional_arg" ]; then
                args+=($optional_arg)
            fi

            # Unlearn
            CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file configs/accelerate/default_config.yaml --main_process_port $MASTER_PORT \
            src/train.py --config-name=unlearn.yaml "${args[@]}" \
            > logs/${task_name}_train.log 2>&1

            # Eval
            CUDA_VISIBLE_DEVICES=0 python src/eval.py \
            experiment=eval/tofu/default.yaml \
            forget_split=${forget_split} \
            model=${model} \
            task_name=${task_name} \
            model.model_args.pretrained_model_name_or_path=saves/unlearn/${task_name} \
            paths.output_dir=saves/unlearn/${task_name}/evals \
            retain_logs_path=saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json \
            > logs/${task_name}_eval.log 2>&1

        done
    done
done

