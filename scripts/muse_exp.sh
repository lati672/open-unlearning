#!/bin/bash

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"


per_device_train_batch_size=4
gradient_accumulation_steps=4


model=Llama-2-7b-hf
#model=Llama-3.2-1B-Instruct

# Allow disabling flash attention override (e.g. for pure CPU eval runs)
use_flash_attention=${USE_FLASH_ATTENTION:-1}
flash_attn_override=()
if [[ "$use_flash_attention" != "0" ]]; then
    flash_attn_override+=("model.model_args.attn_implementation=flash_attention_2")
fi

data_splits=(
    "News"
    "Books"
)

trainers=(
    "GradAscent"
    "GradDiff"
    "NPO"
    "RMU"
)

# #########################################################
# #################### MUSE Unlearning ####################
# #########################################################


for data_split in "${data_splits[@]}"; do
    for trainer in "${trainers[@]}"; do

        task_name=muse_${model}_${data_split}_${trainer}

        CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file configs/accelerate/default_config.yaml --main_process_port $MASTER_PORT \
        src/train.py --config-name=unlearn.yaml \
        experiment=unlearn/muse/default.yaml \
        model=${model} \
        "${flash_attn_override[@]}" \
        data_split=${data_split} \
        trainer=${trainer} \
        task_name=${task_name} \
        retain_logs_path=saves/eval/muse_${model}_${data_split}_retrain/MUSE_EVAL.json \
        trainer.args.per_device_train_batch_size=${per_device_train_batch_size} \
        trainer.args.gradient_accumulation_steps=${gradient_accumulation_steps} \
        trainer.args.ddp_find_unused_parameters=true \
        trainer.args.gradient_checkpointing=true

        CUDA_VISIBLE_DEVICES=0 python src/eval.py \
        experiment=eval/muse/default.yaml \
        data_split=${data_split} \
        task_name=${task_name} \
        model=${model} \
        model.model_args.pretrained_model_name_or_path=saves/unlearn/${task_name} \
        paths.output_dir=saves/unlearn/${task_name}/evals \
        retain_logs_path=saves/eval/muse_${model}_${data_split}_retrain/MUSE_EVAL.json

        find "saves/unlearn/${task_name}" -maxdepth 1 -type f -name "*.safetensors" -print -delete
    done
done



