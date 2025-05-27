#!/bin/bash

# Dynamically assign a free port
export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

# Define model and split combinations
model_name=phi-1_5
forget_retain_splits=(
    "forget01 retain99"
    #"forget05 retain95"
    #"forget10 retain90"
    )

    # Loop over splits
    for split in "${forget_retain_splits[@]}"; do
	    forget_split=$(echo $split | cut -d' ' -f1)
            retain_split=$(echo $split | cut -d' ' -f2)
	    task_name=tofu_${model_name}_${retain_split}

		echo "Running $task_name..."
		HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file configs/accelerate/default_config.yaml --main_process_port $MASTER_PORT \
		src/train.py\
		--config-name=train.yaml \
		experiment=finetune/tofu/default \
		model=${model_name} \
		task_name=${task_name}\
	        data/datasets@data.train=TOFU_QA_${retain_split} 	
	done

