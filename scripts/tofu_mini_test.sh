#!/bin/bash

set -euo pipefail

# Minimal Adaptive RMU run on TOFU forget01.

export MASTER_PORT=$(python - <<'PY'
import socket
s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()
PY
)
echo "Master Port: $MASTER_PORT"

MODEL="Llama-3.2-1B-Instruct"
FORGET_SPLIT="forget01"
HOLDOUT_SPLIT="holdout01"
RETAIN_SPLIT="retain99"
TASK_NAME="tofu_${MODEL}_${FORGET_SPLIT}_AdaptiveRMU_mini"
MODEL_PATH="open-unlearning/tofu_${MODEL}_full"
LOGPROB_PATH="saves/logprobs/TOFU_${FORGET_SPLIT}_${MODEL}/logprobs.json"

# Compute logprobs for the forget01 split if not already present.
if [[ ! -f "${LOGPROB_PATH}" ]]; then
    echo "Computing logprobs at ${LOGPROB_PATH}"
    python src/compute_logprobs.py \
        model="${MODEL}" \
        data/datasets@dataset=TOFU_QA_forget \
        dataset.TOFU_QA_forget.args.hf_args.name="${FORGET_SPLIT}" \
        +dataset_split="${FORGET_SPLIT}" \
        output_dir_base="saves/logprobs"
else
    echo "Logprobs already exist at ${LOGPROB_PATH}, skipping computation."
fi

PER_DEVICE_BATCH=2
GRAD_ACCUM=4
NUM_EPOCHS=3

echo "Running Adaptive RMU on ${MODEL} for ${FORGET_SPLIT}"

CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file configs/accelerate/default_config.yaml --main_process_port "${MASTER_PORT}" \
  src/train.py --config-name=unlearn.yaml \
  experiment=unlearn/tofu/default.yaml \
  trainer=AdaptiveRMU \
  collator=DataCollatorWithLogProbs \
  model="${MODEL}" \
  task_name="${TASK_NAME}" \
  forget_split="${FORGET_SPLIT}" \
  retain_split="${RETAIN_SPLIT}" \
  holdout_split="${HOLDOUT_SPLIT}" \
  model.model_args.pretrained_model_name_or_path="${MODEL_PATH}" \
  retain_logs_path="saves/eval/tofu_${MODEL}_${RETAIN_SPLIT}/TOFU_EVAL.json" \
  trainer.args.per_device_train_batch_size="${PER_DEVICE_BATCH}" \
  trainer.args.gradient_accumulation_steps="${GRAD_ACCUM}" \
  trainer.args.num_train_epochs="${NUM_EPOCHS}" \
  trainer.args.ddp_find_unused_parameters=true \
  trainer.args.gradient_checkpointing=true

# Optional eval using the freshly unlearned model.
CUDA_VISIBLE_DEVICES=0 python src/eval.py \
  experiment=eval/tofu/default.yaml \
  forget_split="${FORGET_SPLIT}" \
  holdout_split="${HOLDOUT_SPLIT}" \
  model="${MODEL}" \
  task_name="${TASK_NAME}" \
  model.model_args.pretrained_model_name_or_path="saves/unlearn/${TASK_NAME}" \
  paths.output_dir="saves/unlearn/${TASK_NAME}/evals" \
  retain_logs_path="saves/eval/tofu_${MODEL}_${RETAIN_SPLIT}/TOFU_EVAL.json"
