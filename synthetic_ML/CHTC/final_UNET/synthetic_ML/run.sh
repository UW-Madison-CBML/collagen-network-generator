#!/bin/bash
echo "$HOSTNAME"
# Set environment variables
rm -rf /tmp/huggingface_cache
mkdir -p /tmp/huggingface_cache
export HF_HOME="/tmp/huggingface_home"

export MPLCONFIGDIR="/tmp/matplotlib_cache"
export WANDB_API_KEY="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"
export WANDB_DIR="/tmp/wandb"
export WANDB_CONFIG_DIR="/tmp/wandb"
export WANDB_CACHE_DIR="/tmp/wandb"

CONFIG="config.yml"
SAVE_DIR="/staging/s/svaren/synthetic_ML/results/"
RUN_NAME="ML_test_2k"

python inference.py \
    --config $CONFIG \
    --save_dir $SAVE_DIR \
    --run_name $RUN_NAME \
    --wandb_project UNET \
    --wandb_team svaren-uni \
    --wandb_dir /tmp/wandb \
    --seed 777
