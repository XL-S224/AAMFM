#!/usr/bin/env bash
# Direct Preference Optimization (DPO) for CDR design, starting from an SFT
# checkpoint. Run from the repository root.
# Config: src/dpo/config/DPO.yaml (set data.train_csv and model.model_path
# before running).
set -euo pipefail

# --- Weights & Biases (optional) -------------------------------------------
# export WANDB_API_KEY=<your-wandb-api-key>
# export WANDB_MODE=offline           # uncomment to disable online logging

# Defaults to the two-GPU example while respecting a caller-provided device list.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
GPU_COUNT=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)

accelerate launch \
    --mixed_precision="bf16" \
    --num_processes="${GPU_COUNT}" \
    --main_process_port=12333 \
    ./src/dpo/training/trainer.py \
    --config ./src/dpo/config/DPO.yaml
