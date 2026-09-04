#!/usr/bin/env bash
# Continual pre-training (CPT) of the base backbone on antibody structures.
# Run from the repository root. Config: src/cpt/config/CPT.yaml
set -euo pipefail

# --- Weights & Biases (optional) -------------------------------------------
# export WANDB_API_KEY=<your-wandb-api-key>
# export WANDB_MODE=offline           # uncomment to disable online logging
# export WANDB_BASE_URL=https://api.wandb.ai

export TRITON_CACHE_DIR=/tmp/triton_cache

# Adjust --gpu_ids / --num_processes to your hardware.
accelerate launch --gpu_ids 0,1,2,3 --num_processes=4 ./src/cpt/training/trainer.py
