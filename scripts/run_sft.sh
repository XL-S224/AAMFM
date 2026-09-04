#!/usr/bin/env bash
# Supervised fine-tuning (SFT) for antigen-conditioned antibody design.
# Run from the repository root. Config: src/sft/config/SFT.yaml
# (set model.model_path in SFT.yaml to your CPT checkpoint).
set -euo pipefail

# --- Weights & Biases (optional) -------------------------------------------
# export WANDB_API_KEY=<your-wandb-api-key>
# export WANDB_MODE=offline           # uncomment to disable online logging
# export WANDB_BASE_URL=https://api.wandb.ai

# Adjust --gpu_ids / --num_processes to your hardware.
accelerate launch --main_process_port 29501 --gpu_ids 0,1,2 --num_processes=3 ./src/sft/training/trainer.py
