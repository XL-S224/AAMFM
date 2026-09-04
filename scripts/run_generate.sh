#!/usr/bin/env bash
# Reference-guided antibody generation. Run evaluation separately with
# scripts/run_evaluate.sh.
#
# Usage: bash scripts/run_generate.sh [GPU_ID]
set -euo pipefail

GPU_ID="${1:-0}"

CHECKPOINT="${CHECKPOINT:?set CHECKPOINT to your checkpoint file}"
ANTIGEN_FEATURES="${ANTIGEN_FEATURES:?set ANTIGEN_FEATURES to your antigen-feature file}"
OUT_DIR="${OUT_DIR:-outputs/reference_logic/h3_t0.7}"
CDR_MODE="${CDR_MODE:-h3}"
TEMPERATURE="${TEMPERATURE:-0.7}"
NUM_TARGETS="${NUM_TARGETS:-60}"
SAMPLES_PER_TARGET="${SAMPLES_PER_TARGET:-10}"

mkdir -p "${OUT_DIR}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python src/eval/generate.py \
  --checkpoint "${CHECKPOINT}" \
  --antigen-features "${ANTIGEN_FEATURES}" \
  --cdr-mode "${CDR_MODE}" \
  --temperature "${TEMPERATURE}" \
  --num-targets "${NUM_TARGETS}" \
  --samples-per-target "${SAMPLES_PER_TARGET}" \
  --mini-batch-size 1 \
  --allow-missing-antigen-features \
  --device cuda \
  --output-dir "${OUT_DIR}"
