#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  bash scripts/run_compute_pll.sh <generation.csv> [cdr] [rabd.json]

Environment overrides:
  CONDA_ENV   Conda environment (default: anti_design)
  GPU_ID      CUDA device ID (default: 0)
  BATCH_SIZE  Masked-position batch size (default: 128)
  OUTPUT_DIR  Output directory (default: input CSV directory)
  RABD_JSON   CDR position metadata used when mode is cdr
EOF
}

if [[ $# -lt 1 || $# -gt 3 ]]; then
  usage
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
GENERATION_CSV="$1"
MODE="${2:-}"
CONDA_ENV="${CONDA_ENV:-anti_design}"
GPU_ID="${GPU_ID:-0}"
BATCH_SIZE="${BATCH_SIZE:-128}"
OUTPUT_DIR="${OUTPUT_DIR:-$(dirname "${GENERATION_CSV}")}"
RABD_JSON="${3:-${RABD_JSON:-${PROJECT_ROOT}/datasets/eval/rabd/rabd.json}}"
CONDA_COMMAND="${CONDA_EXE:-conda}"

if [[ ! -f "${GENERATION_CSV}" ]]; then
  echo "generation CSV not found: ${GENERATION_CSV}" >&2
  exit 2
fi

args=(
  "${PROJECT_ROOT}/src/eval/compute_pll.py"
  "${GENERATION_CSV}"
  --device cuda
  --batch-size "${BATCH_SIZE}"
  --output-dir "${OUTPUT_DIR}"
)

if [[ -z "${MODE}" ]]; then
  :
elif [[ "${MODE}" == "cdr" ]]; then
  if [[ ! -f "${RABD_JSON}" ]]; then
    echo "RAbD JSON not found: ${RABD_JSON}" >&2
    exit 2
  fi
  args+=(--mode cdr --rabd-json "${RABD_JSON}")
else
  echo "unsupported mode: ${MODE}; omit it for antibody PLL or use cdr" >&2
  usage
  exit 2
fi

mkdir -p "${OUTPUT_DIR}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-recompute-pll}"

"${CONDA_COMMAND}" run --no-capture-output -n "${CONDA_ENV}" \
  python "${args[@]}"
