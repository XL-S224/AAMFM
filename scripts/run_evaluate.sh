#!/usr/bin/env bash
# Evaluate a reference-logic generation directory in H3-only mode by default.
# Usage: bash scripts/run_evaluate.sh <designed_dir> [h3only|full] [results_dir]
set -euo pipefail

DESIGNED_DIR="${1:?path to directory with generated PDBs and generation.csv}"
MODE="${2:-h3only}"
RESULTS_DIR="${3:-${DESIGNED_DIR}/result}"

python src/eval/evaluate.py \
  --mode "${MODE}" \
  --designed-dir "${DESIGNED_DIR}" \
  --original-dir datasets/eval/rabd/pdb \
  --usalign-executable tools/USalign \
  --results-dir "${RESULTS_DIR}"
