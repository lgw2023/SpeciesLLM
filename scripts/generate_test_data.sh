#!/usr/bin/env bash
set -euo pipefail

# Generate a small merged dataset for local smoke tests.
# This wrapper resolves paths from its own location, so it can be run from any
# current working directory.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
BATCH_ROOT="${BATCH_ROOT:-${PROJECT_ROOT}/Stage2_SpeciesLLMData}"
OUTPUT_DIR="${OUTPUT_DIR:-${BATCH_ROOT}/all_shuffled_data_test}"
MODE="${MODE:-copy}"
WORKERS="${WORKERS:-16}"
MANIFEST_NAME="${MANIFEST_NAME:-merge_manifest.csv}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"

cmd=(
  "$PYTHON_BIN"
  "${PROJECT_ROOT}/merge_macrogene_rounds_parallel.py"
  --batch-dirs
  "${BATCH_ROOT}/1st_pretrain_data_preprocessed_step4"
  "${BATCH_ROOT}/2nd_pretrain_data_preprocessed_step4"
  "${BATCH_ROOT}/3scbasecount_pretrain_data_preprocessed_step4"
  --batch-names
  1st
  2nd
  3scbasecount
  --output-dir
  "$OUTPUT_DIR"
  --mode
  "$MODE"
  --workers
  "$WORKERS"
  --test-mode
  --manifest-name
  "$MANIFEST_NAME"
)

if [[ "$DRY_RUN" == "1" ]]; then
  cmd+=(--dry-run)
fi

if [[ "$SKIP_EXISTING" == "1" ]]; then
  cmd+=(--skip-existing)
fi

cmd+=("$@")

cd "$PROJECT_ROOT"
echo "[INFO] project root: ${PROJECT_ROOT}"
echo "[INFO] output dir  : ${OUTPUT_DIR}"
printf '[INFO] command     :'
printf ' %q' "${cmd[@]}"
printf '\n'
exec "${cmd[@]}"
