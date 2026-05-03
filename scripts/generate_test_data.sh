#!/usr/bin/env bash
set -euo pipefail

# Generate a small training-ready dataset for local smoke tests.
# This wrapper resolves paths from its own location, so it can be run from any
# current working directory.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
BATCH_ROOT="${BATCH_ROOT:-${PROJECT_ROOT}/Stage2_SpeciesLLMData}"
OUTPUT_DIR="${OUTPUT_DIR:-${BATCH_ROOT}/all_shuffled_test}"
FLATTEN_OUTPUT_DIR="${FLATTEN_OUTPUT_DIR:-${BATCH_ROOT}/all_flatten_data_test}"
MODE="${MODE:-copy}"
WORKERS="${WORKERS:-16}"
MANIFEST_NAME="${MANIFEST_NAME:-merge_manifest.csv}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
SKIP_FLATTEN="${SKIP_FLATTEN:-0}"
FLATTEN_WORKERS="${FLATTEN_WORKERS:-${SHUFFLE_WORKERS:-$WORKERS}}"
ROWS_PER_FILE="${ROWS_PER_FILE:-16384}"
SHUFFLE_SEED="${SHUFFLE_SEED:-42}"
FLATTEN_PATTERN="${FLATTEN_PATTERN:-macrogene_*.parquet}"
FLATTEN_MANIFEST_NAME="${FLATTEN_MANIFEST_NAME:-shuffle_manifest.csv}"
FLATTEN_COMPRESSION="${FLATTEN_COMPRESSION:-snappy}"
FLATTEN_OVERWRITE="${FLATTEN_OVERWRITE:-1}"
VALIDATE_ALL_SCHEMAS="${VALIDATE_ALL_SCHEMAS:-0}"
MAX_FLATTEN_FILES="${MAX_FLATTEN_FILES:-0}"

print_command() {
  local label="$1"
  shift
  printf '[INFO] %s:' "$label"
  printf ' %q' "$@"
  printf '\n'
}

run_command() {
  local label="$1"
  shift
  print_command "$label" "$@"
  "$@"
}

merge_cmd=(
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
  merge_cmd+=(--dry-run)
fi

if [[ "$SKIP_EXISTING" == "1" ]]; then
  merge_cmd+=(--skip-existing)
fi

merge_cmd+=("$@")

flatten_cmd=(
  "$PYTHON_BIN"
  "${PROJECT_ROOT}/shuffle_macrogene_rounds_parallel.py"
  --input-dir
  "$OUTPUT_DIR"
  --output-dir
  "$FLATTEN_OUTPUT_DIR"
  --pattern
  "$FLATTEN_PATTERN"
  --rows-per-file
  "$ROWS_PER_FILE"
  --seed
  "$SHUFFLE_SEED"
  --workers
  "$FLATTEN_WORKERS"
  --compression
  "$FLATTEN_COMPRESSION"
  --manifest-name
  "$FLATTEN_MANIFEST_NAME"
)

if [[ "$FLATTEN_OVERWRITE" == "1" ]]; then
  flatten_cmd+=(--overwrite)
fi

if [[ "$VALIDATE_ALL_SCHEMAS" == "1" ]]; then
  flatten_cmd+=(--validate-all-schemas)
fi

if [[ "$MAX_FLATTEN_FILES" != "0" ]]; then
  flatten_cmd+=(--max-files "$MAX_FLATTEN_FILES")
fi

cd "$PROJECT_ROOT"
echo "[INFO] project root: ${PROJECT_ROOT}"
echo "[INFO] merged species dir : ${OUTPUT_DIR}"
echo "[INFO] flatten output dir : ${FLATTEN_OUTPUT_DIR}"

run_command "merge command" "${merge_cmd[@]}"

if [[ "$SKIP_FLATTEN" == "1" ]]; then
  echo "[INFO] SKIP_FLATTEN=1, skip flatten/shuffle step."
  exit 0
fi

if [[ "$DRY_RUN" == "1" ]]; then
  print_command "flatten command (skipped because DRY_RUN=1)" "${flatten_cmd[@]}"
  echo "[INFO] DRY_RUN=1, merge dry-run completed; flatten step was not executed."
  exit 0
fi

run_command "flatten command" "${flatten_cmd[@]}"
