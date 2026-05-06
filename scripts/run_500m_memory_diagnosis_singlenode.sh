#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Run short single-node 500M memory diagnosis cases.
#
# Intended server usage:
#   cd /data/disk1/SpeciesLLM
#   bash scripts/run_500m_memory_diagnosis_singlenode.sh
#
# Defaults deliberately run only the first 10 parquet files and a few local
# batches per rank. Override any variable below from the shell if needed.
###############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"

timestamp="$(date '+%Y%m%d_%H%M%S')"

WORKDIR="${WORKDIR:-$PROJECT_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-/data/miniconda3/bin/python}"
STAGE2_ROOT="${STAGE2_ROOT:-/data/disk1/SpeciesLLM_obs/Stage2_SpeciesLLMData}"
DATA_PATH="${DATA_PATH:-${STAGE2_ROOT}/all_flatten_data_test_500m}"
EMB_PATH="${EMB_PATH:-${WORKDIR}/Stage2_macrogene_embeddings}"
MODEL_CONFIG_JSON="${MODEL_CONFIG_JSON:-${EMB_PATH}/args_2nd_run.json}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-${WORKDIR}/training_output/memory_diagnosis_500m_${timestamp}}"
RUNNER_LOG="${RUNNER_LOG:-${EXPERIMENT_ROOT}/runner.log}"

NUM_OF_USED_DATA="${NUM_OF_USED_DATA:-10}"
NPROC_PER_NODE="${NPROC_PER_NODE:-5}"
BATCH_SIZE="${BATCH_SIZE:-16}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
EPOCH="${EPOCH:-1}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-3}"
LOG_INTERVAL="${LOG_INTERVAL:-1}"
PROFILE_INTERVAL="${PROFILE_INTERVAL:-1}"
MEMORY_LOG_INTERVAL="${MEMORY_LOG_INTERVAL:-1}"
TENSOR_SHAPE_LOG_INTERVAL="${TENSOR_SHAPE_LOG_INTERVAL:-1}"
METRICS_FLUSH_INTERVAL="${METRICS_FLUSH_INTERVAL:-1}"
SAVE_DATA_INTERVAL="${SAVE_DATA_INTERVAL:-999999999}"
SKIP_FINAL_SAVE="${SKIP_FINAL_SAVE:-true}"
DDP_FIND_UNUSED_PARAMETERS="${DDP_FIND_UNUSED_PARAMETERS:-false}"
VERIFY_NPU="${VERIFY_NPU:-1}"
DRY_RUN="${DRY_RUN:-0}"

if [[ -z "${ASCEND_RT_VISIBLE_DEVICES_VALUE:-}" ]]; then
  device_ids=()
  for ((i = 0; i < NPROC_PER_NODE; i++)); do
    device_ids+=("$i")
  done
  ASCEND_RT_VISIBLE_DEVICES_VALUE="$(IFS=,; printf '%s' "${device_ids[*]}")"
fi

# Comma-separated case names. Use RUNS=00_baseline_current,01_attention_sdpa
# to run a subset.
RUNS="${RUNS:-all}"
RUN_BF16="${RUN_BF16:-0}"

mkdir -p "$EXPERIMENT_ROOT"
touch "$RUNNER_LOG"
exec > >(tee -a "$RUNNER_LOG") 2>&1

log() {
  echo "[$(date '+%F %T')] $*"
}

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

should_run() {
  local name="$1"
  [[ "$RUNS" == "all" ]] && return 0
  case ",${RUNS}," in
    *",${name},"*) return 0 ;;
    *) return 1 ;;
  esac
}

capture_npu_smi() {
  local output="$1"
  if command -v npu-smi >/dev/null 2>&1; then
    npu-smi info > "$output" 2>&1 || true
  else
    printf "npu-smi not found\n" > "$output"
  fi
}

collect_case_logs() {
  local out_dir="$1"
  find "$out_dir" -maxdepth 1 -type f -name 'log.*.txt' -print0 \
    | xargs -0 grep -H -E 'MEMORY|TENSOR_SHAPES|experiment_controls|model_config|number of parameters|runtime_config_overrides' \
    > "${out_dir}/memory_events.log" 2>/dev/null || true
}

write_record_header() {
  local record="$1"
  cat > "$record" <<EOF
# 500M NPU Memory Diagnosis Record

Created: $(date '+%F %T')
Workdir: ${WORKDIR}
Data path: ${DATA_PATH}
Embedding path: ${EMB_PATH}
Model config: ${MODEL_CONFIG_JSON}
First parquet files requested: ${NUM_OF_USED_DATA}
NPROC_PER_NODE: ${NPROC_PER_NODE}
ASCEND_RT_VISIBLE_DEVICES_VALUE: ${ASCEND_RT_VISIBLE_DEVICES_VALUE}
Batch size per rank: ${BATCH_SIZE}
Max local batches per rank: ${MAX_TRAIN_STEPS}

The training script logs torch-npu memory with lines starting with \`MEMORY\`.
For each case, inspect:

- \`after_model_to\`: model parameter storage.
- \`after_static_gene_inputs\`: static macrogene embedding buffers.
- \`after_optimizer_init\`: optimizer object before Adam states are materialized.
- \`after_ddp_wrap\`: DDP buckets/communication setup.
- \`after_forward\`: activation and attention forward footprint.
- \`after_backward\`: gradients plus saved backward state.
- \`after_optimizer_step\`: Adam state materialized after the first step.

Case intent:

| Case | Factor isolated |
|---|---|
| 00_baseline_current | Current behavior: eager attention, no checkpointing, MVC on, zero-prob on, static fp32 |
| 01_attention_sdpa | Attention implementation only |
| 02_gradient_checkpointing | Activation checkpointing only |
| 03_no_mvc_forward | Skip MVC forward/loss branch while keeping module params |
| 04_no_mvc_module | Remove MVC module params and skip MVC forward/loss |
| 05_no_zero_prob | Disable explicit zero-prob heads/losses |
| 06_static_gene_amp_dtype | Store static gene buffers in autocast dtype |
| 07_sdpa_checkpoint | Combined attention + checkpointing best-case probe |
| 08_batch8_baseline | Activation scaling check with half batch size |
| 09_ffn_chunk128 | Feed-forward chunking activation probe |
| 10_amp_bfloat16 | Optional bf16 autocast probe when RUN_BF16=1 |

## Case Results

EOF
}

append_case_record() {
  local record="$1"
  local name="$2"
  local status="$3"
  local out_dir="$4"
  cat >> "$record" <<EOF
### ${name}

Exit status: ${status}
Output dir: ${out_dir}
Console log: ${out_dir}/console.log
Memory events: ${out_dir}/memory_events.log
NPU before: ${out_dir}/npu_smi_before.txt
NPU after: ${out_dir}/npu_smi_after.txt

EOF
}

run_case() {
  local name="$1"
  local description="$2"
  shift 2

  if ! should_run "$name"; then
    log "skip case ${name}"
    return 0
  fi

  local out_dir="${EXPERIMENT_ROOT}/${name}"
  mkdir -p "$out_dir"
  log "run case ${name}: ${description}"

  {
    echo "name=${name}"
    echo "description=${description}"
    printf 'overrides='
    printf '%q ' "$@"
    printf '\n'
  } > "${out_dir}/case.env"

  capture_npu_smi "${out_dir}/npu_smi_before.txt"

  set +e
  (
    export WORKDIR PYTHON_BIN DATA_PATH EMB_PATH MODEL_CONFIG_JSON
    export NUM_OF_USED_DATA NPROC_PER_NODE BATCH_SIZE GRADIENT_ACCUMULATION_STEPS
    export EPOCH MAX_TRAIN_STEPS LOG_INTERVAL PROFILE_INTERVAL MEMORY_LOG_INTERVAL
    export TENSOR_SHAPE_LOG_INTERVAL METRICS_FLUSH_INTERVAL SAVE_DATA_INTERVAL
    export SKIP_FINAL_SAVE DDP_FIND_UNUSED_PARAMETERS VERIFY_NPU DRY_RUN ASCEND_RT_VISIBLE_DEVICES_VALUE
    export OUT_PATH="$out_dir"
    export LOG_ALL_RANKS=false
    export COMPILE=false
    export "$@"
    bash "${SCRIPT_DIR}/launch_singlenode_torchrun.sh"
  ) 2>&1 | tee "${out_dir}/console.log"
  local status="${PIPESTATUS[0]}"
  set -e

  capture_npu_smi "${out_dir}/npu_smi_after.txt"
  collect_case_logs "$out_dir"
  printf '%s\t%s\t%s\n' "$name" "$status" "$out_dir" >> "${EXPERIMENT_ROOT}/case_status.tsv"
  append_case_record "${EXPERIMENT_ROOT}/analysis_record.md" "$name" "$status" "$out_dir"

  if [[ "$status" -ne 0 ]]; then
    log "case ${name} failed with status ${status}; continuing"
  fi
}

log "runner log: ${RUNNER_LOG}"
log "preflight WORKDIR=${WORKDIR}"
log "preflight DATA_PATH=${DATA_PATH}"
log "preflight EMB_PATH=${EMB_PATH}"
log "preflight MODEL_CONFIG_JSON=${MODEL_CONFIG_JSON}"
log "preflight NPROC_PER_NODE=${NPROC_PER_NODE} ASCEND_RT_VISIBLE_DEVICES_VALUE=${ASCEND_RT_VISIBLE_DEVICES_VALUE}"

[[ -d "$WORKDIR" ]] || die "Missing WORKDIR: ${WORKDIR}"
[[ -f "${WORKDIR}/train_MNodes_torchrun_mfu_preindexparquet.py" ]] || die "Missing training entry under ${WORKDIR}"
[[ -d "$DATA_PATH" ]] || die "Missing DATA_PATH: ${DATA_PATH}"
[[ -d "$EMB_PATH" ]] || die "Missing EMB_PATH: ${EMB_PATH}"
[[ -f "$MODEL_CONFIG_JSON" ]] || die "Missing MODEL_CONFIG_JSON: ${MODEL_CONFIG_JSON}"

all_files_path="${EXPERIMENT_ROOT}/all_parquet_files.txt"
selected_files_path="${EXPERIMENT_ROOT}/selected_first_${NUM_OF_USED_DATA}_files.txt"
find "$DATA_PATH" -maxdepth 1 -type f -name '*.parquet' | sort > "$all_files_path"
sed -n "1,${NUM_OF_USED_DATA}p" "$all_files_path" > "$selected_files_path"

selected_count="$(wc -l < "$selected_files_path" | tr -d '[:space:]')"
[[ "$selected_count" -gt 0 ]] || die "No parquet files found under ${DATA_PATH}"
log "available parquet files: $(wc -l < "$all_files_path" | tr -d '[:space:]')"
log "selected parquet files: ${selected_count}"

if (( NUM_OF_USED_DATA % NPROC_PER_NODE != 0 )); then
  log "warning: NUM_OF_USED_DATA=${NUM_OF_USED_DATA} is not divisible by NPROC_PER_NODE=${NPROC_PER_NODE}; current sampler may drop or pad files"
fi

printf 'case\tstatus\tout_dir\n' > "${EXPERIMENT_ROOT}/case_status.tsv"
write_record_header "${EXPERIMENT_ROOT}/analysis_record.md"

log "experiment root: ${EXPERIMENT_ROOT}"
log "selected files listed in ${selected_files_path}"

run_case 00_baseline_current "current defaults" \
  RUNTIME_ATTN_IMPLEMENTATION= \
  RUNTIME_EXPLICIT_ZERO_PROB= \
  RUNTIME_DO_MVC= \
  TRAIN_MVC=true \
  GRADIENT_CHECKPOINTING=false \
  STATIC_GENE_DTYPE=float32 \
  AMP_DTYPE=auto

run_case 01_attention_sdpa "switch eager attention to sdpa" \
  RUNTIME_ATTN_IMPLEMENTATION=sdpa \
  TRAIN_MVC=true \
  GRADIENT_CHECKPOINTING=false \
  STATIC_GENE_DTYPE=float32 \
  AMP_DTYPE=auto

run_case 02_gradient_checkpointing "enable checkpointing with current eager attention" \
  RUNTIME_ATTN_IMPLEMENTATION= \
  TRAIN_MVC=true \
  GRADIENT_CHECKPOINTING=true \
  STATIC_GENE_DTYPE=float32 \
  AMP_DTYPE=auto

run_case 03_no_mvc_forward "skip MVC forward/loss but keep MVC params" \
  TRAIN_MVC=false \
  DDP_FIND_UNUSED_PARAMETERS=true \
  GRADIENT_CHECKPOINTING=false \
  STATIC_GENE_DTYPE=float32 \
  AMP_DTYPE=auto

run_case 04_no_mvc_module "disable MVC module and skip MVC forward/loss" \
  RUNTIME_DO_MVC=false \
  TRAIN_MVC=false \
  GRADIENT_CHECKPOINTING=false \
  STATIC_GENE_DTYPE=float32 \
  AMP_DTYPE=auto

run_case 05_no_zero_prob "disable explicit zero-prob heads/losses" \
  RUNTIME_EXPLICIT_ZERO_PROB=false \
  TRAIN_MVC=true \
  GRADIENT_CHECKPOINTING=false \
  STATIC_GENE_DTYPE=float32 \
  AMP_DTYPE=auto

run_case 06_static_gene_amp_dtype "store static gene buffers in autocast dtype" \
  TRAIN_MVC=true \
  GRADIENT_CHECKPOINTING=false \
  STATIC_GENE_DTYPE=amp \
  AMP_DTYPE=auto

run_case 07_sdpa_checkpoint "combine sdpa attention and checkpointing" \
  RUNTIME_ATTN_IMPLEMENTATION=sdpa \
  TRAIN_MVC=true \
  GRADIENT_CHECKPOINTING=true \
  STATIC_GENE_DTYPE=float32 \
  AMP_DTYPE=auto

run_case 08_batch8_baseline "baseline with batch_size=8 for activation scaling" \
  BATCH_SIZE=8 \
  TRAIN_MVC=true \
  GRADIENT_CHECKPOINTING=false \
  STATIC_GENE_DTYPE=float32 \
  AMP_DTYPE=auto

run_case 09_ffn_chunk128 "feed-forward chunk_size=128 activation probe" \
  RUNTIME_CHUNK_SIZE_FEED_FORWARD=128 \
  TRAIN_MVC=true \
  GRADIENT_CHECKPOINTING=false \
  STATIC_GENE_DTYPE=float32 \
  AMP_DTYPE=auto

if [[ "$RUN_BF16" == "1" ]]; then
  run_case 10_amp_bfloat16 "optional bf16 autocast probe" \
    TRAIN_MVC=true \
    GRADIENT_CHECKPOINTING=false \
    STATIC_GENE_DTYPE=amp \
    AMP_DTYPE=bfloat16
fi

log "done. Record: ${EXPERIMENT_ROOT}/analysis_record.md"
log "status: ${EXPERIMENT_ROOT}/case_status.tsv"
