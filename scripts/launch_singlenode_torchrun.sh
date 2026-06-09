#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# SpeciesLLM single-node launcher for the master server.
#
# Run on the master node:
#   cd /data/disk1/SpeciesLLM
#   bash scripts/launch_singlenode_torchrun.sh
#
# This script intentionally does not read .env from the three-node launcher. It always launches
# one local node only:
#   nnodes=1, node_rank=0, nproc_per_node=8
###############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"

# ---------- single-node torchrun configuration ----------
NNODES=1
NODE_RANK=0
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-12345}"

# ---------- paths ----------
WORKDIR="${WORKDIR:-$PROJECT_ROOT}"
TRAIN_ENTRY="${TRAIN_ENTRY:-train_MNodes_torchrun_mfu_preindexparquet.py}"
PYTHON_BIN="${PYTHON_BIN:-/data/miniconda3/bin/python}"
HOME="${HOME:-/root}"
export HOME

STAGE2_ROOT="${STAGE2_ROOT:-/data/disk1/SpeciesLLM_obs/Stage2_SpeciesLLMData}"
TRAIN_DATASET="${TRAIN_DATASET:-test}"
case "$TRAIN_DATASET" in
  full)
    DEFAULT_DATA_PATH="${STAGE2_ROOT}/all_flatten_data"
    ;;
  test)
    DEFAULT_DATA_PATH="${STAGE2_ROOT}/all_flatten_data_test_500m"
    ;;
  *)
    echo "Unsupported TRAIN_DATASET=${TRAIN_DATASET}. Use full or test." >&2
    exit 1
    ;;
esac

DATA_PATH="${DATA_PATH:-$DEFAULT_DATA_PATH}"
NUM_OF_USED_DATA="${NUM_OF_USED_DATA:-0}"
EMB_ROOT="${EMB_ROOT:-$WORKDIR}"
EMB_PATH="${EMB_PATH:-${EMB_ROOT}/Stage2_macrogene_embeddings}"
MODEL_CONFIG_JSON="${MODEL_CONFIG_JSON:-${EMB_PATH}/args_2nd_run.json}"
OUT_PATH="${OUT_PATH:-training_output}"

# ---------- training parameters ----------
# scripts/pretrain_3node.sh uses 3 nodes * 8 ranks, batch_size=16, grad_accum=4.
# On one node, grad_accum=12 preserves the same effective global batch:
# 8 * 16 * 12 == 24 * 16 * 4.
BATCH_SIZE="${BATCH_SIZE:-16}"
EPOCH="${EPOCH:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-12}"
LEARNING_RATE="${LEARNING_RATE:-0.00001}"
MIN_LR="${MIN_LR:-0.000001}"
DECAY_LR="${DECAY_LR:-true}"
WARMUP_ITERS="${WARMUP_ITERS:-2000}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
LR_DECAY_EPOCHS="${LR_DECAY_EPOCHS:-0}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
SAVE_DATA_INTERVAL="${SAVE_DATA_INTERVAL:-5120000}"
BETA1="${BETA1:-0.9}"
BETA2="${BETA2:-0.95}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
# EMA-based adaptive grad-norm clipping + skip-on-spike. Default true preserves
# the training-script default. Set false to fall back to pure static --grad_clip,
# which matches the pre-stability-feature / first-version gradient handling.
ADAPTIVE_GRAD_CLIP="${ADAPTIVE_GRAD_CLIP:-true}"
# Adaptive grad-clip family. Defaults match the training-script argparse
# defaults, so leaving them unset reproduces the prior launcher behavior.
# They take effect only when ADAPTIVE_GRAD_CLIP=true.
GRAD_CLIP_EMA_BETA="${GRAD_CLIP_EMA_BETA:-0.98}"
GRAD_CLIP_RATIO="${GRAD_CLIP_RATIO:-3.0}"
GRAD_SKIP_RATIO="${GRAD_SKIP_RATIO:-10.0}"
GRAD_SKIP_MAX="${GRAD_SKIP_MAX:-0.0}"
GRAD_CLIP_MIN="${GRAD_CLIP_MIN:-0.5}"
GRAD_CLIP_MAX="${GRAD_CLIP_MAX:-1000.0}"
GRAD_CLIP_WARMUP_STEPS="${GRAD_CLIP_WARMUP_STEPS:-200}"
GRAD_CLIP_MAX_CONSECUTIVE_SKIPS="${GRAD_CLIP_MAX_CONSECUTIVE_SKIPS:-50}"
GRAD_CLIP_EMA_RUNAWAY_FACTOR="${GRAD_CLIP_EMA_RUNAWAY_FACTOR:-2.0}"
GRAD_CLIP_HARD_RAW_NORM_LIMIT="${GRAD_CLIP_HARD_RAW_NORM_LIMIT:-0.0}"
COMPILE="${COMPILE:-false}"
BACKEND="${BACKEND:-hccl}"
DEVICE="${DEVICE:-npu}"
DEVICE_TYPE="${DEVICE_TYPE:-npu}"
S3_REMOTE_DIR_PATH="${S3_REMOTE_DIR_PATH:-}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"
PROFILE_INTERVAL="${PROFILE_INTERVAL:-100}"
NAN_CHECK_INTERVAL="${NAN_CHECK_INTERVAL:-0}"
METRICS_FLUSH_INTERVAL="${METRICS_FLUSH_INTERVAL:-100}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
LOG_ALL_RANKS="${LOG_ALL_RANKS:-false}"

NUM_WORKERS="${NUM_WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-1}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-true}"
PIN_MEMORY="${PIN_MEMORY:-true}"

AMP_DTYPE="${AMP_DTYPE:-auto}"
GEP_LOSS="${GEP_LOSS:-mse}"
HUBER_DELTA="${HUBER_DELTA:-1.0}"
STATIC_GENE_DTYPE="${STATIC_GENE_DTYPE:-float32}"
GENE_EMBEDDING_MODALITIES="${GENE_EMBEDDING_MODALITIES:-esm2,gene_desc,dnaseq}"
RUNTIME_ATTN_IMPLEMENTATION="${RUNTIME_ATTN_IMPLEMENTATION:-}"
RUNTIME_EXPLICIT_ZERO_PROB="${RUNTIME_EXPLICIT_ZERO_PROB:-}"
RUNTIME_DO_MVC="${RUNTIME_DO_MVC:-}"
RUNTIME_CHUNK_SIZE_FEED_FORWARD="${RUNTIME_CHUNK_SIZE_FEED_FORWARD:-}"
TRAIN_MVC="${TRAIN_MVC:-true}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"
MEMORY_LOG_INTERVAL="${MEMORY_LOG_INTERVAL:-0}"
TENSOR_SHAPE_LOG_INTERVAL="${TENSOR_SHAPE_LOG_INTERVAL:-0}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-0}"
SKIP_FINAL_SAVE="${SKIP_FINAL_SAVE:-false}"
DDP_FIND_UNUSED_PARAMETERS="${DDP_FIND_UNUSED_PARAMETERS:-false}"

# ---------- Ascend/HCCL environment ----------
ASCEND_RT_VISIBLE_DEVICES_VALUE="${ASCEND_RT_VISIBLE_DEVICES_VALUE:-0,1,2,3,4,5,6,7}"
HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-7200}"
HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-7200}"
HCCL_WHITELIST_DISABLE="${HCCL_WHITELIST_DISABLE:-1}"
ASCEND_TOOLKIT_HOME="${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/ascend-toolkit}"
ASCEND_ENV_SH="${ASCEND_ENV_SH:-}"
ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-$ASCEND_TOOLKIT_HOME}"
VERIFY_NPU="${VERIFY_NPU:-1}"
DRY_RUN="${DRY_RUN:-0}"

log() {
  echo "[$(date '+%F %T')] $*"
}

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

resolve_ascend_env_sh() {
  local candidate
  for candidate in \
    "$ASCEND_ENV_SH" \
    "/usr/local/Ascend/ascend-toolkit/set_env.sh" \
    "${ASCEND_TOOLKIT_HOME}/set_env.sh" \
    "/usr/local/Ascend/ascend-toolkit/latest/set_env.sh" \
    "${ASCEND_TOOLKIT_HOME}/latest/set_env.sh"; do
    if [[ -n "$candidate" && -f "$candidate" ]]; then
      printf "%s" "$candidate"
      return 0
    fi
  done
  return 1
}

source_ascend_env() {
  local env_sh
  env_sh="$(resolve_ascend_env_sh)" || die "Missing Ascend environment script. Set ASCEND_ENV_SH=/path/to/set_env.sh"
  log "source Ascend environment: ${env_sh}"
  set +u
  # shellcheck source=/dev/null
  source "$env_sh"
  set -u
  export ASCEND_ENV_SH="$env_sh"
}

verify_python_npu() {
  [[ "$VERIFY_NPU" == "1" ]] || return 0
  log "verify Python/NPU environment with: ${PYTHON_BIN}"
  "$PYTHON_BIN" - <<'PY'
import torch
import torch_npu

print("torch:", torch.__version__)
print("torch_npu:", torch_npu.__version__)
print("npu available:", torch.npu.is_available())
PY
}

print_command() {
  printf '%q ' "$@"
  printf '\n'
}

cd "$WORKDIR"

[[ -f "$TRAIN_ENTRY" ]] || die "Missing training entry: ${WORKDIR}/${TRAIN_ENTRY}"
[[ -x "$PYTHON_BIN" ]] || die "PYTHON_BIN is not executable: ${PYTHON_BIN}"
[[ -f "$MODEL_CONFIG_JSON" ]] || die "Missing model config JSON: ${MODEL_CONFIG_JSON}"
[[ -d "$DATA_PATH" ]] || die "Missing training data directory: ${DATA_PATH}"
[[ -d "$EMB_PATH" ]] || die "Missing embedding directory: ${EMB_PATH}"
mkdir -p "$OUT_PATH"

export ASCEND_RT_VISIBLE_DEVICES="$ASCEND_RT_VISIBLE_DEVICES_VALUE"
export HCCL_CONNECT_TIMEOUT HCCL_EXEC_TIMEOUT HCCL_WHITELIST_DISABLE
export ASCEND_TOOLKIT_HOME ASCEND_ENV_SH ASCEND_HOME_PATH
export NNODES NPROC_PER_NODE NODE_RANK MASTER_ADDR MASTER_PORT

TRAIN_ARGS=(
  "--data_path=${DATA_PATH}"
  "--num_of_used_data=${NUM_OF_USED_DATA}"
  "--emb_path=${EMB_PATH}"
  "--gene_embedding_modalities=${GENE_EMBEDDING_MODALITIES}"
  "--config_json=${MODEL_CONFIG_JSON}"
  "--out_path=${OUT_PATH}"
  "--batch_size=${BATCH_SIZE}"
  "--epoch=${EPOCH}"
  "--gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS}"
  "--learning_rate=${LEARNING_RATE}"
  "--min_lr=${MIN_LR}"
  "--decay_lr=${DECAY_LR}"
  "--warmup_iters=${WARMUP_ITERS}"
  "--warmup_ratio=${WARMUP_RATIO}"
  "--lr_decay_epochs=${LR_DECAY_EPOCHS}"
  "--weight_decay=${WEIGHT_DECAY}"
  "--save_data_interval=${SAVE_DATA_INTERVAL}"
  "--beta1=${BETA1}"
  "--beta2=${BETA2}"
  "--grad_clip=${GRAD_CLIP}"
  "--adaptive_grad_clip=${ADAPTIVE_GRAD_CLIP}"
  "--grad_clip_ema_beta=${GRAD_CLIP_EMA_BETA}"
  "--grad_clip_ratio=${GRAD_CLIP_RATIO}"
  "--grad_skip_ratio=${GRAD_SKIP_RATIO}"
  "--grad_skip_max=${GRAD_SKIP_MAX}"
  "--grad_clip_min=${GRAD_CLIP_MIN}"
  "--grad_clip_max=${GRAD_CLIP_MAX}"
  "--grad_clip_warmup_steps=${GRAD_CLIP_WARMUP_STEPS}"
  "--grad_clip_max_consecutive_skips=${GRAD_CLIP_MAX_CONSECUTIVE_SKIPS}"
  "--grad_clip_ema_runaway_factor=${GRAD_CLIP_EMA_RUNAWAY_FACTOR}"
  "--grad_clip_hard_raw_norm_limit=${GRAD_CLIP_HARD_RAW_NORM_LIMIT}"
  "--compile=${COMPILE}"
  "--backend=${BACKEND}"
  "--device=${DEVICE}"
  "--device_type=${DEVICE_TYPE}"
  "--log_interval=${LOG_INTERVAL}"
  "--profile_interval=${PROFILE_INTERVAL}"
  "--nan_check_interval=${NAN_CHECK_INTERVAL}"
  "--metrics_flush_interval=${METRICS_FLUSH_INTERVAL}"
  "--log_level=${LOG_LEVEL}"
  "--log_all_ranks=${LOG_ALL_RANKS}"
  "--num_workers=${NUM_WORKERS}"
  "--prefetch_factor=${PREFETCH_FACTOR}"
  "--persistent_workers=${PERSISTENT_WORKERS}"
  "--pin_memory=${PIN_MEMORY}"
  "--amp_dtype=${AMP_DTYPE}"
  "--gep_loss=${GEP_LOSS}"
  "--huber_delta=${HUBER_DELTA}"
  "--static_gene_dtype=${STATIC_GENE_DTYPE}"
  "--train_mvc=${TRAIN_MVC}"
  "--gradient_checkpointing=${GRADIENT_CHECKPOINTING}"
  "--memory_log_interval=${MEMORY_LOG_INTERVAL}"
  "--tensor_shape_log_interval=${TENSOR_SHAPE_LOG_INTERVAL}"
  "--max_train_steps=${MAX_TRAIN_STEPS}"
  "--skip_final_save=${SKIP_FINAL_SAVE}"
  "--ddp_find_unused_parameters=${DDP_FIND_UNUSED_PARAMETERS}"
)

if [[ -n "$S3_REMOTE_DIR_PATH" ]]; then
  TRAIN_ARGS+=("--s3_remote_dir_path=${S3_REMOTE_DIR_PATH}")
fi
if [[ -n "$RUNTIME_ATTN_IMPLEMENTATION" ]]; then
  TRAIN_ARGS+=("--runtime_attn_implementation=${RUNTIME_ATTN_IMPLEMENTATION}")
fi
if [[ -n "$RUNTIME_EXPLICIT_ZERO_PROB" ]]; then
  TRAIN_ARGS+=("--runtime_explicit_zero_prob=${RUNTIME_EXPLICIT_ZERO_PROB}")
fi
if [[ -n "$RUNTIME_DO_MVC" ]]; then
  TRAIN_ARGS+=("--runtime_do_mvc=${RUNTIME_DO_MVC}")
fi
if [[ -n "$RUNTIME_CHUNK_SIZE_FEED_FORWARD" ]]; then
  TRAIN_ARGS+=("--runtime_chunk_size_feed_forward=${RUNTIME_CHUNK_SIZE_FEED_FORWARD}")
fi

CMD=(
  "$PYTHON_BIN"
  -m
  torch.distributed.run
  "--nproc_per_node=${NPROC_PER_NODE}"
  "--nnodes=${NNODES}"
  "--node_rank=${NODE_RANK}"
  "--master_addr=${MASTER_ADDR}"
  "--master_port=${MASTER_PORT}"
  "$TRAIN_ENTRY"
  "${TRAIN_ARGS[@]}"
)

log "single-node training configuration"
log "WORKDIR=${WORKDIR}"
log "DATA_PATH=${DATA_PATH}"
log "EMB_PATH=${EMB_PATH}"
log "MODEL_CONFIG_JSON=${MODEL_CONFIG_JSON}"
log "NNODES=${NNODES}, NODE_RANK=${NODE_RANK}, NPROC_PER_NODE=${NPROC_PER_NODE}, MASTER=${MASTER_ADDR}:${MASTER_PORT}"
log "BATCH_SIZE=${BATCH_SIZE}, GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS}, EPOCH=${EPOCH}"
log "GRAD_CLIP=${GRAD_CLIP}, ADAPTIVE_GRAD_CLIP=${ADAPTIVE_GRAD_CLIP}, COMPILE=${COMPILE}"
log "GRAD_CLIP_RATIO=${GRAD_CLIP_RATIO}, GRAD_SKIP_RATIO=${GRAD_SKIP_RATIO}, GRAD_SKIP_MAX=${GRAD_SKIP_MAX}, HARD_RAW_NORM_LIMIT=${GRAD_CLIP_HARD_RAW_NORM_LIMIT}, LR_DECAY_EPOCHS=${LR_DECAY_EPOCHS}, GEP_LOSS=${GEP_LOSS}, HUBER_DELTA=${HUBER_DELTA}, AMP_DTYPE=${AMP_DTYPE}"
log "EXPERIMENT AMP_DTYPE=${AMP_DTYPE}, STATIC_GENE_DTYPE=${STATIC_GENE_DTYPE}, GENE_EMBEDDING_MODALITIES=${GENE_EMBEDDING_MODALITIES}, RUNTIME_ATTN_IMPLEMENTATION=${RUNTIME_ATTN_IMPLEMENTATION:-config}"
log "EXPERIMENT TRAIN_MVC=${TRAIN_MVC}, RUNTIME_DO_MVC=${RUNTIME_DO_MVC:-config}, RUNTIME_EXPLICIT_ZERO_PROB=${RUNTIME_EXPLICIT_ZERO_PROB:-config}"
log "EXPERIMENT GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING}, MEMORY_LOG_INTERVAL=${MEMORY_LOG_INTERVAL}, MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS}, SKIP_FINAL_SAVE=${SKIP_FINAL_SAVE}, DDP_FIND_UNUSED_PARAMETERS=${DDP_FIND_UNUSED_PARAMETERS}"
log "ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES}"
log "command:"
print_command "${CMD[@]}"

if [[ "$DRY_RUN" == "1" ]]; then
  log "DRY_RUN=1, skip execution."
  exit 0
fi

source_ascend_env
verify_python_npu

exec "${CMD[@]}"
