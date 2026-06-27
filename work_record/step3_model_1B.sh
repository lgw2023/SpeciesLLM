#!/usr/bin/env bash
set -euo pipefail

# 1B formal pretraining launcher for one explicitly specified full-data
# directory. This script does not know or discover experiment-specific datasets.
#
# Usage:
#   # Prepare command files, sync/check paths, and dry-run.
#   bash work_record/step3_model_1B.sh DATA_PATH=/data/disk1/SpeciesLLM_obs/Stage2_SpeciesLLMData/your_flatten_dir
#
#   # Launch one long-running training job.
#   bash work_record/step3_model_1B.sh ACTION=launch EXPERIMENT_NAME=data_1_3_stable DATA_PATH=/data/.../your_flatten_dir
#
#   # Collect/check one finished job. Reuse the same RUN_ID used for launch.
#   bash work_record/step3_model_1B.sh ACTION=collect EXPERIMENT_NAME=data_1_3_stable RUN_ID=20260515_120000 DATA_PATH=/data/.../your_flatten_dir
#
# Notes:
#   - The script re-execs itself with env -i, so exported shell variables are ignored.
#   - Values from .env are loaded before command-line overrides.
#   - Runtime overrides must be passed after the script path as KEY=VALUE arguments.
#   - Do not use KEY=VALUE before the script path; those inherited environment values
#     are intentionally discarded.

if [[ "${1:-}" != "--__speciesllm-step3-1b-clean-env" ]]; then
  exec /usr/bin/env -i \
    HOME="${HOME:-/root}" \
    PATH="/data/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
    LANG=C.UTF-8 \
    bash "$0" --__speciesllm-step3-1b-clean-env "$@"
fi
shift
RUN_RECORD_ARGS=("$@")

cd /data/disk1/SpeciesLLM

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

usage() {
  sed -n '4,22p' "$0" | sed 's/^# \{0,1\}//'
}

is_allowed_cli_var() {
  case "$1" in
    ENV_FILE|ACTION|RUN_ID|EXPERIMENT_NAME|DATA_PATH|OUT_PATH|COMMAND_DIR|LOG_SUBDIR|\
    PROJECT_ROOT|WORKDIR|PYTHON_BIN|PRETRAIN_CHECKS_PY|STAGE2_ROOT|\
    HOSTS|MASTER_ADDR|MASTER_PORT|NNODES|NPROC_PER_NODE|\
    EMB_ROOT|EMB_PATH|MODEL_CONFIG_JSON|GENE_EMBEDDING_MODALITIES|SKIP_DATA_SYNC|\
    NUM_OF_USED_DATA|BATCH_SIZE|EPOCH|GRADIENT_ACCUMULATION_STEPS|\
    GRADIENT_CHECKPOINTING|LEARNING_RATE|MIN_LR|DECAY_LR|WARMUP_ITERS|\
    WARMUP_RATIO|WEIGHT_DECAY|SAVE_DATA_INTERVAL|BETA1|BETA2|GRAD_CLIP|\
    ADAPTIVE_GRAD_CLIP|GRAD_CLIP_EMA_BETA|GRAD_CLIP_RATIO|GRAD_SKIP_RATIO|GRAD_SKIP_MAX|\
    GRAD_CLIP_MIN|GRAD_CLIP_MAX|GRAD_CLIP_WARMUP_STEPS|\
    COMPILE|BACKEND|DEVICE|DEVICE_TYPE|S3_REMOTE_DIR_PATH|LOG_INTERVAL|\
    PROFILE_INTERVAL|NAN_CHECK_INTERVAL|METRICS_FLUSH_INTERVAL|\
    BATCH_METADATA_LOG_INTERVAL|BATCH_METADATA_TOP_K|LOG_LEVEL|\
    LOG_ALL_RANKS|NUM_WORKERS|PREFETCH_FACTOR|PERSISTENT_WORKERS|PIN_MEMORY|PARQUET_CHUNK_FILES|\
    ASCEND_RT_VISIBLE_DEVICES_VALUE|HCCL_CONNECT_TIMEOUT|\
    HCCL_EXEC_TIMEOUT|HCCL_WHITELIST_DISABLE|ASCEND_TOOLKIT_HOME|\
    ASCEND_ENV_SH|ASCEND_HOME_PATH)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

ENV_FILE=".env"
for arg in "$@"; do
  case "$arg" in
    ENV_FILE=*)
      ENV_FILE="${arg#*=}"
      ;;
  esac
done

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "$ENV_FILE"
  set +a
fi

CLI_KEYS=()
for arg in "$@"; do
  case "$arg" in
    -h|--help|help)
      usage
      exit 0
      ;;
    *=*)
      key="${arg%%=*}"
      value="${arg#*=}"
      [[ "$key" =~ ^[A-Z_][A-Z0-9_]*$ ]] || die "Invalid KEY=VALUE argument: $arg"
      is_allowed_cli_var "$key" || die "Unsupported runtime variable: $key"
      printf -v "$key" "%s" "$value"
      CLI_KEYS+=("$key")
      ;;
    *)
      usage >&2
      die "Unsupported argument: $arg. Use KEY=VALUE arguments after the script path."
      ;;
  esac
done

cli_has() {
  local wanted="$1"
  local key
  for key in "${CLI_KEYS[@]}"; do
    [[ "$key" == "$wanted" ]] && return 0
  done
  return 1
}

write_run_record() {
  "$PYTHON_BIN" scripts/write_run_record.py \
    --repo "$WORKDIR" \
    --out-path "$OUT_PATH" \
    --script "$0" \
    --shell bash \
    -- "${RUN_RECORD_ARGS[@]}"
}

RUN_ID_CONFIGURED=0
OUT_PATH_CONFIGURED=0
[[ -n "${RUN_ID+x}" ]] && RUN_ID_CONFIGURED=1
[[ -n "${OUT_PATH+x}" ]] && OUT_PATH_CONFIGURED=1

run_pretrain_3node() {
  [[ -d "$DATA_PATH" ]] || die "Missing DATA_PATH: ${DATA_PATH}"

  local run_training="0"
  local collect_only="0"
  case "$ACTION" in
    prepare)
      run_training="0"
      ;;
    launch)
      run_training="1"
      ;;
    collect)
      collect_only="1"
      ;;
    *)
      die "Unsupported ACTION=${ACTION}. Use prepare, launch, or collect."
      ;;
  esac

  echo "===== ${ACTION}: ${EXPERIMENT_NAME} ====="
  echo "[INFO] DATA_PATH=${DATA_PATH}"
  echo "[INFO] OUT_PATH=${OUT_PATH}"
  echo "[INFO] COMMAND_DIR=${COMMAND_DIR}"
  echo "[INFO] LOG_SUBDIR=${LOG_SUBDIR}"
  echo "[INFO] MODEL_CONFIG_JSON=${MODEL_CONFIG_JSON}"
  echo "[INFO] BATCH_SIZE=${BATCH_SIZE}"
  echo "[INFO] GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS}"

  bash scripts/pretrain_3node.sh \
    ENV_FILE="$ENV_FILE" \
    MODEL_SCALE=1b \
    PREP_ACTION=commands \
    RESET_TEST_OUTPUT=0 \
    SKIP_EXISTING=1 \
    RUN_TRAINING="$run_training" \
    COLLECT_ONLY="$collect_only" \
    SKIP_DATA_SYNC="$SKIP_DATA_SYNC" \
    PYTHON_BIN="$PYTHON_BIN" \
    PROJECT_ROOT="$PROJECT_ROOT" \
    PRETRAIN_CHECKS_PY="$PRETRAIN_CHECKS_PY" \
    STAGE2_ROOT="$STAGE2_ROOT" \
    COMMAND_DIR="$COMMAND_DIR" \
    WORKDIR="$WORKDIR" \
    LOG_SUBDIR="$LOG_SUBDIR" \
    NODE_LOG_DIR="${WORKDIR}/${LOG_SUBDIR}" \
    HOSTS="$HOSTS" \
    MASTER_ADDR="$MASTER_ADDR" \
    MASTER_PORT="$MASTER_PORT" \
    NNODES="$NNODES" \
    NPROC_PER_NODE="$NPROC_PER_NODE" \
    TRAIN_DATASET=full \
    DATA_PATH="$DATA_PATH" \
    NUM_OF_USED_DATA="$NUM_OF_USED_DATA" \
    EMB_ROOT="$EMB_ROOT" \
    EMB_PATH="$EMB_PATH" \
    MODEL_CONFIG_JSON="$MODEL_CONFIG_JSON" \
    GENE_EMBEDDING_MODALITIES="${GENE_EMBEDDING_MODALITIES:-esm2,gene_desc,dnaseq}" \
    OUT_PATH="$OUT_PATH" \
    TRAIN_OUT_DIR="$OUT_PATH" \
    BATCH_SIZE="$BATCH_SIZE" \
    EPOCH="$EPOCH" \
    GRADIENT_ACCUMULATION_STEPS="$GRADIENT_ACCUMULATION_STEPS" \
    GRADIENT_CHECKPOINTING="$GRADIENT_CHECKPOINTING" \
    LEARNING_RATE="$LEARNING_RATE" \
    MIN_LR="$MIN_LR" \
    DECAY_LR="$DECAY_LR" \
    WARMUP_ITERS="$WARMUP_ITERS" \
    WARMUP_RATIO="$WARMUP_RATIO" \
    WEIGHT_DECAY="$WEIGHT_DECAY" \
    SAVE_DATA_INTERVAL="$SAVE_DATA_INTERVAL" \
    BETA1="$BETA1" \
    BETA2="$BETA2" \
    GRAD_CLIP="$GRAD_CLIP" \
    ADAPTIVE_GRAD_CLIP="$ADAPTIVE_GRAD_CLIP" \
    GRAD_CLIP_EMA_BETA="$GRAD_CLIP_EMA_BETA" \
    GRAD_CLIP_RATIO="$GRAD_CLIP_RATIO" \
    GRAD_SKIP_RATIO="$GRAD_SKIP_RATIO" \
    GRAD_SKIP_MAX="$GRAD_SKIP_MAX" \
    GRAD_CLIP_MIN="$GRAD_CLIP_MIN" \
    GRAD_CLIP_MAX="$GRAD_CLIP_MAX" \
    GRAD_CLIP_WARMUP_STEPS="$GRAD_CLIP_WARMUP_STEPS" \
    COMPILE="$COMPILE" \
    BACKEND="$BACKEND" \
    DEVICE="$DEVICE" \
    DEVICE_TYPE="$DEVICE_TYPE" \
    S3_REMOTE_DIR_PATH="$S3_REMOTE_DIR_PATH" \
    LOG_INTERVAL="$LOG_INTERVAL" \
    PROFILE_INTERVAL="$PROFILE_INTERVAL" \
    NAN_CHECK_INTERVAL="$NAN_CHECK_INTERVAL" \
    METRICS_FLUSH_INTERVAL="$METRICS_FLUSH_INTERVAL" \
    BATCH_METADATA_LOG_INTERVAL="$BATCH_METADATA_LOG_INTERVAL" \
    BATCH_METADATA_TOP_K="$BATCH_METADATA_TOP_K" \
    LOG_LEVEL="$LOG_LEVEL" \
    LOG_ALL_RANKS="$LOG_ALL_RANKS" \
    NUM_WORKERS="$NUM_WORKERS" \
    PREFETCH_FACTOR="$PREFETCH_FACTOR" \
    PERSISTENT_WORKERS="$PERSISTENT_WORKERS" \
    PIN_MEMORY="$PIN_MEMORY" \
    PARQUET_CHUNK_FILES="$PARQUET_CHUNK_FILES" \
    ASCEND_RT_VISIBLE_DEVICES_VALUE="$ASCEND_RT_VISIBLE_DEVICES_VALUE" \
    HCCL_CONNECT_TIMEOUT="$HCCL_CONNECT_TIMEOUT" \
    HCCL_EXEC_TIMEOUT="$HCCL_EXEC_TIMEOUT" \
    HCCL_WHITELIST_DISABLE="$HCCL_WHITELIST_DISABLE" \
    ASCEND_TOOLKIT_HOME="$ASCEND_TOOLKIT_HOME" \
    ASCEND_ENV_SH="$ASCEND_ENV_SH" \
    ASCEND_HOME_PATH="$ASCEND_HOME_PATH"
}

PROJECT_ROOT=${PROJECT_ROOT:-/data/disk1/SpeciesLLM}
WORKDIR=${WORKDIR:-/data/disk1/SpeciesLLM}
PYTHON_BIN=${PYTHON_BIN:-/data/miniconda3/bin/python}
PRETRAIN_CHECKS_PY=${PRETRAIN_CHECKS_PY:-${PROJECT_ROOT}/scripts/pretrain_checks.py}
STAGE2_ROOT=${STAGE2_ROOT:-/data/disk1/SpeciesLLM_obs/Stage2_SpeciesLLMData}

HOSTS=${HOSTS:-7.150.12.45,7.150.15.14,7.150.14.170}
MASTER_ADDR=${MASTER_ADDR:-${HOSTS%%,*}}
MASTER_PORT=${MASTER_PORT:-12345}
NNODES=${NNODES:-3}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}

EMB_ROOT=${EMB_ROOT:-/data/disk1/SpeciesLLM}
EMB_PATH=${EMB_PATH:-${EMB_ROOT}/Stage2_macrogene_embeddings}
MODEL_CONFIG_JSON=${MODEL_CONFIG_JSON:-${EMB_PATH}/args_2nd_run_1b.json}

RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
ACTION=${ACTION:-prepare}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-formal}

# step1_data_1_2_3.sh and step1_data_1_3.sh already sync the generated flatten
# directories to worker nodes by default. Keep this at 1 unless you explicitly
# need step3 to rsync the full training data again.
SKIP_DATA_SYNC=${SKIP_DATA_SYNC:-1}

NUM_OF_USED_DATA=${NUM_OF_USED_DATA:-0}
BATCH_SIZE=${BATCH_SIZE:-96}
EPOCH=${EPOCH:-1}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-true}
LEARNING_RATE=${LEARNING_RATE:-0.00001}
MIN_LR=${MIN_LR:-0.000001}
DECAY_LR=${DECAY_LR:-true}
WARMUP_ITERS=${WARMUP_ITERS:-2000}
WARMUP_RATIO=${WARMUP_RATIO:-0.05}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.1}
SAVE_DATA_INTERVAL=${SAVE_DATA_INTERVAL:-5120000}
BETA1=${BETA1:-0.9}
BETA2=${BETA2:-0.95}
GRAD_CLIP=${GRAD_CLIP:-1.0}
ADAPTIVE_GRAD_CLIP=${ADAPTIVE_GRAD_CLIP:-true}
GRAD_CLIP_EMA_BETA=${GRAD_CLIP_EMA_BETA:-0.98}
GRAD_CLIP_RATIO=${GRAD_CLIP_RATIO:-3.0}
GRAD_SKIP_RATIO=${GRAD_SKIP_RATIO:-10.0}
GRAD_SKIP_MAX=${GRAD_SKIP_MAX:-0.0}
GRAD_CLIP_MIN=${GRAD_CLIP_MIN:-0.5}
GRAD_CLIP_MAX=${GRAD_CLIP_MAX:-1000.0}
GRAD_CLIP_WARMUP_STEPS=${GRAD_CLIP_WARMUP_STEPS:-200}
COMPILE=${COMPILE:-false}
BACKEND=${BACKEND:-hccl}
DEVICE=${DEVICE:-npu}
DEVICE_TYPE=${DEVICE_TYPE:-npu}
S3_REMOTE_DIR_PATH=${S3_REMOTE_DIR_PATH:-}
LOG_INTERVAL=${LOG_INTERVAL:-10}
PROFILE_INTERVAL=${PROFILE_INTERVAL:-100}
NAN_CHECK_INTERVAL=${NAN_CHECK_INTERVAL:-0}
METRICS_FLUSH_INTERVAL=${METRICS_FLUSH_INTERVAL:-100}
BATCH_METADATA_LOG_INTERVAL=${BATCH_METADATA_LOG_INTERVAL:-${LOG_INTERVAL}}
BATCH_METADATA_TOP_K=${BATCH_METADATA_TOP_K:-16}
LOG_LEVEL=${LOG_LEVEL:-INFO}
LOG_ALL_RANKS=${LOG_ALL_RANKS:-false}

NUM_WORKERS=${NUM_WORKERS:-8}
PREFETCH_FACTOR=${PREFETCH_FACTOR:-1}
PERSISTENT_WORKERS=${PERSISTENT_WORKERS:-true}
PIN_MEMORY=${PIN_MEMORY:-true}
PARQUET_CHUNK_FILES=${PARQUET_CHUNK_FILES:-64}

ASCEND_RT_VISIBLE_DEVICES_VALUE=${ASCEND_RT_VISIBLE_DEVICES_VALUE:-0,1,2,3,4,5,6,7}
HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-7200}
HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-7200}
HCCL_WHITELIST_DISABLE=${HCCL_WHITELIST_DISABLE:-1}
ASCEND_TOOLKIT_HOME=${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/ascend-toolkit/latest}
ASCEND_ENV_SH=${ASCEND_ENV_SH:-}
ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-${ASCEND_TOOLKIT_HOME}}

OUT_PATH=${OUT_PATH:-${WORKDIR}/training_output_1b_${EXPERIMENT_NAME}_from_scratch_${RUN_ID}}
COMMAND_DIR=${COMMAND_DIR:-${STAGE2_ROOT}/pretrain_1b_${EXPERIMENT_NAME}_commands_${RUN_ID}}
LOG_SUBDIR=${LOG_SUBDIR:-torchrun_logs/1b_${EXPERIMENT_NAME}_${RUN_ID}}

case "$ACTION" in
  prepare|launch|collect)
    ;;
  *)
    die "Unsupported ACTION=${ACTION}. Use prepare, launch, or collect."
    ;;
esac

if [[ -z "${DATA_PATH:-}" ]]; then
  die "DATA_PATH=/path/to/flatten_data is required. Pass it after the script path or define it in .env."
fi

if [[ "$ACTION" == "collect" && "$RUN_ID_CONFIGURED" != "1" && "$OUT_PATH_CONFIGURED" != "1" ]]; then
  die "ACTION=collect requires RUN_ID from the launch command, or an explicit OUT_PATH. Pass it after the script path or define it in .env."
fi

if [[ "$ACTION" == "launch" ]]; then
  write_run_record
fi

run_pretrain_3node


# Vanilla launch (uses script defaults: LEARNING_RATE=1e-5, GRAD_CLIP=1.0, etc.):
#
# bash work_record/step3_model_1B.sh \
#   ACTION=launch \
#   EXPERIMENT_NAME=data_1_3 \
#   DATA_PATH=/data/disk2/SpeciesLLM_obs/Stage2_SpeciesLLMData/data_1_3_flatten_data_full_no_1st_human_mouse_xxx
#
# Stable-config launch (mirrors 100M data_1_3_stable run that avoided the
# warmup-end loss explosion: lower peak LR, longer warmup, tighter grad_clip,
# higher Adam beta2, NaN-check enabled):
#
# bash work_record/step3_model_1B.sh \
#   ACTION=launch \
#   EXPERIMENT_NAME=data_1_3_stable \
#   DATA_PATH=/data/disk2/SpeciesLLM_obs/Stage2_SpeciesLLMData/data_1_3_flatten_data_full_no_1st_human_mouse_xxx \
#   GRAD_CLIP=0.5 \
#   LEARNING_RATE=0.000001 \
#   MIN_LR=0.0000001 \
#   WARMUP_RATIO=0.10 \
#   BETA2=0.98 \
#   NAN_CHECK_INTERVAL=10
