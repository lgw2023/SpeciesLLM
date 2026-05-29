#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# SpeciesLLM multi-node torchrun launcher for bare-metal servers.
#
# Run on the master node:
#   SSH_PASSWORD='your-password' WORKDIR=/data/disk1/SpeciesLLM \
#   bash scripts/launch_multinode_torchrun.sh
#
# Worker mode is normally started by the launcher through ssh:
#   NODE_RANK=1 bash scripts/launch_multinode_torchrun.sh --worker
#
# All configuration can be overridden with environment variables. The default
# training hyperparameters mirror scripts/launch_singlenode_torchrun.sh.
###############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
SELF_NAME="$(basename "${BASH_SOURCE[0]}")"
ENV_FILE="${ENV_FILE:-$PROJECT_ROOT/.env}"

load_env_defaults() {
  local env_file="$1"
  [[ -f "$env_file" ]] || return 0

  local line key value
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"

    [[ -z "$line" || "$line" == \#* ]] && continue
    [[ "$line" == export[[:space:]]* ]] && line="${line#export }"
    [[ "$line" == *=* ]] || continue

    key="${line%%=*}"
    value="${line#*=}"
    key="${key%"${key##*[![:space:]]}"}"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"

    [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
    [[ -z "${!key+x}" ]] || continue

    if [[ "$value" == \"*\" && "$value" == *\" && ${#value} -ge 2 ]]; then
      value="${value:1:${#value}-2}"
    elif [[ "$value" == \'*\' && "$value" == *\' && ${#value} -ge 2 ]]; then
      value="${value:1:${#value}-2}"
    fi

    export "$key=$value"
  done < "$env_file"
}

load_env_defaults "$ENV_FILE"

# ---------- cluster configuration ----------
# NNODES is recalculated from the final active host list by default. Keep
# AUTO_NNODES=0 if you want NNODES to remain a strict manual value.
NNODES="${NNODES:-3}"
AUTO_NNODES="${AUTO_NNODES:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

MASTER_ADDR="${MASTER_ADDR:-7.150.12.45}"
MASTER_PORT="${MASTER_PORT:-12345}"

# Comma-separated base hosts. Order is the torchrun node_rank order.
HOSTS_CSV="${HOSTS:-7.150.12.45,7.150.15.14,7.150.14.170}"
# Comma-separated optional hosts. Keep AUTO_OPTIONAL_HOSTS=0 for the normal
# 3-node run; set AUTO_OPTIONAL_HOSTS=1 to append reachable optional hosts.
OPTIONAL_HOSTS_CSV="${OPTIONAL_HOSTS:-7.150.8.22}"
AUTO_OPTIONAL_HOSTS="${AUTO_OPTIONAL_HOSTS:-0}"
OPTIONAL_HOST_CONNECT_TIMEOUT="${OPTIONAL_HOST_CONNECT_TIMEOUT:-5}"

SSH_USER="${SSH_USER:-root}"
SSH_KEY="${SSH_KEY-}"
SSH_PASSWORD="${SSH_PASSWORD:-}"
SSH_EXTRA_OPTS="${SSH_EXTRA_OPTS:-}"

# Project directory on every server. It must contain the training entry, data,
# embeddings, and the scripts/ directory after optional SYNC_SELF.
WORKDIR="${WORKDIR:-$PROJECT_ROOT}"
TRAIN_ENTRY="${TRAIN_ENTRY:-train_MNodes_torchrun_mfu_preindexparquet.py}"
PYTHON_BIN="${PYTHON_BIN:-/data/miniconda3/bin/python}"
HOME="${HOME:-/root}"
export HOME

LOG_SUBDIR="${LOG_SUBDIR:-torchrun_logs}"
SYNC_SELF="${SYNC_SELF:-1}"
DRY_RUN="${DRY_RUN:-0}"
LOCAL_NODE_RANK="${LOCAL_NODE_RANK:-0}"

# Visible NPU ids on each node.
ASCEND_RT_VISIBLE_DEVICES_VALUE="${ASCEND_RT_VISIBLE_DEVICES_VALUE:-0,1,2,3,4,5,6,7}"

# Ascend/HCCL environment, matching the ModelArts wrapper where applicable.
HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-7200}"
HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-7200}"
HCCL_WHITELIST_DISABLE="${HCCL_WHITELIST_DISABLE:-1}"
ASCEND_TOOLKIT_HOME="${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/ascend-toolkit}"
ASCEND_ENV_SH="${ASCEND_ENV_SH:-}"
ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-$ASCEND_TOOLKIT_HOME}"

# ---------- training parameters ----------
DEFAULT_OUT_PATH='training_output'

TRAIN_DATASET="${TRAIN_DATASET:-full}"
case "$TRAIN_DATASET" in
  full)
    DEFAULT_DATA_SUBDIR="all_flatten_data"
    ;;
  test)
    DEFAULT_DATA_SUBDIR="all_flatten_data_test"
    ;;
  *)
    echo "Unsupported TRAIN_DATASET=${TRAIN_DATASET}. Use full or test."
    exit 1
    ;;
esac

SERVER_STAGE2_ROOT_DEFAULT="/data/disk1/SpeciesLLM_obs/Stage2_SpeciesLLMData"
if [[ -d "$SERVER_STAGE2_ROOT_DEFAULT" ]]; then
  DATA_ROOT="${DATA_ROOT:-$SERVER_STAGE2_ROOT_DEFAULT}"
else
  DATA_ROOT="${DATA_ROOT:-./Stage2_SpeciesLLMData}"
fi
EMB_ROOT="${EMB_ROOT:-.}"
DEFAULT_DATA_PATH="${DATA_ROOT}/${DEFAULT_DATA_SUBDIR}"
DEFAULT_EMB_PATH="${EMB_ROOT}/Stage2_macrogene_embeddings"

data_path="${data_path:-${DATA_PATH:-$DEFAULT_DATA_PATH}}"
num_of_used_data="${num_of_used_data:-${NUM_OF_USED_DATA:-0}}"
emb_path="${emb_path:-${EMB_PATH:-$DEFAULT_EMB_PATH}}"
config_json="${config_json:-${MODEL_CONFIG_JSON:-${emb_path}/args_2nd_run.json}}"
out_path="${out_path:-${OUT_PATH:-$DEFAULT_OUT_PATH}}"
batch_size="${batch_size:-${BATCH_SIZE:-16}}"
epoch="${epoch:-${EPOCH:-10}}"
gradient_accumulation_steps="${gradient_accumulation_steps:-${GRADIENT_ACCUMULATION_STEPS:-8}}"
gradient_checkpointing="${gradient_checkpointing:-${GRADIENT_CHECKPOINTING:-true}}"
train_mvc="${train_mvc:-${TRAIN_MVC:-true}}"
runtime_do_mvc="${runtime_do_mvc:-${RUNTIME_DO_MVC:-}}"
runtime_explicit_zero_prob="${runtime_explicit_zero_prob:-${RUNTIME_EXPLICIT_ZERO_PROB:-}}"
learning_rate="${learning_rate:-${LEARNING_RATE:-0.00001}}"
min_lr="${min_lr:-${MIN_LR:-0.000001}}"
decay_lr="${decay_lr:-${DECAY_LR:-true}}"
warmup_iters="${warmup_iters:-${WARMUP_ITERS:-2000}}"
warmup_ratio="${warmup_ratio:-${WARMUP_RATIO:-0.05}}"
lr_decay_epochs="${lr_decay_epochs:-${LR_DECAY_EPOCHS:-0}}"
weight_decay="${weight_decay:-${WEIGHT_DECAY:-0.1}}"
save_data_interval="${save_data_interval:-${SAVE_DATA_INTERVAL:-5120000}}"
beta1="${beta1:-${BETA1:-0.9}}"
beta2="${beta2:-${BETA2:-0.95}}"
grad_clip="${grad_clip:-${GRAD_CLIP:-1.0}}"
adaptive_grad_clip="${adaptive_grad_clip:-${ADAPTIVE_GRAD_CLIP:-true}}"
grad_clip_ema_beta="${grad_clip_ema_beta:-${GRAD_CLIP_EMA_BETA:-0.98}}"
grad_clip_ratio="${grad_clip_ratio:-${GRAD_CLIP_RATIO:-3.0}}"
grad_skip_ratio="${grad_skip_ratio:-${GRAD_SKIP_RATIO:-100.0}}"
grad_skip_max="${grad_skip_max:-${GRAD_SKIP_MAX:-100000000000.0}}"
grad_clip_min="${grad_clip_min:-${GRAD_CLIP_MIN:-0.5}}"
grad_clip_max="${grad_clip_max:-${GRAD_CLIP_MAX:-0.5}}"
grad_clip_warmup_steps="${grad_clip_warmup_steps:-${GRAD_CLIP_WARMUP_STEPS:-200}}"
grad_clip_max_consecutive_skips="${grad_clip_max_consecutive_skips:-${GRAD_CLIP_MAX_CONSECUTIVE_SKIPS:-50}}"
grad_clip_ema_runaway_factor="${grad_clip_ema_runaway_factor:-${GRAD_CLIP_EMA_RUNAWAY_FACTOR:-2.0}}"
grad_clip_hard_raw_norm_limit="${grad_clip_hard_raw_norm_limit:-${GRAD_CLIP_HARD_RAW_NORM_LIMIT:-100000000000.0}}"
amp_dtype="${amp_dtype:-${AMP_DTYPE:-float32}}"
gep_loss="${gep_loss:-${GEP_LOSS:-mse}}"
huber_delta="${huber_delta:-${HUBER_DELTA:-1.0}}"
max_train_steps="${max_train_steps:-${MAX_TRAIN_STEPS:-0}}"
init_model_path="${init_model_path:-${INIT_MODEL_PATH:-}}"
init_optimizer_path="${init_optimizer_path:-${INIT_OPTIMIZER_PATH:-}}"
resume_update_step="${resume_update_step:-${RESUME_UPDATE_STEP:-0}}"
resume_start_epoch="${resume_start_epoch:-${RESUME_START_EPOCH:-0}}"
resume_skip_batches="${resume_skip_batches:-${RESUME_SKIP_BATCHES:-0}}"
append_output_logs="${append_output_logs:-${APPEND_OUTPUT_LOGS:-false}}"
compile="${compile:-${COMPILE:-true}}"
backend="${backend:-${BACKEND:-hccl}}"
device="${device:-${DEVICE:-npu}}"
device_type="${device_type:-${DEVICE_TYPE:-npu}}"
s3_remote_dir_path="${s3_remote_dir_path:-${S3_REMOTE_DIR_PATH:-}}"
log_interval="${log_interval:-${LOG_INTERVAL:-10}}"
profile_interval="${profile_interval:-${PROFILE_INTERVAL:-100}}"
nan_check_interval="${nan_check_interval:-${NAN_CHECK_INTERVAL:-0}}"
metrics_flush_interval="${metrics_flush_interval:-${METRICS_FLUSH_INTERVAL:-100}}"
log_level="${log_level:-${LOG_LEVEL:-INFO}}"
log_all_ranks="${log_all_ranks:-${LOG_ALL_RANKS:-false}}"
num_workers="${num_workers:-${NUM_WORKERS:-4}}"
prefetch_factor="${prefetch_factor:-${PREFETCH_FACTOR:-1}}"
persistent_workers="${persistent_workers:-${PERSISTENT_WORKERS:-false}}"
pin_memory="${pin_memory:-${PIN_MEMORY:-true}}"

WORKER_MODE=0
if [[ "${1:-}" == "--worker" ]]; then
  WORKER_MODE=1
  shift
fi

if [[ "$#" -gt 0 ]]; then
  echo "[WARN] Extra arguments are ignored: $*"
fi

if [[ ! -f "$config_json" ]]; then
  echo "Missing fixed model config JSON: $config_json"
  exit 1
fi

log() {
  echo "[$(date '+%F %T')] $*"
}

usage() {
  cat <<USAGE
Usage:
  # Master-node launcher
  HOSTS="host0,host1,host2" OPTIONAL_HOSTS="host3" MASTER_ADDR=host0 WORKDIR=/path/to/SpeciesLLM bash scripts/$SELF_NAME

  # Worker mode, normally used by the launcher
  NODE_RANK=1 bash $SELF_NAME --worker

Important environment variables:
  NNODES, AUTO_NNODES, NPROC_PER_NODE, HOSTS, OPTIONAL_HOSTS, AUTO_OPTIONAL_HOSTS
  OPTIONAL_HOST_CONNECT_TIMEOUT, MASTER_ADDR, MASTER_PORT
  WORKDIR, SSH_USER, SSH_KEY, SSH_PASSWORD, SSH_EXTRA_OPTS, SYNC_SELF, DRY_RUN
  TRAIN_DATASET=full|test
  DATA_ROOT, EMB_ROOT, DATA_PATH/data_path, EMB_PATH/emb_path
  MODEL_CONFIG_JSON/config_json, BATCH_SIZE/batch_size
  NUM_WORKERS/num_workers, PREFETCH_FACTOR/prefetch_factor,
  PERSISTENT_WORKERS/persistent_workers, PIN_MEMORY/pin_memory
USAGE
}

shell_quote() {
  local value="$1"
  printf "'%s'" "${value//\'/\'\\\'\'}"
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Missing required command: $1"
    exit 1
  }
}

split_hosts() {
  HOSTS_ARR=()
  local -a raw_hosts
  local raw_host host
  IFS=',' read -r -a raw_hosts <<< "$HOSTS_CSV"
  for raw_host in "${raw_hosts[@]}"; do
    host="$(trim "$raw_host")"
    [[ -n "$host" ]] && HOSTS_ARR+=("$host")
  done
}

split_optional_hosts() {
  OPTIONAL_HOSTS_ARR=()
  local -a raw_hosts
  local raw_host host
  IFS=',' read -r -a raw_hosts <<< "$OPTIONAL_HOSTS_CSV"
  for raw_host in "${raw_hosts[@]}"; do
    host="$(trim "$raw_host")"
    [[ -n "$host" ]] && OPTIONAL_HOSTS_ARR+=("$host")
  done
}

trim() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf "%s" "$value"
}

is_enabled() {
  local value
  value="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
  case "$value" in
    1|true|yes|y|on)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

join_hosts() {
  local IFS=','
  printf "%s" "$*"
}

host_already_selected() {
  local candidate="$1"
  local host
  for host in "${HOSTS_ARR[@]}"; do
    [[ "$host" == "$candidate" ]] && return 0
  done
  return 1
}

probe_optional_host() {
  local host="$1"
  local -a probe_opts=("${SSH_OPTS[@]}")
  probe_opts+=("-o" "ConnectTimeout=${OPTIONAL_HOST_CONNECT_TIMEOUT}")
  probe_opts+=("-o" "ConnectionAttempts=1")

  if [[ -n "$SSH_PASSWORD" ]]; then
    SSHPASS="$SSH_PASSWORD" sshpass -e ssh "${probe_opts[@]}" "${SSH_USER}@${host}" "true" >/dev/null 2>&1
  else
    ssh "${probe_opts[@]}" "${SSH_USER}@${host}" "true" >/dev/null 2>&1
  fi
}

resolve_launcher_hosts() {
  local host_count
  split_hosts
  host_count="${#HOSTS_ARR[@]}"
  if [[ "$host_count" -eq 0 ]]; then
    echo "HOSTS must contain at least one base host."
    exit 1
  fi

  if is_enabled "$AUTO_OPTIONAL_HOSTS" && [[ -n "$OPTIONAL_HOSTS_CSV" ]]; then
    split_optional_hosts

    local host
    for host in "${OPTIONAL_HOSTS_ARR[@]}"; do
      if host_already_selected "$host"; then
        log "optional host already present in HOSTS, keep existing rank: ${host}"
        continue
      fi

      if [[ "$DRY_RUN" == "1" ]]; then
        log "DRY_RUN=1, include optional host without ssh probe: ${host}"
        HOSTS_ARR+=("$host")
      elif probe_optional_host "$host"; then
        log "optional host available, include: ${host}"
        HOSTS_ARR+=("$host")
      else
        log "optional host unavailable, skip: ${host}"
      fi
    done
  fi

  HOSTS_CSV="$(join_hosts "${HOSTS_ARR[@]}")"
  host_count="${#HOSTS_ARR[@]}"

  if is_enabled "$AUTO_NNODES"; then
    NNODES="$host_count"
  elif [[ "$host_count" -ne "$NNODES" ]]; then
    echo "HOSTS count (${host_count}) must equal NNODES (${NNODES}). HOSTS=${HOSTS_CSV}"
    exit 1
  fi
}

build_ssh_opts() {
  SSH_OPTS=()
  if [[ -n "$SSH_KEY" ]]; then
    SSH_OPTS+=("-i" "$SSH_KEY")
  fi
  if [[ -n "$SSH_PASSWORD" && "$DRY_RUN" != "1" ]]; then
    require_cmd sshpass
  fi
  if [[ -n "$SSH_EXTRA_OPTS" ]]; then
    read -r -a SSH_EXTRA_OPTS_ARR <<< "$SSH_EXTRA_OPTS"
    SSH_OPTS+=("${SSH_EXTRA_OPTS_ARR[@]}")
  fi
}

ssh_display_cmd() {
  if [[ -n "$SSH_PASSWORD" ]]; then
    printf "SSHPASS=*** sshpass -e ssh"
  else
    printf "ssh"
  fi
}

scp_display_cmd() {
  if [[ -n "$SSH_PASSWORD" ]]; then
    printf "SSHPASS=*** sshpass -e scp"
  else
    printf "scp"
  fi
}

ssh_run() {
  local target="$1"
  shift
  if [[ -n "$SSH_PASSWORD" ]]; then
    SSHPASS="$SSH_PASSWORD" sshpass -e ssh ${SSH_OPTS[@]+"${SSH_OPTS[@]}"} "$target" "$@"
  else
    ssh ${SSH_OPTS[@]+"${SSH_OPTS[@]}"} "$target" "$@"
  fi
}

scp_run() {
  local source_path="$1"
  local target_path="$2"
  if [[ -n "$SSH_PASSWORD" ]]; then
    SSHPASS="$SSH_PASSWORD" sshpass -e scp ${SSH_OPTS[@]+"${SSH_OPTS[@]}"} "$source_path" "$target_path"
  else
    scp ${SSH_OPTS[@]+"${SSH_OPTS[@]}"} "$source_path" "$target_path"
  fi
}

is_local_rank() {
  local rank="$1"
  [[ "$rank" == "$LOCAL_NODE_RANK" ]]
}

bool_arg_value() {
  local value
  value="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
  case "$value" in
    1|true|yes|y|on)
      printf "true"
      ;;
    0|false|no|n|off)
      printf "false"
      ;;
    *)
      printf "%s" "$1"
      ;;
  esac
}

build_train_args() {
  TRAIN_ARGS=(
    "--data_path=${data_path}"
    "--num_of_used_data=${num_of_used_data}"
    "--emb_path=${emb_path}"
    "--config_json=${config_json}"
    "--out_path=${out_path}"
    "--batch_size=${batch_size}"
    "--epoch=${epoch}"
    "--gradient_accumulation_steps=${gradient_accumulation_steps}"
    "--gradient_checkpointing=${gradient_checkpointing}"
    "--train_mvc=${train_mvc}"
    "--learning_rate=${learning_rate}"
    "--min_lr=${min_lr}"
    "--decay_lr=$(bool_arg_value "$decay_lr")"
    "--warmup_iters=${warmup_iters}"
    "--warmup_ratio=${warmup_ratio}"
    "--lr_decay_epochs=${lr_decay_epochs}"
    "--weight_decay=${weight_decay}"
    "--save_data_interval=${save_data_interval}"
    "--beta1=${beta1}"
    "--beta2=${beta2}"
    "--grad_clip=${grad_clip}"
    "--adaptive_grad_clip=${adaptive_grad_clip}"
    "--grad_clip_ema_beta=${grad_clip_ema_beta}"
    "--grad_clip_ratio=${grad_clip_ratio}"
    "--grad_skip_ratio=${grad_skip_ratio}"
    "--grad_skip_max=${grad_skip_max}"
    "--grad_clip_min=${grad_clip_min}"
    "--grad_clip_max=${grad_clip_max}"
    "--grad_clip_warmup_steps=${grad_clip_warmup_steps}"
    "--grad_clip_max_consecutive_skips=${grad_clip_max_consecutive_skips}"
    "--grad_clip_ema_runaway_factor=${grad_clip_ema_runaway_factor}"
    "--grad_clip_hard_raw_norm_limit=${grad_clip_hard_raw_norm_limit}"
    "--amp_dtype=${amp_dtype}"
    "--gep_loss=${gep_loss}"
    "--huber_delta=${huber_delta}"
    "--max_train_steps=${max_train_steps}"
    "--resume_update_step=${resume_update_step}"
    "--resume_start_epoch=${resume_start_epoch}"
    "--resume_skip_batches=${resume_skip_batches}"
    "--append_output_logs=${append_output_logs}"
    "--compile=$(bool_arg_value "$compile")"
    "--backend=${backend}"
    "--device=${device}"
    "--device_type=${device_type}"
    "--log_interval=${log_interval}"
    "--profile_interval=${profile_interval}"
    "--nan_check_interval=${nan_check_interval}"
    "--metrics_flush_interval=${metrics_flush_interval}"
    "--log_level=${log_level}"
    "--log_all_ranks=${log_all_ranks}"
    "--num_workers=${num_workers}"
    "--prefetch_factor=${prefetch_factor}"
    "--persistent_workers=${persistent_workers}"
    "--pin_memory=${pin_memory}"
  )

  if [[ -n "$s3_remote_dir_path" ]]; then
    TRAIN_ARGS+=("--s3_remote_dir_path=${s3_remote_dir_path}")
  fi
  if [[ -n "$init_model_path" ]]; then
    TRAIN_ARGS+=("--init_model_path=${init_model_path}")
  fi
  if [[ -n "$init_optimizer_path" ]]; then
    TRAIN_ARGS+=("--init_optimizer_path=${init_optimizer_path}")
  fi
  if [[ -n "$runtime_do_mvc" ]]; then
    TRAIN_ARGS+=("--runtime_do_mvc=${runtime_do_mvc}")
  fi
  if [[ -n "$runtime_explicit_zero_prob" ]]; then
    TRAIN_ARGS+=("--runtime_explicit_zero_prob=${runtime_explicit_zero_prob}")
  fi
}

print_command() {
  printf '%q ' "$@"
  printf '\n'
}

resolve_ascend_env_sh() {
  local candidate
  for candidate in \
    "$ASCEND_ENV_SH" \
    "/usr/local/Ascend/ascend-toolkit/set_env.sh" \
    "${ASCEND_TOOLKIT_HOME}/set_env.sh" \
    "/usr/local/Ascend/ascend-toolkit/latest/set_env.sh" \
    "${ASCEND_TOOLKIT_HOME}/latest/set_env.sh"; do
    [[ -n "$candidate" && -f "$candidate" ]] && {
      printf "%s" "$candidate"
      return 0
    }
  done
  return 1
}

source_ascend_env() {
  local env_sh
  if ! env_sh="$(resolve_ascend_env_sh)"; then
    echo "Missing Ascend environment script. Set ASCEND_ENV_SH=/path/to/set_env.sh" >&2
    exit 1
  fi

  log "source Ascend environment: ${env_sh}"
  set +u
  # shellcheck source=/dev/null
  source "$env_sh"
  set -u
  export ASCEND_ENV_SH="$env_sh"
}

verify_python_npu() {
  log "verify Python/NPU environment with: ${PYTHON_BIN}"
  "$PYTHON_BIN" - <<'PY'
import torch
import torch_npu

print("torch:", torch.__version__)
print("torch_npu:", torch_npu.__version__)
print("npu available:", torch.npu.is_available())
PY
}

run_worker() {
  local node_rank="${NODE_RANK:-}"
  if [[ -z "$node_rank" ]]; then
    echo "NODE_RANK is required in worker mode."
    usage
    exit 1
  fi

  cd "$WORKDIR"
  mkdir -p "$LOG_SUBDIR"

  if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "PYTHON_BIN is not executable: $PYTHON_BIN"
    exit 1
  fi

  export HOME NNODES NPROC_PER_NODE MASTER_ADDR MASTER_PORT NODE_RANK PYTHON_BIN
  export HCCL_CONNECT_TIMEOUT HCCL_EXEC_TIMEOUT HCCL_WHITELIST_DISABLE
  export ASCEND_TOOLKIT_HOME ASCEND_ENV_SH ASCEND_HOME_PATH
  export ASCEND_RT_VISIBLE_DEVICES="$ASCEND_RT_VISIBLE_DEVICES_VALUE"

  build_train_args

  local -a cmd=(
    "$PYTHON_BIN"
    -m
    torch.distributed.run
    "--nproc_per_node=${NPROC_PER_NODE}"
    "--nnodes=${NNODES}"
    "--node_rank=${node_rank}"
    "--master_addr=${MASTER_ADDR}"
    "--master_port=${MASTER_PORT}"
    "$TRAIN_ENTRY"
    "${TRAIN_ARGS[@]}"
  )

  log "worker start: node_rank=${node_rank}, nnodes=${NNODES}, nproc_per_node=${NPROC_PER_NODE}, master=${MASTER_ADDR}:${MASTER_PORT}"
  log "workdir=${WORKDIR}"
  log "ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES}"
  log "command:"
  print_command "${cmd[@]}"

  if [[ "$DRY_RUN" == "1" ]]; then
    log "DRY_RUN=1, skip worker execution."
    return 0
  fi

  source_ascend_env
  verify_python_npu

  "${cmd[@]}"
}

remote_env_assignments() {
  local rank="$1"
  local -a names=(
    HOME
    NNODES NPROC_PER_NODE MASTER_ADDR MASTER_PORT WORKDIR TRAIN_ENTRY PYTHON_BIN LOG_SUBDIR
    DRY_RUN LOCAL_NODE_RANK ASCEND_RT_VISIBLE_DEVICES_VALUE HCCL_CONNECT_TIMEOUT HCCL_EXEC_TIMEOUT
    HCCL_WHITELIST_DISABLE ASCEND_TOOLKIT_HOME ASCEND_ENV_SH ASCEND_HOME_PATH
    data_path num_of_used_data emb_path config_json out_path batch_size epoch
    gradient_accumulation_steps gradient_checkpointing train_mvc runtime_do_mvc runtime_explicit_zero_prob
    learning_rate min_lr decay_lr warmup_iters
    warmup_ratio lr_decay_epochs weight_decay save_data_interval beta1 beta2 grad_clip
    adaptive_grad_clip grad_clip_ema_beta grad_clip_ratio grad_skip_ratio grad_skip_max
    grad_clip_min grad_clip_max grad_clip_warmup_steps grad_clip_max_consecutive_skips
    grad_clip_ema_runaway_factor grad_clip_hard_raw_norm_limit amp_dtype gep_loss huber_delta max_train_steps
    init_model_path init_optimizer_path resume_update_step resume_start_epoch resume_skip_batches
    append_output_logs compile
    backend device device_type s3_remote_dir_path
    log_interval profile_interval nan_check_interval metrics_flush_interval log_level log_all_ranks
    num_workers prefetch_factor persistent_workers pin_memory
  )

  printf "NODE_RANK=%s " "$(shell_quote "$rank")"
  local name value
  for name in "${names[@]}"; do
    value="${!name}"
    printf "%s=%s " "$name" "$(shell_quote "$value")"
  done
}

run_launcher() {
  require_cmd ssh
  require_cmd scp
  build_ssh_opts
  resolve_launcher_hosts

  local self_abs="${SCRIPT_DIR}/${SELF_NAME}"
  local remote_scripts_dir="${WORKDIR}/scripts"
  local remote_self="${remote_scripts_dir}/${SELF_NAME}"
  local remote_log_dir="${WORKDIR}/${LOG_SUBDIR}"

  log "launcher start"
  log "HOSTS(rank order)=${HOSTS_CSV}"
  log "OPTIONAL_HOSTS=${OPTIONAL_HOSTS_CSV}, AUTO_OPTIONAL_HOSTS=${AUTO_OPTIONAL_HOSTS}, AUTO_NNODES=${AUTO_NNODES}"
  log "NNODES=${NNODES}, NPROC_PER_NODE=${NPROC_PER_NODE}, MASTER=${MASTER_ADDR}:${MASTER_PORT}"
  log "WORKDIR=${WORKDIR}, remote_script=${remote_self}"
  log "TRAIN_DATASET=${TRAIN_DATASET}, data_path=${data_path}, emb_path=${emb_path}"
  log "config_json=${config_json}"
  log "max_train_steps=${max_train_steps}, adaptive_grad_clip=${adaptive_grad_clip}"
  log "SYNC_SELF=${SYNC_SELF}, DRY_RUN=${DRY_RUN}, LOCAL_NODE_RANK=${LOCAL_NODE_RANK}"

  local rank host
  for ((rank=0; rank<NNODES; rank++)); do
    host="${HOSTS_ARR[$rank]}"
    log "prepare node_rank=${rank}, host=${host}"

    local mkdir_cmd
    mkdir_cmd="mkdir -p $(shell_quote "$remote_log_dir") $(shell_quote "$remote_scripts_dir")"

    if [[ "$DRY_RUN" == "1" ]]; then
      if is_local_rank "$rank"; then
        echo "local ${mkdir_cmd}"
      else
        echo "$(ssh_display_cmd) ${SSH_OPTS[*]-} ${SSH_USER}@${host} ${mkdir_cmd}"
        if [[ "$SYNC_SELF" == "1" ]]; then
          echo "$(scp_display_cmd) ${SSH_OPTS[*]-} ${self_abs} ${SSH_USER}@${host}:${remote_self}"
        fi
      fi
      continue
    fi

    if is_local_rank "$rank"; then
      mkdir -p "$remote_log_dir" "$remote_scripts_dir"
    else
      ssh_run "${SSH_USER}@${host}" "$mkdir_cmd"
    fi
    if [[ "$SYNC_SELF" == "1" ]] && ! is_local_rank "$rank"; then
      scp_run "$self_abs" "${SSH_USER}@${host}:${remote_self}" >/dev/null
    fi
  done

  for ((rank=0; rank<NNODES; rank++)); do
    host="${HOSTS_ARR[$rank]}"
    local log_file="${remote_log_dir}/node_rank${rank}.log"
    local remote_cmd
    remote_cmd="cd $(shell_quote "$WORKDIR") && nohup env $(remote_env_assignments "$rank") bash $(shell_quote "$remote_self") --worker > $(shell_quote "$log_file") 2>&1 &"

    if [[ "$DRY_RUN" == "1" ]]; then
      if is_local_rank "$rank"; then
        echo "local ${remote_cmd}"
      else
        echo "$(ssh_display_cmd) ${SSH_OPTS[*]-} ${SSH_USER}@${host} ${remote_cmd}"
      fi
    else
      if is_local_rank "$rank"; then
        bash -c "$remote_cmd"
      else
        ssh_run "${SSH_USER}@${host}" "$remote_cmd" &
      fi
    fi

    log "sent start command: rank=${rank}, host=${host}, log=${log_file}"
  done

  if [[ "$DRY_RUN" == "1" ]]; then
    log "DRY_RUN=1, no ssh command was executed."
    return 0
  fi

  wait
  log "all ssh start commands have been sent."
  log "logs: ${remote_log_dir}/node_rank{0..$((NNODES - 1))}.log"
}

if [[ "$WORKER_MODE" == "1" ]]; then
  run_worker
else
  run_launcher
fi
