#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# SpeciesLLM multi-node torchrun launcher for bare-metal servers.
#
# Run on the master node:
#   SSH_PASSWORD='your-password' WORKDIR=/data1/liguowei/SpeciesLLM \
#   bash scripts/train_multinode.sh
#
# Worker mode is normally started by the launcher through ssh:
#   NODE_RANK=1 bash scripts/train_multinode.sh --worker
#
# All configuration can be overridden with environment variables. The default
# training hyperparameters mirror scripts/train_singlenode.sh.
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

LOG_SUBDIR="${LOG_SUBDIR:-torchrun_logs}"
SYNC_SELF="${SYNC_SELF:-1}"
DRY_RUN="${DRY_RUN:-0}"

# Visible NPU ids on each node.
ASCEND_RT_VISIBLE_DEVICES_VALUE="${ASCEND_RT_VISIBLE_DEVICES_VALUE:-0,1,2,3,4,5,6,7}"

# Ascend/HCCL environment, matching the ModelArts wrapper where applicable.
HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-7200}"
HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-7200}"
HCCL_WHITELIST_DISABLE="${HCCL_WHITELIST_DISABLE:-1}"
ASCEND_TOOLKIT_HOME="${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/ascend-toolkit/latest}"
ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-$ASCEND_TOOLKIT_HOME}"

# ---------- training parameters ----------
DEFAULT_OUT_PATH='hs_{hidden_size}_nh_{num_hidden_layers}_na_{num_attention_heads}_hdp_{hidden_dropout_prob}_lr_{learning_rate}_mlr_{min_lr}_wd_{weight_decay}_wr_{warmup_ratio}'

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

DATA_ROOT="${DATA_ROOT:-./Stage2_SpeciesLLMData}"
EMB_ROOT="${EMB_ROOT:-.}"
DEFAULT_DATA_PATH="${DATA_ROOT}/${DEFAULT_DATA_SUBDIR}"
DEFAULT_EMB_PATH="${EMB_ROOT}/Stage2_macrogene_embeddings"

data_path="${data_path:-${DATA_PATH:-$DEFAULT_DATA_PATH}}"
num_of_used_data="${num_of_used_data:-${NUM_OF_USED_DATA:-0}}"
emb_path="${emb_path:-${EMB_PATH:-$DEFAULT_EMB_PATH}}"
seq_len="${seq_len:-${SEQ_LEN:-640}}"
out_path="${out_path:-${OUT_PATH:-$DEFAULT_OUT_PATH}}"
batch_size="${batch_size:-${BATCH_SIZE:-32}}"
epoch="${epoch:-${EPOCH:-10}}"
gradient_accumulation_steps="${gradient_accumulation_steps:-${GRADIENT_ACCUMULATION_STEPS:-8}}"
learning_rate="${learning_rate:-${LEARNING_RATE:-0.00001}}"
min_lr="${min_lr:-${MIN_LR:-0.000001}}"
decay_lr="${decay_lr:-${DECAY_LR:-true}}"
warmup_iters="${warmup_iters:-${WARMUP_ITERS:-2000}}"
warmup_ratio="${warmup_ratio:-${WARMUP_RATIO:-0.05}}"
weight_decay="${weight_decay:-${WEIGHT_DECAY:-0.1}}"
save_data_interval="${save_data_interval:-${SAVE_DATA_INTERVAL:-5120000}}"
beta1="${beta1:-${BETA1:-0.9}}"
beta2="${beta2:-${BETA2:-0.95}}"
grad_clip="${grad_clip:-${GRAD_CLIP:-1.0}}"
compile="${compile:-${COMPILE:-true}}"
backend="${backend:-${BACKEND:-hccl}}"
device="${device:-${DEVICE:-npu}}"
device_type="${device_type:-${DEVICE_TYPE:-npu}}"
s3_remote_dir_path="${s3_remote_dir_path:-${S3_REMOTE_DIR_PATH:-}}"

hidden_size="${hidden_size:-${HIDDEN_SIZE:-1280}}"
num_hidden_layers="${num_hidden_layers:-${NUM_HIDDEN_LAYERS:-24}}"
num_attention_heads="${num_attention_heads:-${NUM_ATTENTION_HEADS:-20}}"
intermediate_size="${intermediate_size:-${INTERMEDIATE_SIZE:-5120}}"
hidden_act="${hidden_act:-${HIDDEN_ACT:-gelu}}"
hidden_dropout_prob="${hidden_dropout_prob:-${HIDDEN_DROPOUT_PROB:-0.1}}"
cell_hidden_size="${cell_hidden_size:-${CELL_HIDDEN_SIZE:-128}}"
attention_probs_dropout_prob="${attention_probs_dropout_prob:-${ATTENTION_PROBS_DROPOUT_PROB:-0.1}}"
type_vocab_size="${type_vocab_size:-${TYPE_VOCAB_SIZE:-2}}"
initializer_range="${initializer_range:-${INITIALIZER_RANGE:-0.02}}"
layer_norm_eps="${layer_norm_eps:-${LAYER_NORM_EPS:-1e-12}}"
_attn_implementation="${_attn_implementation:-${ATTN_IMPLEMENTATION:-eager}}"

use_batch_labels="${use_batch_labels:-${USE_BATCH_LABELS:-false}}"
num_batch_labels="${num_batch_labels:-${NUM_BATCH_LABELS:-62223}}"
use_species_labels="${use_species_labels:-${USE_SPECIES_LABELS:-true}}"
num_species_labels="${num_species_labels:-${NUM_SPECIES_LABELS:-29}}"
use_tissue_labels="${use_tissue_labels:-${USE_TISSUE_LABELS:-true}}"
num_tissue_labels="${num_tissue_labels:-${NUM_TISSUE_LABELS:-336}}"
use_seqmethod_labels="${use_seqmethod_labels:-${USE_SEQMETHOD_LABELS:-true}}"
num_seqmethod_labels="${num_seqmethod_labels:-${NUM_SEQMETHOD_LABELS:-30}}"
use_disease_labels="${use_disease_labels:-${USE_DISEASE_LABELS:-true}}"
num_disease_labels="${num_disease_labels:-${NUM_DISEASE_LABELS:-1921}}"
use_age_labels="${use_age_labels:-${USE_AGE_LABELS:-true}}"
num_age_labels="${num_age_labels:-${NUM_AGE_LABELS:-5}}"
use_sex_labels="${use_sex_labels:-${USE_SEX_LABELS:-true}}"
num_sex_labels="${num_sex_labels:-${NUM_SEX_LABELS:-3}}"
cell_emb_style="${cell_emb_style:-${CELL_EMB_STYLE:-cls}}"
chunk_size_feed_forward="${chunk_size_feed_forward:-${CHUNK_SIZE_FEED_FORWARD:-0}}"
explicit_zero_prob="${explicit_zero_prob:-${EXPLICIT_ZERO_PROB:-true}}"

WORKER_MODE=0
if [[ "${1:-}" == "--worker" ]]; then
  WORKER_MODE=1
  shift
fi

if [[ "$#" -gt 0 ]]; then
  echo "[WARN] Extra arguments are ignored: $*"
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
  SEQ_LEN/seq_len, BATCH_SIZE/batch_size
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

bool_arg_value() {
  local value
  value="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
  case "$value" in
    1|true|yes|y|on)
      printf "true"
      ;;
    0|false|no|n|off)
      # The training script currently declares some boolean flags as
      # type=bool. Passing an empty value is parsed by bool("") as False.
      printf ""
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
    "--seq_len=${seq_len}"
    "--out_path=${out_path}"
    "--batch_size=${batch_size}"
    "--epoch=${epoch}"
    "--gradient_accumulation_steps=${gradient_accumulation_steps}"
    "--learning_rate=${learning_rate}"
    "--min_lr=${min_lr}"
    "--decay_lr=$(bool_arg_value "$decay_lr")"
    "--warmup_iters=${warmup_iters}"
    "--warmup_ratio=${warmup_ratio}"
    "--weight_decay=${weight_decay}"
    "--save_data_interval=${save_data_interval}"
    "--beta1=${beta1}"
    "--beta2=${beta2}"
    "--grad_clip=${grad_clip}"
    "--compile=$(bool_arg_value "$compile")"
    "--backend=${backend}"
    "--device=${device}"
    "--device_type=${device_type}"
    "--hidden_size=${hidden_size}"
    "--num_hidden_layers=${num_hidden_layers}"
    "--num_attention_heads=${num_attention_heads}"
    "--intermediate_size=${intermediate_size}"
    "--hidden_act=${hidden_act}"
    "--hidden_dropout_prob=${hidden_dropout_prob}"
    "--cell_hidden_size=${cell_hidden_size}"
    "--attention_probs_dropout_prob=${attention_probs_dropout_prob}"
    "--type_vocab_size=${type_vocab_size}"
    "--initializer_range=${initializer_range}"
    "--layer_norm_eps=${layer_norm_eps}"
    "--_attn_implementation=${_attn_implementation}"
    "--use_batch_labels=${use_batch_labels}"
    "--num_batch_labels=${num_batch_labels}"
    "--use_species_labels=${use_species_labels}"
    "--num_species_labels=${num_species_labels}"
    "--use_tissue_labels=${use_tissue_labels}"
    "--num_tissue_labels=${num_tissue_labels}"
    "--use_seqmethod_labels=${use_seqmethod_labels}"
    "--num_seqmethod_labels=${num_seqmethod_labels}"
    "--use_disease_labels=${use_disease_labels}"
    "--num_disease_labels=${num_disease_labels}"
    "--use_age_labels=${use_age_labels}"
    "--num_age_labels=${num_age_labels}"
    "--use_sex_labels=${use_sex_labels}"
    "--num_sex_labels=${num_sex_labels}"
    "--cell_emb_style=${cell_emb_style}"
    "--chunk_size_feed_forward=${chunk_size_feed_forward}"
    "--explicit_zero_prob=${explicit_zero_prob}"
  )

  if [[ -n "$s3_remote_dir_path" ]]; then
    TRAIN_ARGS+=("--s3_remote_dir_path=${s3_remote_dir_path}")
  fi
}

print_command() {
  printf '%q ' "$@"
  printf '\n'
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

  export NNODES NPROC_PER_NODE MASTER_ADDR MASTER_PORT NODE_RANK
  export HCCL_CONNECT_TIMEOUT HCCL_EXEC_TIMEOUT HCCL_WHITELIST_DISABLE
  export ASCEND_TOOLKIT_HOME ASCEND_HOME_PATH
  export ASCEND_RT_VISIBLE_DEVICES="$ASCEND_RT_VISIBLE_DEVICES_VALUE"

  build_train_args

  local -a cmd=(
    torchrun
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

  "${cmd[@]}"
}

remote_env_assignments() {
  local rank="$1"
  local -a names=(
    NNODES NPROC_PER_NODE MASTER_ADDR MASTER_PORT WORKDIR TRAIN_ENTRY LOG_SUBDIR
    DRY_RUN ASCEND_RT_VISIBLE_DEVICES_VALUE HCCL_CONNECT_TIMEOUT HCCL_EXEC_TIMEOUT
    HCCL_WHITELIST_DISABLE ASCEND_TOOLKIT_HOME ASCEND_HOME_PATH
    data_path num_of_used_data emb_path seq_len out_path batch_size epoch
    gradient_accumulation_steps learning_rate min_lr decay_lr warmup_iters
    warmup_ratio weight_decay save_data_interval beta1 beta2 grad_clip compile
    backend device device_type s3_remote_dir_path hidden_size num_hidden_layers
    num_attention_heads intermediate_size hidden_act hidden_dropout_prob
    cell_hidden_size attention_probs_dropout_prob type_vocab_size initializer_range
    layer_norm_eps _attn_implementation use_batch_labels num_batch_labels
    use_species_labels num_species_labels use_tissue_labels num_tissue_labels
    use_seqmethod_labels num_seqmethod_labels use_disease_labels num_disease_labels
    use_age_labels num_age_labels use_sex_labels num_sex_labels cell_emb_style
    chunk_size_feed_forward explicit_zero_prob
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
  log "SYNC_SELF=${SYNC_SELF}, DRY_RUN=${DRY_RUN}"

  local rank host
  for ((rank=0; rank<NNODES; rank++)); do
    host="${HOSTS_ARR[$rank]}"
    log "prepare node_rank=${rank}, host=${host}"

    local mkdir_cmd
    mkdir_cmd="mkdir -p $(shell_quote "$remote_log_dir") $(shell_quote "$remote_scripts_dir")"

    if [[ "$DRY_RUN" == "1" ]]; then
      echo "$(ssh_display_cmd) ${SSH_OPTS[*]-} ${SSH_USER}@${host} ${mkdir_cmd}"
      if [[ "$SYNC_SELF" == "1" ]]; then
        echo "$(scp_display_cmd) ${SSH_OPTS[*]-} ${self_abs} ${SSH_USER}@${host}:${remote_self}"
      fi
      continue
    fi

    ssh_run "${SSH_USER}@${host}" "$mkdir_cmd"
    if [[ "$SYNC_SELF" == "1" ]]; then
      scp_run "$self_abs" "${SSH_USER}@${host}:${remote_self}" >/dev/null
    fi
  done

  for ((rank=0; rank<NNODES; rank++)); do
    host="${HOSTS_ARR[$rank]}"
    local log_file="${remote_log_dir}/node_rank${rank}.log"
    local remote_cmd
    remote_cmd="cd $(shell_quote "$WORKDIR") && nohup env $(remote_env_assignments "$rank") bash $(shell_quote "$remote_self") --worker > $(shell_quote "$log_file") 2>&1 &"

    if [[ "$DRY_RUN" == "1" ]]; then
      echo "$(ssh_display_cmd) ${SSH_OPTS[*]-} ${SSH_USER}@${host} ${remote_cmd}"
    else
      ssh_run "${SSH_USER}@${host}" "$remote_cmd" &
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
