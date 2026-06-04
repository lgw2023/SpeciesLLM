#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Launcher for scripts/ddp_grad_sync_probe.py.
#
# Single-node 8-card probe:
#   bash scripts/run_ddp_grad_sync_probe.sh single
#
# Three-node 24-card probe, run on the master node:
#   HOSTS="host0,host1,host2" MASTER_ADDR=host0 WORKDIR=/data/disk1/SpeciesLLM \
#   bash scripts/run_ddp_grad_sync_probe.sh multinode
#
# Worker mode is normally started by the multinode launcher:
#   NODE_RANK=1 bash scripts/run_ddp_grad_sync_probe.sh --worker
###############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
SELF_NAME="$(basename "${BASH_SOURCE[0]}")"
ENV_FILE="${ENV_FILE:-$PROJECT_ROOT/.env}"
WORKER_MODE=0
MODE_ARG=""

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

usage() {
  cat <<USAGE
Usage:
  bash scripts/$SELF_NAME single [KEY=VALUE ...]
  bash scripts/$SELF_NAME multinode [KEY=VALUE ...]
  NODE_RANK=1 bash scripts/$SELF_NAME --worker

Common overrides:
  LOAD_ENV_FILE=0|1
  PYTHON_BIN=/data/miniconda3/bin/python
  WORKDIR=/data/disk1/SpeciesLLM
  ASCEND_RT_VISIBLE_DEVICES_VALUE=0,1,2,3,4,5,6,7
  MASTER_ADDR=host0 MASTER_PORT=29591
  HOSTS=host0,host1,host2
  SSH_USER=root SSH_PASSWORD=... or SSH_KEY=/path/to/key
  PROBE_CASES=all
  PROBE_VERBOSE=1
  DRY_RUN=1
USAGE
}

for arg in "$@"; do
  case "$arg" in
    -h|--help|help)
      usage
      exit 0
      ;;
    single|multinode)
      [[ -z "$MODE_ARG" ]] || die "Mode was already set to $MODE_ARG"
      MODE_ARG="$arg"
      ;;
    --worker)
      WORKER_MODE=1
      ;;
    *=*)
      key="${arg%%=*}"
      value="${arg#*=}"
      [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || die "Invalid KEY=VALUE argument: $arg"
      export "$key=$value"
      ;;
    *)
      usage >&2
      die "Unsupported argument: $arg"
      ;;
  esac
done

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

PRELOAD_MODE="${MODE_ARG:-${MODE:-single}}"
[[ "$WORKER_MODE" == "1" ]] && PRELOAD_MODE="worker"
LOAD_ENV_FILE="${LOAD_ENV_FILE:-}"
if [[ -z "$LOAD_ENV_FILE" ]]; then
  if [[ "$PRELOAD_MODE" == "single" ]]; then
    LOAD_ENV_FILE=0
  else
    LOAD_ENV_FILE=1
  fi
fi

case "$LOAD_ENV_FILE" in
  1|true|TRUE|yes|YES|on|ON)
    load_env_defaults "$ENV_FILE"
    ;;
  0|false|FALSE|no|NO|off|OFF)
    ;;
  *)
    die "LOAD_ENV_FILE must be 0 or 1, got: $LOAD_ENV_FILE"
    ;;
esac

MODE="${MODE_ARG:-${MODE:-single}}"
[[ "$WORKER_MODE" == "1" ]] && MODE="worker"

PYTHON_BIN="${PYTHON_BIN:-/data/miniconda3/bin/python}"
WORKDIR="${WORKDIR:-$PROJECT_ROOT}"
PROBE_ENTRY="${PROBE_ENTRY:-scripts/ddp_grad_sync_probe.py}"
LOG_SUBDIR="${LOG_SUBDIR:-ddp_grad_probe_logs}"
DRY_RUN="${DRY_RUN:-0}"
VERIFY_NPU="${VERIFY_NPU:-1}"
HOME="${HOME:-/root}"
export HOME

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29591}"
NODE_RANK="${NODE_RANK:-}"
LOCAL_NODE_RANK="${LOCAL_NODE_RANK:-0}"

if [[ "$MODE" == "single" ]]; then
  NNODES="${NNODES:-1}"
  MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
elif [[ "$MODE" == "multinode" || "$MODE" == "worker" ]]; then
  NNODES="${NNODES:-3}"
  HOSTS_CSV="${HOSTS:-7.150.12.45,7.150.15.14,7.150.14.170}"
  MASTER_ADDR="${MASTER_ADDR:-${HOSTS_CSV%%,*}}"
else
  usage >&2
  die "Unsupported MODE=$MODE"
fi

PROBE_DEVICE="${PROBE_DEVICE:-npu}"
PROBE_BACKEND="${PROBE_BACKEND:-hccl}"
PROBE_CASES="${PROBE_CASES:-all}"
PROBE_VERBOSE="${PROBE_VERBOSE:-1}"
PROBE_GRADIENT_AS_BUCKET_VIEW="${PROBE_GRADIENT_AS_BUCKET_VIEW:-1}"
PROBE_EXTRA_ARGS="${PROBE_EXTRA_ARGS:-}"

ASCEND_RT_VISIBLE_DEVICES_VALUE="${ASCEND_RT_VISIBLE_DEVICES_VALUE:-0,1,2,3,4,5,6,7}"
HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-7200}"
HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-7200}"
HCCL_WHITELIST_DISABLE="${HCCL_WHITELIST_DISABLE:-1}"
ASCEND_TOOLKIT_HOME="${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/ascend-toolkit}"
ASCEND_ENV_SH="${ASCEND_ENV_SH:-}"
ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-$ASCEND_TOOLKIT_HOME}"

SSH_USER="${SSH_USER:-root}"
SSH_KEY="${SSH_KEY:-}"
SSH_PASSWORD="${SSH_PASSWORD:-}"
SSH_EXTRA_OPTS="${SSH_EXTRA_OPTS:-}"
SYNC_SELF="${SYNC_SELF:-1}"

log() {
  echo "[$(date '+%F %T')] $*"
}

shell_quote() {
  local value="$1"
  printf "'%s'" "${value//\'/\'\\\'\'}"
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

print_command() {
  printf '%q ' "$@"
  printf '\n'
}

split_hosts() {
  HOSTS_ARR=()
  local -a raw_hosts
  local raw_host host
  IFS=',' read -r -a raw_hosts <<< "$HOSTS_CSV"
  for raw_host in "${raw_hosts[@]}"; do
    host="${raw_host#"${raw_host%%[![:space:]]*}"}"
    host="${host%"${host##*[![:space:]]}"}"
    [[ -n "$host" ]] && HOSTS_ARR+=("$host")
  done
}

build_ssh_opts() {
  SSH_OPTS=()
  if [[ -n "$SSH_KEY" ]]; then
    SSH_OPTS+=("-i" "$SSH_KEY")
  fi
  if [[ -n "$SSH_EXTRA_OPTS" ]]; then
    read -r -a SSH_EXTRA_OPTS_ARR <<< "$SSH_EXTRA_OPTS"
    SSH_OPTS+=("${SSH_EXTRA_OPTS_ARR[@]}")
  fi
  if [[ -n "$SSH_PASSWORD" && "$DRY_RUN" != "1" ]]; then
    command -v sshpass >/dev/null || die "sshpass is required when SSH_PASSWORD is set"
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
  [[ "$PROBE_DEVICE" == "npu" ]] || return 0
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
  [[ "$PROBE_DEVICE" == "npu" ]] || return 0
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

build_probe_args() {
  PROBE_ARGS=(
    "--device=${PROBE_DEVICE}"
    "--backend=${PROBE_BACKEND}"
    "--cases=${PROBE_CASES}"
  )
  if is_enabled "$PROBE_VERBOSE"; then
    PROBE_ARGS+=("--verbose")
  fi
  if is_enabled "$PROBE_GRADIENT_AS_BUCKET_VIEW"; then
    PROBE_ARGS+=("--gradient-as-bucket-view")
  else
    PROBE_ARGS+=("--no-gradient-as-bucket-view")
  fi
  if [[ -n "$PROBE_EXTRA_ARGS" ]]; then
    read -r -a PROBE_EXTRA_ARGS_ARR <<< "$PROBE_EXTRA_ARGS"
    PROBE_ARGS+=("${PROBE_EXTRA_ARGS_ARR[@]}")
  fi
}

run_worker() {
  [[ -n "$NODE_RANK" ]] || die "NODE_RANK is required in worker mode"
  cd "$WORKDIR"
  [[ -x "$PYTHON_BIN" ]] || die "PYTHON_BIN is not executable: $PYTHON_BIN"
  [[ -f "$PROBE_ENTRY" ]] || die "Missing probe entry: ${WORKDIR}/${PROBE_ENTRY}"
  mkdir -p "$LOG_SUBDIR"

  export HOME NNODES NPROC_PER_NODE NODE_RANK MASTER_ADDR MASTER_PORT
  export ASCEND_RT_VISIBLE_DEVICES="$ASCEND_RT_VISIBLE_DEVICES_VALUE"
  export HCCL_CONNECT_TIMEOUT HCCL_EXEC_TIMEOUT HCCL_WHITELIST_DISABLE
  export ASCEND_TOOLKIT_HOME ASCEND_ENV_SH ASCEND_HOME_PATH

  build_probe_args
  local -a cmd=(
    "$PYTHON_BIN"
    -m
    torch.distributed.run
    "--nproc_per_node=${NPROC_PER_NODE}"
    "--nnodes=${NNODES}"
    "--node_rank=${NODE_RANK}"
    "--master_addr=${MASTER_ADDR}"
    "--master_port=${MASTER_PORT}"
    "$PROBE_ENTRY"
    "${PROBE_ARGS[@]}"
  )

  log "DDP grad sync probe worker"
  log "WORKDIR=${WORKDIR}"
  log "NNODES=${NNODES}, NODE_RANK=${NODE_RANK}, NPROC_PER_NODE=${NPROC_PER_NODE}, MASTER=${MASTER_ADDR}:${MASTER_PORT}"
  log "ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES}"
  log "PROBE_CASES=${PROBE_CASES}"
  log "command:"
  print_command "${cmd[@]}"

  if [[ "$DRY_RUN" == "1" ]]; then
    log "DRY_RUN=1, skip worker execution."
    return 0
  fi

  source_ascend_env
  verify_python_npu
  exec "${cmd[@]}"
}

remote_env_assignments() {
  local rank="$1"
  local -a names=(
    HOME NNODES NPROC_PER_NODE MASTER_ADDR MASTER_PORT WORKDIR PROBE_ENTRY PYTHON_BIN LOG_SUBDIR
    DRY_RUN VERIFY_NPU ASCEND_RT_VISIBLE_DEVICES_VALUE HCCL_CONNECT_TIMEOUT HCCL_EXEC_TIMEOUT
    HCCL_WHITELIST_DISABLE ASCEND_TOOLKIT_HOME ASCEND_ENV_SH ASCEND_HOME_PATH
    PROBE_DEVICE PROBE_BACKEND PROBE_CASES PROBE_VERBOSE PROBE_GRADIENT_AS_BUCKET_VIEW PROBE_EXTRA_ARGS
  )

  printf "NODE_RANK=%s " "$(shell_quote "$rank")"
  local name value
  for name in "${names[@]}"; do
    value="${!name}"
    printf "%s=%s " "$name" "$(shell_quote "$value")"
  done
}

prepare_multinode() {
  split_hosts
  [[ "${#HOSTS_ARR[@]}" -eq "$NNODES" ]] || die "HOSTS count (${#HOSTS_ARR[@]}) must equal NNODES (${NNODES}). HOSTS=${HOSTS_CSV}"
  [[ "$LOCAL_NODE_RANK" =~ ^[0-9]+$ ]] || die "LOCAL_NODE_RANK must be an integer"
  (( LOCAL_NODE_RANK >= 0 && LOCAL_NODE_RANK < NNODES )) || die "LOCAL_NODE_RANK=${LOCAL_NODE_RANK} is outside [0, $((NNODES - 1))]"
  build_ssh_opts
}

run_single() {
  NNODES=1
  NODE_RANK=0
  MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
  run_worker
}

run_multinode() {
  prepare_multinode

  local self_abs="${SCRIPT_DIR}/${SELF_NAME}"
  local probe_abs="${PROJECT_ROOT}/${PROBE_ENTRY}"
  local remote_scripts_dir="${WORKDIR}/scripts"
  local remote_self="${remote_scripts_dir}/${SELF_NAME}"
  local remote_probe="${WORKDIR}/${PROBE_ENTRY}"
  local remote_log_dir="${WORKDIR}/${LOG_SUBDIR}"

  [[ -f "$probe_abs" ]] || die "Missing local probe entry: $probe_abs"

  log "DDP grad sync probe multinode launcher"
  log "HOSTS(rank order)=${HOSTS_CSV}"
  log "NNODES=${NNODES}, NPROC_PER_NODE=${NPROC_PER_NODE}, MASTER=${MASTER_ADDR}:${MASTER_PORT}"
  log "WORKDIR=${WORKDIR}, PROBE_ENTRY=${PROBE_ENTRY}"
  log "LOCAL_NODE_RANK=${LOCAL_NODE_RANK}, SYNC_SELF=${SYNC_SELF}, DRY_RUN=${DRY_RUN}"

  local rank host mkdir_cmd
  for ((rank=0; rank<NNODES; rank++)); do
    host="${HOSTS_ARR[$rank]}"
    mkdir_cmd="mkdir -p $(shell_quote "$remote_log_dir") $(shell_quote "$remote_scripts_dir")"
    log "prepare node_rank=${rank}, host=${host}"

    if [[ "$DRY_RUN" == "1" ]]; then
      if [[ "$rank" == "$LOCAL_NODE_RANK" ]]; then
        echo "local ${mkdir_cmd}"
      else
        echo "ssh ${SSH_USER}@${host} ${mkdir_cmd}"
        if [[ "$SYNC_SELF" == "1" ]]; then
          echo "scp ${self_abs} ${SSH_USER}@${host}:${remote_self}"
          echo "scp ${probe_abs} ${SSH_USER}@${host}:${remote_probe}"
        fi
      fi
      continue
    fi

    if [[ "$rank" == "$LOCAL_NODE_RANK" ]]; then
      mkdir -p "$remote_log_dir" "$remote_scripts_dir"
    else
      ssh_run "${SSH_USER}@${host}" "$mkdir_cmd"
      if [[ "$SYNC_SELF" == "1" ]]; then
        scp_run "$self_abs" "${SSH_USER}@${host}:${remote_self}" >/dev/null
        scp_run "$probe_abs" "${SSH_USER}@${host}:${remote_probe}" >/dev/null
      fi
    fi
  done

  for ((rank=0; rank<NNODES; rank++)); do
    [[ "$rank" == "$LOCAL_NODE_RANK" ]] && continue
    host="${HOSTS_ARR[$rank]}"
    local log_file="${remote_log_dir}/node_rank${rank}.log"
    local remote_cmd
    remote_cmd="cd $(shell_quote "$WORKDIR") && nohup env $(remote_env_assignments "$rank") bash $(shell_quote "$remote_self") --worker > $(shell_quote "$log_file") 2>&1 &"

    if [[ "$DRY_RUN" == "1" ]]; then
      echo "ssh ${SSH_USER}@${host} ${remote_cmd}"
    else
      ssh_run "${SSH_USER}@${host}" "$remote_cmd"
    fi
    log "sent remote start command: rank=${rank}, host=${host}, log=${log_file}"
  done

  NODE_RANK="$LOCAL_NODE_RANK"
  if [[ "$DRY_RUN" == "1" ]]; then
    log "DRY_RUN=1, local worker command would be:"
    run_worker
    return 0
  fi

  log "starting local worker in foreground; rank0 summary is printed here when LOCAL_NODE_RANK=0"
  run_worker
}

case "$MODE" in
  single)
    run_single
    ;;
  multinode)
    run_multinode
    ;;
  worker)
    run_worker
    ;;
esac
