#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" != "--__speciesllm-smoke-clean-env" ]]; then
  exec /usr/bin/env -i \
    HOME="${HOME:-/root}" \
    PATH="/data/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
    LANG=C.UTF-8 \
    bash "$0" --__speciesllm-smoke-clean-env "$@"
fi
shift

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/smoke_500m_3node.sh [KEY=VALUE ...]

Examples:
  bash scripts/smoke_500m_3node.sh
  bash scripts/smoke_500m_3node.sh PREP_ACTION=all
  bash scripts/smoke_500m_3node.sh RUN_TRAINING=0
  bash scripts/smoke_500m_3node.sh COLLECT_ONLY=1
  bash scripts/smoke_500m_3node.sh COLLECT_ONLY=1 TRAIN_OUT_DIR=/data/disk1/SpeciesLLM/training_output
  bash scripts/smoke_500m_3node.sh SSH_PASSWORD='your-password' PREP_ACTION=commands
  bash scripts/smoke_500m_3node.sh SSH_KEY=/path/to/id_ed25519 PREP_ACTION=commands
  bash scripts/smoke_500m_3node.sh BATCH_SIZE=16 PREP_ACTION=commands SKIP_DATA_SYNC=1
  bash scripts/smoke_500m_3node.sh MODEL_SCALE=100m PREP_ACTION=all
  bash scripts/smoke_500m_3node.sh MODEL_SCALE=1b PREP_ACTION=all

Notes:
  - The script re-execs itself with env -i, so exported shell variables are ignored.
  - .env is loaded explicitly after env -i and before command-line overrides.
  - Runtime overrides must be passed after the script path as KEY=VALUE arguments.
  - Use ENV_FILE=/path/to/env after the script path to load a different env file.
  - Do not use KEY=VALUE before the script path; those inherited environment values are
    intentionally discarded.
USAGE
}

is_allowed_cli_var() {
  case "$1" in
    ENV_FILE|COLLECT_ONLY|PREP_ACTION|RUN_TRAINING|SKIP_DATA_SYNC|MODEL_SCALE|SSH_USER|SSH_KEY|\
    SSH_PASSWORD|SSH_EXTRA_OPTS|PYTHON_BIN|PROJECT_ROOT|PRETRAIN_CHECKS_PY|STAGE2_CHECKS_PY|\
    STAGE2_ROOT|INPUT_1ST|INPUT_2ND|INPUT_3SC|MERGED_TEST_DIR|FLAT_TEST_DIR|\
    COMMAND_DIR|WORKDIR|TRAIN_ENTRY|LOG_SUBDIR|TRAIN_OUTPUT_ROOT|TRAIN_OUT_DIR|NODE_LOG_DIR|EMB_ROOT|\
    EMB_PATH|MODEL_CONFIG_JSON|NNODES|NPROC_PER_NODE|HOSTS|MASTER_ADDR|\
    MASTER_PORT|TRAIN_DATASET|DATA_PATH|NUM_OF_USED_DATA|OUT_PATH|WORKERS|\
    FLATTEN_WORKERS|ROWS_PER_FILE|SHUFFLE_SEED|FLATTEN_DROP_REMAINDER|\
    RESET_TEST_OUTPUT|SKIP_EXISTING|\
    SOURCE_PREFLIGHT_FILES_PER_BATCH|SOURCE_PREFLIGHT_MAX_SCAN|\
    MAX_VALIDATE_ROWS_PER_FILE|MAX_VALIDATE_FILES|REQUIRE_EMBEDDINGS|BATCH_SIZE|\
    EPOCH|GRADIENT_ACCUMULATION_STEPS|LEARNING_RATE|MIN_LR|DECAY_LR|WARMUP_ITERS|\
    GRADIENT_CHECKPOINTING|\
    WARMUP_RATIO|WEIGHT_DECAY|SAVE_DATA_INTERVAL|BETA1|BETA2|GRAD_CLIP|\
    ADAPTIVE_GRAD_CLIP|GRAD_CLIP_EMA_BETA|GRAD_CLIP_RATIO|GRAD_SKIP_RATIO|\
    GRAD_CLIP_MIN|GRAD_CLIP_MAX|GRAD_CLIP_WARMUP_STEPS|COMPILE|\
    BACKEND|DEVICE|DEVICE_TYPE|S3_REMOTE_DIR_PATH|LOG_INTERVAL|PROFILE_INTERVAL|\
    NAN_CHECK_INTERVAL|    METRICS_FLUSH_INTERVAL|LOG_LEVEL|LOG_ALL_RANKS|\
    NUM_WORKERS|PREFETCH_FACTOR|PERSISTENT_WORKERS|PIN_MEMORY|\
    ASCEND_RT_VISIBLE_DEVICES_VALUE|HCCL_CONNECT_TIMEOUT|HCCL_EXEC_TIMEOUT|\
    HCCL_WHITELIST_DISABLE|ASCEND_TOOLKIT_HOME|ASCEND_ENV_SH|ASCEND_HOME_PATH|HOME)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

CLI_KEYS=()
CLI_VALUES=()
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
      if ((${#CLI_KEYS[@]} > 0)); then
        for existing_key in "${CLI_KEYS[@]}"; do
          [[ "$existing_key" != "$key" ]] || die "Duplicate runtime variable: $key"
        done
      fi
      CLI_KEYS+=("$key")
      CLI_VALUES+=("$value")
      ;;
    *)
      usage >&2
      die "Unsupported argument: $arg. Use KEY=VALUE arguments after the script path."
      ;;
  esac
done

get_cli_value() {
  local name="$1"
  local i
  for i in "${!CLI_KEYS[@]}"; do
    if [[ "${CLI_KEYS[$i]}" == "$name" ]]; then
      printf "%s" "${CLI_VALUES[$i]}"
      return 0
    fi
  done
  return 1
}

ENV_KEYS=()
ENV_VALUES=()

load_env_defaults() {
  local env_file="$1"
  [[ -f "$env_file" ]] || return 0

  local line key value existing_key
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

    [[ "$key" =~ ^[A-Z_][A-Z0-9_]*$ ]] || die "Invalid variable name in $env_file: $key"
    is_allowed_cli_var "$key" || die "Unsupported variable in $env_file: $key"

    if [[ "$value" == \"*\" && "$value" == *\" && ${#value} -ge 2 ]]; then
      value="${value:1:${#value}-2}"
    elif [[ "$value" == \'*\' && "$value" == *\' && ${#value} -ge 2 ]]; then
      value="${value:1:${#value}-2}"
    fi

    if ((${#ENV_KEYS[@]} > 0)); then
      for existing_key in "${ENV_KEYS[@]}"; do
        [[ "$existing_key" != "$key" ]] || die "Duplicate variable in $env_file: $key"
      done
    fi
    ENV_KEYS+=("$key")
    ENV_VALUES+=("$value")
  done < "$env_file"
}

get_env_value() {
  local name="$1"
  local i
  for i in "${!ENV_KEYS[@]}"; do
    if [[ "${ENV_KEYS[$i]}" == "$name" ]]; then
      printf "%s" "${ENV_VALUES[$i]}"
      return 0
    fi
  done
  return 1
}

set_default() {
  local name="$1"
  local default_value="$2"
  local cli_value
  if cli_value="$(get_cli_value "$name")"; then
    printf -v "$name" "%s" "$cli_value"
  elif cli_value="$(get_env_value "$name")"; then
    printf -v "$name" "%s" "$cli_value"
  else
    printf -v "$name" "%s" "$default_value"
  fi
  export "$name"
}

cd /data/disk1/SpeciesLLM

command -v ssh >/dev/null
command -v rsync >/dev/null

set_default ENV_FILE .env
load_env_defaults "$ENV_FILE"

set_default HOME "${HOME:-/root}"
set_default COLLECT_ONLY 0
set_default PREP_ACTION commands
set_default RUN_TRAINING 1
set_default SKIP_DATA_SYNC 0
set_default MODEL_SCALE 500m

set_default SSH_USER root
set_default SSH_KEY ""
set_default SSH_PASSWORD ""
set_default SSH_EXTRA_OPTS ""
if [[ -n "$SSH_PASSWORD" ]]; then
  command -v sshpass >/dev/null
fi
SSH_OPTS=()
if [[ -n "$SSH_KEY" ]]; then
  SSH_OPTS+=("-i" "$SSH_KEY")
fi
if [[ -n "$SSH_EXTRA_OPTS" ]]; then
  read -r -a SSH_EXTRA_OPTS_ARR <<< "$SSH_EXTRA_OPTS"
  SSH_OPTS+=("${SSH_EXTRA_OPTS_ARR[@]}")
fi

set_default PYTHON_BIN /data/miniconda3/bin/python
set_default PROJECT_ROOT /data/disk1/SpeciesLLM
set_default PRETRAIN_CHECKS_PY "${PROJECT_ROOT}/scripts/pretrain_checks.py"
if ! get_cli_value PRETRAIN_CHECKS_PY >/dev/null && ! get_env_value PRETRAIN_CHECKS_PY >/dev/null; then
  if legacy_checks_py="$(get_cli_value STAGE2_CHECKS_PY)"; then
    PRETRAIN_CHECKS_PY="$legacy_checks_py"
  elif legacy_checks_py="$(get_env_value STAGE2_CHECKS_PY)"; then
    PRETRAIN_CHECKS_PY="$legacy_checks_py"
  fi
fi
export PRETRAIN_CHECKS_PY

set_default STAGE2_ROOT /data/disk1/SpeciesLLM_obs/Stage2_SpeciesLLMData
set_default INPUT_1ST "${STAGE2_ROOT}/1st_pretrain_data_preprocessed_step4"
set_default INPUT_2ND "${STAGE2_ROOT}/2nd_pretrain_data_preprocessed_step4"
set_default INPUT_3SC "${STAGE2_ROOT}/3scbasecount_pretrain_data_preprocessed_step4"

set_default MERGED_TEST_DIR "${STAGE2_ROOT}/all_shuffled_test_${MODEL_SCALE}"
set_default FLAT_TEST_DIR "${STAGE2_ROOT}/all_flatten_data_test_${MODEL_SCALE}"
set_default COMMAND_DIR "${STAGE2_ROOT}/pretrain_${MODEL_SCALE}_test_commands"

set_default WORKDIR /data/disk1/SpeciesLLM
set_default TRAIN_ENTRY train_MNodes_torchrun_mfu_preindexparquet.py
set_default LOG_SUBDIR torchrun_logs
set_default TRAIN_OUTPUT_ROOT training_output
set_default TRAIN_OUT_DIR ""
set_default NODE_LOG_DIR ""

set_default EMB_ROOT /data/disk1/SpeciesLLM
set_default EMB_PATH /data/disk1/SpeciesLLM/Stage2_macrogene_embeddings
MODEL_CONFIG_JSON_DEFAULT="${EMB_PATH}/args_2nd_run.json"
MODEL_SCALE_LC="$(printf '%s' "$MODEL_SCALE" | tr '[:upper:]' '[:lower:]')"
case "$MODEL_SCALE_LC" in
  100m)
    MODEL_CONFIG_JSON_DEFAULT="${EMB_PATH}/args_2nd_run_100m.json"
    ;;
  1b)
    MODEL_CONFIG_JSON_DEFAULT="${EMB_PATH}/args_2nd_run_1b.json"
    ;;
esac
set_default MODEL_CONFIG_JSON "$MODEL_CONFIG_JSON_DEFAULT"

set_default NNODES 3
set_default NPROC_PER_NODE 8
set_default HOSTS "7.150.12.45,7.150.15.14,7.150.14.170"
set_default MASTER_ADDR 7.150.12.45
set_default MASTER_PORT 12345

set_default TRAIN_DATASET test
set_default DATA_PATH "${FLAT_TEST_DIR}"
set_default NUM_OF_USED_DATA 0
set_default OUT_PATH training_output

set_default WORKERS 16
set_default FLATTEN_WORKERS "$WORKERS"
set_default ROWS_PER_FILE 16384
set_default SHUFFLE_SEED 42
set_default FLATTEN_DROP_REMAINDER 1
set_default RESET_TEST_OUTPUT 0
set_default SKIP_EXISTING 1
set_default SOURCE_PREFLIGHT_FILES_PER_BATCH 3
set_default SOURCE_PREFLIGHT_MAX_SCAN 200
set_default MAX_VALIDATE_ROWS_PER_FILE 16
set_default MAX_VALIDATE_FILES 0
set_default REQUIRE_EMBEDDINGS 1

set_default BATCH_SIZE 16
set_default EPOCH 1
set_default GRADIENT_ACCUMULATION_STEPS 4
set_default GRADIENT_CHECKPOINTING true
set_default LEARNING_RATE 0.00001
set_default MIN_LR 0.000001
set_default DECAY_LR true
set_default WARMUP_ITERS 2000
set_default WARMUP_RATIO 0.05
set_default WEIGHT_DECAY 0.1
set_default SAVE_DATA_INTERVAL 5120000
set_default BETA1 0.9
set_default BETA2 0.95
set_default GRAD_CLIP 1.0
set_default ADAPTIVE_GRAD_CLIP true
set_default GRAD_CLIP_EMA_BETA 0.98
set_default GRAD_CLIP_RATIO 3.0
set_default GRAD_SKIP_RATIO 10.0
set_default GRAD_CLIP_MIN 0.5
set_default GRAD_CLIP_MAX 1000.0
set_default GRAD_CLIP_WARMUP_STEPS 200
set_default COMPILE false
set_default BACKEND hccl
set_default DEVICE npu
set_default DEVICE_TYPE npu
set_default S3_REMOTE_DIR_PATH ""
set_default LOG_INTERVAL 10
set_default PROFILE_INTERVAL 100
set_default NAN_CHECK_INTERVAL 0
set_default METRICS_FLUSH_INTERVAL 100
set_default LOG_LEVEL INFO
set_default LOG_ALL_RANKS false

set_default NUM_WORKERS 8
set_default PREFETCH_FACTOR 1
set_default PERSISTENT_WORKERS true
set_default PIN_MEMORY true

set_default ASCEND_RT_VISIBLE_DEVICES_VALUE "0,1,2,3,4,5,6,7"
set_default HCCL_CONNECT_TIMEOUT 7200
set_default HCCL_EXEC_TIMEOUT 7200
set_default HCCL_WHITELIST_DISABLE 1
set_default ASCEND_TOOLKIT_HOME /usr/local/Ascend/ascend-toolkit/latest
set_default ASCEND_ENV_SH ""
set_default ASCEND_HOME_PATH "${ASCEND_TOOLKIT_HOME}"

ssh_run() {
  local host="$1"
  shift
  if [[ -n "$SSH_PASSWORD" ]]; then
    SSHPASS="$SSH_PASSWORD" sshpass -e ssh ${SSH_OPTS[@]+"${SSH_OPTS[@]}"} "${SSH_USER}@${host}" "$@"
  else
    ssh ${SSH_OPTS[@]+"${SSH_OPTS[@]}"} "${SSH_USER}@${host}" "$@"
  fi
}

rsync_ssh_cmd() {
  local -a cmd
  if [[ -n "$SSH_PASSWORD" ]]; then
    cmd=(sshpass -e ssh)
  else
    cmd=(ssh)
  fi
  if ((${#SSH_OPTS[@]} > 0)); then
    cmd+=("${SSH_OPTS[@]}")
  fi
  printf "%q " "${cmd[@]}"
}

rsync_to_host() {
  local source_path="$1"
  local host="$2"
  local target_path="$3"
  shift 3
  if [[ -n "$SSH_PASSWORD" ]]; then
    SSHPASS="$SSH_PASSWORD" rsync -e "$(rsync_ssh_cmd)" "$@" "$source_path" "${SSH_USER}@${host}:$target_path"
  else
    rsync -e "$(rsync_ssh_cmd)" "$@" "$source_path" "${SSH_USER}@${host}:$target_path"
  fi
}

rsync_from_host() {
  local host="$1"
  local source_path="$2"
  local target_path="$3"
  shift 3
  if [[ -n "$SSH_PASSWORD" ]]; then
    SSHPASS="$SSH_PASSWORD" rsync -e "$(rsync_ssh_cmd)" "$@" "${SSH_USER}@${host}:$source_path" "$target_path"
  else
    rsync -e "$(rsync_ssh_cmd)" "$@" "${SSH_USER}@${host}:$source_path" "$target_path"
  fi
}

shell_quote() {
  local value="$1"
  printf "'%s'" "$(printf "%s" "$value" | sed "s/'/'\\\\''/g")"
}

resolve_workdir_path() {
  local path="$1"
  case "$path" in
    /*)
      printf "%s\n" "$path"
      ;;
    *)
      printf "%s/%s\n" "${WORKDIR%/}" "$path"
      ;;
  esac
}

split_hosts() {
  HOSTS_ARR=()
  local raw_host host
  IFS=',' read -r -a HOSTS_ARR_RAW <<< "$HOSTS"
  for raw_host in "${HOSTS_ARR_RAW[@]}"; do
    host="${raw_host#"${raw_host%%[![:space:]]*}"}"
    host="${host%"${host##*[![:space:]]}"}"
    [[ -n "$host" ]] && HOSTS_ARR+=("$host")
  done
}

validate_project_paths() {
  [[ -d "$PROJECT_ROOT" ]] || die "Missing PROJECT_ROOT: $PROJECT_ROOT"
  [[ -f "$PROJECT_ROOT/$TRAIN_ENTRY" ]] || die "Missing training entry under PROJECT_ROOT: $PROJECT_ROOT/$TRAIN_ENTRY"
  if [[ ! -f "$WORKDIR/$TRAIN_ENTRY" ]]; then
    if [[ "$WORKDIR" != "$PROJECT_ROOT" ]]; then
      die "WORKDIR does not contain the training entry: $WORKDIR/$TRAIN_ENTRY. PROJECT_ROOT is $PROJECT_ROOT. Update .env WORKDIR=$PROJECT_ROOT or override this run with WORKDIR=$PROJECT_ROOT."
    fi
    die "Missing training entry under WORKDIR: $WORKDIR/$TRAIN_ENTRY"
  fi
}

path_preflight_local() {
  validate_project_paths
  [[ -f "$MODEL_CONFIG_JSON" ]] || die "Missing model config on master: $MODEL_CONFIG_JSON"
  [[ -d "$DATA_PATH" ]] || die "Missing training data directory on master: $DATA_PATH"

  local git_rev parquet_count
  git_rev="$(git -C "$WORKDIR" rev-parse --short HEAD 2>/dev/null || true)"
  [[ -n "$git_rev" ]] || git_rev="no-git"
  parquet_count="$(find "$DATA_PATH" -maxdepth 1 -type f -name '*.parquet' | wc -l | tr -d '[:space:]')"
  [[ "$parquet_count" != "0" ]] || die "No parquet files found on master under: $DATA_PATH"

  echo "git: ${git_rev}"
  echo "parquet files: ${parquet_count}"
}

remote_path_preflight_script() {
  cat <<'BASH'
set -e
train_entry="$1"
model_config="$2"
data_path="$3"
workdir="$4"

[[ -f "${workdir}/${train_entry}" ]] || {
  echo "[ERROR] Missing training entry: ${workdir}/${train_entry}" >&2
  exit 1
}
[[ -f "$model_config" ]] || {
  echo "[ERROR] Missing model config: $model_config" >&2
  exit 1
}
[[ -d "$data_path" ]] || {
  echo "[ERROR] Missing training data directory: $data_path" >&2
  exit 1
}

git_rev="$(git -C "$workdir" rev-parse --short HEAD 2>/dev/null || true)"
[[ -n "$git_rev" ]] || git_rev="no-git"
parquet_count="$(find "$data_path" -maxdepth 1 -type f -name '*.parquet' | wc -l | tr -d '[:space:]')"
[[ "$parquet_count" != "0" ]] || {
  echo "[ERROR] No parquet files found under: $data_path" >&2
  exit 1
}

echo "git: ${git_rev}"
echo "parquet files: ${parquet_count}"
BASH
}

remote_mkdirs() {
  local host="$1"
  ssh_run "$host" "
    set -e
    mkdir -p '$PROJECT_ROOT' '$EMB_PATH' '$FLAT_TEST_DIR' '$COMMAND_DIR' '$WORKDIR/$LOG_SUBDIR' '$WORKDIR/training_output'
  "
}

local_mkdirs() {
  mkdir -p "$PROJECT_ROOT" "$EMB_PATH" "$FLAT_TEST_DIR" "$COMMAND_DIR" "$WORKDIR/$LOG_SUBDIR" "$WORKDIR/training_output"
}

sync_dir_to_host() {
  local source_dir="$1"
  local host="$2"
  local target_dir="$3"
  shift 3

  test -d "$source_dir"
  rsync_to_host \
    "${source_dir%/}/" \
    "$host" \
    "${target_dir%/}/" \
    -aH --info=progress2 --delete "$@"
}

sync_dir_to_host_quiet() {
  local source_dir="$1"
  local host="$2"
  local target_dir="$3"
  shift 3

  test -d "$source_dir"
  rsync_to_host \
    "${source_dir%/}/" \
    "$host" \
    "${target_dir%/}/" \
    -aH --delete "$@"
}

sync_code_and_data_to_workers() {
  split_hosts

  local host
  for host in "${HOSTS_ARR[@]}"; do
    if [[ "$host" == "$MASTER_ADDR" ]]; then
      echo "[SYNC] skip master host ${host}; source paths are already local"
      continue
    fi

    echo "[SYNC] prepare ${host}"
    remote_mkdirs "$host"

    echo "[SYNC] project code -> ${host}:${PROJECT_ROOT}"
    sync_dir_to_host_quiet "$PROJECT_ROOT" "$host" "$PROJECT_ROOT" \
      --exclude '/Stage2_macrogene_embeddings/' \
      --exclude '/Stage2_SpeciesLLMData/' \
      --exclude '/training_output/' \
      --exclude '/torchrun_logs/' \
      --exclude '/papers/' \
      --exclude '/.venv/' \
      --exclude '/venv/' \
      --exclude '/__pycache__/' \
      --exclude '*.pt' \
      --exclude '*.pth' \
      --exclude '*.ckpt'

    echo "[SYNC] embeddings -> ${host}:${EMB_PATH}"
    sync_dir_to_host_quiet "$EMB_PATH" "$host" "$EMB_PATH"

    if [[ "$SKIP_DATA_SYNC" == "1" ]]; then
      echo "[SYNC] skip training data for ${host}; expect existing path ${DATA_PATH}"
    else
      echo "[SYNC] training data -> ${host}:${DATA_PATH}"
      sync_dir_to_host "$DATA_PATH" "$host" "$DATA_PATH"
    fi

    echo "[SYNC] generated command files -> ${host}:${COMMAND_DIR}"
    sync_dir_to_host_quiet "$COMMAND_DIR" "$host" "$COMMAND_DIR"
  done
}

check_remote_paths() {
  split_hosts

  local host
  for host in "${HOSTS_ARR[@]}"; do
    echo "===== ${host} ====="
    if [[ "$host" == "$MASTER_ADDR" ]]; then
      path_preflight_local
      continue
    fi

    remote_path_preflight_script | ssh_run "$host" bash -s -- "$TRAIN_ENTRY" "$MODEL_CONFIG_JSON" "$DATA_PATH" "$WORKDIR"
  done
}

resolve_train_out_dir() {
  if [[ -n "$TRAIN_OUT_DIR" ]]; then
    resolve_workdir_path "$TRAIN_OUT_DIR"
    return 0
  fi

  [[ -f "$PRETRAIN_CHECKS_PY" ]] || die "Missing pretraining helper: $PRETRAIN_CHECKS_PY"
  [[ -f "$MODEL_CONFIG_JSON" ]] || die "Missing model config for output path resolution: $MODEL_CONFIG_JSON"

  "$PYTHON_BIN" "$PRETRAIN_CHECKS_PY" resolve-out-dir \
    --out-path "$OUT_PATH" \
    --workdir "$WORKDIR" \
    --config-json "$MODEL_CONFIG_JSON" \
    --learning-rate "$LEARNING_RATE" \
    --min-lr "$MIN_LR" \
    --weight-decay "$WEIGHT_DECAY" \
    --warmup-ratio "$WARMUP_RATIO"
}

collect_training_outputs() {
  split_hosts

  local train_out_dir node_log_dir host quoted_path
  train_out_dir="$(resolve_train_out_dir)"
  if [[ -n "$NODE_LOG_DIR" ]]; then
    node_log_dir="$(resolve_workdir_path "$NODE_LOG_DIR")"
  else
    node_log_dir="${WORKDIR%/}/${LOG_SUBDIR}"
  fi

  export TRAIN_OUT_DIR="$train_out_dir"
  export NODE_LOG_DIR="$node_log_dir"

  mkdir -p "$train_out_dir" "$node_log_dir"

  echo "[COLLECT] training output dir: ${train_out_dir}"
  echo "[COLLECT] launcher log dir: ${node_log_dir}"

  for host in "${HOSTS_ARR[@]}"; do
    echo "[COLLECT] ${host}"

    quoted_path="$(shell_quote "$train_out_dir")"
    if ssh_run "$host" "test -d ${quoted_path}"; then
      rsync_from_host \
        "$host" \
        "${train_out_dir%/}/" \
        "${train_out_dir%/}/" \
        -aH
    else
      echo "[WARN] missing training output dir on ${host}: ${train_out_dir}"
    fi

    quoted_path="$(shell_quote "$node_log_dir")"
    if ssh_run "$host" "test -d ${quoted_path}"; then
      rsync_from_host \
        "$host" \
        "${node_log_dir%/}/" \
        "${node_log_dir%/}/" \
        -aH
    else
      echo "[WARN] missing launcher log dir on ${host}: ${node_log_dir}"
    fi
  done
}

# 已经生成好测试数据时用 commands；需要重新生成测试数据时：
#   bash scripts/smoke_500m_3node.sh PREP_ACTION=all
#
# 训练脚本通过 nohup 在远端后台运行；训练完成后收集各节点输出：
#   bash scripts/smoke_500m_3node.sh COLLECT_ONLY=1
if [[ "$COLLECT_ONLY" == "1" ]]; then
  collect_training_outputs
  bash scripts/pretrain_pipeline.sh check-training
  exit 0
fi

validate_project_paths

bash scripts/pretrain_pipeline.sh "$PREP_ACTION"

local_mkdirs
sync_code_and_data_to_workers
check_remote_paths

LAUNCH_SCRIPT="${COMMAND_DIR}/launch_${MODEL_SCALE}_3nodes.sh"
[[ -f "$LAUNCH_SCRIPT" ]] || die "Missing generated launcher: $LAUNCH_SCRIPT"

# 检查 dry-run，确认每个节点都会使用自己的 /data/disk1 路径。
DRY_RUN=1 bash "$LAUNCH_SCRIPT"

if [[ "$RUN_TRAINING" != "1" ]]; then
  echo "[INFO] RUN_TRAINING is not 1; stop after sync, path check, and dry-run."
  exit 0
fi

bash "$LAUNCH_SCRIPT"

echo "[INFO] Training has been launched through nohup on remote nodes."
echo "[INFO] Monitor logs under each node: ${WORKDIR}/${LOG_SUBDIR}/node_rank*.log"
echo "[INFO] After training finishes, run: bash scripts/smoke_500m_3node.sh COLLECT_ONLY=1"
