set -euo pipefail

cd /data/disk1/SpeciesLLM

command -v ssh >/dev/null
command -v rsync >/dev/null

ENV_FILE="${ENV_FILE:-.env}"

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

export SSH_USER=root
export SSH_PASSWORD="${SSH_PASSWORD:-}"
if [[ -n "$SSH_PASSWORD" ]]; then
  command -v sshpass >/dev/null
fi

export PYTHON_BIN=/data/miniconda3/bin/python
export PROJECT_ROOT=/data/disk1/SpeciesLLM

export STAGE2_ROOT=/data/disk1/SpeciesLLM_obs/Stage2_SpeciesLLMData
export INPUT_1ST="${STAGE2_ROOT}/1st_pretrain_data_preprocessed_step4"
export INPUT_2ND="${STAGE2_ROOT}/2nd_pretrain_data_preprocessed_step4"
export INPUT_3SC="${STAGE2_ROOT}/3scbasecount_pretrain_data_preprocessed_step4"

export MERGED_TEST_DIR="${STAGE2_ROOT}/all_shuffled_test_500m"
export FLAT_TEST_DIR="${STAGE2_ROOT}/all_flatten_data_test_500m"
export COMMAND_DIR="${STAGE2_ROOT}/stage2_500m_test_commands"

export WORKDIR=/data/disk1/SpeciesLLM
export TRAIN_ENTRY=train_MNodes_torchrun_mfu_preindexparquet.py
export LOG_SUBDIR=torchrun_logs

export EMB_ROOT=/data/disk1/SpeciesLLM
export EMB_PATH=/data/disk1/SpeciesLLM/Stage2_macrogene_embeddings
export MODEL_CONFIG_JSON=/data/disk1/SpeciesLLM/Stage2_macrogene_embeddings/args_2nd_run.json

export NNODES=3
export NPROC_PER_NODE=8
export HOSTS="7.150.12.45,7.150.15.14,7.150.14.170"
export MASTER_ADDR=7.150.12.45
export MASTER_PORT=12345

export TRAIN_DATASET=test
export DATA_PATH="${FLAT_TEST_DIR}"
export NUM_OF_USED_DATA=0
export OUT_PATH=training_output

# 保留你自己改过的训练参数
export BATCH_SIZE=64
export EPOCH=1
export GRADIENT_ACCUMULATION_STEPS=4
export LEARNING_RATE=0.00001
export MIN_LR=0.000001
export DECAY_LR=true
export WARMUP_ITERS=2000
export WARMUP_RATIO=0.05
export WEIGHT_DECAY=0.1
export SAVE_DATA_INTERVAL=5120000
export BETA1=0.9
export BETA2=0.95
export GRAD_CLIP=1.0
export COMPILE=false
export BACKEND=hccl
export DEVICE=npu
export DEVICE_TYPE=npu
export S3_REMOTE_DIR_PATH=""

export ASCEND_RT_VISIBLE_DEVICES_VALUE="0,1,2,3,4,5,6,7"
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_EXEC_TIMEOUT=7200
export HCCL_WHITELIST_DISABLE=1
export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest
export ASCEND_HOME_PATH="${ASCEND_TOOLKIT_HOME}"

ssh_run() {
  local host="$1"
  shift
  if [[ -n "$SSH_PASSWORD" ]]; then
    SSHPASS="$SSH_PASSWORD" sshpass -e ssh "${SSH_USER}@${host}" "$@"
  else
    ssh "${SSH_USER}@${host}" "$@"
  fi
}

rsync_to_host() {
  local source_path="$1"
  local host="$2"
  local target_path="$3"
  shift 3
  if [[ -n "$SSH_PASSWORD" ]]; then
    SSHPASS="$SSH_PASSWORD" rsync -e "sshpass -e ssh" "$@" "$source_path" "${SSH_USER}@${host}:$target_path"
  else
    rsync "$@" "$source_path" "${SSH_USER}@${host}:$target_path"
  fi
}

rsync_from_host() {
  local host="$1"
  local source_path="$2"
  local target_path="$3"
  shift 3
  if [[ -n "$SSH_PASSWORD" ]]; then
    SSHPASS="$SSH_PASSWORD" rsync -e "sshpass -e ssh" "$@" "${SSH_USER}@${host}:$source_path" "$target_path"
  else
    rsync "$@" "${SSH_USER}@${host}:$source_path" "$target_path"
  fi
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

remote_mkdirs() {
  local host="$1"
  ssh_run "$host" "
    set -e
    mkdir -p '$PROJECT_ROOT' '$EMB_PATH' '$FLAT_TEST_DIR' '$COMMAND_DIR' '$WORKDIR/$LOG_SUBDIR' '$WORKDIR/training_output'
  "
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
    sync_dir_to_host "$PROJECT_ROOT" "$host" "$PROJECT_ROOT" \
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
    sync_dir_to_host "$EMB_PATH" "$host" "$EMB_PATH"

    echo "[SYNC] training data -> ${host}:${DATA_PATH}"
    sync_dir_to_host "$DATA_PATH" "$host" "$DATA_PATH"

    echo "[SYNC] generated command files -> ${host}:${COMMAND_DIR}"
    sync_dir_to_host "$COMMAND_DIR" "$host" "$COMMAND_DIR"
  done
}

check_remote_paths() {
  split_hosts

  local host
  for host in "${HOSTS_ARR[@]}"; do
    echo "===== ${host} ====="
    ssh_run "$host" "
      set -e
      test -f '$WORKDIR/train_MNodes_torchrun_mfu_preindexparquet.py'
      test -f '$MODEL_CONFIG_JSON'
      test -d '$DATA_PATH'
      cd '$WORKDIR'
      git rev-parse --short HEAD
      ls '$DATA_PATH'/*.parquet | wc -l
    "
  done
}

collect_training_outputs() {
  split_hosts

  local host
  for host in "${HOSTS_ARR[@]}"; do
    echo "[COLLECT] ${host}"
    if ssh_run "$host" "test -d '${WORKDIR%/}/training_output'"; then
      rsync_from_host \
        "$host" \
        "${WORKDIR%/}/training_output/" \
        "${WORKDIR%/}/training_output/" \
        -aH
    fi
    if ssh_run "$host" "test -d '${WORKDIR%/}/${LOG_SUBDIR}'"; then
      rsync_from_host \
        "$host" \
        "${WORKDIR%/}/${LOG_SUBDIR}/" \
        "${WORKDIR%/}/${LOG_SUBDIR}/" \
        -aH
    fi
  done
}

# 已经生成好测试数据时用 commands；需要重新生成测试数据时：
#   PREP_ACTION=all bash work.sh
#
# 训练脚本通过 nohup 在远端后台运行；训练完成后收集各节点输出：
#   COLLECT_ONLY=1 bash work.sh
if [[ "${COLLECT_ONLY:-0}" == "1" ]]; then
  collect_training_outputs
  bash scripts/test_stage2_500m_multinode.sh check-training
  exit 0
fi

PREP_ACTION="${PREP_ACTION:-commands}"
bash scripts/test_stage2_500m_multinode.sh "$PREP_ACTION"

sync_code_and_data_to_workers
check_remote_paths

# 检查 dry-run，确认每个节点都会使用自己的 /data/disk1 路径。
DRY_RUN=1 bash "${COMMAND_DIR}/launch_500m_3nodes.sh"

if [[ "${RUN_TRAINING:-1}" != "1" ]]; then
  echo "[INFO] RUN_TRAINING is not 1; stop after sync, path check, and dry-run."
  exit 0
fi

bash "${COMMAND_DIR}/launch_500m_3nodes.sh"

echo "[INFO] Training has been launched through nohup on remote nodes."
echo "[INFO] Monitor logs under each node: ${WORKDIR}/${LOG_SUBDIR}/node_rank*.log"
echo "[INFO] After training finishes, run: COLLECT_ONLY=1 bash work.sh"
