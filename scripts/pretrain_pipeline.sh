#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Pretraining data + 500M three-node training helper.
#
# This script intentionally uses merge_macrogene_rounds.py --test-mode
# for the small sample. It does not create synthetic parquet inputs.
###############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PRETRAIN_CHECKS_PY="${PRETRAIN_CHECKS_PY:-${STAGE2_CHECKS_PY:-${SCRIPT_DIR}/pretrain_checks.py}}"
HOME="${HOME:-/root}"
export HOME

SERVER_STAGE2_ROOT_DEFAULT="/data/disk1/SpeciesLLM_obs/Stage2_SpeciesLLMData"
if [[ -d "$SERVER_STAGE2_ROOT_DEFAULT" ]]; then
  STAGE2_ROOT="${STAGE2_ROOT:-$SERVER_STAGE2_ROOT_DEFAULT}"
else
  STAGE2_ROOT="${STAGE2_ROOT:-${PROJECT_ROOT}/Stage2_SpeciesLLMData}"
fi

INPUT_1ST="${INPUT_1ST:-${STAGE2_ROOT}/1st_pretrain_data_preprocessed_step4}"
INPUT_2ND="${INPUT_2ND:-${STAGE2_ROOT}/2nd_pretrain_data_preprocessed_step4}"
INPUT_3SC="${INPUT_3SC:-${STAGE2_ROOT}/3scbasecount_pretrain_data_preprocessed_step4}"

MERGED_TEST_DIR="${MERGED_TEST_DIR:-${STAGE2_ROOT}/all_shuffled_test_500m}"
FLAT_TEST_DIR="${FLAT_TEST_DIR:-${STAGE2_ROOT}/all_flatten_data_test_500m}"
COMMAND_DIR="${COMMAND_DIR:-${STAGE2_ROOT}/pretrain_500m_test_commands}"
TRAIN_OUTPUT_ROOT="${TRAIN_OUTPUT_ROOT:-training_output}"

NNODES="${NNODES:-3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
WORLD_SIZE=$((NNODES * NPROC_PER_NODE))
HOSTS="${HOSTS:-7.150.12.45,7.150.15.14,7.150.14.170}"
MASTER_ADDR="${MASTER_ADDR:-${HOSTS%%,*}}"
MASTER_PORT="${MASTER_PORT:-12345}"
WORKDIR="${WORKDIR:-$PROJECT_ROOT}"
TRAIN_ENTRY="${TRAIN_ENTRY:-train_MNodes_torchrun_mfu_preindexparquet.py}"
LOG_SUBDIR="${LOG_SUBDIR:-torchrun_logs}"

EMB_ROOT="${EMB_ROOT:-$WORKDIR}"
EMB_PATH="${EMB_PATH:-${EMB_ROOT}/Stage2_macrogene_embeddings}"
# 固定模型配置文件。与训练入口参数同名/同义的模型结构、label 开关、label 数量都从这里读取；
# 后续如果要改 100M/500M/1B 或 label 规模，只改这个 JSON，不需要改本脚本。
MODEL_CONFIG_JSON="${MODEL_CONFIG_JSON:-${EMB_PATH}/args_2nd_run.json}"

WORKERS="${WORKERS:-16}"
FLATTEN_WORKERS="${FLATTEN_WORKERS:-$WORKERS}"
ROWS_PER_FILE="${ROWS_PER_FILE:-16384}"
SHUFFLE_SEED="${SHUFFLE_SEED:-42}"
FLATTEN_DROP_REMAINDER="${FLATTEN_DROP_REMAINDER:-1}"
RESET_TEST_OUTPUT="${RESET_TEST_OUTPUT:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

# 每个输入批次目录在 preflight 阶段至少抽查多少个有效 parquet 文件。
# 这里的批次指 1st / 2nd / 3scbasecount 三个 step4 输入目录。
SOURCE_PREFLIGHT_FILES_PER_BATCH="${SOURCE_PREFLIGHT_FILES_PER_BATCH:-3}"
# preflight 最多向后扫描多少个 macrogene_*.parquet，用来跳过本机常见的 0 字节占位文件。
# 如果前 200 个文件里找不到足够的有效文件，就提前失败，提示应在真实数据服务器上运行。
SOURCE_PREFLIGHT_MAX_SCAN="${SOURCE_PREFLIGHT_MAX_SCAN:-200}"
# validate-data 阶段每个扁平化 parquet 最多抽查多少行；只读少量行以避免本地/服务器 smoke test 太重。
MAX_VALIDATE_ROWS_PER_FILE="${MAX_VALIDATE_ROWS_PER_FILE:-16}"
# validate-data 阶段最多校验多少个 all_flatten_part_*.parquet；0 表示校验所有扁平化文件。
MAX_VALIDATE_FILES="${MAX_VALIDATE_FILES:-0}"
# 是否强制检查 Stage2_macrogene_embeddings 下三类 macrogene embedding 文件存在且第一维等于 SEQ_LEN。
REQUIRE_EMBEDDINGS="${REQUIRE_EMBEDDINGS:-1}"

# 传给 scripts/launch_multinode_torchrun.sh 的数据集标识；这里默认使用测试扁平化数据，而不是 full 全量数据。
TRAIN_DATASET="${TRAIN_DATASET:-test}"
# 训练脚本实际读取的扁平化 parquet 目录，默认就是本脚本生成的 all_flatten_data_test_500m。
DATA_PATH="${DATA_PATH:-$FLAT_TEST_DIR}"
# 限制训练使用前 N 个 parquet 文件；0 表示使用 DATA_PATH 下所有 parquet 文件。
NUM_OF_USED_DATA="${NUM_OF_USED_DATA:-0}"
# 训练输出目录；相对路径由训练入口解析到项目根目录下。
OUT_PATH="${OUT_PATH:-$TRAIN_OUTPUT_ROOT}"

# smoke test 默认用 per-rank batch 16 和 1 个 epoch，只验证分布式数据分发、前后向、保存权重和日志。
BATCH_SIZE="${BATCH_SIZE:-16}"
EPOCH="${EPOCH:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
LEARNING_RATE="${LEARNING_RATE:-0.00001}"
MIN_LR="${MIN_LR:-0.000001}"
DECAY_LR="${DECAY_LR:-true}"
WARMUP_ITERS="${WARMUP_ITERS:-2000}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
SAVE_DATA_INTERVAL="${SAVE_DATA_INTERVAL:-5120000}"
BETA1="${BETA1:-0.9}"
BETA2="${BETA2:-0.95}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
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

# 模型结构、label 开关、label 数量和 SEQ_LEN 不在 shell 中写默认值；
# load_model_config_from_json 会从 MODEL_CONFIG_JSON 严格加载，缺字段立即退出。

ASCEND_RT_VISIBLE_DEVICES_VALUE="${ASCEND_RT_VISIBLE_DEVICES_VALUE:-0,1,2,3,4,5,6,7}"
HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-7200}"
HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-7200}"
HCCL_WHITELIST_DISABLE="${HCCL_WHITELIST_DISABLE:-1}"
ASCEND_TOOLKIT_HOME="${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/ascend-toolkit}"
ASCEND_ENV_SH="${ASCEND_ENV_SH:-}"
ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-$ASCEND_TOOLKIT_HOME}"

usage() {
  cat <<USAGE
Usage:
  bash scripts/$(basename "$0") all
  bash scripts/$(basename "$0") preflight
  bash scripts/$(basename "$0") generate-flat
  bash scripts/$(basename "$0") validate-data
  bash scripts/$(basename "$0") commands
  bash scripts/$(basename "$0") launch
  bash scripts/$(basename "$0") check-training

Typical server run:
  STAGE2_ROOT=/data/disk1/SpeciesLLM_obs/Stage2_SpeciesLLMData \\
  WORKDIR=/path/to/SpeciesLLM \\
  HOSTS=host0,host1,host2 MASTER_ADDR=host0 \\
  bash scripts/$(basename "$0") all

Notes:
  - generate-flat uses merge_macrogene_rounds.py --test-mode.
  - model/training-entry parameters are loaded from MODEL_CONFIG_JSON.
  - missing required fields in MODEL_CONFIG_JSON fail fast; no shell defaults are used for them.
  - all runs preflight, generate-flat, validate-data, and commands.
  - launch starts scripts/launch_multinode_torchrun.sh. Use DRY_RUN=1 with launch to print only.
  - check-training is intended after the distributed job has finished.
USAGE
}

log() {
  echo "[$(date '+%F %T')] $*"
}

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

load_model_config_from_json() {
  [[ -f "$MODEL_CONFIG_JSON" ]] || die "Missing fixed model config JSON: $MODEL_CONFIG_JSON"
  [[ -f "$PRETRAIN_CHECKS_PY" ]] || die "Missing pretraining helper: $PRETRAIN_CHECKS_PY"

  local assignments
  assignments="$("$PYTHON_BIN" "$PRETRAIN_CHECKS_PY" emit-config-env --config-json "$MODEL_CONFIG_JSON")"
  eval "$assignments"
  log "loaded fixed model config: $MODEL_CONFIG_JSON"
  log "config-derived seq_len=$SEQ_LEN hidden_size=$HIDDEN_SIZE layers=$NUM_HIDDEN_LAYERS heads=$NUM_ATTENTION_HEADS"
}

shell_quote() {
  printf "%q" "$1"
}

print_export() {
  printf "export %s=%q\n" "$1" "$2"
}

print_cmd_multiline() {
  local args=("$@")
  local last=$(( ${#args[@]} - 1 ))
  local i
  for i in "${!args[@]}"; do
    if [[ "$i" -eq "$last" ]]; then
      printf "  %q\n" "${args[$i]}"
    else
      printf "  %q \\\\\n" "${args[$i]}"
    fi
  done
}

print_runtime_env_check() {
  cat <<'BASH'
resolve_ascend_env_sh() {
  local candidate
  for candidate in \
    "${ASCEND_ENV_SH:-}" \
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

ascend_env_sh="$(resolve_ascend_env_sh)" || {
  echo "Missing Ascend environment script. Set ASCEND_ENV_SH=/path/to/set_env.sh" >&2
  exit 1
}
echo "source Ascend environment: ${ascend_env_sh}"
set +u
# shellcheck source=/dev/null
source "$ascend_env_sh"
set -u
export ASCEND_ENV_SH="$ascend_env_sh"

echo "verify Python/NPU environment with: ${PYTHON_BIN}"
"$PYTHON_BIN" - <<'PY'
import torch
import torch_npu

print("torch:", torch.__version__)
print("torch_npu:", torch_npu.__version__)
print("npu available:", torch.npu.is_available())
PY

BASH
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
  [[ "${#HOSTS_ARR[@]}" -eq "$NNODES" ]] || die "HOSTS count (${#HOSTS_ARR[@]}) must equal NNODES (${NNODES}). HOSTS=${HOSTS}"
}

safe_clean_dir() {
  local path="$1"
  [[ -n "$path" && "$path" != "/" ]] || die "Refuse to clean unsafe path: $path"
  case "$path" in
    */all_shuffled_test_500m|*/all_flatten_data_test_500m)
      log "clean $path"
      rm -rf "$path"
      ;;
    *)
      die "Refuse to clean unexpected path: $path"
      ;;
  esac
}

batch_dirs() {
  BATCH_DIRS=("$INPUT_1ST" "$INPUT_2ND" "$INPUT_3SC")
}

preflight() {
  batch_dirs
  log "data root: $STAGE2_ROOT"
  log "input dirs: ${BATCH_DIRS[*]}"
  log "merged test dir: $MERGED_TEST_DIR"
  log "flat test dir: $FLAT_TEST_DIR"
  log "500M config: hidden_size=$HIDDEN_SIZE layers=$NUM_HIDDEN_LAYERS heads=$NUM_ATTENTION_HEADS intermediate_size=$INTERMEDIATE_SIZE"
  "$PYTHON_BIN" "$PRETRAIN_CHECKS_PY" preflight \
    --seq-len "$SEQ_LEN" \
    --files-per-batch "$SOURCE_PREFLIGHT_FILES_PER_BATCH" \
    --max-scan "$SOURCE_PREFLIGHT_MAX_SCAN" \
    "${BATCH_DIRS[@]}"
}

generate_flat() {
  batch_dirs
  if [[ "$RESET_TEST_OUTPUT" == "1" ]]; then
    safe_clean_dir "$MERGED_TEST_DIR"
    safe_clean_dir "$FLAT_TEST_DIR"
  fi

  mkdir -p "$MERGED_TEST_DIR" "$FLAT_TEST_DIR"
  local merge_cmd=(
    "$PYTHON_BIN"
    "${PROJECT_ROOT}/merge_macrogene_rounds.py"
    --batch-dirs
    "${BATCH_DIRS[0]}"
    "${BATCH_DIRS[1]}"
    "${BATCH_DIRS[2]}"
    --batch-names
    1st
    2nd
    3scbasecount
    --output-dir
    "$MERGED_TEST_DIR"
    --mode
    copy
    --workers
    "$WORKERS"
    --test-mode
    --manifest-name
    merge_manifest.csv
  )
  if [[ "$SKIP_EXISTING" == "1" ]]; then
    merge_cmd+=(--skip-existing)
  fi

  local flatten_cmd=(
    "$PYTHON_BIN"
    "${PROJECT_ROOT}/shuffle_flatten_macrogene.py"
    --input-dir
    "$MERGED_TEST_DIR"
    --output-dir
    "$FLAT_TEST_DIR"
    --pattern
    "macrogene_*.parquet"
    --rows-per-file
    "$ROWS_PER_FILE"
    --seed
    "$SHUFFLE_SEED"
    --workers
    "$FLATTEN_WORKERS"
    --compression
    snappy
    --manifest-name
    shuffle_manifest.csv
    --overwrite
    --validate-all-schemas
  )
  if [[ "$FLATTEN_DROP_REMAINDER" == "1" ]]; then
    flatten_cmd+=(--drop-remainder)
  else
    flatten_cmd+=(--keep-remainder)
  fi

  log "merge test-mode command:"
  printf "%q " "${merge_cmd[@]}"; printf "\n"
  "${merge_cmd[@]}"

  log "flatten/shuffle command:"
  printf "%q " "${flatten_cmd[@]}"; printf "\n"
  "${flatten_cmd[@]}"
}

validate_data() {
  mkdir -p "$COMMAND_DIR"
  local -a validate_cmd=(
    "$PYTHON_BIN"
    "$PRETRAIN_CHECKS_PY"
    validate-data
    --flat-dir
    "$FLAT_TEST_DIR"
    --command-dir
    "$COMMAND_DIR"
    --emb-path
    "$EMB_PATH"
    --config-json
    "$MODEL_CONFIG_JSON"
    --world-size
    "$WORLD_SIZE"
    --nnodes
    "$NNODES"
    --nproc-per-node
    "$NPROC_PER_NODE"
    --max-validate-rows-per-file
    "$MAX_VALIDATE_ROWS_PER_FILE"
    --max-validate-files
    "$MAX_VALIDATE_FILES"
  )
  if [[ "$REQUIRE_EMBEDDINGS" == "1" ]]; then
    validate_cmd+=(--require-embeddings)
  fi
  "${validate_cmd[@]}"
}

train_args() {
  TRAIN_ARGS=(
    "--data_path=${DATA_PATH}"
    "--num_of_used_data=${NUM_OF_USED_DATA}"
    "--emb_path=${EMB_PATH}"
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
    "--weight_decay=${WEIGHT_DECAY}"
    "--save_data_interval=${SAVE_DATA_INTERVAL}"
    "--beta1=${BETA1}"
    "--beta2=${BETA2}"
    "--grad_clip=${GRAD_CLIP}"
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
  )
  if [[ -n "$S3_REMOTE_DIR_PATH" ]]; then
    TRAIN_ARGS+=("--s3_remote_dir_path=${S3_REMOTE_DIR_PATH}")
  fi
}

write_torchrun_script() {
  local node_rank="$1"
  local host="$2"
  local script_path="$3"
  train_args
  local cmd=(
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

  {
    echo "#!/usr/bin/env bash"
    echo "set -euo pipefail"
    echo "# Run this on host ${host} (node_rank=${node_rank})."
    printf "cd %q\n" "$WORKDIR"
    print_export HOME "$HOME"
    print_export PYTHON_BIN "$PYTHON_BIN"
    print_export NNODES "$NNODES"
    print_export NPROC_PER_NODE "$NPROC_PER_NODE"
    print_export NODE_RANK "$node_rank"
    print_export MASTER_ADDR "$MASTER_ADDR"
    print_export MASTER_PORT "$MASTER_PORT"
    print_export ASCEND_RT_VISIBLE_DEVICES "$ASCEND_RT_VISIBLE_DEVICES_VALUE"
    print_export HCCL_CONNECT_TIMEOUT "$HCCL_CONNECT_TIMEOUT"
    print_export HCCL_EXEC_TIMEOUT "$HCCL_EXEC_TIMEOUT"
    print_export HCCL_WHITELIST_DISABLE "$HCCL_WHITELIST_DISABLE"
    print_export ASCEND_TOOLKIT_HOME "$ASCEND_TOOLKIT_HOME"
    print_export ASCEND_ENV_SH "$ASCEND_ENV_SH"
    print_export ASCEND_HOME_PATH "$ASCEND_HOME_PATH"
    echo
    print_runtime_env_check
    print_cmd_multiline "${cmd[@]}"
  } > "$script_path"
  chmod +x "$script_path"
}

write_launcher_script() {
  local script_path="$1"
  {
    echo "#!/usr/bin/env bash"
    echo "set -euo pipefail"
    printf "cd %q\n" "$PROJECT_ROOT"
    print_export HOME "$HOME"
    print_export HOSTS "$HOSTS"
    print_export NNODES "$NNODES"
    print_export AUTO_NNODES 0
    print_export NPROC_PER_NODE "$NPROC_PER_NODE"
    print_export MASTER_ADDR "$MASTER_ADDR"
    print_export MASTER_PORT "$MASTER_PORT"
    print_export WORKDIR "$WORKDIR"
    print_export TRAIN_ENTRY "$TRAIN_ENTRY"
    print_export PYTHON_BIN "$PYTHON_BIN"
    print_export LOG_SUBDIR "$LOG_SUBDIR"
    print_export TRAIN_DATASET "$TRAIN_DATASET"
    print_export DATA_ROOT "$STAGE2_ROOT"
    print_export DATA_PATH "$DATA_PATH"
    print_export NUM_OF_USED_DATA "$NUM_OF_USED_DATA"
    print_export EMB_ROOT "$EMB_ROOT"
    print_export EMB_PATH "$EMB_PATH"
    print_export MODEL_CONFIG_JSON "$MODEL_CONFIG_JSON"
    print_export OUT_PATH "$OUT_PATH"
    print_export BATCH_SIZE "$BATCH_SIZE"
    print_export EPOCH "$EPOCH"
    print_export GRADIENT_ACCUMULATION_STEPS "$GRADIENT_ACCUMULATION_STEPS"
    print_export LEARNING_RATE "$LEARNING_RATE"
    print_export MIN_LR "$MIN_LR"
    print_export DECAY_LR "$DECAY_LR"
    print_export WARMUP_ITERS "$WARMUP_ITERS"
    print_export WARMUP_RATIO "$WARMUP_RATIO"
    print_export WEIGHT_DECAY "$WEIGHT_DECAY"
    print_export SAVE_DATA_INTERVAL "$SAVE_DATA_INTERVAL"
    print_export BETA1 "$BETA1"
    print_export BETA2 "$BETA2"
    print_export GRAD_CLIP "$GRAD_CLIP"
    print_export COMPILE "$COMPILE"
    print_export BACKEND "$BACKEND"
    print_export DEVICE "$DEVICE"
    print_export DEVICE_TYPE "$DEVICE_TYPE"
    print_export S3_REMOTE_DIR_PATH "$S3_REMOTE_DIR_PATH"
    print_export LOG_INTERVAL "$LOG_INTERVAL"
    print_export PROFILE_INTERVAL "$PROFILE_INTERVAL"
    print_export NAN_CHECK_INTERVAL "$NAN_CHECK_INTERVAL"
    print_export METRICS_FLUSH_INTERVAL "$METRICS_FLUSH_INTERVAL"
    print_export LOG_LEVEL "$LOG_LEVEL"
    print_export LOG_ALL_RANKS "$LOG_ALL_RANKS"
    print_export ASCEND_RT_VISIBLE_DEVICES_VALUE "$ASCEND_RT_VISIBLE_DEVICES_VALUE"
    print_export HCCL_CONNECT_TIMEOUT "$HCCL_CONNECT_TIMEOUT"
    print_export HCCL_EXEC_TIMEOUT "$HCCL_EXEC_TIMEOUT"
    print_export HCCL_WHITELIST_DISABLE "$HCCL_WHITELIST_DISABLE"
    print_export ASCEND_TOOLKIT_HOME "$ASCEND_TOOLKIT_HOME"
    print_export ASCEND_ENV_SH "$ASCEND_ENV_SH"
    print_export ASCEND_HOME_PATH "$ASCEND_HOME_PATH"
    echo
    echo 'exec bash scripts/launch_multinode_torchrun.sh'
  } > "$script_path"
  chmod +x "$script_path"
}

generate_commands() {
  split_hosts
  mkdir -p "$COMMAND_DIR"

  local launcher="${COMMAND_DIR}/launch_500m_3nodes.sh"
  write_launcher_script "$launcher"

  local rank host script_path
  for ((rank=0; rank<NNODES; rank++)); do
    host="${HOSTS_ARR[$rank]}"
    script_path="${COMMAND_DIR}/torchrun_node_rank${rank}.sh"
    write_torchrun_script "$rank" "$host" "$script_path"
  done

  local summary="${COMMAND_DIR}/README.txt"
  {
    echo "500M three-node pretraining commands"
    echo
    echo "Data path:"
    echo "  $DATA_PATH"
    echo "Embedding path:"
    echo "  $EMB_PATH"
    echo "Output path pattern:"
    echo "  $OUT_PATH"
    echo
    echo "Automatic SSH launcher from master:"
    echo "  DRY_RUN=1 bash $launcher"
    echo "  bash $launcher"
    echo
    echo "Manual per-node torchrun scripts:"
    for ((rank=0; rank<NNODES; rank++)); do
      echo "  node_rank=${rank}, host=${HOSTS_ARR[$rank]}: bash ${COMMAND_DIR}/torchrun_node_rank${rank}.sh"
    done
    echo
    echo "Post-run checks:"
    echo "  bash scripts/$(basename "$0") check-training"
  } > "$summary"

  local dry_run="${COMMAND_DIR}/train_multinode_500m_dry_run.txt"
  if command -v ssh >/dev/null 2>&1 && command -v scp >/dev/null 2>&1; then
    DRY_RUN=1 bash "$launcher" > "$dry_run" 2>&1 || true
  else
    echo "ssh/scp not found; skip launcher dry-run." > "$dry_run"
  fi

  log "wrote command files under $COMMAND_DIR"
  log "launcher: $launcher"
  log "manual node scripts: ${COMMAND_DIR}/torchrun_node_rank{0..$((NNODES - 1))}.sh"
}

launch_training() {
  generate_commands
  log "launch distributed training through scripts/launch_multinode_torchrun.sh"
  bash "${COMMAND_DIR}/launch_500m_3nodes.sh"
}

resolve_train_out_dir() {
  "$PYTHON_BIN" "$PRETRAIN_CHECKS_PY" resolve-out-dir \
    --out-path "$OUT_PATH" \
    --workdir "$WORKDIR" \
    --config-json "$MODEL_CONFIG_JSON" \
    --learning-rate "$LEARNING_RATE" \
    --min-lr "$MIN_LR" \
    --weight-decay "$WEIGHT_DECAY" \
    --warmup-ratio "$WARMUP_RATIO"
}

check_training() {
  local out_dir="${TRAIN_OUT_DIR:-$(resolve_train_out_dir)}"
  local node_log_dir="${NODE_LOG_DIR:-${WORKDIR}/${LOG_SUBDIR}}"
  "$PYTHON_BIN" "$PRETRAIN_CHECKS_PY" check-training \
    --train-out-dir "$out_dir" \
    --node-log-dir "$node_log_dir" \
    --world-size "$WORLD_SIZE" \
    --epoch "$EPOCH"
}

main() {
  local action="${1:-all}"
  case "$action" in
    -h|--help|help)
      usage
      return 0
      ;;
  esac

  load_model_config_from_json

  case "$action" in
    preflight)
      preflight
      ;;
    generate-flat)
      preflight
      generate_flat
      ;;
    validate-data)
      validate_data
      ;;
    commands)
      generate_commands
      ;;
    launch)
      launch_training
      ;;
    check-training)
      check_training
      ;;
    all)
      preflight
      generate_flat
      validate_data
      generate_commands
      ;;
    *)
      usage
      die "unknown action: $action"
      ;;
  esac
}

main "$@"
