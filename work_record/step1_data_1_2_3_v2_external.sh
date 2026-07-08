#!/usr/bin/env bash
set -euo pipefail

# Stage2 v2 全量训练数据（1st + 2nd + 3scbasecount）生成脚本。
# 与 step1_data_1_2_3.sh 使用同一套 merge / shuffle_flatten 流水线，但：
#   - 上游输入切到 *_preprocessed_step4_v2
#   - SHUFFLE_MODE=external（按随机 key 分桶，全局行级打乱）
#
# Usage:
#   # 1) 合并 + external 打平（默认跳过多机同步）
#   bash work_record/step1_data_1_2_3_v2_external.sh
#
#   # 2) 打平完成后做 shuffle 质量验证
#   bash work_record/step1_data_1_2_3_v2_external.sh ACTION=verify
#
#   # 3) 验证通过后同步到其余训练节点
#   bash work_record/step1_data_1_2_3_v2_external.sh ACTION=sync
#
# 覆盖项必须写在脚本路径后面（KEY=VALUE），不能写成 INPUT_1ST=... bash script。
# 本脚本会 re-exec 到干净环境，继承的 shell 环境变量会被丢弃。

if [[ "${1:-}" != "--__speciesllm-step1-data-v2-external-clean-env" ]]; then
  exec /usr/bin/env -i \
    HOME="${HOME:-/root}" \
    PATH="/data/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
    LANG=C.UTF-8 \
    bash "$0" --__speciesllm-step1-data-v2-external-clean-env "$@"
fi
shift

cd /data/disk1/SpeciesLLM

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

usage() {
  sed -n '4,20p' "$0" | sed 's/^# \{0,1\}//'
}

is_allowed_cli_var() {
  case "$1" in
    ENV_FILE|ACTION|RUN_ID|STAGE2_ROOT|INPUT_1ST|INPUT_2ND|INPUT_3SC|\
    MERGED_DIR|FLAT_DIR|WORKERS|ROWS_PER_FILE|SHUFFLE_SEED|SHUFFLE_BUCKETS|\
    SAMPLE_FILES|SKIP_APPLEDOUBLE_CLEANUP|SKIP_DISK_CHECK|PYTHON_BIN|\
    HOSTS|LOCAL_HOST|SSH_USER|SSH_KEY|SSH_PASSWORD|SSH_EXTRA_OPTS)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

CLI_KEYS=()
CLI_VALUES=()
ENV_FILE=".env"
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
      [[ "$key" == "ENV_FILE" ]] && ENV_FILE="$value"
      ;;
    *)
      usage >&2
      die "Unsupported argument: $arg. Use KEY=VALUE arguments after the script path."
      ;;
  esac
done

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "$ENV_FILE"
  set +a
fi

for i in "${!CLI_KEYS[@]}"; do
  printf -v "${CLI_KEYS[$i]}" "%s" "${CLI_VALUES[$i]}"
done

ACTION=${ACTION:-generate}
RUN_ID=${RUN_ID:-v2_604_external_20260708}
STAGE2_ROOT=${STAGE2_ROOT:-/data/disk1/SpeciesLLM_obs/Stage2_SpeciesLLMData}

INPUT_1ST=${INPUT_1ST:-${STAGE2_ROOT}/1st_pretrain_data_preprocessed_step4_v2}
INPUT_2ND=${INPUT_2ND:-${STAGE2_ROOT}/2nd_pretrain_data_preprocessed_step4_v2}
INPUT_3SC=${INPUT_3SC:-${STAGE2_ROOT}/3scbasecount_pretrain_data_preprocessed_step4_v2}

MERGED_DIR=${MERGED_DIR:-${STAGE2_ROOT}/all_merged_full_no_1st_human_mouse_${RUN_ID}}
FLAT_DIR=${FLAT_DIR:-${STAGE2_ROOT}/all_flatten_data_full_no_1st_human_mouse_${RUN_ID}}

WORKERS=${WORKERS:-32}
ROWS_PER_FILE=${ROWS_PER_FILE:-16384}
SHUFFLE_SEED=${SHUFFLE_SEED:-42}
SHUFFLE_BUCKETS=${SHUFFLE_BUCKETS:-512}
SAMPLE_FILES=${SAMPLE_FILES:-80}

SKIP_APPLEDOUBLE_CLEANUP=${SKIP_APPLEDOUBLE_CLEANUP:-0}
SKIP_DISK_CHECK=${SKIP_DISK_CHECK:-0}

PYTHON_BIN=${PYTHON_BIN:-/data/miniconda3/bin/python}
STEP1_SCRIPT=work_record/step1_data_1_2_3.sh
VERIFY_SCRIPT=work_record/step1_verify_shuffle.py

print_paths() {
  cat <<EOF
[INFO] ACTION=${ACTION}
[INFO] RUN_ID=${RUN_ID}
[INFO] STAGE2_ROOT=${STAGE2_ROOT}
[INFO] INPUT_1ST=${INPUT_1ST}
[INFO] INPUT_2ND=${INPUT_2ND}
[INFO] INPUT_3SC=${INPUT_3SC}
[INFO] MERGED_DIR=${MERGED_DIR}
[INFO] FLAT_DIR=${FLAT_DIR}
EOF
}

check_inputs_exist() {
  local missing=0
  for d in "$INPUT_1ST" "$INPUT_2ND" "$INPUT_3SC"; do
    if [[ ! -d "$d" ]]; then
      echo "[ERROR] Missing input directory: $d" >&2
      missing=1
    fi
  done
  [[ "$missing" -eq 0 ]] || die "One or more v2 input directories are missing."
}

check_disk_space() {
  if [[ "$SKIP_DISK_CHECK" == "1" ]]; then
    echo "[INFO] SKIP_DISK_CHECK=1, skip disk space preflight."
    return
  fi
  echo "[INFO] Disk space preflight (external shuffle needs ~1x flatten output as temp space):"
  df -h /data/disk1 "$STAGE2_ROOT"
}

cleanup_appledouble_files() {
  if [[ "$SKIP_APPLEDOUBLE_CLEANUP" == "1" ]]; then
    echo "[INFO] SKIP_APPLEDOUBLE_CLEANUP=1, skip AppleDouble cleanup."
    return
  fi
  echo "[INFO] Removing macOS AppleDouble files under 1st/2nd v2 inputs (merge already skips them)."
  find "$INPUT_1ST" "$INPUT_2ND" -type f -name '._*' -delete
}

action_generate() {
  print_paths
  check_inputs_exist
  check_disk_space
  cleanup_appledouble_files

  bash "$STEP1_SCRIPT" \
    STAGE2_ROOT="$STAGE2_ROOT" \
    INPUT_1ST="$INPUT_1ST" \
    INPUT_2ND="$INPUT_2ND" \
    INPUT_3SC="$INPUT_3SC" \
    RUN_ID="$RUN_ID" \
    MERGED_DIR="$MERGED_DIR" \
    FLAT_DIR="$FLAT_DIR" \
    SHUFFLE_MODE=external \
    SHUFFLE_BUCKETS="$SHUFFLE_BUCKETS" \
    WORKERS="$WORKERS" \
    ROWS_PER_FILE="$ROWS_PER_FILE" \
    SHUFFLE_SEED="$SHUFFLE_SEED" \
    SKIP_SYNC=1

  echo "[DONE] v2 flatten data ready: ${FLAT_DIR}"
  echo "[NEXT] Run: bash work_record/step1_data_1_2_3_v2_external.sh ACTION=verify RUN_ID=${RUN_ID}"
}

action_verify() {
  print_paths

  test -f "$MERGED_DIR/merge_manifest.csv" || die "Missing merge manifest: ${MERGED_DIR}/merge_manifest.csv"
  test -f "$FLAT_DIR/shuffle_manifest.csv" || die "Missing shuffle manifest: ${FLAT_DIR}/shuffle_manifest.csv"

  local part_count
  part_count="$(find "$FLAT_DIR" -maxdepth 1 -name 'all_flatten_part_*.parquet' | wc -l | tr -d ' ')"
  echo "[INFO] flatten part files: ${part_count}"

  awk -F, 'NR>1 {rows += $4} END {printf "rows=%0.f\n", rows}' \
    "$FLAT_DIR/shuffle_manifest.csv"

  "$PYTHON_BIN" "$VERIFY_SCRIPT" "$FLAT_DIR" "$SAMPLE_FILES" \
    | tee "$FLAT_DIR/shuffle_verify_${SAMPLE_FILES}.txt"

  echo "[DONE] shuffle verification log: ${FLAT_DIR}/shuffle_verify_${SAMPLE_FILES}.txt"
  echo "[NEXT] If quality is acceptable, run: bash work_record/step1_data_1_2_3_v2_external.sh ACTION=sync RUN_ID=${RUN_ID}"
}

action_sync() {
  print_paths

  test -d "$FLAT_DIR" || die "Missing flatten directory: ${FLAT_DIR}"
  test -f "$FLAT_DIR/shuffle_manifest.csv" || die "Missing shuffle manifest: ${FLAT_DIR}/shuffle_manifest.csv"

  local -a sync_args=(
    STAGE2_ROOT="$STAGE2_ROOT"
    RUN_ID="$RUN_ID"
    MERGED_DIR="$MERGED_DIR"
    FLAT_DIR="$FLAT_DIR"
    SKIP_MERGE=1
    SKIP_FLATTEN=1
    SKIP_SYNC=0
  )
  if [[ -n "${HOSTS:-}" ]]; then
    sync_args+=(HOSTS="$HOSTS")
  fi
  if [[ -n "${LOCAL_HOST:-}" ]]; then
    sync_args+=(LOCAL_HOST="$LOCAL_HOST")
  fi
  if [[ -n "${SSH_USER:-}" ]]; then
    sync_args+=(SSH_USER="$SSH_USER")
  fi
  if [[ -n "${SSH_KEY:-}" ]]; then
    sync_args+=(SSH_KEY="$SSH_KEY")
  fi
  if [[ -n "${SSH_PASSWORD:-}" ]]; then
    sync_args+=(SSH_PASSWORD="$SSH_PASSWORD")
  fi
  if [[ -n "${SSH_EXTRA_OPTS:-}" ]]; then
    sync_args+=(SSH_EXTRA_OPTS="$SSH_EXTRA_OPTS")
  fi

  bash "$STEP1_SCRIPT" "${sync_args[@]}"

  echo "[DONE] synced flatten data to worker hosts: ${FLAT_DIR}"
}

case "$ACTION" in
  generate)
    action_generate
    ;;
  verify)
    action_verify
    ;;
  sync)
    action_sync
    ;;
  *)
    die "Unsupported ACTION=${ACTION}. Use generate, verify, or sync."
    ;;
esac
