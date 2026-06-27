#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" != "--__speciesllm-step1-data-1-clean-env" ]]; then
  exec /usr/bin/env -i \
    HOME="${HOME:-/root}" \
    PATH="/data/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
    LANG=C.UTF-8 \
    bash "$0" --__speciesllm-step1-data-1-clean-env "$@"
fi
shift

cd /data/disk1/SpeciesLLM

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

usage() {
  cat <<'USAGE'
Usage:
  bash work_record/step1_data_1.sh [KEY=VALUE ...]

Batch 1 only. Unlike step1_data_1_3.sh / step1_data_1_2_3.sh, this keeps every
species in batch 1, INCLUDING Homo_sapiens and Mus_musculus: batch 1 runs alone
here, so there is no other batch to de-duplicate human/mouse against.

Runtime overrides must be passed after the script path as KEY=VALUE arguments.
Inherited shell environment variables are ignored. Values from .env are loaded
before command-line overrides.
USAGE
}

is_allowed_cli_var() {
  case "$1" in
    ENV_FILE|PYTHON_BIN|STAGE2_ROOT|INPUT_1ST|\
    RUN_ID|MERGED_DIR|FLAT_DIR|WORKERS|ROWS_PER_FILE|\
    SHUFFLE_SEED|SHUFFLE_MODE|BATCH_FILES|SHUFFLE_BUCKETS|\
    SKIP_MERGE|SKIP_FLATTEN|SKIP_EXISTING|SKIP_PARTITION_IF_EXISTS|\
    HOSTS|LOCAL_HOST|SSH_USER|SSH_KEY|SSH_PASSWORD|SSH_EXTRA_OPTS|SKIP_SYNC)
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
  export "${CLI_KEYS[$i]}"
done

export PYTHON_BIN=${PYTHON_BIN:-/data/miniconda3/bin/python}
export STAGE2_ROOT=${STAGE2_ROOT:-/data/disk1/SpeciesLLM_obs/Stage2_SpeciesLLMData}

export INPUT_1ST=${INPUT_1ST:-${STAGE2_ROOT}/1st_pretrain_data_preprocessed_step4}

export RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
export MERGED_DIR=${MERGED_DIR:-${STAGE2_ROOT}/data_1_only_merged_full_${RUN_ID}}
export FLAT_DIR=${FLAT_DIR:-${STAGE2_ROOT}/data_1_only_flatten_data_full_${RUN_ID}}

export WORKERS=${WORKERS:-16}
export ROWS_PER_FILE=${ROWS_PER_FILE:-16384}
export SHUFFLE_SEED=${SHUFFLE_SEED:-42}
export SHUFFLE_MODE=${SHUFFLE_MODE:-batch}
export BATCH_FILES=${BATCH_FILES:-2048}
export SHUFFLE_BUCKETS=${SHUFFLE_BUCKETS:-16}
export SKIP_MERGE=${SKIP_MERGE:-0}
export SKIP_FLATTEN=${SKIP_FLATTEN:-0}
export SKIP_EXISTING=${SKIP_EXISTING:-0}
export SKIP_PARTITION_IF_EXISTS=${SKIP_PARTITION_IF_EXISTS:-0}

# Multi-node data sync. Run this script on the first host in HOSTS.
export HOSTS=${HOSTS:-7.150.12.45,7.150.15.14,7.150.14.170}
export LOCAL_HOST=${LOCAL_HOST:-${HOSTS%%,*}}
export SSH_USER=${SSH_USER:-root}
export SSH_KEY=${SSH_KEY:-}
export SSH_PASSWORD=${SSH_PASSWORD:-}
export SSH_EXTRA_OPTS=${SSH_EXTRA_OPTS:-}
export SKIP_SYNC=${SKIP_SYNC:-0}

SSH_OPTS=()
if [[ -n "$SSH_KEY" ]]; then
  SSH_OPTS+=("-i" "$SSH_KEY")
fi
if [[ -n "$SSH_EXTRA_OPTS" ]]; then
  read -r -a SSH_EXTRA_OPTS_ARR <<< "$SSH_EXTRA_OPTS"
  SSH_OPTS+=("${SSH_EXTRA_OPTS_ARR[@]}")
fi
if [[ -n "$SSH_PASSWORD" ]]; then
  command -v sshpass >/dev/null
fi

shell_quote() {
  printf "%q" "$1"
}

print_command() {
  local label="$1"
  shift
  printf '[CMD] %s:' "$label"
  printf ' %q' "$@"
  printf '\n'
}

ssh_run() {
  local host="$1"
  shift
  if [[ -n "$SSH_PASSWORD" ]]; then
    SSHPASS="$SSH_PASSWORD" sshpass -e ssh "${SSH_OPTS[@]}" "${SSH_USER}@${host}" "$@"
  else
    ssh "${SSH_OPTS[@]}" "${SSH_USER}@${host}" "$@"
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

sync_flat_dir_to_host() {
  local host="$1"
  echo "[SYNC] flatten data -> ${host}:${FLAT_DIR}"
  ssh_run "$host" "mkdir -p $(shell_quote "$FLAT_DIR")"
  if [[ -n "$SSH_PASSWORD" ]]; then
    SSHPASS="$SSH_PASSWORD" rsync -e "$(rsync_ssh_cmd)" -aH --info=progress2 --delete \
      "${FLAT_DIR%/}/" "${SSH_USER}@${host}:${FLAT_DIR%/}/"
  else
    rsync -e "$(rsync_ssh_cmd)" -aH --info=progress2 --delete \
      "${FLAT_DIR%/}/" "${SSH_USER}@${host}:${FLAT_DIR%/}/"
  fi
}

sync_flat_dir_to_workers() {
  if [[ "$SKIP_SYNC" == "1" ]]; then
    echo "[SYNC] SKIP_SYNC=1, skip remote data sync."
    return
  fi
  command -v ssh >/dev/null
  command -v rsync >/dev/null

  IFS=',' read -r -a HOSTS_ARR <<< "$HOSTS"
  local raw_host host
  for raw_host in "${HOSTS_ARR[@]}"; do
    host="${raw_host//[[:space:]]/}"
    [[ -n "$host" ]] || continue
    if [[ "$host" == "$LOCAL_HOST" || "$host" == "127.0.0.1" || "$host" == "localhost" ]]; then
      echo "[SYNC] skip local host ${host}; source data is already local"
      continue
    fi
    sync_flat_dir_to_host "$host"
  done
}

mkdir -p "$MERGED_DIR" "$FLAT_DIR"

merge_cmd=(
  "$PYTHON_BIN"
  merge_macrogene_rounds.py
  --batch-dirs
  "$INPUT_1ST"
  --batch-names
  1st
  --output-dir
  "$MERGED_DIR"
  --mode
  copy
  --workers
  "$WORKERS"
  --manifest-name
  merge_manifest.csv
)
if [[ "$SKIP_EXISTING" == "1" ]]; then
  merge_cmd+=(--skip-existing)
fi

flatten_cmd=(
  "$PYTHON_BIN"
  shuffle_flatten_macrogene.py
  --input-dir
  "$MERGED_DIR"
  --output-dir
  "$FLAT_DIR"
  --pattern
  'macrogene_*.parquet'
  --rows-per-file
  "$ROWS_PER_FILE"
  --seed
  "$SHUFFLE_SEED"
  --workers
  "$WORKERS"
  --compression
  snappy
  --manifest-name
  shuffle_manifest.csv
  --merge-manifest
  "${MERGED_DIR}/merge_manifest.csv"
  --shuffle-mode
  "$SHUFFLE_MODE"
  --batch-files
  "$BATCH_FILES"
  --shuffle-buckets
  "$SHUFFLE_BUCKETS"
  --temp-dir
  "${FLAT_DIR}/_shuffle_tmp"
  --overwrite
  --validate-all-schemas
  --keep-remainder
)
if [[ "$SKIP_PARTITION_IF_EXISTS" == "1" ]]; then
  flatten_cmd+=(--skip-partition-if-exists)
fi

if [[ "$SKIP_MERGE" == "1" ]]; then
  echo "[INFO] SKIP_MERGE=1, reuse merged data: ${MERGED_DIR}"
else
  print_command "merge command" "${merge_cmd[@]}"
  "${merge_cmd[@]}"
fi

if [[ "$SKIP_FLATTEN" == "1" ]]; then
  echo "[INFO] SKIP_FLATTEN=1, skip flatten data generation."
else
  print_command "flatten command" "${flatten_cmd[@]}"
  "${flatten_cmd[@]}"
fi

sync_flat_dir_to_workers

echo "[DONE] flatten data: ${FLAT_DIR}"


# 第一次生成批1全量数据（batch 模式：随机文件顺序 + 批内打乱，默认）
# bash work_record/step1_data_1.sh BATCH_FILES=2048 WORKERS=32

# 第二次生成批1全量数据（external 全局打乱，桶数默认 16；ProcessPool 并行 partition）
# 复用第一次的 merged 目录，避免重复 merge：
# bash work_record/step1_data_1.sh \
#   RUN_ID=<第一次的 RUN_ID> \
#   SKIP_MERGE=1 SKIP_SYNC=1 \
#   FLAT_DIR=/data/disk1/SpeciesLLM_obs/Stage2_SpeciesLLMData/data_1_only_flatten_data_full_<RUN_ID>_external \
#   SHUFFLE_MODE=external WORKERS=32
