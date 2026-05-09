#!/usr/bin/env bash
set -euo pipefail

cd /data/disk1/SpeciesLLM

set -a; source .env; set +a

export PYTHON_BIN=/data/miniconda3/bin/python
export STAGE2_ROOT=/data/disk1/SpeciesLLM_obs/Stage2_SpeciesLLMData

export INPUT_1ST=${STAGE2_ROOT}/1st_pretrain_data_preprocessed_step4
export INPUT_2ND=${STAGE2_ROOT}/2nd_pretrain_data_preprocessed_step4
export INPUT_3SC=${STAGE2_ROOT}/3scbasecount_pretrain_data_preprocessed_step4

export RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
export FIRST_VIEW=${FIRST_VIEW:-${STAGE2_ROOT}/views/1st_no_human_mouse_${RUN_ID}}
export MERGED_DIR=${MERGED_DIR:-${STAGE2_ROOT}/all_merged_full_no_1st_human_mouse_${RUN_ID}}
export FLAT_DIR=${FLAT_DIR:-${STAGE2_ROOT}/all_flatten_data_full_no_1st_human_mouse_${RUN_ID}}

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
export UPDATE_DATA_PATH_LINK=${UPDATE_DATA_PATH_LINK:-1}
export DATA_PATH_LINK=${DATA_PATH_LINK:-${STAGE2_ROOT}/all_flatten_data}

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
  ssh_run "$host" "mkdir -p $(shell_quote "$FLAT_DIR") $(shell_quote "$(dirname "$DATA_PATH_LINK")")"
  if [[ -n "$SSH_PASSWORD" ]]; then
    SSHPASS="$SSH_PASSWORD" rsync -e "$(rsync_ssh_cmd)" -aH --info=progress2 --delete \
      "${FLAT_DIR%/}/" "${SSH_USER}@${host}:${FLAT_DIR%/}/"
  else
    rsync -e "$(rsync_ssh_cmd)" -aH --info=progress2 --delete \
      "${FLAT_DIR%/}/" "${SSH_USER}@${host}:${FLAT_DIR%/}/"
  fi

  if [[ "$UPDATE_DATA_PATH_LINK" == "1" ]]; then
    ssh_run "$host" \
      "if [ -e $(shell_quote "$DATA_PATH_LINK") ] && [ ! -L $(shell_quote "$DATA_PATH_LINK") ]; then echo '[SYNC] skip remote DATA_PATH_LINK because it exists and is not a symlink: ${DATA_PATH_LINK}'; else ln -sfn $(shell_quote "$FLAT_DIR") $(shell_quote "$DATA_PATH_LINK"); fi"
  fi
}

update_local_data_path_link() {
  if [[ "$UPDATE_DATA_PATH_LINK" != "1" ]]; then
    return
  fi
  if [[ -e "$DATA_PATH_LINK" && ! -L "$DATA_PATH_LINK" ]]; then
    echo "[SYNC] skip local DATA_PATH_LINK because it exists and is not a symlink: ${DATA_PATH_LINK}"
  else
    ln -sfn "$FLAT_DIR" "$DATA_PATH_LINK"
    echo "[SYNC] local DATA_PATH_LINK -> ${FLAT_DIR}"
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

mkdir -p "$FIRST_VIEW" "$MERGED_DIR" "$FLAT_DIR"

if [[ "$SKIP_MERGE" != "1" ]]; then
  for d in "$INPUT_1ST"/*; do
    [ -d "$d" ] || continue
    species="$(basename "$d")"
    case "$species" in
      Homo_sapiens|Mus_musculus) continue ;;
    esac
    ln -sfn "$d" "$FIRST_VIEW/$species"
  done
fi

merge_cmd=(
  "$PYTHON_BIN"
  merge_macrogene_rounds.py
  --batch-dirs
  "$FIRST_VIEW"
  "$INPUT_2ND"
  "$INPUT_3SC"
  --batch-names
  1st
  2nd
  3scbasecount
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

update_local_data_path_link
sync_flat_dir_to_workers

echo "[DONE] flatten data: ${FLAT_DIR}"
echo "[DONE] training DATA_PATH_LINK: ${DATA_PATH_LINK}"


# 第一次生成全量数据
# BATCH_FILES=2048 WORKERS=32 bash work_record/step1_data_1_2_3.sh

# 第二次生成全量数据（external 全局打乱，桶数默认 16；ProcessPool 并行 partition）
# 先用 --max-files 200 跑 smoke test：在脚本里加 SHUFFLE_EXTRA_ARGS 或临时改 flatten_cmd
# RUN_ID=20260506_165244 \
# SKIP_MERGE=1 SKIP_SYNC=1 UPDATE_DATA_PATH_LINK=0 \
# FLAT_DIR=/data/disk1/SpeciesLLM_obs/Stage2_SpeciesLLMData/all_flatten_data_full_no_1st_human_mouse_20260506_165244_external \
# SHUFFLE_MODE=external WORKERS=32 \
# bash work_record/step1_data_1_2_3.sh