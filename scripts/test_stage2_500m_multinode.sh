#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Stage 2 data + 500M three-node training smoke-test helper.
#
# This script intentionally uses merge_macrogene_rounds_parallel.py --test-mode
# for the small sample. It does not create synthetic parquet inputs.
###############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

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
COMMAND_DIR="${COMMAND_DIR:-${STAGE2_ROOT}/stage2_500m_test_commands}"
TRAIN_OUTPUT_ROOT="${TRAIN_OUTPUT_ROOT:-${STAGE2_ROOT}/stage2_500m_train_outputs}"

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
ROWS_PER_FILE="${ROWS_PER_FILE:-1024}"
SHUFFLE_SEED="${SHUFFLE_SEED:-42}"
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

# 传给 scripts/train_multinode.sh 的数据集标识；这里默认使用测试扁平化数据，而不是 full 全量数据。
TRAIN_DATASET="${TRAIN_DATASET:-test}"
# 训练脚本实际读取的扁平化 parquet 目录，默认就是本脚本生成的 all_flatten_data_test_500m。
DATA_PATH="${DATA_PATH:-$FLAT_TEST_DIR}"
# 限制训练使用前 N 个 parquet 文件；0 表示使用 DATA_PATH 下所有 parquet 文件。
NUM_OF_USED_DATA="${NUM_OF_USED_DATA:-0}"
# 训练输出目录模板。train_MNodes_torchrun_mfu_preindexparquet.py 会用模型结构和超参数填充花括号字段。
if [[ -z "${OUT_PATH+x}" ]]; then
  OUT_PATH="${TRAIN_OUTPUT_ROOT}/stage2_500m_smoke_hs_{hidden_size}_nh_{num_hidden_layers}_na_{num_attention_heads}_hdp_{hidden_dropout_prob}_lr_{learning_rate}_mlr_{min_lr}_wd_{weight_decay}_wr_{warmup_ratio}"
fi

# smoke test 默认用很小的 per-rank batch 和 1 个 epoch，只验证分布式数据分发、前后向、保存权重和日志。
BATCH_SIZE="${BATCH_SIZE:-1}"
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

# 模型结构、label 开关、label 数量和 SEQ_LEN 不在 shell 中写默认值；
# load_model_config_from_json 会从 MODEL_CONFIG_JSON 严格加载，缺字段立即退出。

ASCEND_RT_VISIBLE_DEVICES_VALUE="${ASCEND_RT_VISIBLE_DEVICES_VALUE:-0,1,2,3,4,5,6,7}"
HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-7200}"
HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-7200}"
HCCL_WHITELIST_DISABLE="${HCCL_WHITELIST_DISABLE:-1}"
ASCEND_TOOLKIT_HOME="${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/ascend-toolkit/latest}"
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
  - generate-flat uses merge_macrogene_rounds_parallel.py --test-mode.
  - model/training-entry parameters are loaded from MODEL_CONFIG_JSON.
  - missing required fields in MODEL_CONFIG_JSON fail fast; no shell defaults are used for them.
  - all runs preflight, generate-flat, validate-data, and commands.
  - launch starts the existing SSH launcher. Use DRY_RUN=1 with launch to print only.
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

  local assignments
  assignments="$("$PYTHON_BIN" - "$MODEL_CONFIG_JSON" <<'PY'
import json
import shlex
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
config = json.loads(config_path.read_text(encoding="utf-8"))


def emit(name, value):
    if value is None:
        return
    if isinstance(value, bool):
        value = "true" if value else "false"
    else:
        value = str(value)
    print(f"{name}={shlex.quote(value)}")


def pick_seq_len():
    missing = [
        key for key in ("vocab_size", "max_position_embeddings")
        if key not in config or config[key] is None
    ]
    if missing:
        raise SystemExit(
            f"{config_path}: missing required sequence fields: {', '.join(missing)}"
        )
    vocab_seq_len = int(config["vocab_size"]) - 1
    position_seq_len = int(config["max_position_embeddings"]) - 1
    if vocab_seq_len != position_seq_len:
        raise SystemExit(
            f"{config_path}: vocab_size-1 ({vocab_seq_len}) != "
            f"max_position_embeddings-1 ({position_seq_len})"
        )
    return position_seq_len


emit("SEQ_LEN", pick_seq_len())

field_map = {
    "HIDDEN_SIZE": "hidden_size",
    "NUM_HIDDEN_LAYERS": "num_hidden_layers",
    "NUM_ATTENTION_HEADS": "num_attention_heads",
    "INTERMEDIATE_SIZE": "intermediate_size",
    "HIDDEN_ACT": "hidden_act",
    "HIDDEN_DROPOUT_PROB": "hidden_dropout_prob",
    "CELL_HIDDEN_SIZE": "cell_hidden_size",
    "ATTENTION_PROBS_DROPOUT_PROB": "attention_probs_dropout_prob",
    "TYPE_VOCAB_SIZE": "type_vocab_size",
    "INITIALIZER_RANGE": "initializer_range",
    "LAYER_NORM_EPS": "layer_norm_eps",
    "ATTN_IMPLEMENTATION": "_attn_implementation",
    "USE_BATCH_LABELS": "use_batch_labels",
    "NUM_BATCH_LABELS": "num_batch_labels",
    "USE_SPECIES_LABELS": "use_species_labels",
    "NUM_SPECIES_LABELS": "num_species_labels",
    "USE_TISSUE_LABELS": "use_tissue_labels",
    "NUM_TISSUE_LABELS": "num_tissue_labels",
    "USE_SEQMETHOD_LABELS": "use_seqmethod_labels",
    "NUM_SEQMETHOD_LABELS": "num_seqmethod_labels",
    "USE_DISEASE_LABELS": "use_disease_labels",
    "NUM_DISEASE_LABELS": "num_disease_labels",
    "USE_AGE_LABELS": "use_age_labels",
    "NUM_AGE_LABELS": "num_age_labels",
    "USE_SEX_LABELS": "use_sex_labels",
    "NUM_SEX_LABELS": "num_sex_labels",
    "CELL_EMB_STYLE": "cell_emb_style",
    "CHUNK_SIZE_FEED_FORWARD": "chunk_size_feed_forward",
    "EXPLICIT_ZERO_PROB": "explicit_zero_prob",
}

for env_name, json_key in field_map.items():
    if json_key not in config or config[json_key] is None:
        raise SystemExit(f"{config_path}: missing required field: {json_key}")
    emit(env_name, config[json_key])
PY
)"
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
  log "stage2 root: $STAGE2_ROOT"
  log "input dirs: ${BATCH_DIRS[*]}"
  log "merged test dir: $MERGED_TEST_DIR"
  log "flat test dir: $FLAT_TEST_DIR"
  log "500M config: hidden_size=$HIDDEN_SIZE layers=$NUM_HIDDEN_LAYERS heads=$NUM_ATTENTION_HEADS intermediate_size=$INTERMEDIATE_SIZE"
  "$PYTHON_BIN" - "$SEQ_LEN" "$SOURCE_PREFLIGHT_FILES_PER_BATCH" "$SOURCE_PREFLIGHT_MAX_SCAN" "${BATCH_DIRS[@]}" <<'PY'
import sys
from pathlib import Path

import pyarrow.parquet as pq

seq_len = int(sys.argv[1])
sample_per_batch = int(sys.argv[2])
max_scan = int(sys.argv[3])
batch_dirs = [Path(p) for p in sys.argv[4:]]
required = {
    "X", "soma_joinid", "dataset_id", "assay", "cell_type",
    "development_stage", "disease", "tissue", "sex", "tech_sample",
    "species", "idx",
}

errors = []
for batch_dir in batch_dirs:
    if not batch_dir.exists():
        errors.append(f"missing input directory: {batch_dir}")
        continue
    files = sorted(batch_dir.glob("*/macrogene_*.parquet"))
    print(f"[INFO] {batch_dir}: matched macrogene parquet files={len(files)}")
    if not files:
        errors.append(f"no macrogene parquet files under {batch_dir}")
        continue

    checked = 0
    scanned = 0
    empty = 0
    bad = []
    for path in files:
        scanned += 1
        if scanned > max_scan and checked == 0:
            break
        if path.stat().st_size == 0:
            empty += 1
            continue
        try:
            parquet = pq.ParquetFile(path)
            schema_names = set(parquet.schema_arrow.names)
            missing = sorted(required - schema_names)
            if missing:
                bad.append(f"{path}: missing columns {missing}")
                continue
            if parquet.metadata.num_rows <= 0:
                bad.append(f"{path}: no rows")
                continue
            table = parquet.read_row_group(0, columns=["X"]).slice(0, 1)
            first_x = table.column("X")[0].as_py()
            if len(first_x) != seq_len:
                bad.append(f"{path}: X length {len(first_x)} != seq_len {seq_len}")
                continue
        except Exception as exc:
            bad.append(f"{path}: {type(exc).__name__}: {exc}")
            continue
        checked += 1
        if checked >= sample_per_batch:
            break

    print(f"[INFO] {batch_dir}: checked valid files={checked}, empty placeholders seen={empty}")
    if checked == 0:
        hint = " This is expected on a workstation with placeholder data; run this step on the data server."
        errors.append(f"no valid non-empty parquet files found in sampled scan for {batch_dir}.{hint}")
    if bad:
        errors.extend(bad[:5])

if errors:
    print("[ERROR] source preflight failed:")
    for item in errors:
        print(f"  - {item}")
    sys.exit(1)

print("[OK] source preflight passed")
PY
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
    "${PROJECT_ROOT}/merge_macrogene_rounds_parallel.py"
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
    "${PROJECT_ROOT}/shuffle_macrogene_rounds_parallel.py"
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

  log "merge test-mode command:"
  printf "%q " "${merge_cmd[@]}"; printf "\n"
  "${merge_cmd[@]}"

  log "flatten/shuffle command:"
  printf "%q " "${flatten_cmd[@]}"; printf "\n"
  "${flatten_cmd[@]}"
}

validate_data() {
  mkdir -p "$COMMAND_DIR"
  export FLAT_TEST_DIR COMMAND_DIR EMB_PATH SEQ_LEN WORLD_SIZE NNODES NPROC_PER_NODE
  export MAX_VALIDATE_ROWS_PER_FILE MAX_VALIDATE_FILES REQUIRE_EMBEDDINGS
  export NUM_BATCH_LABELS NUM_SPECIES_LABELS NUM_TISSUE_LABELS NUM_SEQMETHOD_LABELS
  export NUM_DISEASE_LABELS NUM_AGE_LABELS NUM_SEX_LABELS
  "$PYTHON_BIN" <<'PY'
import csv
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

flat_dir = Path(os.environ["FLAT_TEST_DIR"])
command_dir = Path(os.environ["COMMAND_DIR"])
emb_path = Path(os.environ["EMB_PATH"])
seq_len = int(os.environ["SEQ_LEN"])
world_size = int(os.environ["WORLD_SIZE"])
nnodes = int(os.environ["NNODES"])
nproc_per_node = int(os.environ["NPROC_PER_NODE"])
sample_rows = int(os.environ["MAX_VALIDATE_ROWS_PER_FILE"])
max_files = int(os.environ["MAX_VALIDATE_FILES"])
require_embeddings = os.environ["REQUIRE_EMBEDDINGS"] == "1"

required_cols = [
    "X", "soma_joinid", "dataset_id", "assay", "cell_type",
    "development_stage", "disease", "tissue", "sex", "tech_sample",
    "species", "idx",
]
label_limits = {
    "assay": int(os.environ["NUM_SEQMETHOD_LABELS"]),
    "tech_sample": int(os.environ["NUM_BATCH_LABELS"]),
    "species": int(os.environ["NUM_SPECIES_LABELS"]),
    "tissue": int(os.environ["NUM_TISSUE_LABELS"]),
    "disease": int(os.environ["NUM_DISEASE_LABELS"]),
    "development_stage": int(os.environ["NUM_AGE_LABELS"]),
    "sex": int(os.environ["NUM_SEX_LABELS"]),
}

errors = []
warnings = []
files = sorted(flat_dir.glob("all_flatten_part_*.parquet"))
if not files:
    errors.append(f"no all_flatten_part_*.parquet files found in {flat_dir}")

file_rows = []
species_counter = Counter()
validate_files = files if max_files == 0 else files[:max_files]

for path in files:
    try:
        parquet = pq.ParquetFile(path)
        file_rows.append(parquet.metadata.num_rows)
    except Exception as exc:
        errors.append(f"{path}: cannot read parquet metadata: {type(exc).__name__}: {exc}")
        file_rows.append(0)

for path in validate_files:
    try:
        parquet = pq.ParquetFile(path)
        names = parquet.schema_arrow.names
        missing = [col for col in required_cols if col not in names]
        if missing:
            errors.append(f"{path}: missing columns {missing}; found {names}")
            continue

        table = parquet.read(columns=required_cols)
        if sample_rows > 0 and table.num_rows > sample_rows:
            table = table.slice(0, sample_rows)
        df = table.to_pandas()
        if df.empty:
            errors.append(f"{path}: validation sample is empty")
            continue

        for idx, x in enumerate(df["X"]):
            arr = np.asarray(x, dtype=np.float64)
            if arr.shape != (seq_len,):
                errors.append(f"{path}: row {idx} X shape {arr.shape} != ({seq_len},)")
                break
            if not np.isfinite(arr).all():
                errors.append(f"{path}: row {idx} X contains non-finite values")
                break

        for col, limit in label_limits.items():
            vals = pd.to_numeric(df[col], errors="raise")
            if vals.isnull().any():
                errors.append(f"{path}: {col} has null values")
            bad = vals[(vals < 0) | (vals >= limit)]
            if len(bad) > 0:
                errors.append(
                    f"{path}: {col} values outside [0, {limit}): "
                    f"min={int(vals.min())}, max={int(vals.max())}"
                )
    except Exception as exc:
        errors.append(f"{path}: validation failed: {type(exc).__name__}: {exc}")

for path in files:
    try:
        table = pq.read_table(path, columns=["species"])
        species_counter.update(int(x) for x in table.column("species").to_pylist())
    except Exception as exc:
        warnings.append(f"{path}: could not read species distribution: {exc}")

manifest_path = flat_dir / "shuffle_manifest.csv"
if manifest_path.exists():
    with manifest_path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    manifest_total = sum(int(row["num_rows"]) for row in rows)
    metadata_total = sum(file_rows)
    manifest_files = {row["output_file"] for row in rows}
    actual_files = {path.name for path in files}
    if manifest_total != metadata_total:
        errors.append(f"manifest row total {manifest_total} != parquet metadata rows {metadata_total}")
    missing_manifest_files = actual_files - manifest_files
    if missing_manifest_files:
        errors.append(f"manifest missing output files: {sorted(missing_manifest_files)[:5]}")
else:
    warnings.append(f"missing shuffle manifest: {manifest_path}")

if files and len(files) < world_size:
    errors.append(
        f"flat file count {len(files)} < world_size {world_size}; "
        "DistributedFileSampler(drop_last=True) would give empty/invalid rank shards"
    )

if files:
    if len(files) % world_size != 0:
        dropped = len(files) - math.ceil((len(files) - world_size) / world_size) * world_size
        warnings.append(
            f"flat file count {len(files)} is not divisible by world_size {world_size}; "
            f"current drop_last=True sampler will drop {dropped} shuffled file(s) per epoch"
        )

    if len(files) % world_size == 0:
        samples_per_rank = math.ceil(len(files) / world_size)
    else:
        samples_per_rank = math.ceil((len(files) - world_size) / world_size)
    total_size = samples_per_rank * world_size
    if samples_per_rank <= 0:
        errors.append(f"samples_per_rank computed as {samples_per_rank}; need more flat files")
    else:
        indices = list(range(len(files)))[:total_size]
        plan_path = command_dir / "distributed_file_plan_epoch0.csv"
        with plan_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[
                    "node_rank", "local_rank", "global_rank", "num_files",
                    "num_rows", "files",
                ],
            )
            writer.writeheader()
            for rank in range(world_size):
                rank_indices = indices[rank:total_size:world_size]
                rank_files = [files[i] for i in rank_indices]
                writer.writerow({
                    "node_rank": rank // nproc_per_node,
                    "local_rank": rank % nproc_per_node,
                    "global_rank": rank,
                    "num_files": len(rank_files),
                    "num_rows": sum(file_rows[i] for i in rank_indices),
                    "files": ";".join(path.name for path in rank_files),
                })
        print(f"[INFO] wrote distributed file plan: {plan_path}")

embedding_files = [
    "2nd_run_macrogene_features_sum_esm2.npy",
    "2nd_run_macrogene_features_sum_gene_desc.npy",
    "2nd_run_macrogene_features_sum_dnaseq.npy",
]
if emb_path.exists():
    for name in embedding_files:
        path = emb_path / name
        if not path.exists():
            errors.append(f"missing embedding file: {path}")
            continue
        arr = np.load(path, mmap_mode="r")
        if arr.ndim != 2:
            errors.append(f"{path}: expected 2D embedding array, got shape={arr.shape}")
        if arr.shape[0] != seq_len:
            errors.append(f"{path}: first dimension {arr.shape[0]} != seq_len {seq_len}")
        print(f"[INFO] embedding {name}: shape={arr.shape}, dtype={arr.dtype}")
elif require_embeddings:
    errors.append(f"embedding directory does not exist: {emb_path}")
else:
    warnings.append(f"embedding directory does not exist: {emb_path}")

summary = {
    "flat_dir": str(flat_dir),
    "num_files": len(files),
    "total_rows": int(sum(file_rows)),
    "world_size": world_size,
    "seq_len": seq_len,
    "species_distribution": dict(sorted(species_counter.items())),
}
print(json.dumps(summary, indent=2, sort_keys=True))

if warnings:
    print("[WARN] validation warnings:")
    for item in warnings:
        print(f"  - {item}")

if errors:
    print("[ERROR] validation failed:")
    for item in errors:
        print(f"  - {item}")
    sys.exit(1)

print("[OK] flat data validation passed")
PY
}

train_args() {
  TRAIN_ARGS=(
    "--data_path=${DATA_PATH}"
    "--num_of_used_data=${NUM_OF_USED_DATA}"
    "--emb_path=${EMB_PATH}"
    "--seq_len=${SEQ_LEN}"
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
    "--hidden_size=${HIDDEN_SIZE}"
    "--num_hidden_layers=${NUM_HIDDEN_LAYERS}"
    "--num_attention_heads=${NUM_ATTENTION_HEADS}"
    "--intermediate_size=${INTERMEDIATE_SIZE}"
    "--hidden_act=${HIDDEN_ACT}"
    "--hidden_dropout_prob=${HIDDEN_DROPOUT_PROB}"
    "--cell_hidden_size=${CELL_HIDDEN_SIZE}"
    "--attention_probs_dropout_prob=${ATTENTION_PROBS_DROPOUT_PROB}"
    "--type_vocab_size=${TYPE_VOCAB_SIZE}"
    "--initializer_range=${INITIALIZER_RANGE}"
    "--layer_norm_eps=${LAYER_NORM_EPS}"
    "--_attn_implementation=${ATTN_IMPLEMENTATION}"
    "--use_batch_labels=${USE_BATCH_LABELS}"
    "--num_batch_labels=${NUM_BATCH_LABELS}"
    "--use_species_labels=${USE_SPECIES_LABELS}"
    "--num_species_labels=${NUM_SPECIES_LABELS}"
    "--use_tissue_labels=${USE_TISSUE_LABELS}"
    "--num_tissue_labels=${NUM_TISSUE_LABELS}"
    "--use_seqmethod_labels=${USE_SEQMETHOD_LABELS}"
    "--num_seqmethod_labels=${NUM_SEQMETHOD_LABELS}"
    "--use_disease_labels=${USE_DISEASE_LABELS}"
    "--num_disease_labels=${NUM_DISEASE_LABELS}"
    "--use_age_labels=${USE_AGE_LABELS}"
    "--num_age_labels=${NUM_AGE_LABELS}"
    "--use_sex_labels=${USE_SEX_LABELS}"
    "--num_sex_labels=${NUM_SEX_LABELS}"
    "--cell_emb_style=${CELL_EMB_STYLE}"
    "--chunk_size_feed_forward=${CHUNK_SIZE_FEED_FORWARD}"
    "--explicit_zero_prob=${EXPLICIT_ZERO_PROB}"
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
    torchrun
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
    print_export ASCEND_HOME_PATH "$ASCEND_HOME_PATH"
    echo
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
    print_export HOSTS "$HOSTS"
    print_export NNODES "$NNODES"
    print_export AUTO_NNODES 0
    print_export NPROC_PER_NODE "$NPROC_PER_NODE"
    print_export MASTER_ADDR "$MASTER_ADDR"
    print_export MASTER_PORT "$MASTER_PORT"
    print_export WORKDIR "$WORKDIR"
    print_export TRAIN_ENTRY "$TRAIN_ENTRY"
    print_export LOG_SUBDIR "$LOG_SUBDIR"
    print_export TRAIN_DATASET "$TRAIN_DATASET"
    print_export DATA_ROOT "$STAGE2_ROOT"
    print_export DATA_PATH "$DATA_PATH"
    print_export NUM_OF_USED_DATA "$NUM_OF_USED_DATA"
    print_export EMB_ROOT "$EMB_ROOT"
    print_export EMB_PATH "$EMB_PATH"
    print_export SEQ_LEN "$SEQ_LEN"
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
    print_export HIDDEN_SIZE "$HIDDEN_SIZE"
    print_export NUM_HIDDEN_LAYERS "$NUM_HIDDEN_LAYERS"
    print_export NUM_ATTENTION_HEADS "$NUM_ATTENTION_HEADS"
    print_export INTERMEDIATE_SIZE "$INTERMEDIATE_SIZE"
    print_export HIDDEN_ACT "$HIDDEN_ACT"
    print_export HIDDEN_DROPOUT_PROB "$HIDDEN_DROPOUT_PROB"
    print_export CELL_HIDDEN_SIZE "$CELL_HIDDEN_SIZE"
    print_export ATTENTION_PROBS_DROPOUT_PROB "$ATTENTION_PROBS_DROPOUT_PROB"
    print_export TYPE_VOCAB_SIZE "$TYPE_VOCAB_SIZE"
    print_export INITIALIZER_RANGE "$INITIALIZER_RANGE"
    print_export LAYER_NORM_EPS "$LAYER_NORM_EPS"
    print_export ATTN_IMPLEMENTATION "$ATTN_IMPLEMENTATION"
    print_export USE_BATCH_LABELS "$USE_BATCH_LABELS"
    print_export NUM_BATCH_LABELS "$NUM_BATCH_LABELS"
    print_export USE_SPECIES_LABELS "$USE_SPECIES_LABELS"
    print_export NUM_SPECIES_LABELS "$NUM_SPECIES_LABELS"
    print_export USE_TISSUE_LABELS "$USE_TISSUE_LABELS"
    print_export NUM_TISSUE_LABELS "$NUM_TISSUE_LABELS"
    print_export USE_SEQMETHOD_LABELS "$USE_SEQMETHOD_LABELS"
    print_export NUM_SEQMETHOD_LABELS "$NUM_SEQMETHOD_LABELS"
    print_export USE_DISEASE_LABELS "$USE_DISEASE_LABELS"
    print_export NUM_DISEASE_LABELS "$NUM_DISEASE_LABELS"
    print_export USE_AGE_LABELS "$USE_AGE_LABELS"
    print_export NUM_AGE_LABELS "$NUM_AGE_LABELS"
    print_export USE_SEX_LABELS "$USE_SEX_LABELS"
    print_export NUM_SEX_LABELS "$NUM_SEX_LABELS"
    print_export CELL_EMB_STYLE "$CELL_EMB_STYLE"
    print_export CHUNK_SIZE_FEED_FORWARD "$CHUNK_SIZE_FEED_FORWARD"
    print_export EXPLICIT_ZERO_PROB "$EXPLICIT_ZERO_PROB"
    print_export ASCEND_RT_VISIBLE_DEVICES_VALUE "$ASCEND_RT_VISIBLE_DEVICES_VALUE"
    print_export HCCL_CONNECT_TIMEOUT "$HCCL_CONNECT_TIMEOUT"
    print_export HCCL_EXEC_TIMEOUT "$HCCL_EXEC_TIMEOUT"
    print_export HCCL_WHITELIST_DISABLE "$HCCL_WHITELIST_DISABLE"
    print_export ASCEND_TOOLKIT_HOME "$ASCEND_TOOLKIT_HOME"
    print_export ASCEND_HOME_PATH "$ASCEND_HOME_PATH"
    echo
    echo 'exec bash scripts/train_multinode.sh'
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
    echo "Stage 2 500M three-node test commands"
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
  log "launch distributed training through existing scripts/train_multinode.sh"
  bash "${COMMAND_DIR}/launch_500m_3nodes.sh"
}

resolve_train_out_dir() {
  export OUT_PATH WORKDIR HIDDEN_SIZE NUM_HIDDEN_LAYERS NUM_ATTENTION_HEADS
  export HIDDEN_DROPOUT_PROB LEARNING_RATE MIN_LR WEIGHT_DECAY WARMUP_RATIO
  "$PYTHON_BIN" <<'PY'
import os
from pathlib import Path

out_path = os.environ["OUT_PATH"].format(
    hidden_size=int(os.environ["HIDDEN_SIZE"]),
    num_hidden_layers=int(os.environ["NUM_HIDDEN_LAYERS"]),
    num_attention_heads=int(os.environ["NUM_ATTENTION_HEADS"]),
    hidden_dropout_prob=float(os.environ["HIDDEN_DROPOUT_PROB"]),
    learning_rate=float(os.environ["LEARNING_RATE"]),
    min_lr=float(os.environ["MIN_LR"]),
    weight_decay=float(os.environ["WEIGHT_DECAY"]),
    warmup_ratio=float(os.environ["WARMUP_RATIO"]),
)
path = Path(out_path)
if not path.is_absolute():
    path = Path(os.environ["WORKDIR"]) / path
print(path.resolve())
PY
}

check_training() {
  local out_dir="${TRAIN_OUT_DIR:-$(resolve_train_out_dir)}"
  local node_log_dir="${NODE_LOG_DIR:-${WORKDIR}/${LOG_SUBDIR}}"
  export TRAIN_OUT_DIR="$out_dir" NODE_LOG_DIR="$node_log_dir" WORLD_SIZE NNODES NPROC_PER_NODE EPOCH
  "$PYTHON_BIN" <<'PY'
import ast
import csv
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

out_dir = Path(os.environ["TRAIN_OUT_DIR"])
node_log_dir = Path(os.environ["NODE_LOG_DIR"])
world_size = int(os.environ["WORLD_SIZE"])
epoch_count = int(os.environ["EPOCH"])

errors = []
warnings = []

if not out_dir.exists():
    errors.append(f"training output directory does not exist: {out_dir}")
else:
    rank_logs = sorted(out_dir.glob("log.*.txt"))
    loss_logs = sorted(out_dir.glob("loss_to_log.*.txt"))
    all_pt_files = sorted(out_dir.glob("SC-node-*-rank-*-epoch-*-step-*-loss-*.pt"))
    weights = [p for p in all_pt_files if not p.name.endswith(".optimizer.pt")]
    optimizer_weights = sorted(out_dir.glob("SC-node-*-rank-*-epoch-*-step-*-loss-*.optimizer.pt"))

    print(f"[INFO] output dir: {out_dir}")
    print(f"[INFO] rank logs={len(rank_logs)}, loss logs={len(loss_logs)}, weights={len(weights)}, optimizer states={len(optimizer_weights)}")

    if len(rank_logs) < world_size:
        errors.append(f"expected at least {world_size} rank logs, found {len(rank_logs)}")
    if len(loss_logs) < world_size:
        errors.append(f"expected at least {world_size} loss logs, found {len(loss_logs)}")

    final_epoch = epoch_count + 1
    final_pattern = re.compile(rf"SC-node-\d+-rank-\d+-epoch-{final_epoch:02d}-step-0-loss-0\.000000\.pt$")
    final_weights = [p for p in weights if final_pattern.search(p.name)]
    if len(final_weights) < world_size:
        errors.append(f"expected at least {world_size} final model weights for epoch {final_epoch:02d}, found {len(final_weights)}")

    bad_pattern = re.compile(r"(Traceback|RuntimeError|ValueError|out of memory|\bnan\b)", re.IGNORECASE)
    data_pattern = re.compile(r"Node:\s*([^,]+),\s*Rank:\s*([^,]+),\s*Epoch:\s*(\d+),\s*Data:\s*(\[.*\])")
    data_by_epoch = defaultdict(list)
    ranks_with_loss = set()

    for log_path in rank_logs:
        text = log_path.read_text(encoding="utf-8", errors="replace")
        bad = bad_pattern.search(text)
        if bad:
            errors.append(f"{log_path}: found failure marker {bad.group(1)!r}")
        if "loss:" in text:
            ranks_with_loss.add(log_path.name)
        for match in data_pattern.finditer(text):
            node = match.group(1).strip()
            rank = match.group(2).strip()
            epoch = int(match.group(3))
            try:
                files = ast.literal_eval(match.group(4))
            except Exception:
                files = []
                warnings.append(f"{log_path}: could not parse Data list for epoch {epoch}")
            data_by_epoch[epoch].append((node, rank, files, log_path.name))

    if len(ranks_with_loss) < world_size:
        errors.append(f"expected loss lines in {world_size} rank logs, found {len(ranks_with_loss)}")

    for epoch in range(epoch_count):
        records = data_by_epoch.get(epoch, [])
        if len(records) < world_size:
            errors.append(f"epoch {epoch}: expected data assignment records for {world_size} ranks, found {len(records)}")
            continue
        all_files = []
        empty_ranks = []
        for node, rank, files, log_name in records:
            if not files:
                empty_ranks.append(log_name)
            all_files.extend(files)
        if empty_ranks:
            errors.append(f"epoch {epoch}: ranks with empty data assignments: {empty_ranks[:5]}")
        duplicates = [name for name, count in Counter(all_files).items() if count > 1]
        if duplicates:
            errors.append(f"epoch {epoch}: duplicate file assignment across ranks: {duplicates[:10]}")
        print(f"[INFO] epoch {epoch}: assigned files={len(all_files)}, unique files={len(set(all_files))}")

    for loss_path in loss_logs:
        try:
            with loss_path.open(newline="", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            if not rows:
                errors.append(f"{loss_path}: empty loss csv")
        except Exception as exc:
            errors.append(f"{loss_path}: cannot parse loss csv: {exc}")

if node_log_dir.exists():
    node_logs = sorted(node_log_dir.glob("node_rank*.log"))
    print(f"[INFO] node launcher logs={len(node_logs)} in {node_log_dir}")
    bad_pattern = re.compile(r"(Traceback|RuntimeError|ValueError|out of memory|\bnan\b)", re.IGNORECASE)
    for path in node_logs:
        text = path.read_text(encoding="utf-8", errors="replace")
        bad = bad_pattern.search(text)
        if bad:
            errors.append(f"{path}: found failure marker {bad.group(1)!r}")
        if "Complete pretraining!" not in text:
            warnings.append(f"{path}: missing 'Complete pretraining!' marker; job may still be running")
else:
    warnings.append(f"node log directory does not exist: {node_log_dir}")

if warnings:
    print("[WARN] training check warnings:")
    for item in warnings:
        print(f"  - {item}")

if errors:
    print("[ERROR] training check failed:")
    for item in errors:
        print(f"  - {item}")
    sys.exit(1)

print("[OK] training artifacts and logs passed checks")
PY
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
