#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Single-node (1 node x 8 NPUs) regression run, driven through the existing
# multi-node launcher (scripts/launch_multinode_torchrun.sh) in single-node mode.
# This wrapper ONLY sets parameters/env; it does not modify any launcher or the
# training entry. The multi-node launcher already runs node_rank 0 locally
# (no ssh/scp) when HOSTS has a single host and SYNC_SELF=0.
#
#   * INPUTS pinned to the early / first-version files ("necessary file configs"):
#       - 1st-run macrogene embeddings (862 rows, Stage1_macrogene_embeddings),
#         exposed under the selected 2nd_run_* names
#       - early shuffled training data in the old 862-macrogene layout
#       - model config args_1st_run_<size>.json (seq_len=862,
#         use_batch_labels=true, label dims 12028/11/154/28/143/5/3)
#   * CODE + TRAINING RECIPE follow the current repo, matching the second
#       resumed run in:
#       training_output_100m_data_1_2_3_E2_huber_fp32_from_scratch_20260529_160058
#       (huber loss, fp32 AMP, lr 1e-6->1e-7, adaptive clip with norm-skip +
#        aborts OFF and the 1e8 hard raw-norm fuse kept, beta2 0.98, warmup
#        iters 2000, warmup ratio 0.10, epoch 5). Resume/init args are not set
#        by default because this wrapper starts a fresh early-data run.
#
# ---------------------------------------------------------------------------
# Optional:
#   DATA_PATH    flattened, training-ready early parquet dir whose samples carry
#                862 macrogene features. Default:
#                /data/disk1/SpeciesLLM/all_shuffled_data
#
# Embeddings (1st-run, 862 rows) -- pick ONE:
#   SRC_EMB_PATH dir holding 1st_run_macrogene_features_sum_*.npy; a sibling
#                "<dir>_as_2nd_run" of symlinks is built automatically
#                (default: <repo>/Stage1_macrogene_embeddings)
#   EMB_PATH     a dir that already exposes the arrays under 2nd_run_* names
#
#   MODEL_SIZE   100m (default, current architecture) or 500m (early-config
#                structure, kept only for reference). Selects args_1st_run_<size>.json.
#
# Example:
#   bash scripts/launch_singlenode_1st_inputs_current_recipe.sh
#
# Preview without launching:
#   DATA_PATH=... DRY_RUN=1 bash scripts/launch_singlenode_1st_inputs_current_recipe.sh
###############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"

DATA_PATH="${DATA_PATH:-/data/disk1/SpeciesLLM/all_shuffled_data}"

MODEL_SIZE="${MODEL_SIZE:-100m}"
case "$MODEL_SIZE" in
  100m|500m) ;;
  *) echo "[ERROR] MODEL_SIZE must be 100m or 500m (got: ${MODEL_SIZE})" >&2; exit 1 ;;
esac

# --- resolve embeddings: build 2nd_run_* symlinks from 1st_run_* if needed ---
SRC_EMB_PATH="${SRC_EMB_PATH:-${PROJECT_ROOT}/Stage1_macrogene_embeddings}"
GENE_EMBEDDING_MODALITIES="${GENE_EMBEDDING_MODALITIES:-esm2,gene_desc,dnaseq}"
if [[ -z "${EMB_PATH:-}" ]]; then
  EMB_PATH="${SRC_EMB_PATH%/}_as_2nd_run"
  mkdir -p "$EMB_PATH"
  IFS=',' read -r -a embedding_modalities <<< "$GENE_EMBEDDING_MODALITIES"
  for raw_modality in "${embedding_modalities[@]}"; do
    f="${raw_modality//[[:space:]]/}"
    case "$f" in
      esm2|gene_desc|dnaseq) ;;
      *) echo "[ERROR] unsupported GENE_EMBEDDING_MODALITIES item: $f" >&2; exit 1 ;;
    esac
    src="${SRC_EMB_PATH%/}/1st_run_macrogene_features_sum_${f}.npy"
    dst="${EMB_PATH%/}/2nd_run_macrogene_features_sum_${f}.npy"
    [[ -f "$src" ]] || { echo "[ERROR] missing 1st-run embedding: $src" >&2; exit 1; }
    if [[ ! -e "$dst" ]]; then
      ln -s "$src" "$dst"
      echo "[repro] linked $dst -> $src"
    fi
  done
fi

# --- inputs (UPPERCASE env is read by launch_multinode_torchrun.sh) ---
export DATA_PATH EMB_PATH
export GENE_EMBEDDING_MODALITIES
export MODEL_CONFIG_JSON="${MODEL_CONFIG_JSON:-${PROJECT_ROOT}/Stage2_macrogene_embeddings/args_1st_run_${MODEL_SIZE}.json}"
export OUT_PATH="${OUT_PATH:-training_output_${MODEL_SIZE}_1st_inputs_current_recipe}"
export NUM_OF_USED_DATA="${NUM_OF_USED_DATA:-0}"

# --- single-node topology for the multi-node launcher ---
# One host + AUTO_NNODES -> NNODES=1; rank 0 is the local rank, so no ssh/scp.
export HOSTS="${HOSTS:-127.0.0.1}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export NNODES="${NNODES:-1}"
export AUTO_NNODES="${AUTO_NNODES:-1}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export LOCAL_NODE_RANK="${LOCAL_NODE_RANK:-0}"
export SYNC_SELF="${SYNC_SELF:-0}"
export SSH_PASSWORD="${SSH_PASSWORD:-}"
export WORKDIR="${WORKDIR:-${PROJECT_ROOT}}"

# ===== CURRENT 100M recipe (set explicitly: the multi-node launcher defaults
# for GRAD_CLIP_MAX/GRAD_SKIP_*/HARD_RAW_NORM_LIMIT/COMPILE/etc. differ) =====
# Per-rank batch 512 matches the cited run; one node => global batch 8*512*1.
# The cited run used 3 nodes (global 12288); set GRADIENT_ACCUMULATION_STEPS=3
# to match that. Lower BATCH_SIZE + raise grad_accum if 512 OOMs at seq_len=862.
export BATCH_SIZE="${BATCH_SIZE:-512}"
export EPOCH="${EPOCH:-5}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
export GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"
export TRAIN_MVC="${TRAIN_MVC:-true}"
export LEARNING_RATE="${LEARNING_RATE:-0.000001}"
export MIN_LR="${MIN_LR:-0.0000001}"
export DECAY_LR="${DECAY_LR:-true}"
export WARMUP_ITERS="${WARMUP_ITERS:-2000}"
export WARMUP_RATIO="${WARMUP_RATIO:-0.10}"
export LR_DECAY_EPOCHS="${LR_DECAY_EPOCHS:-5}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
export SAVE_DATA_INTERVAL="${SAVE_DATA_INTERVAL:-5120000}"
export BETA1="${BETA1:-0.9}"
export BETA2="${BETA2:-0.98}"

# precision + regression loss
export AMP_DTYPE="${AMP_DTYPE:-float32}"
export GEP_LOSS="${GEP_LOSS:-huber}"
export HUBER_DELTA="${HUBER_DELTA:-5.0}"
export GEP_LOSS_WEIGHT="${GEP_LOSS_WEIGHT:-1.0}"
export ZERO_PROB_LOSS_WEIGHT="${ZERO_PROB_LOSS_WEIGHT:-1.0}"
export GEPC_LOSS_WEIGHT="${GEPC_LOSS_WEIGHT:-0.1}"
export GEPC_ZERO_PROB_LOSS_WEIGHT="${GEPC_ZERO_PROB_LOSS_WEIGHT:-0.1}"

# adaptive grad clip ON, norm-skip + aborts OFF, keep the 1e8 hard fuse
export GRAD_CLIP="${GRAD_CLIP:-1.0}"
export ADAPTIVE_GRAD_CLIP="${ADAPTIVE_GRAD_CLIP:-true}"
export GRAD_CLIP_EMA_BETA="${GRAD_CLIP_EMA_BETA:-0.98}"
export GRAD_CLIP_RATIO="${GRAD_CLIP_RATIO:-3.0}"
export GRAD_CLIP_MIN="${GRAD_CLIP_MIN:-0.5}"
export GRAD_CLIP_MAX="${GRAD_CLIP_MAX:-1000.0}"
export GRAD_CLIP_WARMUP_STEPS="${GRAD_CLIP_WARMUP_STEPS:-200}"
export GRAD_SKIP_RATIO="${GRAD_SKIP_RATIO:-0}"
export GRAD_SKIP_MAX="${GRAD_SKIP_MAX:-0}"
export GRAD_CLIP_MAX_CONSECUTIVE_SKIPS="${GRAD_CLIP_MAX_CONSECUTIVE_SKIPS:-0}"
export GRAD_CLIP_EMA_RUNAWAY_FACTOR="${GRAD_CLIP_EMA_RUNAWAY_FACTOR:-0}"
export GRAD_CLIP_HARD_RAW_NORM_LIMIT="${GRAD_CLIP_HARD_RAW_NORM_LIMIT:-100000000}"

export MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-0}"
export INIT_MODEL_PATH="${INIT_MODEL_PATH:-}"
export INIT_OPTIMIZER_PATH="${INIT_OPTIMIZER_PATH:-}"
export RESUME_UPDATE_STEP="${RESUME_UPDATE_STEP:-0}"
export RESUME_START_EPOCH="${RESUME_START_EPOCH:-0}"
export RESUME_SKIP_BATCHES="${RESUME_SKIP_BATCHES:-0}"
export APPEND_OUTPUT_LOGS="${APPEND_OUTPUT_LOGS:-false}"

export LOG_INTERVAL="${LOG_INTERVAL:-10}"
export PROFILE_INTERVAL="${PROFILE_INTERVAL:-100}"
export NAN_CHECK_INTERVAL="${NAN_CHECK_INTERVAL:-10}"
export METRICS_FLUSH_INTERVAL="${METRICS_FLUSH_INTERVAL:-100}"
export LOG_LEVEL="${LOG_LEVEL:-INFO}"
export LOG_ALL_RANKS="${LOG_ALL_RANKS:-false}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export PREFETCH_FACTOR="${PREFETCH_FACTOR:-1}"
export PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-false}"
export PIN_MEMORY="${PIN_MEMORY:-true}"
export PARQUET_CHUNK_FILES="${PARQUET_CHUNK_FILES:-64}"
export COMPILE="${COMPILE:-false}"
export BACKEND="${BACKEND:-hccl}"
export DEVICE="${DEVICE:-npu}"
export DEVICE_TYPE="${DEVICE_TYPE:-npu}"

echo "[repro] single-node via multi-node launcher: MODEL_SIZE=${MODEL_SIZE}, config=${MODEL_CONFIG_JSON}"
echo "[repro] DATA_PATH=${DATA_PATH}"
echo "[repro] EMB_PATH=${EMB_PATH}"
echo "[repro] WORKDIR=${WORKDIR}"
echo "[repro] training runs detached; logs -> torchrun_logs/node_rank0.log"

exec bash "${SCRIPT_DIR}/launch_multinode_torchrun.sh"
