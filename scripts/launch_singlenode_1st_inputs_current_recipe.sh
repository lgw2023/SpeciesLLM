#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Single-node (1 node x 8 NPUs) regression run:
#   * INPUTS pinned to the early / first-version files:
#       - 1st-run macrogene embeddings (862 rows, Stage1_macrogene_embeddings)
#       - first-batch data in the old 862-macrogene layout
#       - model config args_1st_run_<size>.json  (seq_len=862,
#         use_batch_labels=true, label dims 12028/11/154/28/143/5/3)
#   * EVERYTHING ELSE follows the CURRENT training recipe, matching
#       training_output_100m_data_1_2_3_E2_huber_fp32_from_scratch_20260529_160058
#       (huber loss, fp32 AMP, lr 1e-6->1e-7, adaptive clip with skip/abort OFF
#        and the 1e8 hard raw-norm fuse kept, beta2 0.98, warmup 0.10, epoch 5).
#
# The training entry (train_MNodes_torchrun_mfu_preindexparquet.py) is NOT
# modified. This wrapper sets env and execs scripts/launch_singlenode_torchrun.sh.
#
# ---------------------------------------------------------------------------
# REQUIRED:
#   DATA_PATH    flattened, training-ready first-batch parquet dir whose samples
#                carry 862 macrogene features (collate raises if features != 862)
#
# Embeddings (1st-run, 862 rows) -- the training script hardcodes the
# 2nd_run_macrogene_features_sum_*.npy names, so the 1st-run arrays are exposed
# under those names. Pick ONE:
#   SRC_EMB_PATH dir holding 1st_run_macrogene_features_sum_*.npy; a sibling
#                "<dir>_as_2nd_run" of symlinks is built automatically
#                (default: <repo>/Stage1_macrogene_embeddings)
#   EMB_PATH     a dir that already exposes the arrays under 2nd_run_* names
#
# Optional:
#   MODEL_SIZE   100m (default, matches the cited reference run) or 500m
#                (matches the early first-version config). Selects
#                args_1st_run_<size>.json.
#
# Example:
#   DATA_PATH=/data/disk1/SpeciesLLM_obs/Stage2_SpeciesLLMData/<first_batch_flatten_862> \
#   bash scripts/launch_singlenode_1st_inputs_current_recipe.sh
###############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"

: "${DATA_PATH:?set DATA_PATH to the flattened first-batch (862-macrogene) parquet dir}"

MODEL_SIZE="${MODEL_SIZE:-100m}"
case "$MODEL_SIZE" in
  100m|500m) ;;
  *) echo "[ERROR] MODEL_SIZE must be 100m or 500m (got: ${MODEL_SIZE})" >&2; exit 1 ;;
esac

# --- resolve embeddings: build 2nd_run_* symlinks from 1st_run_* if requested ---
SRC_EMB_PATH="${SRC_EMB_PATH:-${PROJECT_ROOT}/Stage1_macrogene_embeddings}"
if [[ -z "${EMB_PATH:-}" ]]; then
  EMB_PATH="${SRC_EMB_PATH%/}_as_2nd_run"
  mkdir -p "$EMB_PATH"
  for f in esm2 gene_desc dnaseq; do
    src="${SRC_EMB_PATH%/}/1st_run_macrogene_features_sum_${f}.npy"
    dst="${EMB_PATH%/}/2nd_run_macrogene_features_sum_${f}.npy"
    [[ -f "$src" ]] || { echo "[ERROR] missing 1st-run embedding: $src" >&2; exit 1; }
    if [[ ! -e "$dst" ]]; then
      ln -s "$src" "$dst"
      echo "[repro] linked $dst -> $src"
    fi
  done
fi

export DATA_PATH EMB_PATH
export MODEL_CONFIG_JSON="${MODEL_CONFIG_JSON:-${PROJECT_ROOT}/Stage2_macrogene_embeddings/args_1st_run_${MODEL_SIZE}.json}"
export OUT_PATH="${OUT_PATH:-training_output_${MODEL_SIZE}_1st_inputs_current_recipe}"
export NUM_OF_USED_DATA="${NUM_OF_USED_DATA:-0}"

# ===== CURRENT training recipe (matches the cited 100M huber/fp32 run) =====
# Per-rank batch is unchanged from the 3-node reference (512/rank); on one node
# the global batch is 8*512*1 vs the reference 3*8*512*1. Set
# GRADIENT_ACCUMULATION_STEPS=3 to match the reference global batch of 12288.
export BATCH_SIZE="${BATCH_SIZE:-512}"
export EPOCH="${EPOCH:-5}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
export GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"
export LEARNING_RATE="${LEARNING_RATE:-0.000001}"
export MIN_LR="${MIN_LR:-0.0000001}"
export DECAY_LR="${DECAY_LR:-true}"
export WARMUP_RATIO="${WARMUP_RATIO:-0.10}"
export LR_DECAY_EPOCHS="${LR_DECAY_EPOCHS:-5}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
export BETA1="${BETA1:-0.9}"
export BETA2="${BETA2:-0.98}"

# precision + regression loss (current decision: fp32 + huber)
export AMP_DTYPE="${AMP_DTYPE:-float32}"
export GEP_LOSS="${GEP_LOSS:-huber}"
export HUBER_DELTA="${HUBER_DELTA:-5.0}"
export STATIC_GENE_DTYPE="${STATIC_GENE_DTYPE:-float32}"

# adaptive grad clip ON, but norm-skip + aborts OFF, keep the 1e8 hard fuse
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

export NAN_CHECK_INTERVAL="${NAN_CHECK_INTERVAL:-10}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export PREFETCH_FACTOR="${PREFETCH_FACTOR:-1}"
export PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-false}"
export PIN_MEMORY="${PIN_MEMORY:-true}"
export COMPILE="${COMPILE:-false}"

exec bash "${SCRIPT_DIR}/launch_singlenode_torchrun.sh"
