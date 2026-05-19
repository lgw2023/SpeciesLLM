#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" != "--__speciesllm-step2-1b-clean-env" ]]; then
  exec /usr/bin/env -i \
    HOME="${HOME:-/root}" \
    PATH="/data/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
    LANG=C.UTF-8 \
    bash "$0" --__speciesllm-step2-1b-clean-env "$@"
fi
shift

if [[ "$#" -gt 0 ]]; then
  echo "[ERROR] Unsupported arguments: $*" >&2
  echo "Use this script without environment-style overrides; training parameters are passed explicitly inside the script." >&2
  exit 1
fi

# 1B 模型三节点的训练测试

cd /data/disk1/SpeciesLLM

RUN_ID=$(date +%Y%m%d_%H%M%S)
OUT_PATH="/data/disk1/SpeciesLLM/training_output_1b_test_from_scratch_${RUN_ID}"

bash scripts/pretrain_3node.sh \
  MODEL_SCALE=1b \
  PREP_ACTION=all \
  RESET_TEST_OUTPUT=1 \
  SKIP_EXISTING=0 \
  RUN_TRAINING=1 \
  SKIP_DATA_SYNC=0 \
  PYTHON_BIN=/data/miniconda3/bin/python \
  STAGE2_ROOT=/data/disk1/SpeciesLLM_obs/Stage2_SpeciesLLMData \
  WORKDIR=/data/disk1/SpeciesLLM \
  HOSTS=7.150.12.45,7.150.15.14,7.150.14.170 \
  MASTER_ADDR=7.150.12.45 \
  NNODES=3 \
  NPROC_PER_NODE=8 \
  TRAIN_DATASET=test \
  DATA_PATH=/data/disk1/SpeciesLLM_obs/Stage2_SpeciesLLMData/all_flatten_data_test_1b \
  EMB_ROOT=/data/disk1/SpeciesLLM \
  EMB_PATH=/data/disk1/SpeciesLLM/Stage2_macrogene_embeddings \
  MODEL_CONFIG_JSON=/data/disk1/SpeciesLLM/Stage2_macrogene_embeddings/args_2nd_run_1b.json \
  OUT_PATH="$OUT_PATH" \
  BATCH_SIZE=96 \
  EPOCH=1 \
  GRADIENT_ACCUMULATION_STEPS=1 \
  GRADIENT_CHECKPOINTING=true

# 1B 模型三节点的训练测试
# 134 parquet file ~ 6508 seconds
# 28089 parquet file ~ 6508 * 28089 / 134 = 1,364,203 seconds ~ 380 hours ~ 15.8 days
