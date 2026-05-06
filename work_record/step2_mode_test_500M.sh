


# 500M 模型三节点的训练测试

cd /data/disk1/SpeciesLLM

RUN_ID=$(date +%Y%m%d_%H%M%S)
OUT_PATH="/data/disk1/SpeciesLLM_obs/Stage2_SpeciesLLMData/training_output_500m_test_from_scratch_${RUN_ID}"

bash scripts/smoke_500m_3node.sh \
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
  DATA_PATH=/data/disk1/SpeciesLLM_obs/Stage2_SpeciesLLMData/all_flatten_data_test_500m \
  EMB_ROOT=/data/disk1/SpeciesLLM \
  EMB_PATH=/data/disk1/SpeciesLLM/Stage2_macrogene_embeddings \
  MODEL_CONFIG_JSON=/data/disk1/SpeciesLLM/Stage2_macrogene_embeddings/args_2nd_run.json \
  OUT_PATH="$OUT_PATH" \
  BATCH_SIZE=16 \
  EPOCH=1 \
  GRADIENT_ACCUMULATION_STEPS=1