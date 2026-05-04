set -euo pipefail

cd /data/disk1/SpeciesLLM

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

# 只重新生成启动命令，不重新生成测试数据
bash scripts/test_stage2_500m_multinode.sh commands

# 检查 dry-run
DRY_RUN=1 bash "${COMMAND_DIR}/launch_500m_3nodes.sh"

cd '/data/disk1/SpeciesLLM'
WORKDIR='/data/disk1/SpeciesLLM'
emb_path='/data/disk1/SpeciesLLM/Stage2_macrogene_embeddings'
config_json='/data/disk1/SpeciesLLM/Stage2_macrogene_embeddings/args_2nd_run.json'
data_path='/data/disk1/SpeciesLLM_obs/Stage2_SpeciesLLMData/all_flatten_data_test_500m'


for host in 7.150.12.45 7.150.15.14 7.150.14.170; do
  echo "===== ${host} ====="
  ssh root@"${host}" '
    set -e
    test -f /data/disk1/SpeciesLLM/train_MNodes_torchrun_mfu_preindexparquet.py
    test -f /data/disk1/SpeciesLLM/Stage2_macrogene_embeddings/args_2nd_run.json
    test -d /data/disk1/SpeciesLLM_obs/Stage2_SpeciesLLMData/all_flatten_data_test_500m
    cd /data/disk1/SpeciesLLM
    git rev-parse --short HEAD
    ls /data/disk1/SpeciesLLM_obs/Stage2_SpeciesLLMData/all_flatten_data_test_500m/*.parquet | wc -l
  '
done

bash /data/disk1/SpeciesLLM_obs/Stage2_SpeciesLLMData/stage2_500m_test_commands/launch_500m_3nodes.sh
