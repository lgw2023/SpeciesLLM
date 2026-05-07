#!/usr/bin/env bash
set -e

export DEFAULT_INPUT_ROOT="/data/node2_disk2/SpeciesLLM_obs/scbasecount_2026-01-12/scbasecount_processed_02"
export DEFAULT_OUTPUT_BASE="/data/node2_disk3/SpeciesLLM_obs/scbasecount_2026-01-12/scbasecount_processed_03_v2"
export DEFAULT_OLD_LOOKUP_DIR="/data/disk1/SpeciesLLM_obs/scbasecount_demo/data/LOOKUP_categories_unified"
export DEFAULT_NEW_LOOKUP_DIR_NAME="LOOKUP_categories_unified"

cd /data/disk1/SpeciesLLM/scbasecount_demo/code
mkdir -p log_scbasecount_2026-01-12/03_scbasecount_v2
uv run python 03_scbasecount_v2.py &> log_scbasecount_2026-01-12/03_scbasecount_v2/scb5b

wait

## 上传 
/usr/bin/python3 /root/s3_uploader.py --region cn-east-4 --app_token 806279e3-a95e-494d-81e7-12f9ccd57710 --bucket_name bucket-3028 \
    --bucket_path public/SpeciesLLM/scbasecount_2026-01-12/ \
    --lfap /data/node2_disk2/SpeciesLLM_obs/scbasecount_2026-01-12/scbasecount_processed_03_v2

# rm -rf /data/node2_disk2/SpeciesLLM_obs/scbasecount_2026-01-12/scbasecount_processed_02/*/X.zarr/X/*