#!/usr/bin/env bash
set -e

unset DEFAULT_BASE_PATH
unset DEFAULT_SAVE_ROOT
unset HUMAN_SAVE_ROOT
unset MOUSE_SAVE_ROOT

export DEFAULT_BASE_PATH="/data/node2_disk1/SpeciesLLM_obs/scbasecount_2026-01-12/scbasecount_processed"
export DEFAULT_SAVE_ROOT="/data/node2_disk2/SpeciesLLM_obs/scbasecount_2026-01-12/scbasecount_processed_02"
export HUMAN_SAVE_ROOT="/data/node2_disk2/SpeciesLLM_obs/scbasecount_2026-01-12/scbasecount_processed_02"
export MOUSE_SAVE_ROOT="/data/node2_disk2/SpeciesLLM_obs/scbasecount_2026-01-12/scbasecount_processed_02"

cd /data/disk1/SpeciesLLM_obs/scbasecount_demo/code
mkdir -p log_scbasecount_2026-01-12/02_scbasecount_fix
uv run python 02_scbasecount_fix.py --species Homo_sapiens &> log_scbasecount_2026-01-12/02_scbasecount_fix/Homo_sapiens &
uv run python 02_scbasecount_fix.py --species Mus_musculus &> log_scbasecount_2026-01-12/02_scbasecount_fix/Mus_musculus &
uv run python 02_scbasecount_fix.py --exclude-species Homo_sapiens Mus_musculus &> log_scbasecount_2026-01-12/02_scbasecount_fix/exclude-species &
wait


## 上传 
/usr/bin/python3 /root/s3_uploader.py --region cn-east-4 --app_token 806279e3-a95e-494d-81e7-12f9ccd57710 --bucket_name bucket-3028 \
    --bucket_path public/SpeciesLLM/scbasecount_2026-01-12/scbasecount_processed_02/ \
    --lfap /data/node2_disk2/SpeciesLLM_obs/scbasecount_2026-01-12/scbasecount_processed_02/Arabidopsis_thaliana
rm -rf /data/node2_disk1/SpeciesLLM_obs/scbasecount_2026-01-12/scbasecount_processed/Arabidopsis_thaliana/*
/usr/bin/python3 /root/s3_uploader.py --region cn-east-4 --app_token 806279e3-a95e-494d-81e7-12f9ccd57710 --bucket_name bucket-3028 \
    --bucket_path public/SpeciesLLM/scbasecount_2026-01-12/scbasecount_processed_02/ \
    --lfap /data/node2_disk2/SpeciesLLM_obs/scbasecount_2026-01-12/scbasecount_processed_02/Bos_taurus
rm -rf /data/node2_disk1/SpeciesLLM_obs/scbasecount_2026-01-12/scbasecount_processed/Bos_taurus/*
/usr/bin/python3 /root/s3_uploader.py --region cn-east-4 --app_token 806279e3-a95e-494d-81e7-12f9ccd57710 --bucket_name bucket-3028 \
    --bucket_path public/SpeciesLLM/scbasecount_2026-01-12/scbasecount_processed_02/ \
    --lfap /data/node2_disk2/SpeciesLLM_obs/scbasecount_2026-01-12/scbasecount_processed_02/Caenorhabditis_elegans
rm -rf /data/node2_disk1/SpeciesLLM_obs/scbasecount_2026-01-12/scbasecount_processed/Caenorhabditis_elegans/*
/usr/bin/python3 /root/s3_uploader.py --region cn-east-4 --app_token 806279e3-a95e-494d-81e7-12f9ccd57710 --bucket_name bucket-3028 \
    --bucket_path public/SpeciesLLM/scbasecount_2026-01-12/scbasecount_processed_02/ \
    --lfap /data/node2_disk2/SpeciesLLM_obs/scbasecount_2026-01-12/scbasecount_processed_02/Callithrix_jacchus
rm -rf /data/node2_disk1/SpeciesLLM_obs/scbasecount_2026-01-12/scbasecount_processed/Callithrix_jacchus/*
/usr/bin/python3 /root/s3_uploader.py --region cn-east-4 --app_token 806279e3-a95e-494d-81e7-12f9ccd57710 --bucket_name bucket-3028 \
    --bucket_path public/SpeciesLLM/scbasecount_2026-01-12/scbasecount_processed_02/ \
    --lfap /data/node2_disk2/SpeciesLLM_obs/scbasecount_2026-01-12/scbasecount_processed_02/Danio_rerio
rm -rf /data/node2_disk1/SpeciesLLM_obs/scbasecount_2026-01-12/scbasecount_processed/Danio_rerio/*
/usr/bin/python3 /root/s3_uploader.py --region cn-east-4 --app_token 806279e3-a95e-494d-81e7-12f9ccd57710 --bucket_name bucket-3028 \
    --bucket_path public/SpeciesLLM/scbasecount_2026-01-12/scbasecount_processed_02/ \
    --lfap /data/node2_disk2/SpeciesLLM_obs/scbasecount_2026-01-12/scbasecount_processed_02/Drosophila_melanogaster
rm -rf /data/node2_disk1/SpeciesLLM_obs/scbasecount_2026-01-12/scbasecount_processed/Drosophila_melanogaster/*
/usr/bin/python3 /root/s3_uploader.py --region cn-east-4 --app_token 806279e3-a95e-494d-81e7-12f9ccd57710 --bucket_name bucket-3028 \
    --bucket_path public/SpeciesLLM/scbasecount_2026-01-12/scbasecount_processed_02/ \
    --lfap /data/node2_disk2/SpeciesLLM_obs/scbasecount_2026-01-12/scbasecount_processed_02/Equus_caballus
rm -rf /data/node2_disk1/SpeciesLLM_obs/scbasecount_2026-01-12/scbasecount_processed/Equus_caballus/*
/usr/bin/python3 /root/s3_uploader.py --region cn-east-4 --app_token 806279e3-a95e-494d-81e7-12f9ccd57710 --bucket_name bucket-3028 \
    --bucket_path public/SpeciesLLM/scbasecount_2026-01-12/scbasecount_processed_02/ \
    --lfap /data/node2_disk2/SpeciesLLM_obs/scbasecount_2026-01-12/scbasecount_processed_02/Gallus_gallus
rm -rf /data/node2_disk1/SpeciesLLM_obs/scbasecount_2026-01-12/scbasecount_processed/Gallus_gallus/*
/usr/bin/python3 /root/s3_uploader.py --region cn-east-4 --app_token 806279e3-a95e-494d-81e7-12f9ccd57710 --bucket_name bucket-3028 \
    --bucket_path public/SpeciesLLM/scbasecount_2026-01-12/scbasecount_processed_02/ \
    --lfap /data/node2_disk2/SpeciesLLM_obs/scbasecount_2026-01-12/scbasecount_processed_02/Gorilla_gorilla
rm -rf /data/node2_disk1/SpeciesLLM_obs/scbasecount_2026-01-12/scbasecount_processed/Gorilla_gorilla/*
/usr/bin/python3 /root/s3_uploader.py --region cn-east-4 --app_token 806279e3-a95e-494d-81e7-12f9ccd57710 --bucket_name bucket-3028 \
    --bucket_path public/SpeciesLLM/scbasecount_2026-01-12/scbasecount_processed_02/ \
    --lfap /data/node2_disk2/SpeciesLLM_obs/scbasecount_2026-01-12/scbasecount_processed_02/Heterocephalus_glaber
rm -rf /data/node2_disk1/SpeciesLLM_obs/scbasecount_2026-01-12/scbasecount_processed/Heterocephalus_glaber/*
/usr/bin/python3 /root/s3_uploader.py --region cn-east-4 --app_token 806279e3-a95e-494d-81e7-12f9ccd57710 --bucket_name bucket-3028 \
    --bucket_path public/SpeciesLLM/scbasecount_2026-01-12/scbasecount_processed_02/ \
    --lfap /data/node2_disk2/SpeciesLLM_obs/scbasecount_2026-01-12/scbasecount_processed_02/Macaca_mulatta
rm -rf /data/node2_disk1/SpeciesLLM_obs/scbasecount_2026-01-12/scbasecount_processed/Macaca_mulatta/*
/usr/bin/python3 /root/s3_uploader.py --region cn-east-4 --app_token 806279e3-a95e-494d-81e7-12f9ccd57710 --bucket_name bucket-3028 \
    --bucket_path public/SpeciesLLM/scbasecount_2026-01-12/scbasecount_processed_02/ \
    --lfap /data/node2_disk2/SpeciesLLM_obs/scbasecount_2026-01-12/scbasecount_processed_02/Oryctolagus_cuniculus
rm -rf /data/node2_disk1/SpeciesLLM_obs/scbasecount_2026-01-12/scbasecount_processed/Oryctolagus_cuniculus/*
/usr/bin/python3 /root/s3_uploader.py --region cn-east-4 --app_token 806279e3-a95e-494d-81e7-12f9ccd57710 --bucket_name bucket-3028 \
    --bucket_path public/SpeciesLLM/scbasecount_2026-01-12/scbasecount_processed_02/ \
    --lfap /data/node2_disk2/SpeciesLLM_obs/scbasecount_2026-01-12/scbasecount_processed_02/Oryza_sativa
rm -rf /data/node2_disk1/SpeciesLLM_obs/scbasecount_2026-01-12/scbasecount_processed/Oryza_sativa/*
/usr/bin/python3 /root/s3_uploader.py --region cn-east-4 --app_token 806279e3-a95e-494d-81e7-12f9ccd57710 --bucket_name bucket-3028 \
    --bucket_path public/SpeciesLLM/scbasecount_2026-01-12/scbasecount_processed_02/ \
    --lfap /data/node2_disk2/SpeciesLLM_obs/scbasecount_2026-01-12/scbasecount_processed_02/Ovis_aries
rm -rf /data/node2_disk1/SpeciesLLM_obs/scbasecount_2026-01-12/scbasecount_processed/Ovis_aries/*
/usr/bin/python3 /root/s3_uploader.py --region cn-east-4 --app_token 806279e3-a95e-494d-81e7-12f9ccd57710 --bucket_name bucket-3028 \
    --bucket_path public/SpeciesLLM/scbasecount_2026-01-12/scbasecount_processed_02/ \
    --lfap /data/node2_disk2/SpeciesLLM_obs/scbasecount_2026-01-12/scbasecount_processed_02/Pan_troglodytes
rm -rf /data/node2_disk1/SpeciesLLM_obs/scbasecount_2026-01-12/scbasecount_processed/Pan_troglodytes/*
/usr/bin/python3 /root/s3_uploader.py --region cn-east-4 --app_token 806279e3-a95e-494d-81e7-12f9ccd57710 --bucket_name bucket-3028 \
    --bucket_path public/SpeciesLLM/scbasecount_2026-01-12/scbasecount_processed_02/ \
    --lfap /data/node2_disk2/SpeciesLLM_obs/scbasecount_2026-01-12/scbasecount_processed_02/Solanum_lycopersicum
rm -rf /data/node2_disk1/SpeciesLLM_obs/scbasecount_2026-01-12/scbasecount_processed/Solanum_lycopersicum/*
/usr/bin/python3 /root/s3_uploader.py --region cn-east-4 --app_token 806279e3-a95e-494d-81e7-12f9ccd57710 --bucket_name bucket-3028 \
    --bucket_path public/SpeciesLLM/scbasecount_2026-01-12/scbasecount_processed_02/ \
    --lfap /data/node2_disk2/SpeciesLLM_obs/scbasecount_2026-01-12/scbasecount_processed_02/Sus_scrofa
rm -rf /data/node2_disk1/SpeciesLLM_obs/scbasecount_2026-01-12/scbasecount_processed/Sus_scrofa/*
/usr/bin/python3 /root/s3_uploader.py --region cn-east-4 --app_token 806279e3-a95e-494d-81e7-12f9ccd57710 --bucket_name bucket-3028 \
    --bucket_path public/SpeciesLLM/scbasecount_2026-01-12/scbasecount_processed_02/ \
    --lfap /data/node2_disk2/SpeciesLLM_obs/scbasecount_2026-01-12/scbasecount_processed_02/Zea_mays
rm -rf /data/node2_disk1/SpeciesLLM_obs/scbasecount_2026-01-12/scbasecount_processed/Zea_mays/*
/usr/bin/python3 /root/s3_uploader.py --region cn-east-4 --app_token 806279e3-a95e-494d-81e7-12f9ccd57710 --bucket_name bucket-3028 \
    --bucket_path public/SpeciesLLM/scbasecount_2026-01-12/scbasecount_processed_02/ \
    --lfap /data/node2_disk2/SpeciesLLM_obs/scbasecount_2026-01-12/scbasecount_processed_02/Homo_sapiens
rm -rf /data/node2_disk1/SpeciesLLM_obs/scbasecount_2026-01-12/scbasecount_processed/Homo_sapiens/*
/usr/bin/python3 /root/s3_uploader.py --region cn-east-4 --app_token 806279e3-a95e-494d-81e7-12f9ccd57710 --bucket_name bucket-3028 \
    --bucket_path public/SpeciesLLM/scbasecount_2026-01-12/scbasecount_processed_02/ \
    --lfap /data/node2_disk2/SpeciesLLM_obs/scbasecount_2026-01-12/scbasecount_processed_02/Mus_musculus
rm -rf /data/node2_disk1/SpeciesLLM_obs/scbasecount_2026-01-12/scbasecount_processed/Mus_musculus/*