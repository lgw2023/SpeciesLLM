#!/usr/bin/env bash
set -e

export DEFAULT_INPUT_ROOT="/data/node2_disk2/SpeciesLLM_obs/scbasecount_2026-01-12/scbasecount_processed_02"
export DEFAULT_OUTPUT_BASE="/data/node2_disk3/SpeciesLLM_obs/scbasecount_2026-01-12/scbasecount_processed_03_v2"
export DEFAULT_OLD_LOOKUP_DIR="/data/disk1/SpeciesLLM_obs/scbasecount_demo/data/LOOKUP_categories_unified"
export DEFAULT_NEW_LOOKUP_DIR_NAME="LOOKUP_categories_unified"

cd /data/disk1/SpeciesLLM_obs/scbasecount_demo/code
mkdir -p log_scbasecount_2026-01-12/03_scbasecount_v2
uv run python 03_scbasecount_v2.py &> log_scbasecount_2026-01-12/03_scbasecount_v2/scb5b
