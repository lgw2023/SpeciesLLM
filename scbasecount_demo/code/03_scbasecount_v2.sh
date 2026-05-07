#!/usr/bin/env bash
set -e

export DEFAULT_INPUT_ROOT="/data/disk1/SpeciesLLM_obs/scbasecount_demo/scbasecount_processed_02"
export DEFAULT_OUTPUT_BASE="/data/disk1/SpeciesLLM_obs/scbasecount_demo/scbasecount_processed_03_v2"
export DEFAULT_OLD_LOOKUP_DIR="/data/disk1/SpeciesLLM_obs/scbasecount_demo/data/LOOKUP_categories_unified"
export DEFAULT_NEW_LOOKUP_DIR_NAME="LOOKUP_categories_unified"

cd /data/disk1/SpeciesLLM/scbasecount_demo/code
mkdir -p log/03_scbasecount_v2
uv run python 03_scbasecount_v2.py &> log/03_scbasecount_v2/scb5b

