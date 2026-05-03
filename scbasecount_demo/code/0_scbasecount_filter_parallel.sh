#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="/data/disk1/SpeciesLLM_obs/scbasecount_demo"
PY_SCRIPT="${SCRIPT_DIR}/0_scbasecount_filter.py"
LOG_DIR="${SCRIPT_DIR}/log/0_scbasecount_filter"
MAX_JOBS="${MAX_JOBS:-4}"

mkdir -p "${LOG_DIR}"

run_species() {
  local species="$1"
  local coding_csv="$2"
  local gene_id_map_json="$3"

  echo "[start] ${species}"
  (
    cd "${SCRIPT_DIR}"

    uv run python "${PY_SCRIPT}" \
      --dir "${BASE}/scbasecount_raw/${species}" \
      --coding-genes-csv "${coding_csv}" \
      --gene-id-map-json "${gene_id_map_json}" \
      --min-genes 200 \
      --outdir "${BASE}/scbasecount_processed/${species}"

    /usr/bin/python3 /root/s3_uploader.py \
      --region cn-east-4 \
      --app_token 806279e3-a95e-494d-81e7-12f9ccd57710 \
      --bucket_name bucket-3028 \
      --bucket_path bucket-3028/public/SpeciesLLM/scbasecount_demo/scbasecount_processed/ \
      --lfap "${BASE}/scbasecount_processed/${species}"

    rm -rf "${BASE}/scbasecount_raw/${species}"/*
  ) &> "${LOG_DIR}/${species}.log"
  echo "[done] ${species}"
}

wait_for_slot() {
  local max_jobs="$1"
  while [ "$(jobs -rp | wc -l)" -ge "${max_jobs}" ]; do
    wait -n
  done
}

declare -a PARALLEL_SPECIES=(
  "Arabidopsis_thaliana|${BASE}/data/gene_symbols/Arabidopsis_thaliana.csv|${BASE}/data/gene_id_to_gene_symbol_2nd_pretrain/Arabidopsis_thaliana.TAIR10.gene_ID_to_gene_symbol.json"
  "Bos_taurus|${BASE}/data/gene_symbols/Bos_taurus.gene_symbols.csv|${BASE}/data/gene_id_to_gene_symbol_2nd_pretrain/Bos_taurus.ARS-UCD2.0.gene_ID_to_gene_symbol.json"
  "Caenorhabditis_elegans|${BASE}/data/gene_symbols/Caenorhabditis_elegans.csv|${BASE}/data/gene_id_to_gene_symbol_2nd_pretrain/Caenorhabditis_elegans.WBcel235.gene_ID_to_gene_symbol.json"
  "Callithrix_jacchus|${BASE}/data/gene_symbols/Callithrix_jacchus.gene_symbols.csv|${BASE}/data/gene_id_to_gene_symbol_2nd_pretrain/Callithrix_jacchus.mCalJac1.pat.X.gene_ID_to_gene_symbol.json"
  "Danio_rerio|${BASE}/data/gene_symbols/Danio_rerio.gene_symbols.csv|${BASE}/data/gene_id_to_gene_symbol_2nd_pretrain/Danio_rerio.GRCz11.gene_ID_to_gene_symbol.json"
  "Drosophila_melanogaster|${BASE}/data/gene_symbols/Drosophila_melanogaster.gene_symbols.csv|${BASE}/data/gene_id_to_gene_symbol_2nd_pretrain/Drosophila_melanogaster.BDGP6.54.gene_ID_to_gene_symbol.json"
  "Equus_caballus|${BASE}/data/gene_symbols/Equus_caballus.gene_symbols.csv|${BASE}/data/gene_id_to_gene_symbol_2nd_pretrain/Equus_caballus.EquCab3.0.gene_ID_to_gene_symbol.json"
  "Gallus_gallus|${BASE}/data/gene_symbols/Gallus_gallus.gene_symbols.csv|${BASE}/data/gene_id_to_gene_symbol_2nd_pretrain/Gallus_gallus.bGalGal1.mat.broiler.GRCg7b.gene_ID_to_gene_symbol.json"
  "Gorilla_gorilla|${BASE}/data/gene_symbols/Gorilla_gorilla.gene_symbols.csv|${BASE}/data/gene_id_to_gene_symbol_2nd_pretrain/Gorilla_gorilla.gorGor4.gene_ID_to_gene_symbol.json"
  "Heterocephalus_glaber|${BASE}/data/gene_symbols/Heterocephalus_glaber.csv|${BASE}/data/gene_id_to_gene_symbol_2nd_pretrain/Heterocephalus_glaber_male.Naked_mole-rat_paternal.gene_ID_to_gene_symbol.json"
  "Macaca_mulatta|${BASE}/data/gene_symbols/Macaca_mulatta.gene_symbols.csv|${BASE}/data/gene_id_to_gene_symbol_2nd_pretrain/Macaca_mulatta.Mmul_10.gene_ID_to_gene_symbol.json"
  "Oryctolagus_cuniculus|${BASE}/data/gene_symbols/Oryctolagus_cuniculus.gene_symbols.csv|${BASE}/data/gene_id_to_gene_symbol_2nd_pretrain/Oryctolagus_cuniculus.OryCun2.0.gene_ID_to_gene_symbol.json"
  "Oryza_sativa|${BASE}/data/gene_symbols/Oryza_sativa.gene_symbols.csv|${BASE}/data/gene_id_to_gene_symbol_2nd_pretrain/Oryza_sativa.IRGSP-1.0.gene_ID_to_gene_symbol.json"
  "Ovis_aries|${BASE}/data/gene_symbols/Ovis_aries.gene_symbols.csv|${BASE}/data/gene_id_to_gene_symbol_2nd_pretrain/Ovis_aries.Oar_v3.1.gene_ID_to_gene_symbol.json"
  "Pan_troglodytes|${BASE}/data/gene_symbols/Pan_troglodytes.gene_symbols.csv|${BASE}/data/gene_id_to_gene_symbol_2nd_pretrain/Pan_troglodytes.Pan_tro_3.0.gene_ID_to_gene_symbol.json"
  "Schistosoma_mansoni|${BASE}/data/gene_symbols/Schistosoma_mansoni.csv|${BASE}/data/gene_id_to_gene_symbol_2nd_pretrain/Schistosoma_mansoni.Smansoni_v7.gene_ID_to_gene_symbol.json"
  "Solanum_lycopersicum|${BASE}/data/gene_symbols/Solanum_lycopersicum.csv|${BASE}/data/gene_id_to_gene_symbol_2nd_pretrain/Solanum_lycopersicum.SL3.0.gene_ID_to_gene_symbol.json"
  "Sus_scrofa|${BASE}/data/gene_symbols/Sus_scrofa.gene_symbols.csv|${BASE}/data/gene_id_to_gene_symbol_2nd_pretrain/Sus_scrofa.Sscrofa11.1.gene_ID_to_gene_symbol.json"
  "Zea_mays|${BASE}/data/gene_symbols/Zea_mays.csv|${BASE}/data/gene_id_to_gene_symbol_2nd_pretrain/Zea_mays.Zm-B73-REFERENCE-NAM-5.0.gene_ID_to_gene_symbol.json"
)

declare -a SERIAL_SPECIES=(
  "Homo_sapiens|${BASE}/data/gene_symbols/Homo_sapiens.gene_symbol.csv|${BASE}/data/gene_id_to_gene_symbol_2nd_pretrain/Homo_sapiens.GRCh38.gene_ID_to_gene_symbol.json"
  "Mus_musculus|${BASE}/data/gene_symbols/Mus_musculus.gene_symbol.csv|${BASE}/data/gene_id_to_gene_symbol_2nd_pretrain/Mus_musculus.GRCm39.gene_ID_to_gene_symbol.json"
)

echo "[info] MAX_JOBS=${MAX_JOBS}"
echo "[info] run non-human/mouse species in parallel"

for item in "${PARALLEL_SPECIES[@]}"; do
  IFS="|" read -r species coding_csv gene_id_map_json <<< "${item}"
  wait_for_slot "${MAX_JOBS}"
  run_species "${species}" "${coding_csv}" "${gene_id_map_json}" &
done

wait

echo "[info] run Homo_sapiens and Mus_musculus serially"

for item in "${SERIAL_SPECIES[@]}"; do
  IFS="|" read -r species coding_csv gene_id_map_json <<< "${item}"
  run_species "${species}" "${coding_csv}" "${gene_id_map_json}"
done

echo "[info] all species finished"
