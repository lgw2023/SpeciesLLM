#!/usr/bin/env bash
set -e

cd /data/disk1/SpeciesLLM/scbasecount_demo/code
mkdir -p log_scbasecount_2026-01-12/0_scbasecount_filter

uv run python 0_scbasecount_filter.py \
  --dir /data/disk3/scbasecount_2026-01-12/h5ad/Gene/Arabidopsis_thaliana \
  --coding-genes-csv /data/disk1/SpeciesLLM_obs/scbasecount_demo/data/gene_symbols/Arabidopsis_thaliana.csv \
  --gene-id-map-json /data/disk1/SpeciesLLM_obs/scbasecount_demo/data/gene_id_to_gene_symbol_2nd_pretrain/Arabidopsis_thaliana.TAIR10.gene_ID_to_gene_symbol.json \
  --min-genes 200 \
  --outdir /data/node2_disk1/SpeciesLLM_obs/scbasecount_2026-01-12/scbasecount_processed/Arabidopsis_thaliana &> log_scbasecount_2026-01-12/0_scbasecount_filter/Arabidopsis_thaliana &

uv run python 0_scbasecount_filter.py \
  --dir /data/disk3/scbasecount_2026-01-12/h5ad/Gene/Bos_taurus \
  --coding-genes-csv /data/disk1/SpeciesLLM_obs/scbasecount_demo/data/gene_symbols/Bos_taurus.gene_symbols.csv \
  --gene-id-map-json /data/disk1/SpeciesLLM_obs/scbasecount_demo/data/gene_id_to_gene_symbol_2nd_pretrain/Bos_taurus.ARS-UCD2.0.gene_ID_to_gene_symbol.json \
  --min-genes 200 \
  --outdir /data/node2_disk1/SpeciesLLM_obs/scbasecount_2026-01-12/scbasecount_processed/Bos_taurus &> log_scbasecount_2026-01-12/0_scbasecount_filter/Bos_taurus &

uv run python 0_scbasecount_filter.py \
  --dir /data/disk3/scbasecount_2026-01-12/h5ad/Gene/Caenorhabditis_elegans \
  --coding-genes-csv /data/disk1/SpeciesLLM_obs/scbasecount_demo/data/gene_symbols/Caenorhabditis_elegans.csv \
  --gene-id-map-json /data/disk1/SpeciesLLM_obs/scbasecount_demo/data/gene_id_to_gene_symbol_2nd_pretrain/Caenorhabditis_elegans.WBcel235.gene_ID_to_gene_symbol.json \
  --min-genes 200 \
  --outdir /data/node2_disk1/SpeciesLLM_obs/scbasecount_2026-01-12/scbasecount_processed/Caenorhabditis_elegans &> log_scbasecount_2026-01-12/0_scbasecount_filter/Caenorhabditis_elegans &

uv run python 0_scbasecount_filter.py \
  --dir /data/disk3/scbasecount_2026-01-12/h5ad/Gene/Callithrix_jacchus \
  --coding-genes-csv /data/disk1/SpeciesLLM_obs/scbasecount_demo/data/gene_symbols/Callithrix_jacchus.gene_symbols.csv \
  --gene-id-map-json /data/disk1/SpeciesLLM_obs/scbasecount_demo/data/gene_id_to_gene_symbol_2nd_pretrain/Callithrix_jacchus.mCalJac1.pat.X.gene_ID_to_gene_symbol.json \
  --min-genes 200 \
  --outdir /data/node2_disk1/SpeciesLLM_obs/scbasecount_2026-01-12/scbasecount_processed/Callithrix_jacchus &> log_scbasecount_2026-01-12/0_scbasecount_filter/Callithrix_jacchus &

uv run python 0_scbasecount_filter.py \
  --dir /data/disk3/scbasecount_2026-01-12/h5ad/Gene/Danio_rerio \
  --coding-genes-csv /data/disk1/SpeciesLLM_obs/scbasecount_demo/data/gene_symbols/Danio_rerio.gene_symbols.csv \
  --gene-id-map-json /data/disk1/SpeciesLLM_obs/scbasecount_demo/data/gene_id_to_gene_symbol_2nd_pretrain/Danio_rerio.GRCz11.gene_ID_to_gene_symbol.json \
  --min-genes 200 \
  --outdir /data/node2_disk1/SpeciesLLM_obs/scbasecount_2026-01-12/scbasecount_processed/Danio_rerio &> log_scbasecount_2026-01-12/0_scbasecount_filter/Danio_rerio &

wait

uv run python 0_scbasecount_filter.py \
  --dir /data/disk3/scbasecount_2026-01-12/h5ad/Gene/Drosophila_melanogaster \
  --coding-genes-csv /data/disk1/SpeciesLLM_obs/scbasecount_demo/data/gene_symbols/Drosophila_melanogaster.gene_symbols.csv \
  --gene-id-map-json /data/disk1/SpeciesLLM_obs/scbasecount_demo/data/gene_id_to_gene_symbol_2nd_pretrain/Drosophila_melanogaster.BDGP6.54.gene_ID_to_gene_symbol.json \
  --min-genes 200 \
  --outdir /data/node2_disk1/SpeciesLLM_obs/scbasecount_2026-01-12/scbasecount_processed/Drosophila_melanogaster &> log_scbasecount_2026-01-12/0_scbasecount_filter/Drosophila_melanogaster &

uv run python 0_scbasecount_filter.py \
  --dir /data/disk3/scbasecount_2026-01-12/h5ad/Gene/Equus_caballus \
  --coding-genes-csv /data/disk1/SpeciesLLM_obs/scbasecount_demo/data/gene_symbols/Equus_caballus.gene_symbols.csv \
  --gene-id-map-json /data/disk1/SpeciesLLM_obs/scbasecount_demo/data/gene_id_to_gene_symbol_2nd_pretrain/Equus_caballus.EquCab3.0.gene_ID_to_gene_symbol.json \
  --min-genes 200 \
  --outdir /data/node2_disk1/SpeciesLLM_obs/scbasecount_2026-01-12/scbasecount_processed/Equus_caballus &> log_scbasecount_2026-01-12/0_scbasecount_filter/Equus_caballus &

uv run python 0_scbasecount_filter.py \
  --dir /data/disk3/scbasecount_2026-01-12/h5ad/Gene/Gorilla_gorilla \
  --coding-genes-csv /data/disk1/SpeciesLLM_obs/scbasecount_demo/data/gene_symbols/Gorilla_gorilla.gene_symbols.csv \
  --gene-id-map-json /data/disk1/SpeciesLLM_obs/scbasecount_demo/data/gene_id_to_gene_symbol_2nd_pretrain/Gorilla_gorilla.gorGor4.gene_ID_to_gene_symbol.json \
  --min-genes 200 \
  --outdir /data/node2_disk1/SpeciesLLM_obs/scbasecount_2026-01-12/scbasecount_processed/Gorilla_gorilla &> log_scbasecount_2026-01-12/0_scbasecount_filter/Gorilla_gorilla &

uv run python 0_scbasecount_filter.py \
  --dir /data/disk3/scbasecount_2026-01-12/h5ad/Gene/Heterocephalus_glaber \
  --coding-genes-csv /data/disk1/SpeciesLLM_obs/scbasecount_demo/data/gene_symbols/Heterocephalus_glaber.csv \
  --gene-id-map-json /data/disk1/SpeciesLLM_obs/scbasecount_demo/data/gene_id_to_gene_symbol_2nd_pretrain/Heterocephalus_glaber_male.Naked_mole-rat_paternal.gene_ID_to_gene_symbol.json \
  --min-genes 200 \
  --outdir /data/node2_disk1/SpeciesLLM_obs/scbasecount_2026-01-12/scbasecount_processed/Heterocephalus_glaber &> log_scbasecount_2026-01-12/0_scbasecount_filter/Heterocephalus_glaber &

uv run python 0_scbasecount_filter.py \
  --dir /data/disk3/scbasecount_2026-01-12/h5ad/Gene/Macaca_mulatta \
  --coding-genes-csv /data/disk1/SpeciesLLM_obs/scbasecount_demo/data/gene_symbols/Macaca_mulatta.gene_symbols.csv \
  --gene-id-map-json /data/disk1/SpeciesLLM_obs/scbasecount_demo/data/gene_id_to_gene_symbol_2nd_pretrain/Macaca_mulatta.Mmul_10.gene_ID_to_gene_symbol.json \
  --min-genes 200 \
  --outdir /data/node2_disk1/SpeciesLLM_obs/scbasecount_2026-01-12/scbasecount_processed/Macaca_mulatta &> log_scbasecount_2026-01-12/0_scbasecount_filter/Macaca_mulatta &

wait

uv run python 0_scbasecount_filter.py \
  --dir /data/disk3/scbasecount_2026-01-12/h5ad/Gene/Oryctolagus_cuniculus \
  --coding-genes-csv /data/disk1/SpeciesLLM_obs/scbasecount_demo/data/gene_symbols/Oryctolagus_cuniculus.gene_symbols.csv \
  --gene-id-map-json /data/disk1/SpeciesLLM_obs/scbasecount_demo/data/gene_id_to_gene_symbol_2nd_pretrain/Oryctolagus_cuniculus.OryCun2.0.gene_ID_to_gene_symbol.json \
  --min-genes 200 \
  --outdir /data/node2_disk1/SpeciesLLM_obs/scbasecount_2026-01-12/scbasecount_processed/Oryctolagus_cuniculus &> log_scbasecount_2026-01-12/0_scbasecount_filter/Oryctolagus_cuniculus &

uv run python 0_scbasecount_filter.py \
  --dir /data/disk3/scbasecount_2026-01-12/h5ad/Gene/Oryza_sativa \
  --coding-genes-csv /data/disk1/SpeciesLLM_obs/scbasecount_demo/data/gene_symbols/Oryza_sativa.gene_symbols.csv \
  --gene-id-map-json /data/disk1/SpeciesLLM_obs/scbasecount_demo/data/gene_id_to_gene_symbol_2nd_pretrain/Oryza_sativa.IRGSP-1.0.gene_ID_to_gene_symbol.json \
  --min-genes 200 \
  --outdir /data/node2_disk1/SpeciesLLM_obs/scbasecount_2026-01-12/scbasecount_processed/Oryza_sativa &> log_scbasecount_2026-01-12/0_scbasecount_filter/Oryza_sativa &

uv run python 0_scbasecount_filter.py \
  --dir /data/disk3/scbasecount_2026-01-12/h5ad/Gene/Ovis_aries \
  --coding-genes-csv /data/disk1/SpeciesLLM_obs/scbasecount_demo/data/gene_symbols/Ovis_aries.gene_symbols.csv \
  --gene-id-map-json /data/disk1/SpeciesLLM_obs/scbasecount_demo/data/gene_id_to_gene_symbol_2nd_pretrain/Ovis_aries.Oar_v3.1.gene_ID_to_gene_symbol.json \
  --min-genes 200 \
  --outdir /data/node2_disk1/SpeciesLLM_obs/scbasecount_2026-01-12/scbasecount_processed/Ovis_aries &> log_scbasecount_2026-01-12/0_scbasecount_filter/Ovis_aries &

uv run python 0_scbasecount_filter.py \
  --dir /data/disk3/scbasecount_2026-01-12/h5ad/Gene/Pan_troglodytes \
  --coding-genes-csv /data/disk1/SpeciesLLM_obs/scbasecount_demo/data/gene_symbols/Pan_troglodytes.gene_symbols.csv \
  --gene-id-map-json /data/disk1/SpeciesLLM_obs/scbasecount_demo/data/gene_id_to_gene_symbol_2nd_pretrain/Pan_troglodytes.Pan_tro_3.0.gene_ID_to_gene_symbol.json \
  --min-genes 200 \
  --outdir /data/node2_disk1/SpeciesLLM_obs/scbasecount_2026-01-12/scbasecount_processed/Pan_troglodytes &> log_scbasecount_2026-01-12/0_scbasecount_filter/Pan_troglodytes &

uv run python 0_scbasecount_filter.py \
  --dir /data/disk3/scbasecount_2026-01-12/h5ad/Gene/Schistosoma_mansoni \
  --coding-genes-csv /data/disk1/SpeciesLLM_obs/scbasecount_demo/data/gene_symbols/Schistosoma_mansoni.csv \
  --gene-id-map-json /data/disk1/SpeciesLLM_obs/scbasecount_demo/data/gene_id_to_gene_symbol_2nd_pretrain/Schistosoma_mansoni.Smansoni_v7.gene_ID_to_gene_symbol.json \
  --min-genes 200 \
  --outdir /data/node2_disk1/SpeciesLLM_obs/scbasecount_2026-01-12/scbasecount_processed/Schistosoma_mansoni &> log_scbasecount_2026-01-12/0_scbasecount_filter/Schistosoma_mansoni &

wait

uv run python 0_scbasecount_filter.py \
  --dir /data/disk3/scbasecount_2026-01-12/h5ad/Gene/Solanum_lycopersicum \
  --coding-genes-csv /data/disk1/SpeciesLLM_obs/scbasecount_demo/data/gene_symbols/Solanum_lycopersicum.csv \
  --gene-id-map-json /data/disk1/SpeciesLLM_obs/scbasecount_demo/data/gene_id_to_gene_symbol_2nd_pretrain/Solanum_lycopersicum.SL3.0.gene_ID_to_gene_symbol.json \
  --min-genes 200 \
  --outdir /data/node2_disk1/SpeciesLLM_obs/scbasecount_2026-01-12/scbasecount_processed/Solanum_lycopersicum &> log_scbasecount_2026-01-12/0_scbasecount_filter/Solanum_lycopersicum &

uv run python 0_scbasecount_filter.py \
  --dir /data/disk3/scbasecount_2026-01-12/h5ad/Gene/Sus_scrofa \
  --coding-genes-csv /data/disk1/SpeciesLLM_obs/scbasecount_demo/data/gene_symbols/Sus_scrofa.gene_symbols.csv \
  --gene-id-map-json /data/disk1/SpeciesLLM_obs/scbasecount_demo/data/gene_id_to_gene_symbol_2nd_pretrain/Sus_scrofa.Sscrofa11.1.gene_ID_to_gene_symbol.json \
  --min-genes 200 \
  --outdir /data/node2_disk1/SpeciesLLM_obs/scbasecount_2026-01-12/scbasecount_processed/Sus_scrofa &> log_scbasecount_2026-01-12/0_scbasecount_filter/Sus_scrofa &

uv run python 0_scbasecount_filter.py \
  --dir /data/disk3/scbasecount_2026-01-12/h5ad/Gene/Zea_mays \
  --coding-genes-csv /data/disk1/SpeciesLLM_obs/scbasecount_demo/data/gene_symbols/Zea_mays.csv \
  --gene-id-map-json /data/disk1/SpeciesLLM_obs/scbasecount_demo/data/gene_id_to_gene_symbol_2nd_pretrain/Zea_mays.Zm-B73-REFERENCE-NAM-5.0.gene_ID_to_gene_symbol.json \
  --min-genes 200 \
  --outdir /data/node2_disk1/SpeciesLLM_obs/scbasecount_2026-01-12/scbasecount_processed/Zea_mays &> log_scbasecount_2026-01-12/0_scbasecount_filter/Zea_mays &

uv run python 0_scbasecount_filter.py \
  --dir /data/disk3/scbasecount_2026-01-12/h5ad/Gene/Gallus_gallus \
  --coding-genes-csv /data/disk1/SpeciesLLM_obs/scbasecount_demo/data/gene_symbols/Gallus_gallus.gene_symbols.csv \
  --gene-id-map-json /data/disk1/SpeciesLLM_obs/scbasecount_demo/data/gene_id_to_gene_symbol_2nd_pretrain/Gallus_gallus.bGalGal1.mat.broiler.GRCg7b.gene_ID_to_gene_symbol.json \
  --min-genes 200 \
  --outdir /data/node2_disk1/SpeciesLLM_obs/scbasecount_2026-01-12/scbasecount_processed/Gallus_gallus &> log_scbasecount_2026-01-12/0_scbasecount_filter/Gallus_gallus &

wait

uv run python 0_scbasecount_filter.py \
  --dir /data/disk3/scbasecount_2026-01-12/h5ad/Gene/Homo_sapiens \
  --coding-genes-csv /data/disk1/SpeciesLLM_obs/scbasecount_demo/data/gene_symbols/Homo_sapiens.gene_symbol.csv \
  --gene-id-map-json /data/disk1/SpeciesLLM_obs/scbasecount_demo/data/gene_id_to_gene_symbol_2nd_pretrain/Homo_sapiens.GRCh38.gene_ID_to_gene_symbol.json \
  --min-genes 200 \
  --outdir /data/node2_disk1/SpeciesLLM_obs/scbasecount_2026-01-12/scbasecount_processed/Homo_sapiens &> log_scbasecount_2026-01-12/0_scbasecount_filter/Homo_sapiens &

uv run python 0_scbasecount_filter.py \
  --dir /data/disk3/scbasecount_2026-01-12/h5ad/Gene/Mus_musculus \
  --coding-genes-csv /data/disk1/SpeciesLLM_obs/scbasecount_demo/data/gene_symbols/Mus_musculus.gene_symbol.csv \
  --gene-id-map-json /data/disk1/SpeciesLLM_obs/scbasecount_demo/data/gene_id_to_gene_symbol_2nd_pretrain/Mus_musculus.GRCm39.gene_ID_to_gene_symbol.json \
  --min-genes 200 \
  --outdir /data/node2_disk1/SpeciesLLM_obs/scbasecount_2026-01-12/scbasecount_processed/Mus_musculus &> log_scbasecount_2026-01-12/0_scbasecount_filter/Mus_musculus &

wait


## 上传  这会产生 bucket-3028/public/SpeciesLLM/scbasecount_2026-01-12/scbasecount_processed/Arabidopsis_thaliana 的远程文件夹
/usr/bin/python3 /root/s3_uploader.py --region cn-east-4 --app_token 806279e3-a95e-494d-81e7-12f9ccd57710 --bucket_name bucket-3028 \
    --bucket_path public/SpeciesLLM/scbasecount_2026-01-12/scbasecount_processed/ \
    --lfap /data/node2_disk1/SpeciesLLM_obs/scbasecount_2026-01-12/scbasecount_processed/Arabidopsis_thaliana
# rm -rf /data/disk3/scbasecount_2026-01-12/h5ad/Gene/Arabidopsis_thaliana/*

## 上传  这会产生 bucket-3028/public/SpeciesLLM/scbasecount_2026-01-12/scbasecount_processed/Bos_taurus 的远程文件夹
/usr/bin/python3 /root/s3_uploader.py --region cn-east-4 --app_token 806279e3-a95e-494d-81e7-12f9ccd57710 --bucket_name bucket-3028 \
    --bucket_path public/SpeciesLLM/scbasecount_2026-01-12/scbasecount_processed/ \
    --lfap /data/node2_disk1/SpeciesLLM_obs/scbasecount_2026-01-12/scbasecount_processed/Bos_taurus
# rm -rf /data/disk3/scbasecount_2026-01-12/h5ad/Gene/Bos_taurus/*

## 上传  这会产生 bucket-3028/public/SpeciesLLM/scbasecount_2026-01-12/scbasecount_processed/Caenorhabditis_elegans 的远程文件夹
/usr/bin/python3 /root/s3_uploader.py --region cn-east-4 --app_token 806279e3-a95e-494d-81e7-12f9ccd57710 --bucket_name bucket-3028 \
    --bucket_path public/SpeciesLLM/scbasecount_2026-01-12/scbasecount_processed/ \
    --lfap /data/node2_disk1/SpeciesLLM_obs/scbasecount_2026-01-12/scbasecount_processed/Caenorhabditis_elegans
# rm -rf /data/disk3/scbasecount_2026-01-12/h5ad/Gene/Caenorhabditis_elegans/*

## 上传  这会产生 bucket-3028/public/SpeciesLLM/scbasecount_2026-01-12/scbasecount_processed/Callithrix_jacchus 的远程文件夹
/usr/bin/python3 /root/s3_uploader.py --region cn-east-4 --app_token 806279e3-a95e-494d-81e7-12f9ccd57710 --bucket_name bucket-3028 \
    --bucket_path public/SpeciesLLM/scbasecount_2026-01-12/scbasecount_processed/ \
    --lfap /data/node2_disk1/SpeciesLLM_obs/scbasecount_2026-01-12/scbasecount_processed/Callithrix_jacchus
# rm -rf /data/disk3/scbasecount_2026-01-12/h5ad/Gene/Callithrix_jacchus/*

## 上传  这会产生 bucket-3028/public/SpeciesLLM/scbasecount_2026-01-12/scbasecount_processed/Danio_rerio 的远程文件夹
/usr/bin/python3 /root/s3_uploader.py --region cn-east-4 --app_token 806279e3-a95e-494d-81e7-12f9ccd57710 --bucket_name bucket-3028 \
    --bucket_path public/SpeciesLLM/scbasecount_2026-01-12/scbasecount_processed/ \
    --lfap /data/node2_disk1/SpeciesLLM_obs/scbasecount_2026-01-12/scbasecount_processed/Danio_rerio
# rm -rf /data/disk3/scbasecount_2026-01-12/h5ad/Gene/Danio_rerio/*

## 上传  这会产生 bucket-3028/public/SpeciesLLM/scbasecount_2026-01-12/scbasecount_processed/Drosophila_melanogaster 的远程文件夹
/usr/bin/python3 /root/s3_uploader.py --region cn-east-4 --app_token 806279e3-a95e-494d-81e7-12f9ccd57710 --bucket_name bucket-3028 \
    --bucket_path public/SpeciesLLM/scbasecount_2026-01-12/scbasecount_processed/ \
    --lfap /data/node2_disk1/SpeciesLLM_obs/scbasecount_2026-01-12/scbasecount_processed/Drosophila_melanogaster
# rm -rf /data/disk3/scbasecount_2026-01-12/h5ad/Gene/Drosophila_melanogaster/*

## 上传  这会产生 bucket-3028/public/SpeciesLLM/scbasecount_2026-01-12/scbasecount_processed/Equus_caballus 的远程文件夹
/usr/bin/python3 /root/s3_uploader.py --region cn-east-4 --app_token 806279e3-a95e-494d-81e7-12f9ccd57710 --bucket_name bucket-3028 \
    --bucket_path public/SpeciesLLM/scbasecount_2026-01-12/scbasecount_processed/ \
    --lfap /data/node2_disk1/SpeciesLLM_obs/scbasecount_2026-01-12/scbasecount_processed/Equus_caballus
# rm -rf /data/disk3/scbasecount_2026-01-12/h5ad/Gene/Equus_caballus/*

## 上传  这会产生 bucket-3028/public/SpeciesLLM/scbasecount_2026-01-12/scbasecount_processed/Gallus_gallus 的远程文件夹
/usr/bin/python3 /root/s3_uploader.py --region cn-east-4 --app_token 806279e3-a95e-494d-81e7-12f9ccd57710 --bucket_name bucket-3028 \
    --bucket_path public/SpeciesLLM/scbasecount_2026-01-12/scbasecount_processed/ \
    --lfap /data/node2_disk1/SpeciesLLM_obs/scbasecount_2026-01-12/scbasecount_processed/Gallus_gallus
# rm -rf /data/disk3/scbasecount_2026-01-12/h5ad/Gene/Gallus_gallus/*

## 上传  这会产生 bucket-3028/public/SpeciesLLM/scbasecount_2026-01-12/scbasecount_processed/Gorilla_gorilla 的远程文件夹
/usr/bin/python3 /root/s3_uploader.py --region cn-east-4 --app_token 806279e3-a95e-494d-81e7-12f9ccd57710 --bucket_name bucket-3028 \
    --bucket_path public/SpeciesLLM/scbasecount_2026-01-12/scbasecount_processed/ \
    --lfap /data/node2_disk1/SpeciesLLM_obs/scbasecount_2026-01-12/scbasecount_processed/Gorilla_gorilla
# rm -rf /data/disk3/scbasecount_2026-01-12/h5ad/Gene/Gorilla_gorilla/*

## 上传  这会产生 bucket-3028/public/SpeciesLLM/scbasecount_2026-01-12/scbasecount_processed/Heterocephalus_glaber 的远程文件夹
/usr/bin/python3 /root/s3_uploader.py --region cn-east-4 --app_token 806279e3-a95e-494d-81e7-12f9ccd57710 --bucket_name bucket-3028 \
    --bucket_path public/SpeciesLLM/scbasecount_2026-01-12/scbasecount_processed/ \
    --lfap /data/node2_disk1/SpeciesLLM_obs/scbasecount_2026-01-12/scbasecount_processed/Heterocephalus_glaber
# rm -rf /data/disk3/scbasecount_2026-01-12/h5ad/Gene/Heterocephalus_glaber/*

## 上传  这会产生 bucket-3028/public/SpeciesLLM/scbasecount_2026-01-12/scbasecount_processed/Macaca_mulatta 的远程文件夹
/usr/bin/python3 /root/s3_uploader.py --region cn-east-4 --app_token 806279e3-a95e-494d-81e7-12f9ccd57710 --bucket_name bucket-3028 \
    --bucket_path public/SpeciesLLM/scbasecount_2026-01-12/scbasecount_processed/ \
    --lfap /data/node2_disk1/SpeciesLLM_obs/scbasecount_2026-01-12/scbasecount_processed/Macaca_mulatta
# rm -rf /data/disk3/scbasecount_2026-01-12/h5ad/Gene/Macaca_mulatta/*

## 上传  这会产生 bucket-3028/public/SpeciesLLM/scbasecount_2026-01-12/scbasecount_processed/Oryctolagus_cuniculus 的远程文件夹
/usr/bin/python3 /root/s3_uploader.py --region cn-east-4 --app_token 806279e3-a95e-494d-81e7-12f9ccd57710 --bucket_name bucket-3028 \
    --bucket_path public/SpeciesLLM/scbasecount_2026-01-12/scbasecount_processed/ \
    --lfap /data/node2_disk1/SpeciesLLM_obs/scbasecount_2026-01-12/scbasecount_processed/Oryctolagus_cuniculus
# rm -rf /data/disk3/scbasecount_2026-01-12/h5ad/Gene/Oryctolagus_cuniculus/*

## 上传  这会产生 bucket-3028/public/SpeciesLLM/scbasecount_2026-01-12/scbasecount_processed/Oryza_sativa 的远程文件夹
/usr/bin/python3 /root/s3_uploader.py --region cn-east-4 --app_token 806279e3-a95e-494d-81e7-12f9ccd57710 --bucket_name bucket-3028 \
    --bucket_path public/SpeciesLLM/scbasecount_2026-01-12/scbasecount_processed/ \
    --lfap /data/node2_disk1/SpeciesLLM_obs/scbasecount_2026-01-12/scbasecount_processed/Oryza_sativa
# rm -rf /data/disk3/scbasecount_2026-01-12/h5ad/Gene/Oryza_sativa/*

## 上传  这会产生 bucket-3028/public/SpeciesLLM/scbasecount_2026-01-12/scbasecount_processed/Ovis_aries 的远程文件夹
/usr/bin/python3 /root/s3_uploader.py --region cn-east-4 --app_token 806279e3-a95e-494d-81e7-12f9ccd57710 --bucket_name bucket-3028 \
    --bucket_path public/SpeciesLLM/scbasecount_2026-01-12/scbasecount_processed/ \
    --lfap /data/node2_disk1/SpeciesLLM_obs/scbasecount_2026-01-12/scbasecount_processed/Ovis_aries
# rm -rf /data/disk3/scbasecount_2026-01-12/h5ad/Gene/Ovis_aries/*

## 上传  这会产生 bucket-3028/public/SpeciesLLM/scbasecount_2026-01-12/scbasecount_processed/Pan_troglodytes 的远程文件夹
/usr/bin/python3 /root/s3_uploader.py --region cn-east-4 --app_token 806279e3-a95e-494d-81e7-12f9ccd57710 --bucket_name bucket-3028 \
    --bucket_path public/SpeciesLLM/scbasecount_2026-01-12/scbasecount_processed/ \
    --lfap /data/node2_disk1/SpeciesLLM_obs/scbasecount_2026-01-12/scbasecount_processed/Pan_troglodytes
# rm -rf /data/disk3/scbasecount_2026-01-12/h5ad/Gene/Pan_troglodytes/*

## 上传  这会产生 bucket-3028/public/SpeciesLLM/scbasecount_2026-01-12/scbasecount_processed/Schistosoma_mansoni 的远程文件夹
/usr/bin/python3 /root/s3_uploader.py --region cn-east-4 --app_token 806279e3-a95e-494d-81e7-12f9ccd57710 --bucket_name bucket-3028 \
    --bucket_path public/SpeciesLLM/scbasecount_2026-01-12/scbasecount_processed/ \
    --lfap /data/node2_disk1/SpeciesLLM_obs/scbasecount_2026-01-12/scbasecount_processed/Schistosoma_mansoni
# rm -rf /data/disk3/scbasecount_2026-01-12/h5ad/Gene/Schistosoma_mansoni/*

## 上传  这会产生 bucket-3028/public/SpeciesLLM/scbasecount_2026-01-12/scbasecount_processed/Solanum_lycopersicum 的远程文件夹
/usr/bin/python3 /root/s3_uploader.py --region cn-east-4 --app_token 806279e3-a95e-494d-81e7-12f9ccd57710 --bucket_name bucket-3028 \
    --bucket_path public/SpeciesLLM/scbasecount_2026-01-12/scbasecount_processed/ \
    --lfap /data/node2_disk1/SpeciesLLM_obs/scbasecount_2026-01-12/scbasecount_processed/Solanum_lycopersicum
# rm -rf /data/disk3/scbasecount_2026-01-12/h5ad/Gene/Solanum_lycopersicum/*

## 上传  这会产生 bucket-3028/public/SpeciesLLM/scbasecount_2026-01-12/scbasecount_processed/Sus_scrofa 的远程文件夹
/usr/bin/python3 /root/s3_uploader.py --region cn-east-4 --app_token 806279e3-a95e-494d-81e7-12f9ccd57710 --bucket_name bucket-3028 \
    --bucket_path public/SpeciesLLM/scbasecount_2026-01-12/scbasecount_processed/ \
    --lfap /data/node2_disk1/SpeciesLLM_obs/scbasecount_2026-01-12/scbasecount_processed/Sus_scrofa
# rm -rf /data/disk3/scbasecount_2026-01-12/h5ad/Gene/Sus_scrofa/*

## 上传  这会产生 bucket-3028/public/SpeciesLLM/scbasecount_2026-01-12/scbasecount_processed/Zea_mays 的远程文件夹
/usr/bin/python3 /root/s3_uploader.py --region cn-east-4 --app_token 806279e3-a95e-494d-81e7-12f9ccd57710 --bucket_name bucket-3028 \
    --bucket_path public/SpeciesLLM/scbasecount_2026-01-12/scbasecount_processed/ \
    --lfap /data/node2_disk1/SpeciesLLM_obs/scbasecount_2026-01-12/scbasecount_processed/Zea_mays
# rm -rf /data/disk3/scbasecount_2026-01-12/h5ad/Gene/Zea_mays/*

## 上传  这会产生 bucket-3028/public/SpeciesLLM/scbasecount_2026-01-12/scbasecount_processed/Homo_sapiens 的远程文件夹
/usr/bin/python3 /root/s3_uploader.py --region cn-east-4 --app_token 806279e3-a95e-494d-81e7-12f9ccd57710 --bucket_name bucket-3028 \
    --bucket_path public/SpeciesLLM/scbasecount_2026-01-12/scbasecount_processed/ \
    --lfap /data/node2_disk1/SpeciesLLM_obs/scbasecount_2026-01-12/scbasecount_processed/Homo_sapiens
# rm -rf /data/disk3/scbasecount_2026-01-12/h5ad/Gene/Homo_sapiens/*

## 上传  这会产生 bucket-3028/public/SpeciesLLM/scbasecount_2026-01-12/scbasecount_processed/Mus_musculus 的远程文件夹
/usr/bin/python3 /root/s3_uploader.py --region cn-east-4 --app_token 806279e3-a95e-494d-81e7-12f9ccd57710 --bucket_name bucket-3028 \
    --bucket_path public/SpeciesLLM/scbasecount_2026-01-12/scbasecount_processed/ \
    --lfap /data/node2_disk1/SpeciesLLM_obs/scbasecount_2026-01-12/scbasecount_processed/Mus_musculus
# rm -rf /data/disk3/scbasecount_2026-01-12/h5ad/Gene/Mus_musculus/*

