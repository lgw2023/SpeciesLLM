# 第一步
0_scbasecount_filter.py：
用标准化后的metadata填充obs
过滤非编码基因，保存编码基因
用min gene = 200过滤细胞

用法：
python 0_scbasecount_filter.py \
  --dir h5ad文件路径
  --coding-genes-csv 编码基因csv路径
  --gene-id-map-json gene id与gene symbol匹配的JSON路径
  --min-genes 200 \
  --outdir 输出路径

例如：
python 0_scbasecount_filter.py \
  --dir /ibex/project/c2307/arc-virtual-cell-atlas/scBaseCount/scbasecount/Zea_mays \
  --coding-genes-csv /ibex/project/c2307/datasets/gene_symbol2protein/gene_symbols/Zea_mays.csv \
  --gene-id-map-json /ibex/project/c2307/datasets/gene_symbol2protein/gene_id_to_gene_symbol_2nd_pretrain/Zea_mays.Zm-B73-REFERENCE-NAM-5.0.gene_ID_to_gene_symbol.json \
  --min-genes 200 \
  --outdir /ibex/project/c2307/arc-virtual-cell-atlas/scBaseCount/scbasecount_processed/Zea_mays

当前机器中的路径替换规则：
/ibex/project/c2307/arc-virtual-cell-atlas/scBaseCount/scbasecount = /data/disk1/SpeciesLLM_obs/scbasecount_demo/scbasecount_raw
/ibex/project/c2307/datasets/gene_symbol2protein/gene_symbols = /data/disk1/SpeciesLLM_obs/scbasecount_demo/data/gene_symbols
/ibex/project/c2307/datasets/gene_symbol2protein/gene_id_to_gene_symbol_2nd_pretrain = /data/disk1/SpeciesLLM_obs/scbasecount_demo/data/gene_id_to_gene_symbol_2nd_pretrain
/ibex/project/c2307/arc-virtual-cell-atlas/scBaseCount/scbasecount_processed = /data/disk1/SpeciesLLM_obs/scbasecount_demo/scbasecount_processed

```bash
python 0_scbasecount_filter.py \
  --dir /data/disk1/SpeciesLLM_obs/scbasecount_demo/scbasecount_raw/Arabidopsis_thaliana \
  --coding-genes-csv /data/disk1/SpeciesLLM_obs/scbasecount_demo/data/gene_symbols/Arabidopsis_thaliana.csv \
  --gene-id-map-json /data/disk1/SpeciesLLM_obs/scbasecount_demo/data/gene_id_to_gene_symbol_2nd_pretrain/Arabidopsis_thaliana.TAIR10.gene_ID_to_gene_symbol.json \
  --min-genes 200 \
  --outdir /data/disk1/SpeciesLLM_obs/scbasecount_demo/scbasecount_processed/Arabidopsis_thaliana &> log/0_scbasecount_filter/Arabidopsis_thaliana

python 0_scbasecount_filter.py \
  --dir /data/disk1/SpeciesLLM_obs/scbasecount_demo/scbasecount_raw/Bos_taurus \
  --coding-genes-csv /data/disk1/SpeciesLLM_obs/scbasecount_demo/data/gene_symbols/Bos_taurus.gene_symbols.csv \
  --gene-id-map-json /data/disk1/SpeciesLLM_obs/scbasecount_demo/data/gene_id_to_gene_symbol_2nd_pretrain/Bos_taurus.ARS-UCD2.0.gene_ID_to_gene_symbol.json \
  --min-genes 200 \
  --outdir /data/disk1/SpeciesLLM_obs/scbasecount_demo/scbasecount_processed/Bos_taurus &> log/0_scbasecount_filter/Bos_taurus

python 0_scbasecount_filter.py \
  --dir /data/disk1/SpeciesLLM_obs/scbasecount_demo/scbasecount_raw/Caenorhabditis_elegans \
  --coding-genes-csv /data/disk1/SpeciesLLM_obs/scbasecount_demo/data/gene_symbols/Caenorhabditis_elegans.csv \
  --gene-id-map-json /data/disk1/SpeciesLLM_obs/scbasecount_demo/data/gene_id_to_gene_symbol_2nd_pretrain/Caenorhabditis_elegans.WBcel235.gene_ID_to_gene_symbol.json \
  --min-genes 200 \
  --outdir /data/disk1/SpeciesLLM_obs/scbasecount_demo/scbasecount_processed/Caenorhabditis_elegans &> log/0_scbasecount_filter/Caenorhabditis_elegans

python 0_scbasecount_filter.py \
  --dir /data/disk1/SpeciesLLM_obs/scbasecount_demo/scbasecount_raw/Callithrix_jacchus \
  --coding-genes-csv /data/disk1/SpeciesLLM_obs/scbasecount_demo/data/gene_symbols/Callithrix_jacchus.gene_symbols.csv \
  --gene-id-map-json /data/disk1/SpeciesLLM_obs/scbasecount_demo/data/gene_id_to_gene_symbol_2nd_pretrain/Callithrix_jacchus.mCalJac1.pat.X.gene_ID_to_gene_symbol.json \
  --min-genes 200 \
  --outdir /data/disk1/SpeciesLLM_obs/scbasecount_demo/scbasecount_processed/Callithrix_jacchus &> log/0_scbasecount_filter/Callithrix_jacchus

python 0_scbasecount_filter.py \
  --dir /data/disk1/SpeciesLLM_obs/scbasecount_demo/scbasecount_raw/Danio_rerio \
  --coding-genes-csv /data/disk1/SpeciesLLM_obs/scbasecount_demo/data/gene_symbols/Danio_rerio.gene_symbols.csv \
  --gene-id-map-json /data/disk1/SpeciesLLM_obs/scbasecount_demo/data/gene_id_to_gene_symbol_2nd_pretrain/Danio_rerio.GRCz11.gene_ID_to_gene_symbol.json \
  --min-genes 200 \
  --outdir /data/disk1/SpeciesLLM_obs/scbasecount_demo/scbasecount_processed/Danio_rerio &> log/0_scbasecount_filter/Danio_rerio

python 0_scbasecount_filter.py \
  --dir /data/disk1/SpeciesLLM_obs/scbasecount_demo/scbasecount_raw/Drosophila_melanogaster \
  --coding-genes-csv /data/disk1/SpeciesLLM_obs/scbasecount_demo/data/gene_symbols/Drosophila_melanogaster.gene_symbols.csv \
  --gene-id-map-json /data/disk1/SpeciesLLM_obs/scbasecount_demo/data/gene_id_to_gene_symbol_2nd_pretrain/Drosophila_melanogaster.BDGP6.54.gene_ID_to_gene_symbol.json \
  --min-genes 200 \
  --outdir /data/disk1/SpeciesLLM_obs/scbasecount_demo/scbasecount_processed/Drosophila_melanogaster &> log/0_scbasecount_filter/Drosophila_melanogaster

python 0_scbasecount_filter.py \
  --dir /data/disk1/SpeciesLLM_obs/scbasecount_demo/scbasecount_raw/Equus_caballus \
  --coding-genes-csv /data/disk1/SpeciesLLM_obs/scbasecount_demo/data/gene_symbols/Equus_caballus.gene_symbols.csv \
  --gene-id-map-json /data/disk1/SpeciesLLM_obs/scbasecount_demo/data/gene_id_to_gene_symbol_2nd_pretrain/Equus_caballus.EquCab3.0.gene_ID_to_gene_symbol.json \
  --min-genes 200 \
  --outdir /data/disk1/SpeciesLLM_obs/scbasecount_demo/scbasecount_processed/Equus_caballus &> log/0_scbasecount_filter/Equus_caballus

python 0_scbasecount_filter.py \
  --dir /data/disk1/SpeciesLLM_obs/scbasecount_demo/scbasecount_raw/Gallus_gallus \
  --coding-genes-csv /data/disk1/SpeciesLLM_obs/scbasecount_demo/data/gene_symbols/Gallus_gallus.gene_symbols.csv \
  --gene-id-map-json /data/disk1/SpeciesLLM_obs/scbasecount_demo/data/gene_id_to_gene_symbol_2nd_pretrain/Gallus_gallus.bGalGal1.mat.broiler.GRCg7b.gene_ID_to_gene_symbol.json \
  --min-genes 200 \
  --outdir /data/disk1/SpeciesLLM_obs/scbasecount_demo/scbasecount_processed/Gallus_gallus &> log/0_scbasecount_filter/Gallus_gallus

python 0_scbasecount_filter.py \
  --dir /data/disk1/SpeciesLLM_obs/scbasecount_demo/scbasecount_raw/Gorilla_gorilla \
  --coding-genes-csv /data/disk1/SpeciesLLM_obs/scbasecount_demo/data/gene_symbols/Gorilla_gorilla.gene_symbols.csv \
  --gene-id-map-json /data/disk1/SpeciesLLM_obs/scbasecount_demo/data/gene_id_to_gene_symbol_2nd_pretrain/Gorilla_gorilla.gorGor4.gene_ID_to_gene_symbol.json \
  --min-genes 200 \
  --outdir /data/disk1/SpeciesLLM_obs/scbasecount_demo/scbasecount_processed/Gorilla_gorilla &> log/0_scbasecount_filter/Gorilla_gorilla

python 0_scbasecount_filter.py \
  --dir /data/disk1/SpeciesLLM_obs/scbasecount_demo/scbasecount_raw/Heterocephalus_glaber \
  --coding-genes-csv /data/disk1/SpeciesLLM_obs/scbasecount_demo/data/gene_symbols/Heterocephalus_glaber.csv \
  --gene-id-map-json /data/disk1/SpeciesLLM_obs/scbasecount_demo/data/gene_id_to_gene_symbol_2nd_pretrain/Heterocephalus_glaber_male.Naked_mole-rat_paternal.gene_ID_to_gene_symbol.json \
  --min-genes 200 \
  --outdir /data/disk1/SpeciesLLM_obs/scbasecount_demo/scbasecount_processed/Heterocephalus_glaber &> log/0_scbasecount_filter/Heterocephalus_glaber

python 0_scbasecount_filter.py \
  --dir /data/disk1/SpeciesLLM_obs/scbasecount_demo/scbasecount_raw/Homo_sapiens \
  --coding-genes-csv /data/disk1/SpeciesLLM_obs/scbasecount_demo/data/gene_symbols/Homo_sapiens.gene_symbol.csv \
  --gene-id-map-json /data/disk1/SpeciesLLM_obs/scbasecount_demo/data/gene_id_to_gene_symbol_2nd_pretrain/Homo_sapiens.GRCh38.gene_ID_to_gene_symbol.json \
  --min-genes 200 \
  --outdir /data/disk1/SpeciesLLM_obs/scbasecount_demo/scbasecount_processed/Homo_sapiens &> log/0_scbasecount_filter/Homo_sapiens

python 0_scbasecount_filter.py \
  --dir /data/disk1/SpeciesLLM_obs/scbasecount_demo/scbasecount_raw/Macaca_mulatta \
  --coding-genes-csv /data/disk1/SpeciesLLM_obs/scbasecount_demo/data/gene_symbols/Macaca_mulatta.gene_symbols.csv \
  --gene-id-map-json /data/disk1/SpeciesLLM_obs/scbasecount_demo/data/gene_id_to_gene_symbol_2nd_pretrain/Macaca_mulatta.Mmul_10.gene_ID_to_gene_symbol.json \
  --min-genes 200 \
  --outdir /data/disk1/SpeciesLLM_obs/scbasecount_demo/scbasecount_processed/Macaca_mulatta &> log/0_scbasecount_filter/Macaca_mulatta

python 0_scbasecount_filter.py \
  --dir /data/disk1/SpeciesLLM_obs/scbasecount_demo/scbasecount_raw/Mus_musculus \
  --coding-genes-csv /data/disk1/SpeciesLLM_obs/scbasecount_demo/data/gene_symbols/Mus_musculus.gene_symbol.csv \
  --gene-id-map-json /data/disk1/SpeciesLLM_obs/scbasecount_demo/data/gene_id_to_gene_symbol_2nd_pretrain/Mus_musculus.GRCm39.gene_ID_to_gene_symbol.json \
  --min-genes 200 \
  --outdir /data/disk1/SpeciesLLM_obs/scbasecount_demo/scbasecount_processed/Mus_musculus &> log/0_scbasecount_filter/Mus_musculus

python 0_scbasecount_filter.py \
  --dir /data/disk1/SpeciesLLM_obs/scbasecount_demo/scbasecount_raw/Oryctolagus_cuniculus \
  --coding-genes-csv /data/disk1/SpeciesLLM_obs/scbasecount_demo/data/gene_symbols/Oryctolagus_cuniculus.gene_symbols.csv \
  --gene-id-map-json /data/disk1/SpeciesLLM_obs/scbasecount_demo/data/gene_id_to_gene_symbol_2nd_pretrain/Oryctolagus_cuniculus.OryCun2.0.gene_ID_to_gene_symbol.json \
  --min-genes 200 \
  --outdir /data/disk1/SpeciesLLM_obs/scbasecount_demo/scbasecount_processed/Oryctolagus_cuniculus &> log/0_scbasecount_filter/Oryctolagus_cuniculus

python 0_scbasecount_filter.py \
  --dir /data/disk1/SpeciesLLM_obs/scbasecount_demo/scbasecount_raw/Oryza_sativa \
  --coding-genes-csv /data/disk1/SpeciesLLM_obs/scbasecount_demo/data/gene_symbols/Oryza_sativa.gene_symbols.csv \
  --gene-id-map-json /data/disk1/SpeciesLLM_obs/scbasecount_demo/data/gene_id_to_gene_symbol_2nd_pretrain/Oryza_sativa.IRGSP-1.0.gene_ID_to_gene_symbol.json \
  --min-genes 200 \
  --outdir /data/disk1/SpeciesLLM_obs/scbasecount_demo/scbasecount_processed/Oryza_sativa &> log/0_scbasecount_filter/Oryza_sativa

python 0_scbasecount_filter.py \
  --dir /data/disk1/SpeciesLLM_obs/scbasecount_demo/scbasecount_raw/Ovis_aries \
  --coding-genes-csv /data/disk1/SpeciesLLM_obs/scbasecount_demo/data/gene_symbols/Ovis_aries.gene_symbols.csv \
  --gene-id-map-json /data/disk1/SpeciesLLM_obs/scbasecount_demo/data/gene_id_to_gene_symbol_2nd_pretrain/Ovis_aries.Oar_v3.1.gene_ID_to_gene_symbol.json \
  --min-genes 200 \
  --outdir /data/disk1/SpeciesLLM_obs/scbasecount_demo/scbasecount_processed/Ovis_aries &> log/0_scbasecount_filter/Ovis_aries

python 0_scbasecount_filter.py \
  --dir /data/disk1/SpeciesLLM_obs/scbasecount_demo/scbasecount_raw/Pan_troglodytes \
  --coding-genes-csv /data/disk1/SpeciesLLM_obs/scbasecount_demo/data/gene_symbols/Pan_troglodytes.gene_symbols.csv \
  --gene-id-map-json /data/disk1/SpeciesLLM_obs/scbasecount_demo/data/gene_id_to_gene_symbol_2nd_pretrain/Pan_troglodytes.Pan_tro_3.0.gene_ID_to_gene_symbol.json \
  --min-genes 200 \
  --outdir /data/disk1/SpeciesLLM_obs/scbasecount_demo/scbasecount_processed/Pan_troglodytes &> log/0_scbasecount_filter/Pan_troglodytes

python 0_scbasecount_filter.py \
  --dir /data/disk1/SpeciesLLM_obs/scbasecount_demo/scbasecount_raw/Schistosoma_mansoni \
  --coding-genes-csv /data/disk1/SpeciesLLM_obs/scbasecount_demo/data/gene_symbols/Schistosoma_mansoni.csv \
  --gene-id-map-json /data/disk1/SpeciesLLM_obs/scbasecount_demo/data/gene_id_to_gene_symbol_2nd_pretrain/Schistosoma_mansoni.Smansoni_v7.gene_ID_to_gene_symbol.json \
  --min-genes 200 \
  --outdir /data/disk1/SpeciesLLM_obs/scbasecount_demo/scbasecount_processed/Schistosoma_mansoni &> log/0_scbasecount_filter/Schistosoma_mansoni

python 0_scbasecount_filter.py \
  --dir /data/disk1/SpeciesLLM_obs/scbasecount_demo/scbasecount_raw/Solanum_lycopersicum \
  --coding-genes-csv /data/disk1/SpeciesLLM_obs/scbasecount_demo/data/gene_symbols/Solanum_lycopersicum.csv \
  --gene-id-map-json /data/disk1/SpeciesLLM_obs/scbasecount_demo/data/gene_id_to_gene_symbol_2nd_pretrain/Solanum_lycopersicum.SL3.0.gene_ID_to_gene_symbol.json \
  --min-genes 200 \
  --outdir /data/disk1/SpeciesLLM_obs/scbasecount_demo/scbasecount_processed/Solanum_lycopersicum &> log/0_scbasecount_filter/Solanum_lycopersicum

python 0_scbasecount_filter.py \
  --dir /data/disk1/SpeciesLLM_obs/scbasecount_demo/scbasecount_raw/Sus_scrofa \
  --coding-genes-csv /data/disk1/SpeciesLLM_obs/scbasecount_demo/data/gene_symbols/Sus_scrofa.gene_symbols.csv \
  --gene-id-map-json /data/disk1/SpeciesLLM_obs/scbasecount_demo/data/gene_id_to_gene_symbol_2nd_pretrain/Sus_scrofa.Sscrofa11.1.gene_ID_to_gene_symbol.json \
  --min-genes 200 \
  --outdir /data/disk1/SpeciesLLM_obs/scbasecount_demo/scbasecount_processed/Sus_scrofa &> log/0_scbasecount_filter/Sus_scrofa

python 0_scbasecount_filter.py \
  --dir /data/disk1/SpeciesLLM_obs/scbasecount_demo/scbasecount_raw/Zea_mays \
  --coding-genes-csv /data/disk1/SpeciesLLM_obs/scbasecount_demo/data/gene_symbols/Zea_mays.csv \
  --gene-id-map-json /data/disk1/SpeciesLLM_obs/scbasecount_demo/data/gene_id_to_gene_symbol_2nd_pretrain/Zea_mays.Zm-B73-REFERENCE-NAM-5.0.gene_ID_to_gene_symbol.json \
  --min-genes 200 \
  --outdir /data/disk1/SpeciesLLM_obs/scbasecount_demo/scbasecount_processed/Zea_mays &> log/0_scbasecount_filter/Zea_mays
```


#  第二步
02_scbasecount.py：
scTab的02步骤，把每个物种的obs, var, X分别存储，X分块存成稠密矩阵

# 第三步
03_scbasecount.py:
scTab的03步骤，对X做log1p，把obs转换成INT,然后写成按obs存储的parquet数据表，并存储obs的lookup表
这里为了双方多个数据集INT与obs对应关系一致，assay, cell_type, development_stage, disease, sex, species, tissue已经用sample_metadata转换好了编码
dataset_id, soma_joinid, tech_sample在已有INT编号的基础上逐号续写

# data文件目录解释
scbasecount_demo/data:
  gene_id_to_gene_symbol_2nd_pretrain： gene id与gene symbol匹配的JSON文件
  gene_symbols： 编码基因文件
  scbasecount_sample_metadata_processed：每个物种路径下是标准化disease和tissue后的sample_metadata_with_mondo_tissue.parquet
  LOOKUP_categories_unified：obs转换INT后，查询编码对应的文件
