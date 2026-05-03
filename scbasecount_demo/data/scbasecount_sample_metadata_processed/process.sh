#!/usr/bin/env bash
set -euo pipefail
trap 'echo "[!] Error at line $LINENO" >&2' ERR

# ---- 可配置参数 ----
BUCKET="gs://arc-scbasecount"
DATE="2025-02-25"
DATASET="GeneFull_Ex50pAS"
GSDK="./google-cloud-sdk/bin/gsutil"

# 需要处理的物种列表
SPECIES=(
  Pan_troglodytes Oryza_sativa Macaca_mulatta Gallus_gallus
  Oryctolagus_cuniculus Caenorhabditis_elegans Schistosoma_mansoni Sus_scrofa
  Drosophila_melanogaster Arabidopsis_thaliana Solanum_lycopersicum Bos_taurus
  Mus_musculus Homo_sapiens Equus_caballus Gorilla_gorilla
  Heterocephalus_glaber Zea_mays Danio_rerio Ovis_aries Callithrix_jacchus
)

# ---- 一次性准备 ----
cp -r /home/ma-user/anaconda3/work/* ./ || true

# 确保 conda 可在非交互 shell 中使用；若你环境已能直接 `conda activate` 可删掉下一行
eval "$(conda shell.bash hook)"
conda activate base

# 依赖一次性安装（你原来每个物种都装了一次，这里统一到顶层）
pip install -q pyarrow

# ---- 下载 h5ad 数据 ----
echo "==> Downloading h5ad folders..."
for sp in "${SPECIES[@]}"; do
  echo "  - ${sp}"
  # -m 开多线程；若不想覆盖已有文件可加 -n
  "${GSDK}" -m cp -r "${BUCKET}/${DATE}/h5ad/${DATASET}/${sp}/" .
done

# ---- 下载 metadata 到对应物种目录 ----
echo "==> Downloading metadata..."
for sp in "${SPECIES[@]}"; do
  echo "  - ${sp}"
  mkdir -p "${sp}"
  "${GSDK}" cp "${BUCKET}/${DATE}/metadata/${DATASET}/${sp}/sample_metadata.parquet" "${sp}/"
  "${GSDK}" cp "${BUCKET}/${DATE}/metadata/${DATASET}/${sp}/obs_metadata.parquet"     "${sp}/"
done

# ---- 快速检查：打印每个物种的 sample_metadata & 统计 h5ad 数量 ----
python - <<'PY'
import os, glob, pyarrow.dataset as ds
species = [
  "Pan_troglodytes","Oryza_sativa","Macaca_mulatta","Gallus_gallus",
  "Oryctolagus_cuniculus","Caenorhabditis_elegans","Schistosoma_mansoni","Sus_scrofa",
  "Drosophila_melanogaster","Arabidopsis_thaliana","Solanum_lycopersicum","Bos_taurus",
  "Mus_musculus","Homo_sapiens","Equus_caballus","Gorilla_gorilla",
  "Heterocephalus_glaber","Zea_mays","Danio_rerio","Ovis_aries","Callithrix_jacchus",
]
for sp in species:
    print(f"\n--- {sp} ---")
    try:
        tbl = ds.dataset(os.path.join(sp, "sample_metadata.parquet"), format="parquet").to_table()
        print(tbl.to_pandas())
    except Exception as e:
        print(f"[warn] cannot read sample_metadata: {e}")
    n = len(glob.glob(os.path.join(sp, "*.h5ad")))
    print(f"h5ad files: {n}")
PY

# ---- 你的校验脚本（按原逻辑示例执行一个物种；如需全量可自己循环 SPECIES）----
SPECIES=(
  Pan_troglodytes Oryza_sativa Macaca_mulatta Gallus_gallus
  Oryctolagus_cuniculus Caenorhabditis_elegans Schistosoma_mansoni Sus_scrofa
  Drosophila_melanogaster Arabidopsis_thaliana Solanum_lycopersicum Bos_taurus
  Mus_musculus Homo_sapiens Equus_caballus Gorilla_gorilla
  Heterocephalus_glaber Zea_mays Danio_rerio Ovis_aries Callithrix_jacchus
)
echo "==> validate_h5ad_vs_metadata..."
for sp in "${SPECIES[@]}"; do
    # 在 Shell 层面判断文件夹是否存在
    if [ -d "${sp}" ]; then
        echo "validate_h5ad_vs_metadata ${sp}"
        python validate_h5ad_vs_metadata.py --base_dir=${sp}
        cat ${sp}/h5ad_validation_report.csv
        cat ${sp}/h5ad_validation_report.csv | grep -i "False"
    else
        echo "WARNING: 文件夹 ${sp} 不存在，跳过validate_h5ad_vs_metadata。"
    fi
done

# ---- 你的校验脚本（按原逻辑示例执行一个物种；如需全量可自己循环 SPECIES）----

SPECIES=(
  Pan_troglodytes Oryza_sativa Macaca_mulatta Gallus_gallus
  Oryctolagus_cuniculus Caenorhabditis_elegans Schistosoma_mansoni Sus_scrofa
  Drosophila_melanogaster Arabidopsis_thaliana Solanum_lycopersicum Bos_taurus
  Mus_musculus Homo_sapiens Equus_caballus Gorilla_gorilla
  Heterocephalus_glaber Zea_mays Danio_rerio Ovis_aries Callithrix_jacchus
)

echo "==> validate_h5ad_vs_metadata..."
for sp in "${SPECIES[@]}"; do
    # 在 Shell 层面判断文件夹是否存在
    if [ -d "${sp}" ]; then
        echo "validate_h5ad_vs_metadata ${sp}"
        export species=${sp}
        /home/ma-user/anaconda3/envs/PyTorch-2.1.0/bin/python -c 'import os;import moxing as mox;mox.file.copy_parallel("./" + os.environ.get("species"), "s3://bucket-ai-for-medicine/liguowei/scbasecount/" + os.environ.get("species"))'
    else
        echo "WARNING: 文件夹 ${sp} 不存在，跳过复制。"
    fi
done
