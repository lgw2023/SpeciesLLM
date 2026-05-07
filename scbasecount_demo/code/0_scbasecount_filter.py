#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import sys
import pandas as pd
import numpy as np

import json
import json as _json

import anndata as ad
from scipy import sparse
ad.settings.allow_write_nullable_strings = True
import os
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")


TECH10X_MAP = {
    "3_prime_gex": "10x 3' transcription profiling",
    "5_prime_gex": "10x 5' transcription profiling",
    "flex": "10x gene expression flex",
    "vdj": "10x immune profiling",
}

def log(msg): print(f"[process_h5ad] {msg}", flush=True)

def to_lower_underscore(x):
    if pd.isna(x): return None
    return str(x).strip().lower().replace(" ", "_")

def ensure_csr(X):
    return X.tocsr() if sparse.issparse(X) else X

def n_genes_by_counts(X):
    if sparse.issparse(X):
        X = ensure_csr(X)
        return np.diff(X.indptr)
    arr = np.asarray(X)
    return np.asarray((arr > 0).sum(axis=1)).ravel()

def shape_report(adata, tag):
    layers_info = []
    if hasattr(adata, "layers") and isinstance(adata.layers, dict) and len(adata.layers) > 0:
        for k, v in adata.layers.items():
            try:
                layers_info.append((k, tuple(v.shape)))
            except Exception:
                layers_info.append((k, "shape_error"))
    else:
        layers_info = "none"
    raw_info = "none"
    if getattr(adata, "raw", None) is not None:
        try:
            raw_info = {"X": tuple(adata.raw.X.shape),
                        "var": adata.raw.var.shape[0],
                        "obs": tuple(adata.raw.obs.shape)}
        except Exception:
            raw_info = "error"
    log(f"{tag}: X={adata.X.shape}, var={adata.var.shape[0]}, obs={adata.obs.shape}, layers={layers_info}, raw={raw_info}")

def load_coding_genes(csv_path: Path) -> pd.Index:
    df = pd.read_csv(csv_path)
    cand = [c for c in df.columns if "gene" in c.lower()]
    col = cand[0] if cand else df.columns[0]
    genes = df[col].astype(str).str.strip().str.lower()
    genes = genes[genes != ""].dropna().drop_duplicates()
    log(f"编码基因CSV：列='{col}', 基因数={len(genes)}")
    return pd.Index(genes)


def map_assay(tech_val):
    tech = to_lower_underscore(tech_val) if tech_val is not None else None
    return TECH10X_MAP.get(tech, "unknown")


def _norm_gene_id(x: str) -> str:
    x = str(x).strip().lower()
    if "." in x:
        x = x.split(".", 1)[0]
    return x

def load_gene_id_map(json_path: Path) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        raw = _json.load(f)
    id2sym = {}
    for k, v in raw.items():
        if v is None:
            continue
        v_str = str(v).strip()
        if v_str == "":
            continue
        id2sym[_norm_gene_id(k)] = v_str
    log(f"加载 gene id→symbol 映射：{len(id2sym)} 条（来自 {json_path}）")
    return id2sym

def map_mixed_gene_symbols_to_symbols(adata: ad.AnnData, id2sym: dict):
    if "gene_symbols" not in adata.var.columns:
        raise ValueError("adata.var 缺少 'gene_symbols' 列。")

    sym = adata.var["gene_symbols"].astype(str).str.strip()
    mapped = sym.map(lambda x: id2sym.get(_norm_gene_id(x), x))
    adata.var["gene_symbols"] = mapped
    n_changed = int((mapped != sym).sum())
    log(f"[gene-id-map] 基于 gene_symbols 完成映射：{n_changed} / {len(sym)} 条被替换为 symbol")

def sanitize_obs_for_h5ad(adata: ad.AnnData):
    obs = adata.obs.copy()

    for col in obs.columns:
        s = obs[col]
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s) or pd.api.types.is_categorical_dtype(s):
            continue
        if s.dtype == object:
            def _cast(x):
                if x is None or (isinstance(x, float) and np.isnan(x)):
                    return ""
                if isinstance(x, (list, dict, tuple, set, np.ndarray)):
                    try:
                        return json.dumps(x, ensure_ascii=False)
                    except Exception:
                        return str(x)
                return str(x)

            obs[col] = s.map(_cast).astype("string")  # 统一成 pandas StringDtype

    adata.obs = obs

    adata.obs_names = adata.obs_names.astype(str)
    adata.var_names = adata.var_names.astype(str)

def fill_obs_simple(adata: ad.AnnData, smeta: pd.DataFrame):
    if "SRX_accession" not in adata.obs.columns:
        raise ValueError("adata.obs 缺少 'SRX_accession' 列。")
    if "srx_accession" not in smeta.columns:
        raise ValueError("sample_metadata_with_mondo_tissue.parquet 缺少 'srx_accession' 列。")

    obs = adata.obs.copy()
    obs["SRX_accession"] = obs["SRX_accession"].astype(str)

    want = ["srx_accession", "organism", "tissue", "tech_10x", "mondo_id"]
    use_cols = [c for c in want if c in smeta.columns]
    smeta_use = smeta[use_cols].copy()
    smeta_use["srx_accession"] = smeta_use["srx_accession"].astype(str)
    smeta_use = smeta_use.set_index("srx_accession")

    obs = obs.join(smeta_use, on="SRX_accession", how="left")

    if "organism" in obs.columns:
        obs["species"] = obs["organism"].map(lambda x: to_lower_underscore(x) or "unknown")
    else:
        obs["species"] = "unknown"
    obs["tissue"] = obs["tissue"].astype(str) if "tissue" in obs.columns else "unknown"
    obs["assay"] = obs["tech_10x"].apply(map_assay) if "tech_10x" in obs.columns else "unknown"

    obs["dataset_id"] = obs["SRX_accession"].astype("string").str.strip().str.lower()
    obs["sex"] = "unknown"
    obs["development_stage"] = "unknown"
    obs["tech_sample"] = obs["dataset_id"] + "_" + obs["assay"]
    obs["disease"] = obs["mondo_id"].astype("string").str.strip().str.lower().fillna("unknown")
    obs["soma_joinid"] = obs["tech_sample"] + "_" + obs.index.astype(str)

    for c in ["organism", "tech_10x", "mondo_id"]:
        if c in obs.columns:
            del obs[c]

    adata.obs = obs



def subset_vars_keep_and_append_keep_dups(adata: ad.AnnData, keep_symbols: pd.Index) -> ad.AnnData:
    if "gene_symbols" not in adata.var.columns:
        raise ValueError("adata.var 缺少 'gene_symbols' 列。")

    current_raw = adata.var["gene_symbols"]
    current = current_raw.astype(str).str.strip().str.lower().fillna("")
    adata.var["gene_symbols"] = current

    nunique_before = current_raw.nunique(dropna=False)
    nunique_lower = current.nunique(dropna=False)
    dup_count = int(current.duplicated(keep="first").sum())
    log(f"[keep+append] gene_symbols 原唯一数={nunique_before}, 小写后唯一数={nunique_lower}, 重复列数={dup_count}")

    keep_symbols = pd.Index(pd.Series(keep_symbols).astype(str).str.strip().str.lower().dropna().unique())
    log(f"[keep+append] 编码基因数={len(keep_symbols)}")

    mask_keep_existing = current.isin(keep_symbols)
    kept_cols = int(mask_keep_existing.sum())
    log(f"[keep+append] 当前矩阵中属于编码基因的列数（含重复）={kept_cols} / {adata.n_vars}")
    adata._inplace_subset_var(mask_keep_existing.values)

    present_set = set(adata.var["gene_symbols"].astype(str))
    missing = [g for g in keep_symbols if g not in present_set]
    log(f"[keep+append] 需要补零的编码基因数={len(missing)}")

    if len(missing) == 0:
        shape_report(adata, "after keep+append (no-missing)")
        return adata

    n_obs = adata.n_obs
    if sparse.issparse(adata.X):
        zeros_block = sparse.csr_matrix((n_obs, len(missing)), dtype=adata.X.dtype)
    else:
        zeros_block = np.zeros((n_obs, len(missing)), dtype=np.asarray(adata.X).dtype)

    zero_var = pd.DataFrame({"gene_symbols": missing}, index=pd.Index(missing, name=None))
    zeros_adata = ad.AnnData(X=zeros_block, obs=adata.obs.copy(), var=zero_var)

    adata_new = ad.concat([adata, zeros_adata], axis=1, join="outer", merge="same")

    adata_new.var["gene_symbols"] = adata_new.var["gene_symbols"].astype(str).str.lower()
    shape_report(adata_new, "after keep+append (concat)")
    return adata_new

def collapse_duplicate_genes_sum(adata: ad.AnnData, handle_raw="drop") -> ad.AnnData:
    if "gene_symbols" not in adata.var.columns:
        raise ValueError("adata.var 缺少 'gene_symbols' 列。")

    gs = adata.var["gene_symbols"].astype(str).str.strip().str.lower()
    adata.var["gene_symbols"] = gs

    n_obs, n_vars = adata.n_obs, adata.n_vars
    if n_vars == 0:
        X_empty = sparse.csr_matrix((n_obs, 0), dtype=adata.X.dtype) if sparse.issparse(adata.X) \
                  else np.zeros((n_obs, 0), dtype=np.asarray(adata.X).dtype)
        new_var = pd.DataFrame({"gene_symbols": pd.Index([], dtype="object")})
        new_var.index = new_var["gene_symbols"]
        new = ad.AnnData(X=X_empty, obs=adata.obs.copy(), var=new_var)
        if handle_raw == "drop":
            new.raw = None
        shape_report(new, "after collapse (n_vars==0)")
        return new

    labels, uniques = pd.factorize(gs.to_numpy(), sort=False)  # labels: 每列所属组ID
    n_groups = len(uniques)
    dup_groups = (pd.Series(labels).value_counts() > 1).sum()
    log(f"[collapse] 合并后基因数={n_groups}（其中重复基因组数={dup_groups}）")

    rows = np.arange(n_vars, dtype=int)
    cols = labels.astype(int)
    dtypeX = (adata.X.dtype if sparse.issparse(adata.X) else np.asarray(adata.X).dtype)
    M = sparse.csr_matrix((np.ones_like(rows, dtype=dtypeX), (rows, cols)), shape=(n_vars, n_groups))

    if sparse.issparse(adata.X):
        X_new = adata.X @ M
    else:
        X_new = (sparse.csr_matrix(np.asarray(adata.X)) @ M).toarray()

    new_layers = {}
    if hasattr(adata, "layers") and len(adata.layers) > 0:
        for k in list(adata.layers.keys()):
            L = adata.layers[k]
            if sparse.issparse(L):
                new_layers[k] = L @ M
            else:
                new_layers[k] = (sparse.csr_matrix(np.asarray(L)) @ M).toarray()


    new_raw = None if handle_raw == "drop" else adata.raw

    new_var = pd.DataFrame({"gene_symbols": uniques})
    new_var.index = new_var["gene_symbols"]

    new = ad.AnnData(X=X_new, obs=adata.obs.copy(), var=new_var)
    if new_layers:
        for k, v in new_layers.items():
            new.layers[k] = v
    new.raw = new_raw

    assert new.X.shape[1] == new.var.shape[0], f"post-collapse shape mismatch: X={new.X.shape}, var={new.var.shape}"
    if hasattr(new, "layers") and len(new.layers) > 0:
        for k, L in new.layers.items():
            assert L.shape[1] == new.var.shape[0], f"layer '{k}' shape {L.shape} != var {new.var.shape[0]}"

    shape_report(new, "after collapse")
    return new

def filter_cells_by_min_genes(adata: ad.AnnData, min_genes: int = 200):
    counts = n_genes_by_counts(adata.X)
    thresh = int(min_genes)
    keep = counts >= thresh
    log(f"[filter200] 每细胞>0基因数：min={counts.min() if counts.size>0 else 'NA'}, "
        f"median={np.median(counts) if counts.size>0 else 'NA'}, max={counts.max() if counts.size>0 else 'NA'}")
    log(f"[filter200] 保留细胞数={int(keep.sum())} / {adata.n_obs} (阈值={thresh})")
    if keep.sum() == 0:
        log("警告：过滤后无细胞保留，跳过此文件。")
        return False
    adata._inplace_subset_obs(keep)
    shape_report(adata, "after filter200")
    return True


def process_one(h5ad_path: Path, smeta: pd.DataFrame, coding_genes: pd.Index,
                min_genes: int, outdir: Path, overwrite: bool, id2sym: dict):

    log(f"处理：{h5ad_path.name}")
    adata = ad.read_h5ad(h5ad_path)
    shape_report(adata, "loaded")



    if id2sym:
        map_mixed_gene_symbols_to_symbols(adata, id2sym)
    else:
        log("[gene-id-map] 未提供映射或为空，跳过 gene id→symbol 转换")



    adata = subset_vars_keep_and_append_keep_dups(adata, keep_symbols=coding_genes)
    adata = collapse_duplicate_genes_sum(adata, handle_raw="drop")
    if not filter_cells_by_min_genes(adata, min_genes=min_genes):
        return
    fill_obs_simple(adata, smeta)
    log(f"[fill_obs] 目标列已生成："
        f"species={ 'species' in adata.obs.columns }, "
        f"tissue={ 'tissue' in adata.obs.columns }, "
        f"assay={ 'assay' in adata.obs.columns }, "
        f"dataset_id={ 'dataset_id' in adata.obs.columns }, "
        f"sex={ 'sex' in adata.obs.columns }, "
        f"development_stage={ 'development_stage' in adata.obs.columns }, "
        f"tech_sample={ 'tech_sample' in adata.obs.columns }, "
        f"disease={ 'disease' in adata.obs.columns }, "
        f"soma_joinid={ 'soma_joinid' in adata.obs.columns }")
    shape_report(adata, "after fill_obs")
    sanitize_obs_for_h5ad(adata)
    if "gene_symbols" in adata.var.columns:
        adata.var.index = adata.var["gene_symbols"].astype(str)
        adata.var = adata.var.drop(columns=["gene_symbols"])
    else:
        log("[save] 警告：var 中未找到 'gene_symbols' 列可删除；将按当前索引写出。")

    if outdir is None:
        out_path = h5ad_path if overwrite else h5ad_path.with_name(h5ad_path.stem + ".processed.h5ad")
    else:
        outdir.mkdir(parents=True, exist_ok=True)
        out_path = outdir / (h5ad_path.name if overwrite else (h5ad_path.stem + ".processed.h5ad"))
    adata.write_h5ad(out_path)
    log(f"已保存：{out_path}")

def main():
    parser = argparse.ArgumentParser(description="批处理 h5ad：编码基因筛选+补零 -> 合并重复基因 -> 过滤min genes -> 填充obs")
    parser.add_argument("--dir", required=True, help="含 .h5ad 与 sample_metadata_with_mondo_tissue.parquet 的目录")
    parser.add_argument("--coding-genes-csv", required=True, help="编码基因 CSV（全小写，含表头）")
    parser.add_argument("--min-genes", type=int, default=200)
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--gene-id-map-json", default=None,
                        help="gene id→symbol 映射 JSON（key=id, value=symbol）。若不提供，将尝试在 --dir 下寻找 gene_id2symbol.json")

    args = parser.parse_args()

    base = Path(args.dir).expanduser().resolve()
    if not base.exists():
        sys.exit(f"目录不存在：{base}")

    smeta_fp = base / "sample_metadata_with_mondo_tissue.parquet"
    if not smeta_fp.exists():
        sys.exit(f"缺少文件：{smeta_fp}")

    smeta = pd.read_parquet(smeta_fp)
    coding_genes = load_coding_genes(Path(args.coding_genes_csv))

    gene_map_fp = None
    if args.gene_id_map_json:
        candidate = Path(args.gene_id_map_json).expanduser().resolve()
        if not candidate.exists():
            sys.exit(f"gene id→symbol JSON 不存在：{candidate}")
        gene_map_fp = candidate
    else:
        candidate = base / "gene_id2symbol.json"
        if candidate.exists():
            gene_map_fp = candidate

    id2sym = load_gene_id_map(gene_map_fp) if gene_map_fp else {}


    h5ads = sorted(base.glob("*.h5ad"))
    if not h5ads:
        sys.exit(f"目录中未发现 .h5ad 文件：{base}")

    outdir = Path(args.outdir).expanduser().resolve() if args.outdir else None
    for fp in h5ads:
        try:
            process_one(fp, smeta, coding_genes, args.min_genes, outdir, args.overwrite, id2sym=id2sym)
        except Exception as e:
            log(f"处理失败：{fp.name} | {e}")
            try:
                adata_dbg = ad.read_h5ad(fp)
                shape_report(adata_dbg, "debug reload (original)")
            except Exception as e2:
                log(f"debug reload failed: {e2}")
    log("全部处理完成。")

if __name__ == "__main__":
    main()
