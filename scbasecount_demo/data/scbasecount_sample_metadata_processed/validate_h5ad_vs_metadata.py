#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
校验 Oryza_sativa/ 下的 h5ad 与 metadata 一致性，并输出 CSV 报告。
- 单文件级别：n_obs vs sample_metadata.obs_count
- 单细胞级别：gene_count、umi_count vs obs_metadata
- 集合级别：cell_barcode 集合是否一致

用法：
    python validate_h5ad_vs_metadata.py [--base_dir Oryza_sativa]

输出：
    {base_dir}/h5ad_validation_report.csv
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import anndata as ad


def normalize_counts(s: pd.Series) -> pd.Series:
    """
    将计数列转为 Int64（能容纳缺失值 NA），容忍 float/字符串等输入。
    会先转为数值、四舍五入，再转为 Int64。
    """
    s2 = pd.to_numeric(s, errors="coerce")
    # 计数应为整数，这里做一次 round（通常只是 *.0）
    return s2.round().astype("Int64")


def load_sample_metadata(sample_path: Path) -> pd.Series:
    """
    读取 sample_metadata.parquet，返回：
      index = srx_accession（字符串），value = obs_count（int）
    """
    df = pd.read_parquet(sample_path)
    # 兼容大小写
    if "srx_accession" not in df.columns:
        # 有些表用 "SRX_accession"
        if "SRX_accession" in df.columns:
            df = df.rename(columns={"SRX_accession": "srx_accession"})
        else:
            raise KeyError("sample_metadata.parquet 缺少列：srx_accession / SRX_accession")

    if "obs_count" not in df.columns:
        raise KeyError("sample_metadata.parquet 缺少列：obs_count")

    df["srx_accession"] = df["srx_accession"].astype(str)
    # 如果同一 SRX 有重复行，优先取首次出现（通常不会重复）
    s = (df[["srx_accession", "obs_count"]]
         .drop_duplicates(subset=["srx_accession"])
         .set_index("srx_accession")["obs_count"])
    s = pd.to_numeric(s, errors="coerce").astype("Int64")
    return s


def load_obs_metadata(obs_meta_path: Path) -> pd.DataFrame:
    """
    读取 obs_metadata.parquet，规范列名并归一化 gene/umi 为 Int64。
    返回包含列：
      srx_accession, cell_barcode, gene_count, umi_count
    """
    obs = pd.read_parquet(obs_meta_path)

    # 统一列名
    rename_map = {}
    if "SRX_accession" in obs.columns:
        rename_map["SRX_accession"] = "srx_accession"
    obs = obs.rename(columns=rename_map)

    required = {"srx_accession", "cell_barcode"}
    missing = required - set(obs.columns)
    if missing:
        raise KeyError(f"obs_metadata.parquet 缺少列：{missing}")

    obs["srx_accession"] = obs["srx_accession"].astype(str)
    if "gene_count" not in obs.columns:
        obs["gene_count"] = pd.Series([pd.NA] * len(obs), dtype="Int64")
    if "umi_count" not in obs.columns:
        obs["umi_count"] = pd.Series([pd.NA] * len(obs), dtype="Int64")

    obs["gene_count"] = normalize_counts(obs["gene_count"])
    obs["umi_count"] = normalize_counts(obs["umi_count"])
    obs["cell_barcode"] = obs["cell_barcode"].astype(str)
    return obs[["srx_accession", "cell_barcode", "gene_count", "umi_count"]]


def validate_one_h5ad(h5_path: Path,
                      sample_obs_count_map: pd.Series,
                      obs_meta_all: pd.DataFrame) -> dict:
    """
    校验单个 h5ad 文件，返回一行结果（作为字典）。
    """
    srx = h5_path.stem  # 文件名去 .h5ad
    rec = {
        "file_name": h5_path.name,
        "srx_accession": srx,
        "error": "",
        "h5ad_n_obs": pd.NA,
        "metadata_obs_count": pd.NA,
        "obs_count_match": pd.NA,

        "h5ad_n_cells": pd.NA,
        "obsmeta_n_cells": pd.NA,
        "cells_overlap": pd.NA,
        "cells_only_in_h5ad": pd.NA,
        "cells_only_in_obsmeta": pd.NA,
        "cell_barcode_set_equal": pd.NA,

        "has_gene_count_col_in_h5ad": pd.NA,
        "has_umi_count_col_in_h5ad": pd.NA,
        "gene_count_mismatches": pd.NA,
        "gene_count_match_rate": pd.NA,
        "umi_count_mismatches": pd.NA,
        "umi_count_match_rate": pd.NA,

        "per_cell_checks_pass": pd.NA,
        "overall_pass": pd.NA,
    }

    try:
        adata = ad.read_h5ad(str(h5_path), backed="r")  # 不加载矩阵到内存
    except Exception as e:
        rec["error"] = f"读取 h5ad 失败: {e}"
        return rec

    try:
        # --- 文件级别 ---
        n_obs = int(adata.n_obs)
        rec["h5ad_n_obs"] = n_obs
        rec["h5ad_n_cells"] = n_obs

        expected_obs = sample_obs_count_map.get(srx, pd.NA)
        rec["metadata_obs_count"] = expected_obs
        
        if pd.isna(expected_obs):
            rec["obs_count_match"] = False     # ⚠️ 保持为布尔
        else:
            rec["obs_count_match"] = (n_obs == int(expected_obs))

        # --- 细胞级别 / 集合级别 ---
        has_gene = "gene_count" in adata.obs.columns
        has_umi = "umi_count" in adata.obs.columns
        rec["has_gene_count_col_in_h5ad"] = bool(has_gene)
        rec["has_umi_count_col_in_h5ad"] = bool(has_umi)

        # 直接从 obs 抽列，并把索引（条形码）下放为一列
        h5_df = adata.obs[["gene_count", "umi_count"]].copy()
        h5_df.index = h5_df.index.astype(str)
        h5_df = h5_df.rename_axis("cell_barcode").reset_index()

        h5_df["gene_count_h5"] = normalize_counts(h5_df.pop("gene_count"))
        h5_df["umi_count_h5"]  = normalize_counts(h5_df.pop("umi_count"))
        # 保留需要的列
        h5_df = h5_df[["cell_barcode", "gene_count_h5", "umi_count_h5"]]

        meta_df = obs_meta_all.loc[obs_meta_all["srx_accession"] == srx,
                                   ["cell_barcode", "gene_count", "umi_count"]].copy()
        meta_df = meta_df.rename(columns={
            "gene_count": "gene_count_meta",
            "umi_count": "umi_count_meta",
        })
        meta_df["cell_barcode"] = meta_df["cell_barcode"].astype(str)

        rec["obsmeta_n_cells"] = int(len(meta_df))

        merged = h5_df.merge(meta_df, on="cell_barcode", how="inner", validate="one_to_one")
        n_inter = int(len(merged))
        only_in_h5 = int(len(h5_df) - n_inter)
        only_in_meta = int(len(meta_df) - n_inter)

        rec["cells_overlap"] = n_inter
        rec["cells_only_in_h5ad"] = only_in_h5
        rec["cells_only_in_obsmeta"] = only_in_meta
        rec["cell_barcode_set_equal"] = (only_in_h5 == 0 and only_in_meta == 0)

        # --- gene_count 对比（仅在两边都有值时计入比对）---
        gene_comp_mask = merged["gene_count_h5"].notna() & merged["gene_count_meta"].notna()
        if gene_comp_mask.any():
            gene_mismatches = int((merged.loc[gene_comp_mask, "gene_count_h5"] !=
                                   merged.loc[gene_comp_mask, "gene_count_meta"]).sum())
            rec["gene_count_mismatches"] = gene_mismatches
            rec["gene_count_match_rate"] = float(
                (gene_comp_mask.sum() - gene_mismatches) / gene_comp_mask.sum()
            )
        else:
            rec["gene_count_mismatches"] = pd.NA
            rec["gene_count_match_rate"] = pd.NA

        # --- umi_count 对比 ---
        umi_comp_mask = merged["umi_count_h5"].notna() & merged["umi_count_meta"].notna()
        if umi_comp_mask.any():
            umi_mismatches = int((merged.loc[umi_comp_mask, "umi_count_h5"] !=
                                  merged.loc[umi_comp_mask, "umi_count_meta"]).sum())
            rec["umi_count_mismatches"] = umi_mismatches
            rec["umi_count_match_rate"] = float(
                (umi_comp_mask.sum() - umi_mismatches) / umi_comp_mask.sum()
            )
        else:
            rec["umi_count_mismatches"] = pd.NA
            rec["umi_count_match_rate"] = pd.NA

        # ✅ 别再让 pd.NA 参与布尔计算：用 pd.isna(...) 显式判断
        gm = rec["gene_count_mismatches"]
        um = rec["umi_count_mismatches"]
        gene_ok = (pd.isna(gm) or gm == 0)
        umi_ok  = (pd.isna(um) or um == 0)

        per_cell_ok = bool(rec["cell_barcode_set_equal"]) and gene_ok and umi_ok

        # overall：确保这里左右两侧都是布尔值（前面 obs_count_match 已保证为 bool）
        rec["per_cell_checks_pass"] = per_cell_ok
        rec["overall_pass"] = bool(rec["obs_count_match"] and per_cell_ok)

    except Exception as e:
        rec["error"] = f"校验失败: {e}"
    finally:
        # 释放 backed 模式文件句柄（某些 anndata 版本中可访问 .file）
        try:
            if getattr(adata, "file", None) is not None:
                adata.file.close()
        except Exception:
            pass

    return rec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", default="Oryza_sativa", help="包含 h5ad 与 parquet 的目录")
    args = parser.parse_args()

    base = Path(args.base_dir)
    sample_path = base / "sample_metadata.parquet"
    obs_meta_path = base / "obs_metadata.parquet"

    if not sample_path.exists():
        raise FileNotFoundError(f"未找到: {sample_path}")
    if not obs_meta_path.exists():
        raise FileNotFoundError(f"未找到: {obs_meta_path}")

    sample_obs_count_map = load_sample_metadata(sample_path)
    obs_meta_all = load_obs_metadata(obs_meta_path)

    rows = []
    for h5 in sorted(base.glob("*.h5ad")):
        rows.append(validate_one_h5ad(h5, sample_obs_count_map, obs_meta_all))

    if not rows:
        print(f"目录中未发现 .h5ad 文件: {base}")
        return

    out_df = pd.DataFrame(rows)
    out_csv = base / "h5ad_validation_report.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"校验完成，报告已输出：{out_csv}")


if __name__ == "__main__":
    main()
