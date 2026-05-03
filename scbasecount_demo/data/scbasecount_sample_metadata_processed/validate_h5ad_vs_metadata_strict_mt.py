#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
多线程版：更严格的 h5ad 与 metadata 校验脚本

与单线程严格版一致的校验逻辑（见下），但对每个 .h5ad 文件并发处理，加速大批量样本的校验。
默认使用 ThreadPoolExecutor（读 h5ad + 解压通常是 I/O 为主；每个线程各自打开文件，互不共享句柄）。

检查内容：
1) 单文件级别：
   - h5ad.n_obs == sample_metadata.obs_count
   - obs_metadata 中该 SRX 的行数 == sample_metadata.obs_count
2) 集合级别：
   - h5ad 中的条形码集合与 obs_metadata 对应 SRX 的条形码集合完全一致
   - 统计两侧仅存在的条形码数量
   - 检测重复条形码
3) 单细胞级别：
   - gene_count、umi_count 是否全为整数（允许极小浮点误差；不会四舍五入）
   - h5ad 与 obs_metadata 按条形码逐一比对，两列分别统计不一致数量与匹配率
4) 其他一致性：
   - 若 h5ad.obs 存在 SRX_accession 列，检查其是否全部等于文件名中的 SRX

输出：
    {base_dir}/h5ad_validation_report.csv

用法：
    python validate_h5ad_vs_metadata_strict_mt.py --base_dir Oryza_sativa \
        [--barcode_strip_suffix "-1"] \
        [--workers 8]
"""

import argparse
from pathlib import Path
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Tuple
import pandas as pd
import numpy as np
import anndata as ad

# ------------------------ 公用工具函数 ------------------------

def to_int_if_close(x: pd.Series, *, tol: float = 1e-6) -> Tuple[pd.Series, int]:
    """
    将数值列转换为 Int64（严格流：不会四舍五入）。
    - 非数字 -> NA
    - 浮点值若与最近整数的差的绝对值 <= tol，视为该整数
    - 其余保留为 NA，并计数返回（non_integer_n）
    返回 (Int64系列, non_integer_n)
    """
    x_num = pd.to_numeric(x, errors="coerce")
    nearest = np.rint(x_num)
    diff = (x_num - nearest).abs()
    ok = diff.le(tol) | x_num.isna()
    non_integer_n = int((~ok & x_num.notna()).sum())
    out = x_num.where(ok, pd.NA).round().astype("Int64")
    return out, non_integer_n

def normalize_barcodes(s: pd.Series, strip_suffix: str = "-1") -> pd.Series:
    """
    统一条形码格式：去空白、转为字符串、可选剥离后缀（如 10x 的“-1”）。
    """
    s2 = s.astype(str).str.strip()
    if strip_suffix:
        s2 = s2.str.replace(f"{strip_suffix}$", "", regex=True)
    return s2

def load_sample_metadata(sample_path: Path) -> pd.DataFrame:
    """
    读取 sample_metadata.parquet，返回 DataFrame，索引为 srx_accession。
    需要列：srx_accession/SRX_accession, obs_count
    """
    df = pd.read_parquet(sample_path)
    if "srx_accession" not in df.columns:
        if "SRX_accession" in df.columns:
            df = df.rename(columns={"SRX_accession": "srx_accession"})
        else:
            raise KeyError("sample_metadata.parquet 缺少列：srx_accession / SRX_accession")
    if "obs_count" not in df.columns:
        raise KeyError("sample_metadata.parquet 缺少列：obs_count")
    df["srx_accession"] = df["srx_accession"].astype(str)
    df = df.drop_duplicates(subset=["srx_accession"]).set_index("srx_accession")
    df["obs_count"] = pd.to_numeric(df["obs_count"], errors="coerce").astype("Int64")
    return df

def load_obs_metadata(obs_meta_path: Path, *, barcode_strip_suffix: str) -> pd.DataFrame:
    """
    读取 obs_metadata.parquet，返回包含：
      srx_accession, cell_barcode, gene_count, umi_count
    """
    obs = pd.read_parquet(obs_meta_path)
    if "SRX_accession" in obs.columns and "srx_accession" not in obs.columns:
        obs = obs.rename(columns={"SRX_accession": "srx_accession"})
    required = {"srx_accession", "cell_barcode"}
    missing = required - set(obs.columns)
    if missing:
        raise KeyError(f"obs_metadata.parquet 缺少列：{missing}")
    obs["srx_accession"] = obs["srx_accession"].astype(str)
    obs["cell_barcode"] = normalize_barcodes(obs["cell_barcode"], strip_suffix=barcode_strip_suffix)
    if "gene_count" not in obs.columns:
        obs["gene_count"] = pd.Series([pd.NA]*len(obs), dtype="Int64")
    if "umi_count" not in obs.columns:
        obs["umi_count"] = pd.Series([pd.NA]*len(obs), dtype="Int64")
    return obs[["srx_accession", "cell_barcode", "gene_count", "umi_count"]]

# ------------------------ 单文件校验逻辑 ------------------------

def validate_one_h5ad(h5_path: Path,
                      sample_df: pd.DataFrame,
                      obs_meta_by_srx: Dict[str, pd.DataFrame],
                      *,
                      barcode_strip_suffix: str) -> dict:
    """
    校验单个 h5ad，返回结果字典（单行）。
    注意：函数内不修改传入的 DataFrame/字典，线程安全。
    """
    srx = h5_path.stem
    rec = {
        "file_name": h5_path.name,
        "srx_accession": srx,
        "error": "",

        # 基本规模
        "h5ad_n_obs": pd.NA,
        "metadata_obs_count": pd.NA,
        "obs_count_match": False,

        # obs_metadata 规模
        "obsmeta_n_cells": pd.NA,
        "obsmeta_rows_match_sample_obs_count": pd.NA,

        # 集合级别
        "cells_overlap": pd.NA,
        "cells_only_in_h5ad": pd.NA,
        "cells_only_in_obsmeta": pd.NA,
        "cell_barcode_set_equal": False,
        "dup_barcodes_in_h5ad": pd.NA,
        "dup_barcodes_in_obsmeta": pd.NA,

        # 列存在性
        "has_gene_count_col_in_h5ad": False,
        "has_umi_count_col_in_h5ad": False,
        "has_SRX_accession_col_in_h5ad": False,

        # 单细胞比对
        "non_integer_gene_in_h5ad": pd.NA,
        "non_integer_gene_in_obsmeta": pd.NA,
        "gene_count_mismatches": pd.NA,
        "gene_count_match_rate": pd.NA,

        "non_integer_umi_in_h5ad": pd.NA,
        "non_integer_umi_in_obsmeta": pd.NA,
        "umi_count_mismatches": pd.NA,
        "umi_count_match_rate": pd.NA,

        # 其他一致性
        "srx_accession_all_match_in_h5ad": pd.NA,

        # 汇总
        "per_cell_checks_pass": False,
        "overall_pass": False,
    }

    # 取 obs_metadata 子表（可为空表）
    meta = obs_meta_by_srx.get(srx, None)
    if meta is None:
        meta = pd.DataFrame(columns=["cell_barcode", "gene_count", "umi_count"])

    try:
        adata = ad.read_h5ad(str(h5_path), backed="r")
    except Exception as e:
        rec["error"] = f"读取 h5ad 失败: {e}"
        return rec

    try:
        # ---------- 文件级别 ----------
        n_obs = int(adata.n_obs)
        rec["h5ad_n_obs"] = n_obs
        expected_obs = sample_df.get("obs_count").get(srx, pd.NA) if srx in sample_df.index else pd.NA
        rec["metadata_obs_count"] = expected_obs
        rec["obs_count_match"] = (not pd.isna(expected_obs)) and (n_obs == int(expected_obs))

        # ---------- 取出 h5ad.obs ----------
        obs = adata.obs.copy()
        obs.index = normalize_barcodes(obs.index.astype(str), strip_suffix=barcode_strip_suffix)
        obs = obs.rename_axis("cell_barcode").reset_index()

        # SRX_accession 列校验（若存在）
        rec["has_SRX_accession_col_in_h5ad"] = ("SRX_accession" in obs.columns) or ("srx_accession" in obs.columns)
        if "SRX_accession" in obs.columns:
            rec["srx_accession_all_match_in_h5ad"] = bool(obs["SRX_accession"].astype(str).eq(srx).all())
        elif "srx_accession" in obs.columns:
            rec["srx_accession_all_match_in_h5ad"] = bool(obs["srx_accession"].astype(str).eq(srx).all())

        # 基因/UMI 列存在性
        rec["has_gene_count_col_in_h5ad"] = "gene_count" in obs.columns
        rec["has_umi_count_col_in_h5ad"]  = "umi_count"  in obs.columns

        # 将 h5ad 的 gene/umi 转为严格整数（不四舍五入；仅容忍 1e-6 漂移）
        if rec["has_gene_count_col_in_h5ad"]:
            obs["gene_count_h5"], nonint_gene_h5 = to_int_if_close(obs["gene_count"])
        else:
            obs["gene_count_h5"], nonint_gene_h5 = pd.Series([pd.NA]*len(obs), dtype="Int64"), pd.NA
        if rec["has_umi_count_col_in_h5ad"]:
            obs["umi_count_h5"], nonint_umi_h5 = to_int_if_close(obs["umi_count"])
        else:
            obs["umi_count_h5"], nonint_umi_h5 = pd.Series([pd.NA]*len(obs), dtype="Int64"), pd.NA
        rec["non_integer_gene_in_h5ad"] = nonint_gene_h5
        rec["non_integer_umi_in_h5ad"]  = nonint_umi_h5

        # ---------- obs_metadata 子集 ----------
        # 这里的 meta 已经在主进程按 SRX 分好；仅做本地副本防止副作用
        meta_local = meta.copy()
        rec["obsmeta_n_cells"] = int(len(meta_local))
        if not pd.isna(expected_obs):
            rec["obsmeta_rows_match_sample_obs_count"] = (int(len(meta_local)) == int(expected_obs))
        else:
            rec["obsmeta_rows_match_sample_obs_count"] = False

        # ---------- 重复条形码检测 ----------
        rec["dup_barcodes_in_h5ad"] = int((obs["cell_barcode"].value_counts() > 1).sum())
        rec["dup_barcodes_in_obsmeta"] = int((meta_local["cell_barcode"].value_counts() > 1).sum())

        # ---------- 集合比较 ----------
        h5_bcs = set(obs["cell_barcode"])
        meta_bcs = set(meta_local["cell_barcode"])
        inter = h5_bcs & meta_bcs
        rec["cells_overlap"] = len(inter)
        rec["cells_only_in_h5ad"] = len(h5_bcs - meta_bcs)
        rec["cells_only_in_obsmeta"] = len(meta_bcs - h5_bcs)
        rec["cell_barcode_set_equal"] = (h5_bcs == meta_bcs)

        # ---------- 合并逐细胞比较（仅交集参与） ----------
        if len(inter) > 0:
            obs_sub = obs.loc[obs["cell_barcode"].isin(inter), ["cell_barcode", "gene_count_h5", "umi_count_h5"]]
            meta_sub = meta_local.loc[meta_local["cell_barcode"].isin(inter), ["cell_barcode", "gene_count", "umi_count"]]

            merged = obs_sub.merge(meta_sub, on="cell_barcode", how="inner", validate="one_to_one")
            # 将 obs_metadata 的 gene/umi 严格转整数（不四舍五入）
            merged["gene_count_meta"], nonint_gene_meta = to_int_if_close(merged["gene_count"])
            merged["umi_count_meta"],  nonint_umi_meta  = to_int_if_close(merged["umi_count"])
            rec["non_integer_gene_in_obsmeta"] = nonint_gene_meta
            rec["non_integer_umi_in_obsmeta"]  = nonint_umi_meta

            # 仅在两侧都有整数时参与比较
            gene_mask = merged["gene_count_h5"].notna() & merged["gene_count_meta"].notna()
            if gene_mask.any():
                gene_mis = int((merged.loc[gene_mask, "gene_count_h5"] != merged.loc[gene_mask, "gene_count_meta"]).sum())
                rec["gene_count_mismatches"] = gene_mis
                rec["gene_count_match_rate"] = float((gene_mask.sum() - gene_mis) / gene_mask.sum())
            else:
                rec["gene_count_mismatches"] = pd.NA
                rec["gene_count_match_rate"] = pd.NA

            umi_mask = merged["umi_count_h5"].notna() & merged["umi_count_meta"].notna()
            if umi_mask.any():
                umi_mis = int((merged.loc[umi_mask, "umi_count_h5"] != merged.loc[umi_mask, "umi_count_meta"]).sum())
                rec["umi_count_mismatches"] = umi_mis
                rec["umi_count_match_rate"] = float((umi_mask.sum() - umi_mis) / umi_mask.sum())
            else:
                rec["umi_count_mismatches"] = pd.NA
                rec["umi_count_match_rate"] = pd.NA

        # ---------- 汇总布尔 ----------
        no_dup = (rec["dup_barcodes_in_h5ad"] == 0) and (rec["dup_barcodes_in_obsmeta"] == 0)
        gene_ok = (pd.isna(rec["gene_count_mismatches"]) or rec["gene_count_mismatches"] == 0)
        umi_ok  = (pd.isna(rec["umi_count_mismatches"])  or rec["umi_count_mismatches"]  == 0)
        nonint_ok = ( (pd.isna(rec["non_integer_gene_in_h5ad"]) or rec["non_integer_gene_in_h5ad"] == 0) and
                      (pd.isna(rec["non_integer_gene_in_obsmeta"]) or rec["non_integer_gene_in_obsmeta"] == 0) and
                      (pd.isna(rec["non_integer_umi_in_h5ad"]) or rec["non_integer_umi_in_h5ad"] == 0) and
                      (pd.isna(rec["non_integer_umi_in_obsmeta"]) or rec["non_integer_umi_in_obsmeta"] == 0) )

        per_cell_ok = bool(rec["cell_barcode_set_equal"] and gene_ok and umi_ok and no_dup and nonint_ok)
        rec["per_cell_checks_pass"] = per_cell_ok

        overall_ok = bool(rec["obs_count_match"] and
                          bool(rec["obsmeta_rows_match_sample_obs_count"]) and
                          per_cell_ok)
        rec["overall_pass"] = overall_ok

    except Exception as e:
        rec["error"] = f"校验失败: {e}"
    finally:
        try:
            if getattr(adata, "file", None) is not None:
                adata.file.close()
        except Exception:
            pass

    return rec

# ------------------------ 主流程（多线程） ------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", default="Oryza_sativa", help="包含 h5ad 与 parquet 的目录")
    ap.add_argument("--barcode_strip_suffix", default="-1",
                    help="条形码标准化时剥离的后缀；设为空串可禁用")
    ap.add_argument("--workers", type=int, default=8, help="并发线程数（建议 4~32；过大不一定更快）")
    args = ap.parse_args()

    base = Path(args.base_dir)
    sample_path = base / "sample_metadata.parquet"
    obs_meta_path = base / "obs_metadata.parquet"
    out_csv = base / "h5ad_validation_report.csv"

    if not sample_path.exists():
        raise FileNotFoundError(f"未找到: {sample_path}")
    if not obs_meta_path.exists():
        raise FileNotFoundError(f"未找到: {obs_meta_path}")

    # 读取元数据
    sample_df = load_sample_metadata(sample_path)
    obs_meta_all = load_obs_metadata(obs_meta_path, barcode_strip_suffix=args.barcode_strip_suffix)

    # 预分片：按 SRX 切分 obs_metadata，避免每个线程都对整表筛选
    obs_meta_by_srx: Dict[str, pd.DataFrame] = {
        srx: sub[["cell_barcode", "gene_count", "umi_count"]].copy()
        for srx, sub in obs_meta_all.groupby("srx_accession", sort=False)
    }

    # 收集 h5ad 文件
    h5_files = sorted(base.glob("*.h5ad"))
    if not h5_files:
        print(f"目录中未发现 .h5ad 文件: {base}")
        return

    # 线程数
    max_workers = max(1, min(args.workers, len(h5_files)))

    # 并发执行
    rows = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(validate_one_h5ad, h5, sample_df, obs_meta_by_srx,
                      barcode_strip_suffix=args.barcode_strip_suffix): h5
            for h5 in h5_files
        }
        for fut in as_completed(futures):
            h5 = futures[fut]
            try:
                rec = fut.result()
            except Exception as e:
                rec = {
                    "file_name": h5.name,
                    "srx_accession": h5.stem,
                    "error": f"线程异常: {e}",
                    "overall_pass": False
                }
            rows.append(rec)

    # 汇总输出
    out_df = pd.DataFrame(rows)
    # 统一列顺序
    preferred_order = [
        "file_name", "srx_accession", "error",
        "overall_pass", "per_cell_checks_pass",
        "obs_count_match", "obsmeta_rows_match_sample_obs_count",
        "cell_barcode_set_equal",
        "gene_count_mismatches", "umi_count_mismatches",
        "gene_count_match_rate", "umi_count_match_rate",
        "dup_barcodes_in_h5ad", "dup_barcodes_in_obsmeta",
        "non_integer_gene_in_h5ad", "non_integer_gene_in_obsmeta",
        "non_integer_umi_in_h5ad", "non_integer_umi_in_obsmeta",
        "h5ad_n_obs", "metadata_obs_count", "obsmeta_n_cells",
        "cells_overlap", "cells_only_in_h5ad", "cells_only_in_obsmeta",
        "has_gene_count_col_in_h5ad", "has_umi_count_col_in_h5ad",
        "has_SRX_accession_col_in_h5ad", "srx_accession_all_match_in_h5ad",
    ]
    cols = [c for c in preferred_order if c in out_df.columns] + \
           [c for c in out_df.columns if c not in preferred_order]
    out_df = out_df[cols].sort_values("file_name")
    out_df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"校验完成，报告已输出：{out_csv}")
    # 打印简短概览
    total = len(out_df)
    bad = int((~out_df["overall_pass"].fillna(False)).sum())
    print(f"样本数: {total}, overall_pass=False 的样本: {bad}")

if __name__ == "__main__":
    main()
