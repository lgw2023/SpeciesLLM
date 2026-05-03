#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from os.path import join, isdir
import anndata as ad
import dask
import dask.array as da
import pandas as pd
import numpy as np
import shutil
import time
from tqdm import tqdm
from dask.distributed import Client, LocalCluster

# 如只想保留 obs 的少数列，按需修改
COLUMNS_OF_INTEREST = ["dataset_id","species","tissue","assay","sex",
                       "development_stage","disease","tech_sample",
                       "soma_joinid","cell_type"]

# ========== 基础读取函数 ==========
def read_obs_backed(path):
    return ad.read_h5ad(path, backed="r").obs

def read_var_backed(path):
    return ad.read_h5ad(path, backed="r").var

def read_X_full(path):
    return ad.read_h5ad(path).X  # 可能很大：谨慎内存

def get_n_obs(path):
    return ad.read_h5ad(path, backed="r").n_obs

# ========== 目录扫描与一致性检查 ==========
def build_species_file_map(base_path: str):
    """
    假设 base_path 下每个子目录名即物种名，目录里全是该物种的 .h5ad 文件。
    返回：
      species_to_files: {species_name: [file1, file2, ...]}
      species_order:    物种名按字典序或出现顺序（这里用字典序）
    """
    species_to_files = {}
    for name in sorted(os.listdir(base_path)):
        sp_dir = join(base_path, name)
        if not isdir(sp_dir):
            continue
        files = sorted([join(sp_dir, f) for f in os.listdir(sp_dir) if f.endswith(".h5ad")])
        if files:
            species_to_files[name] = files
    species_order = sorted(species_to_files.keys())
    return species_to_files, species_order

def ensure_same_var_for_species(files_for_species):
    """
    用第一个文件 var.index 作为标准，确保同物种下所有文件 var 完全一致。
    """
    v0 = read_var_backed(files_for_species[0])
    v0_idx = v0.index
    for f in files_for_species[1:]:
        v = read_var_backed(f).index
        if not v.equals(v0_idx):
            raise ValueError(
                f"var/index 不一致：\n - 标准: {files_for_species[0]}\n - 不一致: {f}\n"
                "请确保同一物种目录下所有 h5ad 的 var 完全一致（同序同长）。"
            )
    return v0, v0_idx

# ========== 物种级处理 ==========
def process_one_species(species_name, files_for_species, save_path, chunk_size=16384):
    os.makedirs(save_path, exist_ok=True)
    print(f"\n=== Processing species: {species_name}  ->  {save_path} ===")

    # —— var：取首文件并强校验一致性
    var_df, var_index = ensure_same_var_for_species(files_for_species)
    var_path = join(save_path, "var.parquet")
    var_df.to_parquet(var_path, compression="snappy")
    print(f"[{species_name}] var saved at {var_path}")

    # —— obs：全量拼接（同物种目录，无需再筛选）
    obs_parts = []
    for f in tqdm(files_for_species, desc=f"[{species_name}] Reading obs"):
        sub = read_obs_backed(f).copy()
        # 可选：压缩 object 列为 category
        for col in sub.columns:
            if sub[col].dtype == object:
                sub[col] = sub[col].astype("category")
        obs_parts.append(sub)

    if not obs_parts:
        print(f"[{species_name}] No obs rows found, skip.")
        return

    obs = pd.concat(obs_parts, ignore_index=True)

    # 如只想保存少数列，启用下列两行
    keep = [c for c in COLUMNS_OF_INTEREST if c in obs.columns]
    if keep:  # 若有匹配列再筛，否则保留全部
        obs = obs[keep]

    # parquet 前把 object 转为 str（与原脚本习惯一致）
    for col in obs.columns:
        if obs[col].dtype == object:
            obs[col] = obs[col].astype(str)

    obs_path = join(save_path, "obs.parquet")
    obs.to_parquet(obs_path, compression="snappy")
    print(f"[{species_name}] obs saved at {obs_path}")

    # —— X：全量纵向拼接
    print(f"[{species_name}] Building delayed X blocks...")
    X_blocks = []
    n_cols = len(var_index)

    def _read_X_whole(path):
        X = read_X_full(path)
        if hasattr(X, "tocsr"):  # 稀疏
            X = X.tocsr()[:]     # 整块
        else:                    # 稠密
            X = X[:]
        if X.dtype != np.float32:
            X = X.astype(np.float32)
        return X

    # 逐文件延迟加载，拼接时需要每个文件的 n_obs
    for f in tqdm(files_for_species, desc=f"[{species_name}] Scheduling X"):
        n_rows = int(get_n_obs(f))
        X_delayed = da.from_delayed(
            dask.delayed(_read_X_whole)(f),
            shape=(n_rows, n_cols),
            dtype="float32"
        )
        X_blocks.append(X_delayed)

    X = da.concatenate(X_blocks, axis=0).compute_chunk_sizes()
    X = X.rechunk((chunk_size, -1)).persist()
    print(f"[{species_name}] Final X shape: {X.shape}")

    # —— 写 Zarr（注意：会稠密化）
    zarr_store_path = join(save_path, "X.zarr")
    shutil.rmtree(zarr_store_path, ignore_errors=True)
    da.to_zarr(
        X.map_blocks(lambda xx: xx.toarray() if hasattr(xx, "toarray") else xx, dtype="float32"),
        zarr_store_path,
        component="X",
        compute=True,
        compressor="default",
        order="C"
    )
    print(f"[{species_name}] X saved at {zarr_store_path}")

# ========== 主程序 ==========
def main():
    # Dask 本地集群（兼容 SLURM）
    n_workers_per_node = int(os.getenv("SLURM_NTASKS_PER_NODE", 1))
    cpus_per_task = int(os.getenv("SLURM_CPUS_PER_TASK", 8))
    cluster = LocalCluster(n_workers=n_workers_per_node, threads_per_worker=cpus_per_task)
    client = Client(cluster)
    print(client)

    # ===== 修改成你的路径 =====
    # # 源代码路径
    # BASE_PATH = "/ibex/project/c2307/arc-virtual-cell-atlas/scBaseCount/scbasecount_processed"  # 物种/ 目录
    # SAVE_ROOT = "/ibex/project/c2307/arc-virtual-cell-atlas/scBaseCount/scbasecount_processed_02"
    # 当前机器中的路径，直接替换
    BASE_PATH = "/data/disk1/SpeciesLLM_obs/scbasecount_demo/scbasecount_processed"  # 物种/ 目录
    SAVE_ROOT = "/data/disk1/SpeciesLLM_obs/scbasecount_demo/scbasecount_processed_02"

    # =========================

    overall_start = time.time()

    species_to_files, species_order = build_species_file_map(BASE_PATH)
    if not species_order:
        print(f"No species directories with .h5ad under {BASE_PATH}. Exit.")
        return

    print(f"Discovered species: {species_order}")
    for sp in species_order:
        species_save_path = join(SAVE_ROOT, sp)
        process_one_species(sp, species_to_files[sp], species_save_path)

    client.close()
    overall_end = time.time()
    print(f"All species done. Total time: {overall_end - overall_start:.2f}s.")

if __name__ == "__main__":
    main()
