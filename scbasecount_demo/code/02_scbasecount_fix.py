#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gc
import argparse
import os
import shutil
import time
from os.path import isdir, join

import anndata as ad
import numcodecs
import numpy as np
import pandas as pd
import zarr
from tqdm import tqdm

# 如只想保留 obs 的少数列，按需修改
COLUMNS_OF_INTEREST = [
    "dataset_id",
    "species",
    "tissue",
    "assay",
    "sex",
    "development_stage",
    "disease",
    "tech_sample",
    "soma_joinid",
    "cell_type",
]

DEFAULT_CHUNK_SIZE = 16384
DEFAULT_X_READ_CHUNK_SIZE = 2048
DEFAULT_BASE_PATH = "/data/disk1/SpeciesLLM_obs/scbasecount_demo/scbasecount_processed"
DEFAULT_SAVE_ROOT = "/data/disk1/SpeciesLLM_obs/scbasecount_demo/scbasecount_processed_02"
HUMAN_SAVE_ROOT = "/data/disk2/SpeciesLLM_obs/scbasecount_demo/scbasecount_processed_02"
MOUSE_SAVE_ROOT = "/data/disk3/SpeciesLLM_obs/scbasecount_demo/scbasecount_processed_02"

DEFAULT_ZARR_COMPRESSOR = numcodecs.Blosc(
    cname="lz4",
    clevel=5,
    shuffle=numcodecs.blosc.SHUFFLE,
    blocksize=0,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split per-species h5ad into obs.parquet, var.parquet and dense X.zarr."
    )
    parser.add_argument(
        "--species",
        nargs="+",
        default=None,
        help="Only process the listed species names.",
    )
    parser.add_argument(
        "--exclude-species",
        nargs="+",
        default=None,
        help="Process all species except the listed species names.",
    )
    return parser.parse_args()


def close_backed_adata(adata):
    try:
        adata.file.close()
    except Exception:
        pass


# ========== 基础读取函数 ==========
def read_obs_backed(path):
    adata = ad.read_h5ad(path, backed="r")
    try:
        return adata.obs.copy()
    finally:
        close_backed_adata(adata)


def read_var_backed(path):
    adata = ad.read_h5ad(path, backed="r")
    try:
        return adata.var.copy()
    finally:
        close_backed_adata(adata)


def read_var_index_backed(path):
    adata = ad.read_h5ad(path, backed="r")
    try:
        return adata.var_names.copy()
    finally:
        close_backed_adata(adata)


def get_n_obs(path):
    adata = ad.read_h5ad(path, backed="r")
    try:
        return int(adata.n_obs)
    finally:
        close_backed_adata(adata)


def iter_dense_X_blocks(path, read_chunk_size):
    adata = ad.read_h5ad(path, backed="r")
    try:
        n_obs = int(adata.n_obs)
        for start in range(0, n_obs, read_chunk_size):
            end = min(start + read_chunk_size, n_obs)
            block = adata.X[start:end]
            if hasattr(block, "toarray"):
                block = block.toarray()
            else:
                block = np.asarray(block)
            if block.dtype != np.float32:
                block = block.astype(np.float32, copy=False)
            yield np.ascontiguousarray(block)
    finally:
        close_backed_adata(adata)


# ========== 目录扫描与一致性检查 ==========
def build_species_file_map(base_path: str):
    """
    假设 base_path 下每个子目录名即物种名，目录里全是该物种的 .h5ad 文件。
    返回：
      species_to_files: {species_name: [file1, file2, ...]}
      species_order:    物种名按字典序
    """
    species_to_files = {}
    for name in sorted(os.listdir(base_path)):
        sp_dir = join(base_path, name)
        if not isdir(sp_dir):
            continue
        files = sorted(
            [join(sp_dir, f) for f in os.listdir(sp_dir) if f.endswith(".h5ad")]
        )
        if files:
            species_to_files[name] = files
    species_order = sorted(species_to_files.keys())
    return species_to_files, species_order


def ensure_same_var_for_species(files_for_species):
    """
    用第一个文件 var.index 作为标准，确保同物种下所有文件 var 完全一致。
    """
    v0 = read_var_backed(files_for_species[0])
    v0_idx = v0.index.copy()
    for f in tqdm(files_for_species[1:], desc="Checking var/index", leave=False):
        v_idx = read_var_index_backed(f)
        if not v_idx.equals(v0_idx):
            raise ValueError(
                f"var/index 不一致：\n - 标准: {files_for_species[0]}\n - 不一致: {f}\n"
                "请确保同一物种目录下所有 h5ad 的 var 完全一致（同序同长）。"
            )
    return v0, v0_idx


def resolve_save_root(species_name, save_root):
    if species_name == "Homo_sapiens":
        return HUMAN_SAVE_ROOT
    if species_name == "Mus_musculus":
        return MOUSE_SAVE_ROOT
    return save_root


def filter_species_order(species_order, include_species=None, exclude_species=None):
    include_set = set(include_species or [])
    exclude_set = set(exclude_species or [])

    if include_set:
        missing = sorted(include_set - set(species_order))
        if missing:
            raise ValueError(f"--species 中有不存在的物种目录: {missing}")
        filtered = [sp for sp in species_order if sp in include_set]
    else:
        filtered = list(species_order)

    if exclude_set:
        filtered = [sp for sp in filtered if sp not in exclude_set]

    if not filtered:
        raise ValueError("筛选后没有需要处理的物种。")

    return filtered


def write_X_zarr_streaming(
    species_name,
    files_for_species,
    zarr_store_path,
    total_rows,
    n_cols,
    chunk_size,
    read_chunk_size,
):
    print(f"[{species_name}] Writing X.zarr in streaming mode...")
    shutil.rmtree(zarr_store_path, ignore_errors=True)

    zarr_group = zarr.open_group(zarr_store_path, mode="w")
    zarr_array = zarr_group.create_dataset(
        "X",
        shape=(total_rows, n_cols),
        chunks=(min(chunk_size, total_rows), n_cols),
        dtype="float32",
        compressor=DEFAULT_ZARR_COMPRESSOR,
        order="C",
        fill_value=0.0,
    )

    offset = 0
    for f in tqdm(files_for_species, desc=f"[{species_name}] Writing X"):
        file_rows = 0
        for block in iter_dense_X_blocks(f, read_chunk_size):
            block_rows = int(block.shape[0])
            zarr_array[offset : offset + block_rows, :] = block
            offset += block_rows
            file_rows += block_rows
        expected_rows = get_n_obs(f)
        if file_rows != expected_rows:
            raise ValueError(
                f"[{species_name}] X 行数写入不一致: {f}, expected={expected_rows}, got={file_rows}"
            )

    if offset != total_rows:
        raise ValueError(
            f"[{species_name}] X 总行数写入不一致: expected={total_rows}, got={offset}"
        )

    print(f"[{species_name}] Final X shape: {(total_rows, n_cols)}")
    print(f"[{species_name}] X saved at {zarr_store_path}")


# ========== 物种级处理 ==========
def process_one_species(
    species_name,
    files_for_species,
    save_root,
    chunk_size=DEFAULT_CHUNK_SIZE,
    read_chunk_size=DEFAULT_X_READ_CHUNK_SIZE,
):
    save_path = join(resolve_save_root(species_name, save_root), species_name)
    os.makedirs(save_path, exist_ok=True)
    print(f"\n=== Processing species: {species_name}  ->  {save_path} ===")

    # —— var：取首文件并强校验一致性
    var_df, var_index = ensure_same_var_for_species(files_for_species)
    var_path = join(save_path, "var.parquet")
    var_df.to_parquet(var_path, compression="snappy")
    print(f"[{species_name}] var saved at {var_path}")

    # —— obs：保持原逻辑，确保最终内容与列选择规则不变
    obs_parts = []
    for f in tqdm(files_for_species, desc=f"[{species_name}] Reading obs"):
        sub = read_obs_backed(f)
        for col in sub.columns:
            if sub[col].dtype == object:
                sub[col] = sub[col].astype("category")
        obs_parts.append(sub)

    if not obs_parts:
        print(f"[{species_name}] No obs rows found, skip.")
        return

    obs = pd.concat(obs_parts, ignore_index=True)
    keep = [c for c in COLUMNS_OF_INTEREST if c in obs.columns]
    if keep:
        obs = obs[keep]

    for col in obs.columns:
        if obs[col].dtype == object:
            obs[col] = obs[col].astype(str)

    obs_path = join(save_path, "obs.parquet")
    obs.to_parquet(obs_path, compression="snappy")
    print(f"[{species_name}] obs saved at {obs_path}")

    total_rows = int(obs.shape[0])
    n_cols = len(var_index)

    # 在开始写 X 前尽量释放 obs/var 的内存，避免与稠密块峰值叠加。
    del obs_parts
    del obs
    del var_df
    gc.collect()

    zarr_store_path = join(save_path, "X.zarr")
    write_X_zarr_streaming(
        species_name=species_name,
        files_for_species=files_for_species,
        zarr_store_path=zarr_store_path,
        total_rows=total_rows,
        n_cols=n_cols,
        chunk_size=chunk_size,
        read_chunk_size=read_chunk_size,
    )


# ========== 主程序 ==========
def main():
    args = parse_args()
    base_path = os.getenv("SCBASECOUNT_BASE_PATH", DEFAULT_BASE_PATH)
    save_root = os.getenv("SCBASECOUNT_SAVE_ROOT", DEFAULT_SAVE_ROOT)
    chunk_size = int(os.getenv("SCBASECOUNT_CHUNK_SIZE", DEFAULT_CHUNK_SIZE))
    read_chunk_size = int(
        os.getenv("SCBASECOUNT_X_READ_CHUNK_SIZE", DEFAULT_X_READ_CHUNK_SIZE)
    )

    overall_start = time.time()

    species_to_files, species_order = build_species_file_map(base_path)
    if not species_order:
        print(f"No species directories with .h5ad under {base_path}. Exit.")
        return

    species_order = filter_species_order(
        species_order,
        include_species=args.species,
        exclude_species=args.exclude_species,
    )

    print(f"Discovered species: {species_order}")
    print(
        "Output roots: "
        f"default={save_root}, human={HUMAN_SAVE_ROOT}, mouse={MOUSE_SAVE_ROOT}"
    )
    print(
        f"X write mode: final_zarr_chunk_rows={chunk_size}, "
        f"backed_read_chunk_rows={read_chunk_size}"
    )

    for sp in species_order:
        process_one_species(
            species_name=sp,
            files_for_species=species_to_files[sp],
            save_root=save_root,
            chunk_size=chunk_size,
            read_chunk_size=read_chunk_size,
        )

    overall_end = time.time()
    print(f"All species done. Total time: {overall_end - overall_start:.2f}s.")


if __name__ == "__main__":
    main()
