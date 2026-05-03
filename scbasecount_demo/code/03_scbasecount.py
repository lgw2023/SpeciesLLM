#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from os.path import join, isdir

import dask
import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from dask.distributed import Client, LocalCluster
from scipy.sparse import issparse
from sklearn.utils import sparsefuncs


# -----------------------------
# Default config (可经命令行覆盖，如 --input-root、-o / --output-base)
# -----------------------------
# # 源代码路径
# DEFAULT_INPUT_ROOT = "/ibex/project/c2307/arc-virtual-cell-atlas/scBaseCount/scbasecount_processed_02"
# DEFAULT_OUTPUT_BASE = "/ibex/project/c2307/arc-virtual-cell-atlas/scBaseCount/scbasecount_processed_03_v2"
# DEFAULT_OLD_LOOKUP_DIR = "/ibex/project/c2307/datasets/2nd_pretrain_data_preprocessed_step3/LOOKUP_categories_unified/"
# DEFAULT_NEW_LOOKUP_DIR_NAME = "LOOKUP_categories_unified"
# 当前机器中的路径，直接替换
DEFAULT_INPUT_ROOT = "/data/disk1/SpeciesLLM_obs/scbasecount_demo/scbasecount_processed_02"
DEFAULT_OUTPUT_BASE = "/data/disk1/SpeciesLLM_obs/scbasecount_demo/scbasecount_processed_03_v1"
DEFAULT_OLD_LOOKUP_DIR = "/data/disk1/SpeciesLLM_obs/scbasecount_demo/data/LOOKUP_categories_unified"
DEFAULT_NEW_LOOKUP_DIR_NAME = "LOOKUP_categories_unified"

DEFAULT_NORMALIZATION = "sf-log1p"  # or "raw"
DEFAULT_CHUNK_SIZE = 16384
DEFAULT_ROW_GROUP_SIZE = 1024

COLUMNS_WITH_LOOKUP = {
    "assay": "assay.parquet",
    "cell_type": "cell_type.parquet",
    "development_stage": "development_stage.parquet",
    "disease": "disease.parquet",
    "sex": "sex.parquet",
    "species": "species.parquet",
    "tissue": "tissue.parquet",
    #下面的这三个逐号续写
    "dataset_id": "dataset_id.parquet",
    "soma_joinid": "soma_joinid.parquet",
    "tech_sample": "tech_sample.parquet",
}
# 统一后的 species 查找表在 LOOKUP 中的唯一 label 数（与数据探查 log 一致）
EXPECTED_SPECIES_LOOKUP_LABELS = 29

LATIN_TO_COMMON = {
    "homo_sapiens": "human",
    "mus_musculus": "mouse",
    "chlorocebus_sabaeus": "green monkey",
    "rattus_norvegicus": "rat",
    "pan_troglodytes": "chimpanzee",
    "pan_paniscus": "bonobo",
    "macaca_mulatta": "macaque",
    "macaca_fascicularis": "crab-eating macaque",
    "gorilla_gorilla": "gorilla",
    "gallus_gallus": "chicken",
    "callithrix_jacchus": "marmoset",
}
COMMON_TO_LATIN = {v: k for k, v in LATIN_TO_COMMON.items()}


# -----------------------------
# Helpers: normalization
# -----------------------------
def sf_normalize(X):
    X = X.copy()
    counts = np.array(X.sum(axis=1))
    counts += counts == 0.0
    sf = 10000.0 / counts
    if issparse(X):
        sparsefuncs.inplace_row_scale(X, sf)
    else:
        np.multiply(X, sf.reshape((-1, 1)), out=X)
    return X


def sf_log1p_norm(x):
    x = sf_normalize(x)
    return np.log1p(x).astype("f4")


def preprocess_count_matrix(x, normalization):
    if normalization == "sf-log1p":
        return x.map_blocks(sf_log1p_norm, dtype="f4")
    elif normalization == "raw":
        return x
    else:
        raise ValueError('NORMALIZATION must be "sf-log1p" or "raw"')


@dask.delayed
def convert_to_dataframe(x, start, end):
    return pd.DataFrame(
        {"X": [arr.squeeze().astype("f4") for arr in np.vsplit(x, x.shape[0])]},
        index=pd.RangeIndex(start, end),
    )


# -----------------------------
# Helpers: io and normalization of labels
# -----------------------------
def list_species_dirs(input_root):
    return sorted([d for d in os.listdir(input_root) if isdir(join(input_root, d))])


def load_obs(sp_path):
    obs_path = join(sp_path, "obs.parquet")
    if not os.path.exists(obs_path):
        return None
    obs = pd.read_parquet(obs_path).reset_index(drop=True)

    # 若缺 cell_type：填 "unknown"
    if "cell_type" not in obs.columns:
        obs["cell_type"] = "unknown"
    obs["cell_type"] = obs["cell_type"].fillna("unknown")

    return obs


def to_latin_snake(x: str) -> str:
    if x is None or pd.isna(x):
        return x
    s = str(x).strip().lower().replace(" ", "_")
    return COMMON_TO_LATIN.get(s, s)


def normalize_value_for_match(col, val):
    if pd.isna(val):
        return val
    if col == "disease":
        s = str(val).strip().lower()
        if s == "health":
            s = "normal"
        return s
    if col == "species":
        return to_latin_snake(val)
    return val


def normalize_label_for_store(col, val):
    if val is None or pd.isna(val):
        return val
    s = str(val)
    if col == "disease":
        s = s.strip().lower()
        if s == "health":
            s = "normal"
        return s
    if col == "species":
        return to_latin_snake(s)
    return s


def read_old_lookup(old_lookup_dir, col):
    fn = COLUMNS_WITH_LOOKUP[col]
    path = join(old_lookup_dir, fn)
    if not os.path.exists(path):
        df = pd.DataFrame({"label": []})
        df.index.name = None
        return df.reset_index(drop=True)
    df = pd.read_parquet(path)
    if "label" not in df.columns:
        if df.shape[1] == 1:
            df = df.rename(columns={df.columns[0]: "label"})
        else:
            raise ValueError(f"{path} 没有 label 列，且无法推断")
    return df.reset_index(drop=True)


# -----------------------------
# Pass-1: collect all values, then build unified lookups
# -----------------------------
def collect_all_values(input_root):
    all_vals = {c: set() for c in COLUMNS_WITH_LOOKUP.keys()}
    species_dirs = list_species_dirs(input_root)

    for sp in species_dirs:
        sp_path = join(input_root, sp)
        obs = load_obs(sp_path)
        if obs is None:
            print(f"[SKIP] {sp}: missing obs.parquet")
            continue

        for col in COLUMNS_WITH_LOOKUP.keys():
            if col not in obs.columns:
                continue
            series = obs[col].map(lambda x: normalize_value_for_match(col, x))
            for v in series.dropna().unique().tolist():
                all_vals[col].add(v)

    return all_vals


def build_unified_lookup(old_lookup_dir, new_lookup_dir, all_values_per_col):
    os.makedirs(new_lookup_dir, exist_ok=True)
    label2id_maps = {}

    for col, values in all_values_per_col.items():
        old_df = read_old_lookup(old_lookup_dir, col)

        transformed_old_labels = []
        for lbl in old_df["label"].astype(str).tolist():
            if col == "species":
                transformed_old_labels.append(
                    normalize_label_for_store(col, COMMON_TO_LATIN.get(lbl.strip().lower(), lbl))
                )
            else:
                transformed_old_labels.append(normalize_label_for_store(col, lbl))

        label2id = {lbl: i for i, lbl in enumerate(transformed_old_labels)}
        next_id = len(label2id)

        for raw in sorted(values):
            if pd.isna(raw):
                continue
            store_label = normalize_label_for_store(col, raw)
            if store_label in label2id:
                continue
            label2id[store_label] = next_id
            next_id += 1

        new_df = pd.DataFrame({"label": [None] * len(label2id)})
        for lbl, i in label2id.items():
            new_df.iloc[i, 0] = lbl
        out_path = join(new_lookup_dir, COLUMNS_WITH_LOOKUP[col])
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        new_df.to_parquet(out_path, index=True)

        match_key_to_id = {}
        if col == "disease":
            for lbl, i in label2id.items():
                match_key_to_id[str(lbl).lower()] = i
        elif col == "species":
            for lbl, i in label2id.items():
                latin = to_latin_snake(lbl)
                match_key_to_id[latin] = i
            for common, latin in COMMON_TO_LATIN.items():
                if latin in match_key_to_id:
                    match_key_to_id[common] = match_key_to_id[latin]
        else:
            for lbl, i in label2id.items():
                match_key_to_id[str(lbl)] = i

        label2id_maps[col] = match_key_to_id

    return label2id_maps


def rebuild_label2id_maps_from_lookup_dir(new_lookup_dir):
    """
    从已写出的 unified lookup parquet 重建 Pass-2 所需的 label2id_maps，
    与 build_unified_lookup 末尾构造 match_key_to_id 的逻辑一致。
    行号 i 即该 label 的 id（与写盘时 new_df.iloc[i,0]=lbl 一致）。
    """
    label2id_maps = {}
    for col, fn in COLUMNS_WITH_LOOKUP.items():
        path = join(new_lookup_dir, fn)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"--skip-pass1 需要已生成的完整 lookup，缺失文件: {path}"
            )
        df = pd.read_parquet(path).reset_index(drop=True)
        if "label" not in df.columns:
            if df.shape[1] == 1:
                df = df.rename(columns={df.columns[0]: "label"})
            else:
                raise ValueError(f"{path} 没有 label 列，且无法推断")

        label2id = {}
        for i, lbl in enumerate(df["label"].tolist()):
            if pd.isna(lbl):
                continue
            store_label = normalize_label_for_store(col, lbl)
            label2id[store_label] = i

        match_key_to_id = {}
        if col == "disease":
            for lbl, i in label2id.items():
                match_key_to_id[str(lbl).lower()] = i
        elif col == "species":
            for lbl, i in label2id.items():
                latin = to_latin_snake(lbl)
                match_key_to_id[latin] = i
            for common, latin in COMMON_TO_LATIN.items():
                if latin in match_key_to_id:
                    match_key_to_id[common] = match_key_to_id[latin]
        else:
            for lbl, i in label2id.items():
                match_key_to_id[str(lbl)] = i

        label2id_maps[col] = match_key_to_id

    return label2id_maps


def check_unified_lookups(new_lookup_dir):
    """
    读取 new_lookup_dir 下各 lookup parquet，按 LOOKUP 探查 log 风格打印结构；
    并校验 species.parquet 中 label 唯一值个数为 EXPECTED_SPECIES_LOOKUP_LABELS。
    """
    print("=" * 80)
    print("LOOKUP_categories_unified 数据目录探查 (PASS-1 校验)")
    print(f"目录: {new_lookup_dir}")
    print("=" * 80)

    for col_key, fn in COLUMNS_WITH_LOOKUP.items():
        path = join(new_lookup_dir, fn)
        print("-" * 80)
        if not os.path.exists(path):
            if col_key == "species":
                print(f"[FATAL] 必须存在但缺失: {path}")
                sys.exit(1)
            print(f"[WARN] 缺失: {fn}，跳过")
            continue

        st = os.stat(path)
        size_b = st.st_size
        pf = pq.ParquetFile(path)
        nrows = pf.metadata.num_rows
        print(f"文件: {fn}")
        print(f"大小: {size_b} bytes ({size_b / (1024**2):.2f} MB)")
        print(f"Parquet 行数: {nrows}")
        print(f"行组数: {pf.num_row_groups}")
        print("Schema (每列):")
        for j in range(len(pf.schema)):
            col = pf.schema.column(j)
            print("  <ParquetColumnSchema>")
            print(f"  name: {col.name}")
            print(f"  path: {getattr(col, 'path', col.name)}")
            print(f"  max_definition_level: {col.max_definition_level}")
            print(f"  max_repetition_level: {col.max_repetition_level}")
            print(f"  physical_type: {col.physical_type}")
            lt = col.logical_type
            print(f"  logical_type: {str(lt) if lt is not None else 'None'}")
            cv = getattr(col, "converted_type", None)
            print(f"  converted_type (legacy): {cv if cv is not None else 'NONE'}")

        df = pd.read_parquet(path)
        print(f"DataFrame shape: {df.shape}")
        print("列 dtypes:")
        print(df.dtypes.to_string())
        print("前 5 行:")
        print(df.head(5).to_string())
        if col_key == "species":
            if "label" not in df.columns:
                print("[FATAL] species.parquet 无 label 列")
                sys.exit(1)
            n_uniq = df["label"].nunique()
            n_len = len(df)
            if n_uniq != n_len:
                print(f"[FATAL] species.parquet 存在重复 label: 行数={n_len}, nunique={n_uniq}")
                sys.exit(1)
            if n_uniq != EXPECTED_SPECIES_LOOKUP_LABELS:
                print(
                    f"[FATAL] species.parquet 唯一 label 数应为 {EXPECTED_SPECIES_LOOKUP_LABELS}，"
                    f"实际 nunique={n_uniq}（行数={n_len}）"
                )
                sys.exit(1)
            print(
                f"[OK] species: label 唯一值个数 = {n_uniq}（期望 {EXPECTED_SPECIES_LOOKUP_LABELS}）"
            )

    print("\n[PASS-1] Unified lookups 校验完成。")


# -----------------------------
# Pass-2: map each species directory and write parquet
# -----------------------------
def map_obs_with_unified_ids(obs, label2id_maps):

    obs = obs.copy()

    for col in COLUMNS_WITH_LOOKUP.keys():
        if col not in obs.columns:
            continue
        keys = obs[col].map(lambda x: normalize_value_for_match(col, x))
        mapper = label2id_maps.get(col, {})
        mapped = keys.map(lambda x: mapper.get(str(x), -1))
        obs[col] = mapped.astype("i8")

    return obs


def process_species_dir(
    sp_name,
    sp_path,
    output_base,
    label2id_maps,
    normalization,
    chunk_size,
    row_group_size,
):
    print(f"\n{'='*60}\nProcessing species dir: {sp_name}\n{'='*60}")
    species_output_path = join(output_base, sp_name)
    os.makedirs(species_output_path, exist_ok=True)

    obs = load_obs(sp_path)
    if obs is None:
        print(f"[SKIP] {sp_name}: missing obs.parquet")
        return

    # 使用统一新 lookup 做整列映射
    obs_mapped = map_obs_with_unified_ids(obs, label2id_maps)
    obs_mapped["idx"] = np.arange(len(obs_mapped), dtype="i8")

    zarr_store_path = join(sp_path, "X.zarr", "X")
    if not os.path.exists(zarr_store_path):
        print(f"[SKIP] {sp_name}: missing X.zarr/X")
        return

    try:
        X = preprocess_count_matrix(da.from_zarr(zarr_store_path), normalization)
    except Exception as e:
        print(f"[ERROR] {sp_name}: load zarr failed: {e}")
        return

    # 行数对齐到 row group 的倍数
    n_samples = int(X.shape[0])
    n_samples = (n_samples // row_group_size) * row_group_size
    if n_samples == 0:
        print(f"[SKIP] {sp_name}: no full row_group")
        return

    X = X[:n_samples].rechunk((chunk_size, -1))
    obs_mapped = obs_mapped.iloc[:n_samples].copy()

    starts = [0] + list(np.cumsum(X.chunks[0]))[:-1]
    ends = list(np.cumsum(X.chunks[0]))
    divisions = [0] + list(np.cumsum(X.chunks[0]))
    divisions[-1] = divisions[-1] - 1

    ddf_X = dd.from_delayed(
        [convert_to_dataframe(arr, s, e) for arr, s, e in zip(X.to_delayed().flatten().tolist(), starts, ends)],
        divisions=divisions,
    )

    obs_mapped = obs_mapped.iloc[:n_samples].copy()
    obs_mapped.index = pd.RangeIndex(0, n_samples)

    ddf_obs = dd.from_pandas(obs_mapped, chunksize=chunk_size)
    ddf_obs = ddf_obs.repartition(divisions=ddf_X.divisions)

    ddf = dd.multi.concat([ddf_X, ddf_obs], axis=1)

    keep_cols = ["X"] + list(COLUMNS_WITH_LOOKUP.keys()) + ["idx"]
    keep_cols = [c for c in keep_cols if c in ddf.columns]
    ddf = ddf[keep_cols]

    schema_fields = [("X", pa.list_(pa.float32()))]
    for col in keep_cols:
        if col == "X":
            continue
        schema_fields.append((col, pa.int64()))
    schema = pa.schema(schema_fields)

    out_parquet = join(species_output_path, "obs")
    print(f"[WRITE] {sp_name} → {out_parquet}")
    ddf.to_parquet(
        out_parquet,
        engine="pyarrow",
        schema=schema,
        write_metadata_file=True,
        row_group_size=row_group_size,
    )
    print(f"[DONE] {sp_name}")


# -----------------------------
# CLI / main
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Unify categorical IDs across species and export parquet datasets.")
    p.add_argument("--input-root", default=DEFAULT_INPUT_ROOT, help="Root dir containing species subdirs")
    p.add_argument(
        "-o",
        "--output-base",
        default=DEFAULT_OUTPUT_BASE,
        help="各物种结果输出根目录（覆盖上方 DEFAULT_OUTPUT_BASE；与 --new-lookup-dir 配合时后者可单独指定 LOOKUP 目录）",
    )
    p.add_argument("--old-lookup-dir", default=DEFAULT_OLD_LOOKUP_DIR, help="Existing LOOKUP_categories dir (read-only)")
    p.add_argument(
        "--new-lookup-dir",
        default=None,
        help="Where to write the new unified lookup (default: <output-base>/LOOKUP_categories_unified)",
    )
    p.add_argument("--normalization", default=DEFAULT_NORMALIZATION, choices=["sf-log1p", "raw"])
    p.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    p.add_argument("--row-group-size", type=int, default=DEFAULT_ROW_GROUP_SIZE)
    p.add_argument("--workers", type=int, default=int(os.getenv("SLURM_NTASKS_PER_NODE", 1)))
    p.add_argument("--threads-per-worker", type=int, default=int(os.getenv("SLURM_CPUS_PER_TASK", 16)))
    p.add_argument(
        "--species-jobs",
        type=int,
        default=1,
        help="Pass-2: number of species dirs to process concurrently (threads). 1 = sequential (default). "
        "Shares one Dask cluster; raise gradually to avoid memory/IO spikes.",
    )
    p.add_argument(
        "--skip-pass1",
        action="store_true",
        help="跳过 collect_all_values / build_unified_lookup，从 --new-lookup-dir 下已有 parquet "
        "重建 label2id_maps 并继续 Pass-2（需 COLUMNS_WITH_LOOKUP 对应文件齐全）。",
    )
    return p.parse_args()


def main():
    args = parse_args()

    input_root = args.input_root
    output_base = args.output_base
    old_lookup_dir = args.old_lookup_dir
    new_lookup_dir = args.new_lookup_dir or join(output_base, DEFAULT_NEW_LOOKUP_DIR_NAME)

    os.makedirs(output_base, exist_ok=True)
    os.makedirs(new_lookup_dir, exist_ok=True)

    print(f"[CONFIG] input_root       = {input_root}")
    print(f"[CONFIG] output_base      = {output_base}")
    print(f"[CONFIG] old_lookup_dir   = {old_lookup_dir} (read-only)")
    print(f"[CONFIG] new_lookup_dir   = {new_lookup_dir} (write)")
    print(f"[CONFIG] normalization    = {args.normalization}")
    print(f"[CONFIG] chunk_size       = {args.chunk_size}")
    print(f"[CONFIG] row_group_size   = {args.row_group_size}")
    print(f"[CONFIG] workers x threads= {args.workers} x {args.threads_per_worker}")
    print(f"[CONFIG] species_jobs (Pass-2) = {args.species_jobs}")

    if args.skip_pass1:
        print("\n[PASS-1] --skip-pass1: loading unified lookups from disk (no rescan / rebuild)...")
        label2id_maps = rebuild_label2id_maps_from_lookup_dir(new_lookup_dir)
        print(f"[LOOKUP] Loaded label2id_maps from: {new_lookup_dir}")
    else:
        # Pass-1: 收集全集并构建统一 lookup
        print("\n[PASS-1] Collecting all values across species...")
        all_values = collect_all_values(input_root)
        print("[PASS-1] Building unified lookups...")
        label2id_maps = build_unified_lookup(old_lookup_dir, new_lookup_dir, all_values)
        print(f"[LOOKUP] Unified lookups saved under: {new_lookup_dir}")

    print("[PASS-1] Checking unified lookups...")
    check_unified_lookups(new_lookup_dir)

    # Pass-2: 逐物种处理（可选多线程并行；每物种仍独立读写，不改变数值逻辑）
    print("\n[PASS-2] Processing each species directory...")
    species_list = list_species_dirs(input_root)

    cluster = LocalCluster(n_workers=args.workers, threads_per_worker=args.threads_per_worker)
    client = Client(cluster)

    def _pass2_one(sp):
        process_species_dir(
            sp_name=sp,
            sp_path=join(input_root, sp),
            output_base=output_base,
            label2id_maps=label2id_maps,
            normalization=args.normalization,
            chunk_size=args.chunk_size,
            row_group_size=args.row_group_size,
        )

    if args.species_jobs <= 1:
        for sp in species_list:
            _pass2_one(sp)
    else:
        with ThreadPoolExecutor(max_workers=args.species_jobs) as ex:
            futures = {ex.submit(_pass2_one, sp): sp for sp in species_list}
            for fut in as_completed(futures):
                fut.result()

    print("\nAll processing completed.")
    client.restart()


if __name__ == "__main__":
    dask.config.set({"dataframe.convert-string": False})
    main()
