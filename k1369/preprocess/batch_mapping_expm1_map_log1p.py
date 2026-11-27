"""Map normalized gene expression before log1p.

Input X must be natural-log1p normalized GENE expression (not mapped data).
Computes log1p(align_and_sum_duplicate_genes(expm1(X)) @ W).
Normalization is preserved; raw counts are not recovered. No renormalization.
See ../README.md for deployment paths and usage.
"""
import os
import glob
import argparse
import threading
from datetime import datetime

import anndata as ad
import numpy as np
import pandas as pd
from dask import delayed, compute

try:
    import scipy.sparse as sp
except Exception:  # pragma: no cover
    sp = None


# =========================
# Thread-safe caches
# =========================
_CACHE_LOCK = threading.Lock()
_LOG_LOCK = threading.Lock()
_WEIGHT_CACHE = {}
_VAR_CACHE = {}


def _to_lower_index(index_like):
    return pd.Index([str(x).lower() for x in index_like])


def _build_lower_groups(index_like):
    lower_index = _to_lower_index(index_like)
    groups = {}
    for idx, gene in enumerate(lower_index):
        groups.setdefault(gene, []).append(idx)
    return lower_index, groups


def _duplicate_stats(lower_index):
    duplicated = lower_index.duplicated(keep=False)
    duplicate_rows = int(duplicated.sum())
    duplicate_names = int(lower_index[duplicated].nunique()) if duplicate_rows else 0
    return duplicate_names, duplicate_rows


def _write_log(log_path, message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with _LOG_LOCK:
        with open(log_path, "a") as f:
            f.write(f"[{timestamp}] {message}\n")


def _collapse_weight_rows_by_lower(W, lower_index, target_dtype=np.float32):
    duplicate_names, duplicate_rows = _duplicate_stats(lower_index)
    if duplicate_rows == 0:
        return W, lower_index, {
            "weight_duplicate_gene_names": 0,
            "weight_duplicate_rows": 0,
            "weight_unique_gene_count": len(lower_index),
        }

    codes, unique_values = pd.factorize(lower_index, sort=False)
    counts = np.bincount(codes).astype(np.float32)

    if sp is not None and sp.issparse(W):
        rows = codes
        cols = np.arange(len(codes))
        values = 1.0 / counts[codes]
        group_matrix = sp.csr_matrix((values, (rows, cols)), shape=(len(unique_values), W.shape[0]))
        W_collapsed = (group_matrix @ W).tocsr()
        if W_collapsed.dtype != target_dtype:
            W_collapsed = W_collapsed.astype(target_dtype)
    else:
        W_arr = np.asarray(W, dtype=target_dtype)
        W_collapsed = np.zeros((len(unique_values), W_arr.shape[1]), dtype=target_dtype)
        np.add.at(W_collapsed, codes, W_arr)
        W_collapsed /= counts[:, None]

    return W_collapsed, pd.Index(unique_values), {
        "weight_duplicate_gene_names": duplicate_names,
        "weight_duplicate_rows": duplicate_rows,
        "weight_unique_gene_count": len(unique_values),
    }


def _matrix_row_sums(M):
    if sp is not None and sp.issparse(M):
        return np.asarray(M.sum(axis=1)).ravel()
    return np.asarray(M.sum(axis=1)).ravel()


def _validate_weight_matrix(W, source_path):
    if sp is not None and sp.issparse(W):
        if not np.isfinite(W.data).all():
            raise ValueError(f"Weight matrix contains NaN/Inf values: {source_path}")
        min_value = float(W.data.min()) if W.data.size else 0.0
    else:
        if not np.isfinite(W).all():
            raise ValueError(f"Weight matrix contains NaN/Inf values: {source_path}")
        min_value = float(np.min(W)) if W.size else 0.0

    if min_value < -1e-6:
        raise ValueError(f"Weight matrix contains negative values below tolerance: {source_path}")

    row_sums = _matrix_row_sums(W)
    if row_sums.size == 0 or np.any(row_sums <= 0):
        raise ValueError(f"Weight matrix contains zero-sum rows: {source_path}")

    return {
        "weight_row_sum_min": float(row_sums.min()),
        "weight_row_sum_median": float(np.median(row_sums)),
        "weight_row_sum_max": float(row_sums.max()),
    }


def _load_var_cache(var_parquet_path):
    with _CACHE_LOCK:
        cached = _VAR_CACHE.get(var_parquet_path)
        if cached is not None:
            return cached

    var_df = pd.read_parquet(var_parquet_path)

    if "feature_name" in var_df.columns:
        gene_order_raw = pd.Index(var_df["feature_name"])
    else:
        gene_order_raw = var_df.index
    gene_order_lower, lower_to_x_indices = _build_lower_groups(gene_order_raw)
    duplicate_names, duplicate_columns = _duplicate_stats(gene_order_lower)
    payload = {
        "gene_order_raw": gene_order_raw,
        "gene_order_lower": gene_order_lower,
        "lower_to_x_indices": lower_to_x_indices,
        "input_unique_gene_count": len(lower_to_x_indices),
        "input_duplicate_gene_names": duplicate_names,
        "input_duplicate_columns": duplicate_columns,
    }

    with _CACHE_LOCK:
        _VAR_CACHE[var_parquet_path] = payload
    return payload


def _load_weight_cache(weight_h5ad_path, target_dtype=np.float32):
    with _CACHE_LOCK:
        cached = _WEIGHT_CACHE.get(weight_h5ad_path)
        if cached is not None:
            return cached

    weight_adata = ad.read_h5ad(weight_h5ad_path)
    expected_axis = [f"MacroGene-{i}" for i in range(1369)]
    if list(weight_adata.var_names) != expected_axis:
        raise ValueError(f"{weight_h5ad_path}: expected ordered K1369 macrogene axis")
    W = weight_adata.X
    if sp is not None and sp.issparse(W):
        W = W.tocsr()
        if W.dtype != target_dtype:
            W = W.astype(target_dtype)
    else:
        W = np.asarray(W, dtype=target_dtype, order="C")

    weight_gene_names_lower_raw = _to_lower_index(weight_adata.obs_names)
    W, weight_gene_names_lower, duplicate_payload = _collapse_weight_rows_by_lower(
        W, weight_gene_names_lower_raw, target_dtype=target_dtype
    )
    weight_qc = _validate_weight_matrix(W, weight_h5ad_path)

    payload = {
        "W": W,
        "weight_gene_names_raw": weight_adata.obs_names,
        "weight_gene_names_lower": weight_gene_names_lower,
        "macrogene_names": weight_adata.var_names,
        **duplicate_payload,
        **weight_qc,
    }

    with _CACHE_LOCK:
        _WEIGHT_CACHE[weight_h5ad_path] = payload
    return payload


def _prepare_mapping(var_parquet_path, weight_h5ad_path):
    var_info = _load_var_cache(var_parquet_path)
    weight_info = _load_weight_cache(weight_h5ad_path)

    gene_order_raw = var_info["gene_order_raw"]
    gene_order_lower = var_info["gene_order_lower"]
    lower_to_x_indices = var_info["lower_to_x_indices"]
    weight_gene_names_lower = weight_info["weight_gene_names_lower"]

    common_gene_names = sorted(set(gene_order_lower).intersection(set(weight_gene_names_lower)))
    if not common_gene_names:
        raise ValueError("No overlapping genes found!")

    x_groups = [np.asarray(lower_to_x_indices[gene], dtype=np.int64) for gene in common_gene_names]
    matched_input_columns = int(sum(len(group) for group in x_groups))
    matched_input_column_idx = np.asarray(
        sorted({int(idx) for group in x_groups for idx in group}), dtype=np.int64
    )
    w_idx = weight_gene_names_lower.get_indexer(common_gene_names)
    if np.any(w_idx < 0):
        raise ValueError("Internal error: matched weight genes produced negative indices.")

    mapping = {
        "X_groups": x_groups,
        "matched_input_column_idx": matched_input_column_idx,
        "W_idx": w_idx,
        "matched_gene_count": len(common_gene_names),
        "input_gene_count": len(gene_order_raw),
        "input_unique_gene_count": var_info["input_unique_gene_count"],
        "input_duplicate_gene_names": var_info["input_duplicate_gene_names"],
        "input_duplicate_columns": var_info["input_duplicate_columns"],
        "matched_input_columns": matched_input_columns,
        "weight_gene_count": len(weight_info["weight_gene_names_raw"]),
        "weight_unique_gene_count": weight_info["weight_unique_gene_count"],
        "weight_duplicate_gene_names": weight_info["weight_duplicate_gene_names"],
        "weight_duplicate_rows": weight_info["weight_duplicate_rows"],
        "matched_input_unique_fraction": len(common_gene_names) / max(var_info["input_unique_gene_count"], 1),
        "matched_input_column_fraction": matched_input_columns / max(len(gene_order_raw), 1),
        "matched_weight_fraction": len(common_gene_names) / max(weight_info["weight_unique_gene_count"], 1),
        "weight_row_sum_min": weight_info["weight_row_sum_min"],
        "weight_row_sum_median": weight_info["weight_row_sum_median"],
        "weight_row_sum_max": weight_info["weight_row_sum_max"],
        "macrogene_names": weight_info["macrogene_names"],
    }
    return mapping, weight_info


def _extract_x_matrix(series, dtype=np.float32):
    # Keep one contiguous dense matrix; object->matrix conversion is unavoidable here,
    # but we enforce float32 to reduce memory.
    return np.asarray(np.stack(series.to_numpy()), dtype=dtype, order="C")


def _align_x_chunk(X_chunk, X_groups, out_dtype=np.float32):
    if all(len(group) == 1 for group in X_groups):
        x_idx = np.asarray([group[0] for group in X_groups], dtype=np.int64)
        return np.asarray(X_chunk[:, x_idx], dtype=out_dtype, order="C")

    X_aligned = np.empty((X_chunk.shape[0], len(X_groups)), dtype=out_dtype, order="C")
    for j, group in enumerate(X_groups):
        if len(group) == 1:
            X_aligned[:, j] = X_chunk[:, group[0]]
        else:
            X_aligned[:, j] = X_chunk[:, group].sum(axis=1)
    return X_aligned


def _require_nonnegative_finite(matrix, label):
    values = matrix.data if sp is not None and sp.issparse(matrix) else np.asarray(matrix)
    if not np.isfinite(values).all():
        raise ValueError(f"{label} contains NaN/Inf (possibly numerical overflow).")
    if values.size and np.min(values) < 0:
        raise ValueError(f"{label} contains negative values; expected nonnegative expression/weights.")


def _matmul_chunked(X_cell, X_groups, W, W_idx, chunk_size=2048, out_dtype=np.float32):
    """Invert log per original column BEFORE duplicate-gene summation.

    Only the current gene chunk is exponentiated. QC totals are on the
    normalized linear scale, before the output log1p operation.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if X_cell.ndim != 2:
        raise ValueError("X must be a two-dimensional cell-by-gene matrix")
    W_aligned = W[W_idx, :]
    if sp is None or not sp.issparse(W_aligned):
        W_aligned = np.asarray(W_aligned, dtype=out_dtype, order="C")
    _require_nonnegative_finite(W_aligned, "Aligned weights")
    matched_idx = np.unique(np.concatenate(X_groups))
    result = np.empty((X_cell.shape[0], W.shape[1]), dtype=out_dtype)
    totals = dict(input_total=0.0, matched_total=0.0, macro_total=0.0)
    for start in range(0, len(X_cell), chunk_size):
        end = min(start + chunk_size, len(X_cell))
        # A copy is essential: do not mutate the original log1p input.
        linear = np.array(X_cell[start:end], dtype=out_dtype, order="C", copy=True)
        _require_nonnegative_finite(linear, "Input gene log1p expression")
        with np.errstate(over="raise", invalid="raise"):
            np.expm1(linear, out=linear)
            _require_nonnegative_finite(linear, "Recovered normalized gene expression")
            totals["input_total"] += float(linear.sum(dtype=np.float64))
            totals["matched_total"] += float(linear[:, matched_idx].sum(dtype=np.float64))
            aligned = _align_x_chunk(linear, X_groups, out_dtype=out_dtype)
            macro = aligned @ W_aligned
            if sp is not None and sp.issparse(macro):
                macro = macro.toarray()
            macro = np.asarray(macro, dtype=out_dtype)
            _require_nonnegative_finite(macro, "Linear macrogene expression")
            totals["macro_total"] += float(macro.sum(dtype=np.float64))
            np.log1p(macro, out=macro)
        result[start:end] = macro
    return result, totals


@delayed
def process_single_part(
    species,
    obs_parquet_path,
    var_parquet_path,
    weight_h5ad_path,
    output_path,
    output_dir,
    log_path,
    chunk_size=2048,
):
    try:
        print(f"[{species}] Processing: {obs_parquet_path}")
        os.makedirs(output_dir, exist_ok=True)

        mapping, weight_info = _prepare_mapping(var_parquet_path, weight_h5ad_path)
        X_groups = mapping["X_groups"]
        W_idx = mapping["W_idx"]
        W = weight_info["W"]

        print(f"[{species}] Gene stats for {os.path.basename(obs_parquet_path)}:")
        print(f"    - Input gene count:      {mapping['input_gene_count']}")
        print(f"    - Input unique genes:    {mapping['input_unique_gene_count']}")
        print(f"    - Weight raw/unique:     {mapping['weight_gene_count']} / {mapping['weight_unique_gene_count']}")
        print(f"    - Matched unique genes:  {mapping['matched_gene_count']}")
        print(f"    - Matched input columns: {mapping['matched_input_columns']}")

        df = pd.read_parquet(obs_parquet_path)
        X_cell = (_extract_x_matrix(df["X"], dtype=np.float32) if len(df)
                  else np.empty((0, mapping["input_gene_count"]), dtype=np.float32))
        if X_cell.ndim != 2 or X_cell.shape[1] != mapping["input_gene_count"]:
            raise ValueError("X width does not match var.parquet gene count; use gene-level input.")
        X_macro, totals = _matmul_chunked(
            X_cell, X_groups, W, W_idx, chunk_size=chunk_size, out_dtype=np.float32
        )
        input_total = totals["input_total"]
        matched_total = totals["matched_total"]
        macro_total = totals["macro_total"]

        unmatched_loss_fraction = 1.0 - (matched_total / input_total) if input_total > 0 else 0.0
        macro_vs_matched_fraction = (macro_total - matched_total) / matched_total if matched_total > 0 else 0.0
        macro_vs_input_fraction = macro_total / input_total if input_total > 0 else 0.0
        qc_message = (
            f"QC transform=expm1_mapping_log1p totals_scale=normalized_linear species={species} part={os.path.basename(obs_parquet_path)} "
            f"cells={X_cell.shape[0]} input_genes={mapping['input_gene_count']} "
            f"input_unique_genes={mapping['input_unique_gene_count']} "
            f"input_duplicate_gene_names={mapping['input_duplicate_gene_names']} "
            f"input_duplicate_columns={mapping['input_duplicate_columns']} "
            f"weight_genes={mapping['weight_gene_count']} "
            f"weight_unique_genes={mapping['weight_unique_gene_count']} "
            f"weight_duplicate_gene_names={mapping['weight_duplicate_gene_names']} "
            f"weight_duplicate_rows={mapping['weight_duplicate_rows']} "
            f"matched_unique_genes={mapping['matched_gene_count']} "
            f"matched_input_columns={mapping['matched_input_columns']} "
            f"matched_input_unique_fraction={mapping['matched_input_unique_fraction']:.6f} "
            f"matched_input_column_fraction={mapping['matched_input_column_fraction']:.6f} "
            f"matched_weight_fraction={mapping['matched_weight_fraction']:.6f} "
            f"input_total={input_total:.6f} matched_total={matched_total:.6f} "
            f"macro_total={macro_total:.6f} "
            f"unmatched_loss_fraction={unmatched_loss_fraction:.8f} "
            f"macro_vs_matched_fraction={macro_vs_matched_fraction:.8f} "
            f"macro_vs_input_fraction={macro_vs_input_fraction:.8f} "
            f"weight_row_sum_min={mapping['weight_row_sum_min']:.8f} "
            f"weight_row_sum_median={mapping['weight_row_sum_median']:.8f} "
            f"weight_row_sum_max={mapping['weight_row_sum_max']:.8f}"
        )
        print(f"[{species}] {qc_message}")
        _write_log(log_path, qc_message)

        # Free the largest temporary early.
        del X_cell

        # Keep only one DataFrame object instead of copy().
        df["X"] = [row.tolist() for row in X_macro]
        del X_macro

        # Atomic rename prevents interrupted writes from being skipped on resume.
        temporary_path = output_path + ".tmp"
        try:
            df.to_parquet(temporary_path, index=False)
            os.replace(temporary_path, output_path)
        finally:
            if os.path.exists(temporary_path):
                os.remove(temporary_path)
        print(f"[{species}] Saved macrogene: {output_path}")
        return {"status": "ok", "species": species, "obs": obs_parquet_path, "output": output_path}

    except Exception as e:
        _write_log(log_path, f"ERROR species={species} part={obs_parquet_path} error={str(e)}")
        print(f"[{species}] Failed processing {obs_parquet_path}")
        return {"status": "error", "species": species, "obs": obs_parquet_path, "error": str(e)}


def process_from_csv_with_dir(
    csv_path,
    sep=",",
    log_path="expm1_map_log1p_qc.log",
    scheduler="threads",
    num_workers=2,
    chunk_size=2048,
    skip_existing=True,
    fresh_log=False,
):
    if chunk_size <= 0 or num_workers <= 0:
        raise ValueError("chunk_size and num_workers must be positive")
    if scheduler not in ("threads", "synchronous", "single-threaded"):
        raise ValueError("Use threads or synchronous for the shared weight cache and log lock")
    df = pd.read_csv(csv_path, sep=sep)
    required = ["species", "obs_parquet_path", "var_parquet_path", "weight_h5ad_path", "output_path"]
    if not set(required).issubset(df.columns) or df[required].isna().any().any() or df.empty:
        raise ValueError("Configuration must contain nonempty species/input/var/weight/output columns")
    if df["species"].duplicated().any() or df["output_path"].map(os.path.realpath).duplicated().any():
        raise ValueError("Species and output directories must be unique")
    # Validate every task before creating any outputs. Never write into input data.
    for _, row in df.iterrows():
        source = os.path.realpath(row["obs_parquet_path"])
        target = os.path.realpath(row["output_path"])
        if os.path.commonpath([source, target]) in (source, target):
            raise ValueError("Input and output directories must be separate")
        if not glob.glob(os.path.join(source, "part.*.parquet")):
            raise ValueError(f"No gene-level part.*.parquet input in {source}; step4 may already be mapped")
        for key in ("var_parquet_path", "weight_h5ad_path"):
            if not os.path.isfile(row[key]):
                raise FileNotFoundError(row[key])
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    print(df)
    tasks = []
    skipped = []

    if fresh_log and os.path.exists(log_path):
        os.remove(log_path)

    # Write lookup exactly once per species in the parent process.
    for _, row in df.iterrows():
        species = row["species"]
        output_dir = row["output_path"]
        weight_h5ad_path = row["weight_h5ad_path"]
        os.makedirs(output_dir, exist_ok=True)
        marker = os.path.join(output_dir, "expm1_mapping_log1p.version")
        if not os.path.exists(marker):
            if glob.glob(os.path.join(output_dir, "macrogene_*.parquet")):
                raise ValueError(f"Unmarked mapping outputs in {output_dir}; choose a new output directory")
            with open(marker, "w") as handle:
                handle.write("natural_log1p -> expm1 -> duplicate_sum -> mapping -> natural_log1p\n")

        _, weight_info = _prepare_mapping(row["var_parquet_path"], weight_h5ad_path)
        lookup_path = os.path.join(output_dir, f"{species}_macrogene_lookup.parquet")
        if not os.path.exists(lookup_path):
            pd.DataFrame({"macrogene": weight_info["macrogene_names"]}).to_parquet(lookup_path, index=False)
            print(f"[{species}] Saved lookup to: {lookup_path}")

    for _, row in df.iterrows():
        species = row["species"]
        obs_dir = row["obs_parquet_path"]
        var_parquet_path = row["var_parquet_path"]
        weight_h5ad_path = row["weight_h5ad_path"]
        output_dir = row["output_path"]

        obs_files = sorted(glob.glob(os.path.join(obs_dir, "part.*.parquet")))
        if not obs_files:
            print(f"[{species}] No part.*.parquet files found in {obs_dir}")
            continue

        for obs_file in obs_files:
            part_id = os.path.basename(obs_file).split(".")[-2]
            output_path = os.path.join(output_dir, f"macrogene_{part_id}.parquet")
            if skip_existing and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                skipped.append({"species": species, "obs": obs_file, "output": output_path})
                continue

            task = process_single_part(
                species=species,
                obs_parquet_path=obs_file,
                var_parquet_path=var_parquet_path,
                weight_h5ad_path=weight_h5ad_path,
                output_path=output_path,
                output_dir=output_dir,
                log_path=log_path,
                chunk_size=chunk_size,
            )
            tasks.append(task)

    print(
        f"Prepared {len(tasks)} task(s); skipped {len(skipped)} existing output file(s). "
        f"skip_existing={skip_existing}"
    )
    if skipped:
        _write_log(
            log_path,
            f"RESUME skipped_existing={len(skipped)} scheduled_tasks={len(tasks)}",
        )

    if not tasks:
        print("No tasks to run. All outputs already exist.")
        return {"results": [], "skipped": skipped}

    results = compute(*tasks, scheduler=scheduler, num_workers=num_workers)
    failures = [r for r in results if not r or r.get("status") != "ok"]
    if failures:
        preview = "; ".join(
            f"{r.get('species', 'unknown')}:{r.get('obs', 'unknown')}:{r.get('error', 'unknown')}"
            for r in failures[:5]
            if r
        )
        raise RuntimeError(
            f"{len(failures)} macrogene mapping task(s) failed. "
            f"See {log_path} for details. First failures: {preview}"
        )
    return {"results": results, "skipped": skipped}


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Gene expm1 -> mapping -> macrogene log1p (natural logarithms)."
    )
    parser.add_argument("--csv", default="task_config_1st_expm1_map_log1p.csv", help="Task configuration CSV.")
    parser.add_argument("--sep", default=",", help="Task configuration delimiter.")
    parser.add_argument("--log", default="expm1_map_log1p_qc.log", help="Internal QC/error log path.")
    parser.add_argument("--scheduler", default="threads", help="Dask scheduler.")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of concurrent Dask workers.")
    parser.add_argument("--chunk-size", type=int, default=2048, help="Cell chunk size for matrix multiplication.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute outputs even when macrogene_*.parquet already exists.",
    )
    parser.add_argument(
        "--fresh-log",
        action="store_true",
        help="Delete the internal log before starting this run.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    process_from_csv_with_dir(
        args.csv,
        sep=args.sep,
        log_path=args.log,
        scheduler=args.scheduler,
        num_workers=args.num_workers,
        chunk_size=args.chunk_size,
        skip_existing=not args.overwrite,
        fresh_log=args.fresh_log,
    )
