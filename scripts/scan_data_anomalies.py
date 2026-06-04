#!/usr/bin/env python3
"""Full-dataset anomaly scan for the flattened macrogene parquet training data.

Complements ``pretrain_checks.py validate-data`` (which samples the first N rows
per file and only checks X-shape / finiteness / label range). This scanner reads
**every row** and focuses on the dimension that the existing check misses and
that actually drives the GEP huber divergence: the *magnitude / distribution* of
the expression vector ``X``.

Why magnitude matters here: in the collator the Huber regression target
(``target_values``) is the raw ``X`` as stored — there is no log1p / normalize /
clamp / binning. So a finite-but-large ``X`` passes ``isfinite`` yet pushes
``loss_gep`` up (Huber is linear beyond delta), and a sparsity shift pushes
``loss_zero_prob`` (BCE on ``X > 0``). This tool surfaces both.

Outputs (CSV):
  <prefix>_per_file.csv     one row per parquet file with distribution stats
  <prefix>_per_dataset.csv  aggregates by dataset_id (trace a bad source)
  <prefix>_per_species.csv  aggregates by species id

It also prints the worst files / groups so you can eyeball outliers and
cross-reference them against the divergence (e.g. epoch-2 file-shuffle order).

Read-only. Run on the box that holds the data:
  python scripts/scan_data_anomalies.py \
      --data-path /data/disk1/SpeciesLLM_obs/Stage2_SpeciesLLMData/all_flatten_data_full_no_1st_human_mouse_20260506_165244_external \
      --workers 16 --out-prefix /tmp/data_scan
"""
import argparse
import glob
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow.compute as pc

# Columns we touch. X is required; the rest are best-effort for tracing/labels.
X_COL = "X"
TRACE_COLS = ["dataset_id", "species"]
LABEL_COLS = ["assay", "tech_sample", "species", "tissue", "disease", "sex", "development_stage"]


def _percentiles(a, qs):
    if a.size == 0:
        return [float("nan")] * len(qs)
    return [float(x) for x in np.percentile(a, qs)]


def scan_file(path, seq_len, sample_rows):
    """Stream one parquet file by row-group; return (per_file_dict, group_records)."""
    res = {
        "file": os.path.basename(path),
        "n_rows": 0,
        "n_scanned": 0,
        "x_shape_bad": 0,
        "x_nonfinite_vals": 0,
        "x_nonfinite_cells": 0,
        "val_min": float("inf"),
        "val_max": float("-inf"),
        "error": "",
    }
    cell_max_all = []   # per-cell max value
    cell_sum_all = []   # per-cell library size (sum)
    cell_nnz_all = []   # per-cell number of expressed genes
    label_minmax = {}   # col -> [min, max]
    group_rows = []     # (group_kind, group_key, n, val_max, libsize_max)

    try:
        pf = pq.ParquetFile(path)
        names = set(pf.schema_arrow.names)
        if X_COL not in names:
            res["error"] = f"missing column {X_COL}; found {sorted(names)[:8]}..."
            return res, group_rows
        res["n_rows"] = pf.metadata.num_rows
        present_trace = [c for c in TRACE_COLS if c in names]
        present_label = [c for c in LABEL_COLS if c in names]
        read_cols = [X_COL] + sorted(set(present_trace) | set(present_label))

        scanned = 0
        for rg in range(pf.num_row_groups):
            if sample_rows and scanned >= sample_rows:
                break
            table = pf.read_row_group(rg, columns=read_cols)
            n = table.num_rows
            if sample_rows and scanned + n > sample_rows:
                table = table.slice(0, sample_rows - scanned)
                n = table.num_rows
            scanned += n

            col = table.column(X_COL).combine_chunks()
            lengths = pc.list_value_length(col).to_numpy(zero_copy_only=False)
            if seq_len is not None:
                res["x_shape_bad"] += int((lengths != seq_len).sum())
            # Flatten + per-cell reduceat (cells are non-empty -> reduceat is safe).
            offsets = col.offsets.to_numpy()
            vals = col.values.to_numpy(zero_copy_only=False).astype(np.float64, copy=False)
            starts = offsets[:-1]
            if starts.size == 0:
                continue
            finite = np.isfinite(vals)
            if not finite.all():
                res["x_nonfinite_vals"] += int((~finite).sum())
                bad_per_cell = np.add.reduceat((~finite).astype(np.int64), starts)
                res["x_nonfinite_cells"] += int((bad_per_cell > 0).sum())
                vals = np.where(finite, vals, 0.0)  # neutralize for stats below

            cmax = np.maximum.reduceat(vals, starts)
            csum = np.add.reduceat(vals, starts)
            cnnz = np.add.reduceat((vals > 0).astype(np.int64), starts)
            cell_max_all.append(cmax)
            cell_sum_all.append(csum)
            cell_nnz_all.append(cnnz)
            res["val_min"] = min(res["val_min"], float(vals.min()))
            res["val_max"] = max(res["val_max"], float(vals.max()))

            for c in present_label:
                mm = pc.min_max(table.column(c)).as_py()
                lo, hi = mm.get("min"), mm.get("max")
                if lo is None:
                    continue
                cur = label_minmax.setdefault(c, [lo, hi])
                cur[0] = min(cur[0], lo)
                cur[1] = max(cur[1], hi)

            # group reductions (dataset_id, species)
            for kind in present_trace:
                g = pd.DataFrame({
                    "key": table.column(kind).to_pylist(),
                    "cmax": cmax, "csum": csum,
                }).groupby("key", dropna=False)
                agg = g.agg(n=("cmax", "size"), val_max=("cmax", "max"),
                            libsize_max=("csum", "max"))
                for key, r in agg.iterrows():
                    group_rows.append((kind, key, int(r["n"]),
                                       float(r["val_max"]), float(r["libsize_max"])))

        res["n_scanned"] = scanned
        if cell_max_all:
            cmax = np.concatenate(cell_max_all)
            csum = np.concatenate(cell_sum_all)
            cnnz = np.concatenate(cell_nnz_all)
            res["cellmax_p50"], res["cellmax_p99"], res["cellmax_p999"], res["cellmax_max"] = \
                _percentiles(cmax, [50, 99, 99.9, 100])
            res["libsize_p50"], res["libsize_p99"], res["libsize_max"] = \
                _percentiles(csum, [50, 99, 100])
            res["nnz_p50"], res["nnz_max"] = _percentiles(cnnz, [50, 100])
            res["sparsity_mean"] = float(1.0 - cnnz.mean() / (seq_len if seq_len else cnnz.max() or 1))
        for c, (lo, hi) in label_minmax.items():
            res[f"label_{c}_min"] = lo
            res[f"label_{c}_max"] = hi
    except Exception as exc:  # noqa: BLE001 - report and keep going
        res["error"] = f"{type(exc).__name__}: {exc}"
    return res, group_rows


def merge_groups(all_group_rows):
    if not all_group_rows:
        return pd.DataFrame(), pd.DataFrame()
    df = pd.DataFrame(all_group_rows, columns=["kind", "key", "n", "val_max", "libsize_max"])
    out = {}
    for kind, sub in df.groupby("kind"):
        g = sub.groupby("key").agg(n_cells=("n", "sum"), val_max=("val_max", "max"),
                                   libsize_max=("libsize_max", "max")).reset_index()
        out[kind] = g.sort_values("val_max", ascending=False)
    return out.get("dataset_id", pd.DataFrame()), out.get("species", pd.DataFrame())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-path", help="Directory containing *.parquet")
    ap.add_argument("--glob", help="Explicit glob (overrides --data-path)")
    ap.add_argument("--seq-len", type=int, default=None,
                    help="Expected X length; if set, counts shape mismatches and computes sparsity.")
    ap.add_argument("--sample-rows", type=int, default=0,
                    help="Rows per file to scan (0 = ALL rows; the point of this tool).")
    ap.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 8))
    ap.add_argument("--out-prefix", default="data_scan")
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    pattern = args.glob or (os.path.join(args.data_path, "*.parquet") if args.data_path else None)
    if not pattern:
        ap.error("provide --data-path or --glob")
    files = sorted(glob.glob(pattern))
    if not files:
        ap.error(f"no parquet files matched {pattern}")
    print(f"[INFO] scanning {len(files)} files with {args.workers} workers "
          f"({'ALL rows' if not args.sample_rows else str(args.sample_rows)+' rows/file'})")

    per_file, all_groups = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(scan_file, f, args.seq_len, args.sample_rows): f for f in files}
        done = 0
        for fut in as_completed(futs):
            res, groups = fut.result()
            per_file.append(res)
            all_groups.extend(groups)
            done += 1
            if done % 50 == 0 or done == len(files):
                print(f"  scanned {done}/{len(files)}")

    fdf = pd.DataFrame(per_file).sort_values("val_max", ascending=False)
    fdf.to_csv(f"{args.out_prefix}_per_file.csv", index=False)
    ddf, sdf = merge_groups(all_groups)
    if not ddf.empty:
        ddf.to_csv(f"{args.out_prefix}_per_dataset.csv", index=False)
    if not sdf.empty:
        sdf.to_csv(f"{args.out_prefix}_per_species.csv", index=False)

    # ---- report ----
    errs = fdf[fdf["error"] != ""]
    if len(errs):
        print(f"\n[!!] {len(errs)} files failed to read:")
        for _, r in errs.head(args.top).iterrows():
            print(f"   {r['file']}: {r['error']}")
    bad_struct = fdf[(fdf.get("x_shape_bad", 0) > 0) | (fdf.get("x_nonfinite_cells", 0) > 0)]
    if len(bad_struct):
        print(f"\n[!!] {len(bad_struct)} files have shape/non-finite issues:")
        cols = ["file", "n_rows", "x_shape_bad", "x_nonfinite_cells", "x_nonfinite_vals"]
        print(bad_struct[cols].head(args.top).to_string(index=False))

    show = [c for c in ["file", "n_rows", "val_max", "cellmax_p999", "libsize_max",
                        "libsize_p99", "sparsity_mean"] if c in fdf.columns]
    print(f"\n=== TOP {args.top} files by max X value (magnitude outliers) ===")
    print(fdf[show].head(args.top).to_string(index=False))

    glob_valmax = fdf["val_max"].replace(-np.inf, np.nan).dropna()
    if len(glob_valmax):
        med = glob_valmax.median()
        ratio = (glob_valmax.max() / med) if med else float("inf")
        verdict = ("BIG SPREAD -> likely scale inconsistency / unnormalized shards"
                   if ratio >= 3 else
                   "tight (max/median < 3x) -> consistent scale across files, no outlier shards")
        print(f"\n[stat] per-file val_max distribution: "
              f"median={med:.3g}  p95={glob_valmax.quantile(.95):.3g}  "
              f"max={glob_valmax.max():.3g}  (max/median={ratio:.2f}x) -> {verdict}")
    if not ddf.empty:
        print(f"\n=== TOP {args.top} dataset_id by max X value (trace the source) ===")
        print(ddf.head(args.top).to_string(index=False))
    if not sdf.empty:
        print(f"\n=== species by max X value ===")
        print(sdf.to_string(index=False))
    print(f"\n[done] wrote {args.out_prefix}_per_file.csv "
          f"(+ _per_dataset.csv / _per_species.csv if present)")


if __name__ == "__main__":
    main()
