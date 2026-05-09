#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Flatten and shuffle macrogene parquet files into training-ready chunks.

This script is the final data layout step after macrogene round merging. It
recursively reads species-organized parquet files such as:

  merged_by_species/
    Homo_sapiens/macrogene_0.parquet
    Mus_musculus/macrogene_0.parquet

and writes a flat directory that train_MNodes_torchrun_mfu_preindexparquet.py
can consume directly:

  all_flatten_data/
    all_flatten_part_0.parquet
    all_flatten_part_1.parquet

The implementation intentionally follows shuffle_species.py: load parquet data
into pandas, shuffle rows globally, then split into fixed-size parquet chunks.
"""

from __future__ import annotations

import argparse
import csv
import gc
import math
import multiprocessing as mp
import os
import re
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.parquet as pq

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


SCHEMA = pa.schema([
    ("X", pa.list_(pa.float64())),
    ("soma_joinid", pa.int64()),
    ("dataset_id", pa.int64()),
    ("assay", pa.int64()),
    ("cell_type", pa.int64()),
    ("development_stage", pa.int64()),
    ("disease", pa.int64()),
    ("tissue", pa.int64()),
    ("sex", pa.int64()),
    ("tech_sample", pa.int64()),
    ("species", pa.int64()),
    ("idx", pa.int64()),
])
SCHEMA_COLUMNS = [field.name for field in SCHEMA]
SHUFFLE_KEY_COLUMN = "__shuffle_key"
TEMP_SCHEMA = SCHEMA.append(pa.field(SHUFFLE_KEY_COLUMN, pa.float64()))


def log_step(step_name: str, start_time: float) -> None:
    mem_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
    elapsed = time.time() - start_time
    print(f"[{step_name}] elapsed={elapsed:.2f}s memory={mem_mb:.2f}MB", flush=True)


def release_unused_memory() -> None:
    gc.collect()
    try:
        pa.default_memory_pool().release_unused()
    except Exception:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Flatten species-organized macrogene parquet files, shuffle rows globally, and write training chunks."
    )
    parser.add_argument(
        "--input-dir",
        nargs="+",
        required=True,
        help="One or more input directories/files. Directories are searched recursively by --pattern.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Flat output directory for all_flatten_part_*.parquet files.",
    )
    parser.add_argument(
        "--pattern",
        default="macrogene_*.parquet",
        help="Recursive parquet filename pattern to read from input directories. Default: macrogene_*.parquet",
    )
    parser.add_argument(
        "--rows-per-file",
        type=int,
        default=16384,
        help="Number of rows per output parquet file. Default: 16384",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for pandas row shuffle. Default: 42",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(16, os.cpu_count() or 8),
        help="Parallel parquet read / metadata workers. Default: min(16, cpu_count)",
    )
    parser.add_argument(
        "--compression",
        default="snappy",
        help="Output parquet compression. Default: snappy",
    )
    parser.add_argument(
        "--output-prefix",
        default="all_flatten_part_",
        help="Output parquet filename prefix. Default: all_flatten_part_",
    )
    parser.add_argument(
        "--manifest-name",
        default="shuffle_manifest.csv",
        help="Output manifest CSV filename. Default: shuffle_manifest.csv",
    )
    parser.add_argument(
        "--shuffle-mode",
        choices=["in-memory", "external", "batch"],
        default="in-memory",
        help=(
            "Shuffle implementation. in-memory preserves the original pandas "
            "full-load behavior. external uses a disk-backed random-key bucket "
            "shuffle for full datasets that do not fit in RAM. batch randomizes "
            "input file order, loads file batches in parallel, shuffles each "
            "batch in memory, and streams output chunks. Default: in-memory"
        ),
    )
    parser.add_argument(
        "--batch-files",
        type=int,
        default=2048,
        help=(
            "Number of input parquet files to load per batch for "
            "--shuffle-mode batch. Increase on high-RAM nodes. Default: 2048"
        ),
    )
    parser.add_argument(
        "--shuffle-buckets",
        type=int,
        default=16,
        help=(
            "Number of disk buckets for --shuffle-mode external. Default: 16. "
            "Each bucket is read into RAM as one pyarrow Table during the write "
            "phase, so total_rows / buckets * row_size must fit (peak ~2x during "
            "sort)."
        ),
    )
    parser.add_argument(
        "--pyarrow-threads-per-worker",
        type=int,
        default=4,
        help=(
            "pyarrow CPU/IO threads per partition worker process for "
            "--shuffle-mode external. Total threads = --workers * this. Default: 4"
        ),
    )
    parser.add_argument(
        "--temp-dir",
        default=None,
        help=(
            "Temporary directory for --shuffle-mode external. Default: "
            "<output-dir>/_shuffle_tmp"
        ),
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep external-shuffle temporary bucket files after success.",
    )
    parser.add_argument(
        "--skip-partition-if-exists",
        action="store_true",
        help=(
            "If --temp-dir already contains bucket_*_chunk_*.parquet files from "
            "a previous successful partition phase, skip the partition phase and "
            "go straight to the write phase. Useful for recovering from a write "
            "phase failure without redoing the partition."
        ),
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Use only the first N matched input files, useful for smoke tests. Default: 0 means all files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing output parquet files with the selected prefix before writing.",
    )
    parser.add_argument(
        "--drop-remainder",
        dest="drop_remainder",
        action="store_true",
        default=True,
        help=(
            "Drop trailing rows that do not fill a complete --rows-per-file output "
            "parquet file. This is the default behavior."
        ),
    )
    parser.add_argument(
        "--keep-remainder",
        dest="drop_remainder",
        action="store_false",
        help="Keep the final partial output parquet file instead of dropping it.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only scan files and count rows from parquet metadata; do not read full data or write outputs.",
    )
    parser.add_argument(
        "--validate-all-schemas",
        action="store_true",
        help="Validate required columns on every input file. By default only the first file is checked.",
    )
    return parser.parse_args()


def iter_input_files(input_paths: Iterable[str], pattern: str) -> List[Path]:
    files: List[Path] = []
    for raw_path in input_paths:
        path = Path(raw_path).expanduser().resolve()
        if path.is_file():
            if path.match(pattern) or pattern == "*.parquet":
                files.append(path)
            continue
        if not path.exists():
            raise FileNotFoundError(f"Input path does not exist: {path}")
        if not path.is_dir():
            raise NotADirectoryError(f"Input path is not a directory or file: {path}")
        files.extend(p.resolve() for p in path.rglob(pattern) if p.is_file())

    unique_files = sorted(set(files))
    if not unique_files:
        raise FileNotFoundError(f"No input files matched pattern {pattern!r}")
    return unique_files


def validate_schema(file_path: Path) -> None:
    schema = pq.read_schema(file_path)
    names = set(schema.names)
    missing = [name for name in SCHEMA_COLUMNS if name not in names]
    if missing:
        raise ValueError(
            f"Input file is missing required columns: {file_path}\n"
            f"Missing columns: {missing}\n"
            f"Found columns: {schema.names}"
        )


def validate_input_schemas(files: List[Path], validate_all: bool, workers: int) -> None:
    if validate_all:
        run_parallel(validate_schema, files, workers, desc="validate schema")
    else:
        validate_schema(files[0])


def parquet_num_rows(file_path: Path) -> int:
    return pq.ParquetFile(file_path).metadata.num_rows


def read_parquet_file(file_path: Path) -> pd.DataFrame:
    return pd.read_parquet(file_path, engine="pyarrow", columns=SCHEMA_COLUMNS)


def progress_iter(iterable, total: int, desc: str):
    if tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc, unit="file")


def run_parallel(fn, files: List[Path], workers: int, desc: str):
    if workers <= 1:
        return [fn(path) for path in progress_iter(files, len(files), desc)]

    results = [None] * len(files)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_index = {executor.submit(fn, path): i for i, path in enumerate(files)}
        for future in progress_iter(as_completed(future_to_index), len(files), desc):
            index = future_to_index[future]
            results[index] = future.result()
    return results


def count_rows(files: List[Path], workers: int) -> int:
    row_counts = run_parallel(parquet_num_rows, files, workers, desc="count rows")
    return int(sum(row_counts))


def read_all_data(files: List[Path], workers: int) -> pd.DataFrame:
    start = time.time()
    print(f"[INFO] Reading {len(files)} parquet files with workers={workers}", flush=True)
    frames = run_parallel(read_parquet_file, files, workers, desc="read parquet")
    df = pd.concat(frames, ignore_index=True, copy=False)
    log_step("Load Parquet with Pandas", start)
    return df


def prepare_output_dir(output_dir: Path, output_prefix: str, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_outputs = sorted(output_dir.glob(f"{output_prefix}*.parquet"))
    if existing_outputs and not overwrite:
        raise FileExistsError(
            f"Output directory already contains {len(existing_outputs)} files matching "
            f"{output_prefix}*.parquet: {output_dir}\n"
            "Use --overwrite or choose a fresh output directory."
        )
    if overwrite:
        for path in existing_outputs:
            path.unlink()


def prepare_temp_dir(temp_dir: Path, overwrite: bool) -> None:
    if temp_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Temporary directory already exists: {temp_dir}\n"
                "Use --overwrite or choose a fresh --temp-dir."
            )
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)


def write_manifest(manifest_path: Path, rows: List[dict]) -> None:
    with manifest_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=["output_file", "start_row", "end_row_exclusive", "num_rows"],
        )
        writer.writeheader()
        writer.writerows(rows)


def chunk_counts(total_rows: int, rows_per_file: int, drop_remainder: bool) -> tuple[int, int, int]:
    if drop_remainder:
        usable_rows = (total_rows // rows_per_file) * rows_per_file
    else:
        usable_rows = total_rows
    dropped_rows = total_rows - usable_rows
    num_files = math.ceil(usable_rows / rows_per_file) if usable_rows else 0
    return usable_rows, dropped_rows, num_files


def write_shuffled_chunks(
    df: pd.DataFrame,
    output_dir: Path,
    rows_per_file: int,
    output_prefix: str,
    compression: str,
    manifest_name: str,
    drop_remainder: bool,
) -> None:
    start = time.time()
    total_rows = len(df)
    usable_rows, dropped_rows, num_files = chunk_counts(
        total_rows=total_rows,
        rows_per_file=rows_per_file,
        drop_remainder=drop_remainder,
    )
    print(
        f"[INFO] total rows={total_rows}, usable rows={usable_rows}, "
        f"dropped rows={dropped_rows}, output files={num_files}",
        flush=True,
    )

    manifest_rows: List[dict] = []
    iterator = range(num_files)
    if tqdm is not None:
        iterator = tqdm(iterator, desc="write parquet", unit="file")

    for i in iterator:
        start_row = i * rows_per_file
        end_row = min((i + 1) * rows_per_file, usable_rows)
        chunk = df.iloc[start_row:end_row]
        output_name = f"{output_prefix}{i}.parquet"
        output_path = output_dir / output_name
        table = pa.Table.from_pandas(chunk, schema=SCHEMA, preserve_index=False)
        pq.write_table(table, output_path, compression=compression)
        manifest_rows.append({
            "output_file": output_name,
            "start_row": start_row,
            "end_row_exclusive": end_row,
            "num_rows": end_row - start_row,
        })

    write_manifest(output_dir / manifest_name, manifest_rows)
    log_step("Write shuffled parquet chunks", start)


def write_output_chunk(
    chunk: pd.DataFrame,
    output_dir: Path,
    output_prefix: str,
    output_index: int,
    compression: str,
) -> str:
    output_name = f"{output_prefix}{output_index}.parquet"
    output_path = output_dir / output_name
    chunk = chunk[SCHEMA_COLUMNS]
    table = pa.Table.from_pandas(chunk, schema=SCHEMA, preserve_index=False)
    pq.write_table(table, output_path, compression=compression)
    return output_name


def _set_pyarrow_threads_init(threads: int) -> None:
    """ProcessPoolExecutor initializer; bound pyarrow CPU/IO threads per worker.

    Total system threads = workers * threads. Default 32 * 4 = 128 leaves headroom
    on a 192-core box.
    """
    pa.set_cpu_count(threads)
    pa.set_io_thread_count(threads)


def _partition_chunk_for_external(
    files: List[str],
    file_indices: List[int],
    chunk_index: int,
    seed: int,
    buckets: int,
    temp_dir: str,
) -> int:
    """Process pool worker.

    Reads a slice of input parquet files and partitions rows into per-(bucket,
    chunk) parquet files written under ``temp_dir`` as
    ``bucket_{bid:05d}_chunk_{chunk_index:05d}.parquet``. Each worker owns its
    own writers, so workers never contend on output files.

    Stays in pyarrow throughout — never converts to pandas. The list<float64>
    X column is expensive to round-trip through pandas object dtype.
    """
    temp_dir_path = Path(temp_dir)
    writers: Dict[int, pq.ParquetWriter] = {}
    rows_total = 0
    try:
        for file_idx, file_path_str in zip(file_indices, files):
            table = pq.read_table(Path(file_path_str), columns=SCHEMA_COLUMNS)
            n = table.num_rows
            if n == 0:
                continue

            file_rng = np.random.default_rng([seed, file_idx])
            keys = file_rng.random(n, dtype=np.float64)
            bucket_ids = np.minimum((keys * buckets).astype(np.int64), buckets - 1)

            table = table.append_column(SHUFFLE_KEY_COLUMN, pa.array(keys))

            sort_order = np.argsort(bucket_ids, kind="stable")
            sorted_bucket_ids = bucket_ids[sort_order]
            sorted_table = table.take(pa.array(sort_order))

            boundaries = np.searchsorted(sorted_bucket_ids, np.arange(buckets + 1))

            for bucket_id in range(buckets):
                start = int(boundaries[bucket_id])
                end = int(boundaries[bucket_id + 1])
                if end <= start:
                    continue
                bucket_table = sorted_table.slice(start, end - start)
                writer = writers.get(bucket_id)
                if writer is None:
                    bucket_path = (
                        temp_dir_path
                        / f"bucket_{bucket_id:05d}_chunk_{chunk_index:05d}.parquet"
                    )
                    writer = pq.ParquetWriter(bucket_path, TEMP_SCHEMA, compression="snappy")
                    writers[bucket_id] = writer
                writer.write_table(bucket_table)
                rows_total += end - start
    finally:
        for writer in writers.values():
            writer.close()
    return rows_total


def partition_external_shuffle_buckets(
    files: List[Path],
    temp_dir: Path,
    buckets: int,
    seed: int,
    workers: int = 1,
    pyarrow_threads_per_worker: int = 4,
) -> None:
    start = time.time()
    n_chunks = max(1, min(workers, len(files)))

    file_chunks: List[List[str]] = [[] for _ in range(n_chunks)]
    index_chunks: List[List[int]] = [[] for _ in range(n_chunks)]
    for i, fp in enumerate(files):
        c = i % n_chunks
        file_chunks[c].append(str(fp))
        index_chunks[c].append(i)

    print(
        f"[INFO] External partition (parallel): files={len(files)} buckets={buckets} "
        f"workers={workers} chunks={n_chunks} pyarrow_threads={pyarrow_threads_per_worker}",
        flush=True,
    )

    if workers <= 1:
        _set_pyarrow_threads_init(pyarrow_threads_per_worker)
        iterator = range(n_chunks)
        if tqdm is not None:
            iterator = tqdm(
                iterator, total=n_chunks, desc="partition shuffle (chunks)", unit="chunk"
            )
        for c in iterator:
            _partition_chunk_for_external(
                file_chunks[c], index_chunks[c], c, seed, buckets, str(temp_dir),
            )
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=ctx,
            initializer=_set_pyarrow_threads_init,
            initargs=(pyarrow_threads_per_worker,),
        ) as executor:
            futures = [
                executor.submit(
                    _partition_chunk_for_external,
                    file_chunks[c], index_chunks[c], c, seed, buckets, str(temp_dir),
                )
                for c in range(n_chunks)
            ]
            completed_iter = as_completed(futures)
            if tqdm is not None:
                completed_iter = tqdm(
                    completed_iter,
                    total=len(futures),
                    desc="partition shuffle (chunks)",
                    unit="chunk",
                )
            for future in completed_iter:
                future.result()

    log_step("Partition external shuffle buckets", start)


def _list_bucket_chunks(temp_dir: Path) -> Dict[int, List[Path]]:
    """Group ``bucket_{bid}_chunk_{cid}.parquet`` files under temp_dir by bucket id."""
    pattern = re.compile(r"^bucket_(\d+)_chunk_\d+\.parquet$")
    result: Dict[int, List[Path]] = {}
    for path in temp_dir.iterdir():
        match = pattern.match(path.name)
        if not match:
            continue
        bid = int(match.group(1))
        result.setdefault(bid, []).append(path)
    for bid in result:
        result[bid].sort()
    return result


def _safe_take_chunk_rows(table: pa.Table, target_values: int = 1_500_000_000) -> int:
    """Compute safe rows-per-take to avoid int32 offset overflow on list columns.

    pa.list_<T> uses int32 offsets, capping the values buffer at 2^31 ≈ 2.15B
    elements. For ``X: list<float64>`` with average list length L, ``table.take``
    of N rows produces ~N*L values. We pick N so N*L stays under
    ``target_values`` (default 1.5B; ~30% margin under 2^31).
    """
    max_avg_len = 0.0
    for field in table.schema:
        if pa.types.is_list(field.type):
            col = table.column(field.name)
            n_rows = col.length()
            if n_rows == 0:
                continue
            total_values = sum(len(chunk.values) for chunk in col.chunks)
            avg = total_values / n_rows
            if avg > max_avg_len:
                max_avg_len = avg
    if max_avg_len <= 0:
        return table.num_rows
    return max(1, min(int(target_values / max_avg_len), table.num_rows))


def _read_concat_sort_bucket_arrow(chunk_paths: List[Path]) -> Optional[pa.Table]:
    """Read all chunks for one bucket, concat, sort by random key, drop key column.

    The take is done in sub-batches sized by ``_safe_take_chunk_rows`` to avoid
    int32 offset overflow on the X: list<float64> column. Sub-results are
    glued back as a ChunkedArray-based Table (preserves chunk boundaries —
    each chunk independently fits in int32 offsets).

    Returns a pyarrow Table whose rows are in random order (a quantile-band slice
    of the global random-key shuffle). Returns None if the bucket is empty.
    """
    if not chunk_paths:
        return None
    tables = [pq.read_table(p) for p in chunk_paths]
    table = pa.concat_tables(tables)
    del tables
    if table.num_rows == 0:
        return None

    keys = table.column(SHUFFLE_KEY_COLUMN).combine_chunks().to_numpy(zero_copy_only=False)
    indices = np.argsort(keys, kind="stable")
    del keys

    table_no_key = table.drop_columns([SHUFFLE_KEY_COLUMN])
    del table

    take_rows = _safe_take_chunk_rows(table_no_key)
    n = len(indices)
    sub_tables: List[pa.Table] = []
    for i in range(0, n, take_rows):
        sub_idx = pa.array(indices[i:i + take_rows])
        sub_tables.append(table_no_key.take(sub_idx))
    del table_no_key, indices

    return pa.concat_tables(sub_tables)


def _materialize_table_copy(table: pa.Table) -> pa.Table:
    """Force a deep copy via Arrow IPC.

    Sliced pyarrow Tables share buffers with the parent — a small ``pending``
    slice can keep a 100GB+ parent alive across iterations. IPC round-trip
    decouples the buffers so the parent can be released.
    """
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return pa.ipc.open_stream(sink.getvalue()).read_all()


def write_external_shuffled_chunks(
    temp_dir: Path,
    output_dir: Path,
    rows_per_file: int,
    output_prefix: str,
    compression: str,
    manifest_name: str,
    drop_remainder: bool,
    workers: int = 1,
) -> None:
    start = time.time()
    output_index = 0
    rows_seen = 0
    rows_written = 0
    manifest_rows: List[dict] = []
    pending: Optional[pa.Table] = None

    bucket_chunks = _list_bucket_chunks(temp_dir)
    if not bucket_chunks:
        raise FileNotFoundError(
            f"No partition chunk files (bucket_*_chunk_*.parquet) found under {temp_dir}"
        )

    print(
        f"[INFO] External write: buckets={len(bucket_chunks)} workers={workers}",
        flush=True,
    )

    bucket_iter: Iterable[Tuple[int, List[Path]]] = sorted(bucket_chunks.items())
    if tqdm is not None:
        bucket_iter = tqdm(
            bucket_iter,
            total=len(bucket_chunks),
            desc="write shuffled buckets",
            unit="bucket",
        )

    for bucket_id, chunk_paths in bucket_iter:
        table = _read_concat_sort_bucket_arrow(chunk_paths)
        if table is None:
            continue
        rows_seen += table.num_rows

        if pending is not None:
            table = pa.concat_tables([pending, table])
            pending = None

        n = table.num_rows
        full_rows = (n // rows_per_file) * rows_per_file
        chunks_in_bucket = full_rows // rows_per_file

        chunk_args: List[Tuple[pa.Table, Path]] = []
        for i in range(chunks_in_bucket):
            start_row = i * rows_per_file
            chunk_table = table.slice(start_row, rows_per_file)
            output_name = f"{output_prefix}{output_index}.parquet"
            output_path = output_dir / output_name
            chunk_args.append((chunk_table, output_path))
            manifest_rows.append({
                "output_file": output_name,
                "start_row": rows_written,
                "end_row_exclusive": rows_written + rows_per_file,
                "num_rows": rows_per_file,
            })
            rows_written += rows_per_file
            output_index += 1

        if chunk_args:
            if workers <= 1:
                for ct, op in chunk_args:
                    pq.write_table(ct, op, compression=compression)
            else:
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = [
                        executor.submit(pq.write_table, ct, op, compression=compression)
                        for ct, op in chunk_args
                    ]
                    for future in futures:
                        future.result()

        if full_rows < n:
            pending = _materialize_table_copy(table.slice(full_rows, n - full_rows))
        del table, chunk_args
        release_unused_memory()

    dropped_rows = pending.num_rows if (pending is not None and drop_remainder) else 0
    if not drop_remainder and pending is not None and pending.num_rows > 0:
        output_name = f"{output_prefix}{output_index}.parquet"
        output_path = output_dir / output_name
        pq.write_table(pending, output_path, compression=compression)
        manifest_rows.append({
            "output_file": output_name,
            "start_row": rows_written,
            "end_row_exclusive": rows_written + pending.num_rows,
            "num_rows": pending.num_rows,
        })
        rows_written += pending.num_rows
        output_index += 1

    print(
        f"[INFO] total rows={rows_seen}, usable rows={rows_written}, "
        f"dropped rows={dropped_rows}, output files={output_index}",
        flush=True,
    )
    write_manifest(output_dir / manifest_name, manifest_rows)
    log_step("Write external shuffled parquet chunks", start)


def write_batch_shuffled_chunks(
    files: List[Path],
    output_dir: Path,
    rows_per_file: int,
    output_prefix: str,
    compression: str,
    manifest_name: str,
    drop_remainder: bool,
    seed: int,
    workers: int,
    batch_files: int,
) -> None:
    if batch_files <= 0:
        raise ValueError("--batch-files must be positive")

    start = time.time()
    rng = np.random.default_rng(seed)
    file_indices = rng.permutation(len(files)).tolist()
    shuffled_files = [files[i] for i in file_indices]

    output_index = 0
    rows_seen = 0
    rows_written = 0
    manifest_rows: List[dict] = []
    pending = pd.DataFrame(columns=SCHEMA_COLUMNS)
    num_batches = math.ceil(len(shuffled_files) / batch_files)

    print(
        f"[INFO] Batch shuffle: input files={len(files)}, batch_files={batch_files}, "
        f"batches={num_batches}, workers={workers}, seed={seed}",
        flush=True,
    )

    for batch_index in range(num_batches):
        batch_start = batch_index * batch_files
        batch_end = min(batch_start + batch_files, len(shuffled_files))
        batch_paths = shuffled_files[batch_start:batch_end]

        batch_timer = time.time()
        print(
            f"[INFO] Batch {batch_index + 1}/{num_batches}: "
            f"reading files {batch_start}:{batch_end}",
            flush=True,
        )
        df = read_all_data(batch_paths, workers)
        rows_seen += len(df)

        if not pending.empty:
            df = pd.concat([pending, df], ignore_index=True, copy=False)
            pending = pd.DataFrame(columns=SCHEMA_COLUMNS)

        shuffle_seed = int((seed + batch_index) % (2**32 - 1))
        print(
            f"[INFO] Batch {batch_index + 1}/{num_batches}: "
            f"shuffling rows={len(df)} seed={shuffle_seed}",
            flush=True,
        )
        df = df.sample(frac=1, random_state=shuffle_seed).reset_index(drop=True)

        full_rows = (len(df) // rows_per_file) * rows_per_file
        iterator = range(0, full_rows, rows_per_file)
        if tqdm is not None:
            iterator = tqdm(iterator, desc=f"write batch {batch_index + 1}", unit="chunk")
        for start_row in iterator:
            end_row = start_row + rows_per_file
            chunk = df.iloc[start_row:end_row].copy()
            output_name = write_output_chunk(
                chunk=chunk,
                output_dir=output_dir,
                output_prefix=output_prefix,
                output_index=output_index,
                compression=compression,
            )
            manifest_rows.append({
                "output_file": output_name,
                "start_row": rows_written,
                "end_row_exclusive": rows_written + len(chunk),
                "num_rows": len(chunk),
            })
            rows_written += len(chunk)
            output_index += 1
            del chunk

        if full_rows < len(df):
            pending = df.iloc[full_rows:].copy()

        del df
        release_unused_memory()
        log_step(f"Batch {batch_index + 1}/{num_batches}", batch_timer)

    dropped_rows = len(pending) if drop_remainder else 0
    if not drop_remainder and not pending.empty:
        output_name = write_output_chunk(
            chunk=pending,
            output_dir=output_dir,
            output_prefix=output_prefix,
            output_index=output_index,
            compression=compression,
        )
        manifest_rows.append({
            "output_file": output_name,
            "start_row": rows_written,
            "end_row_exclusive": rows_written + len(pending),
            "num_rows": len(pending),
        })
        rows_written += len(pending)
        output_index += 1

    print(
        f"[INFO] total rows={rows_seen}, usable rows={rows_written}, "
        f"dropped rows={dropped_rows}, output files={output_index}",
        flush=True,
    )
    write_manifest(output_dir / manifest_name, manifest_rows)
    log_step("Write batch shuffled parquet chunks", start)


def external_shuffle(
    files: List[Path],
    output_dir: Path,
    rows_per_file: int,
    output_prefix: str,
    compression: str,
    manifest_name: str,
    drop_remainder: bool,
    seed: int,
    buckets: int,
    temp_dir: Path,
    keep_temp: bool,
    overwrite: bool,
    workers: int = 1,
    pyarrow_threads_per_worker: int = 4,
    skip_partition_if_exists: bool = False,
) -> None:
    if buckets <= 0:
        raise ValueError("--shuffle-buckets must be positive")

    existing_bucket_chunks: Dict[int, List[Path]] = {}
    if skip_partition_if_exists and temp_dir.exists():
        existing_bucket_chunks = _list_bucket_chunks(temp_dir)
    existing_chunks = bool(existing_bucket_chunks)
    if existing_chunks and len(existing_bucket_chunks) != buckets:
        bucket_ids = sorted(existing_bucket_chunks)
        raise ValueError(
            f"Existing partition data under {temp_dir} contains "
            f"{len(existing_bucket_chunks)} bucket(s) "
            f"(id range {bucket_ids[0]}..{bucket_ids[-1]}), but "
            f"--shuffle-buckets={buckets}. This usually means _shuffle_tmp "
            f"was created by a different SHUFFLE_BUCKETS value or is incomplete. "
            f"Rerun without --skip-partition-if-exists, remove the temp dir, "
            f"or set --shuffle-buckets to match the existing partition if intentional."
        )
    if existing_chunks:
        print(
            f"[INFO] External shuffle: reusing existing partition data under {temp_dir} "
            f"(--skip-partition-if-exists). Skipping partition phase.",
            flush=True,
        )
    else:
        prepare_temp_dir(temp_dir, overwrite=overwrite)

    print(
        f"[INFO] External shuffle: buckets={buckets}, temp_dir={temp_dir}, seed={seed}, "
        f"workers={workers}, pyarrow_threads_per_worker={pyarrow_threads_per_worker}",
        flush=True,
    )
    if not existing_chunks:
        partition_external_shuffle_buckets(
            files=files,
            temp_dir=temp_dir,
            buckets=buckets,
            seed=seed,
            workers=workers,
            pyarrow_threads_per_worker=pyarrow_threads_per_worker,
        )
    write_external_shuffled_chunks(
        temp_dir=temp_dir,
        output_dir=output_dir,
        rows_per_file=rows_per_file,
        output_prefix=output_prefix,
        compression=compression,
        manifest_name=manifest_name,
        drop_remainder=drop_remainder,
        workers=workers,
    )
    if not keep_temp:
        shutil.rmtree(temp_dir)


def main() -> None:
    args = parse_args()
    if args.rows_per_file <= 0:
        raise ValueError("--rows-per-file must be positive")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")

    output_dir = Path(args.output_dir).expanduser().resolve()

    scan_start = time.time()
    files = iter_input_files(args.input_dir, args.pattern)
    if args.max_files:
        files = files[:args.max_files]
    print(f"[INFO] matched input files={len(files)}", flush=True)
    print(f"[INFO] first input file={files[0]}", flush=True)
    print(f"[INFO] output dir={output_dir}", flush=True)
    log_step("Scan input files", scan_start)

    validate_input_schemas(files, args.validate_all_schemas, args.workers)

    if args.dry_run:
        rows = count_rows(files, args.workers)
        usable_rows, dropped_rows, num_files = chunk_counts(
            total_rows=rows,
            rows_per_file=args.rows_per_file,
            drop_remainder=args.drop_remainder,
        )
        print(f"[DRY-RUN] input files={len(files)}", flush=True)
        print(f"[DRY-RUN] total rows={rows}", flush=True)
        print(f"[DRY-RUN] usable rows={usable_rows}", flush=True)
        print(f"[DRY-RUN] dropped rows={dropped_rows}", flush=True)
        print(f"[DRY-RUN] output files={num_files}", flush=True)
        return

    prepare_output_dir(output_dir, args.output_prefix, args.overwrite)

    if args.shuffle_mode == "batch":
        write_batch_shuffled_chunks(
            files=files,
            output_dir=output_dir,
            rows_per_file=args.rows_per_file,
            output_prefix=args.output_prefix,
            compression=args.compression,
            manifest_name=args.manifest_name,
            drop_remainder=args.drop_remainder,
            seed=args.seed,
            workers=args.workers,
            batch_files=args.batch_files,
        )
    elif args.shuffle_mode == "external":
        temp_dir = (
            Path(args.temp_dir).expanduser().resolve()
            if args.temp_dir is not None
            else output_dir / "_shuffle_tmp"
        )
        external_shuffle(
            files=files,
            output_dir=output_dir,
            rows_per_file=args.rows_per_file,
            output_prefix=args.output_prefix,
            compression=args.compression,
            manifest_name=args.manifest_name,
            drop_remainder=args.drop_remainder,
            seed=args.seed,
            buckets=args.shuffle_buckets,
            temp_dir=temp_dir,
            keep_temp=args.keep_temp,
            overwrite=args.overwrite,
            workers=args.workers,
            pyarrow_threads_per_worker=args.pyarrow_threads_per_worker,
            skip_partition_if_exists=args.skip_partition_if_exists,
        )
    else:
        df = read_all_data(files, args.workers)

        shuffle_start = time.time()
        print(f"[INFO] Shuffling rows with seed={args.seed}", flush=True)
        df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
        log_step("Shuffle rows", shuffle_start)

        write_shuffled_chunks(
            df=df,
            output_dir=output_dir,
            rows_per_file=args.rows_per_file,
            output_prefix=args.output_prefix,
            compression=args.compression,
            manifest_name=args.manifest_name,
            drop_remainder=args.drop_remainder,
        )
    print(f"[DONE] Shuffled data written to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
