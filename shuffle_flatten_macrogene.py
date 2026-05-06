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
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List

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


def log_step(step_name: str, start_time: float) -> None:
    mem_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
    elapsed = time.time() - start_time
    print(f"[{step_name}] elapsed={elapsed:.2f}s memory={mem_mb:.2f}MB", flush=True)


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
