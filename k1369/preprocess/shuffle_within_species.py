#!/usr/bin/env python3
"""Shuffle cells across sources within each species, preserving every row.

Run beside shuffle_species.py (shared disk-bucket and output-writer helpers).
Only macrogene_*.parquet are cell inputs; lookup files are copied separately.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from shuffle_species import BucketWriterPool, FixedSizeOutputWriter, log, split_into_buckets

DEFAULT_INPUT = Path(__file__).resolve().parent / '2nd_pretrain_data_preprocessed_step4_K1500_map_then_log1p'
SOURCE_COLUMNS = ['dataset_id', 'tissue', 'tech_sample']


def source_counts(table: pa.Table) -> Counter:
    return Counter(zip(*(table[c].to_pylist() for c in SOURCE_COLUMNS)))


def inspect_species(folder: Path):
    files = sorted(folder.glob('macrogene_*.parquet'))
    if not files:
        raise ValueError(f'No cell parquet files in {folder}')
    schema = None
    rows = 0
    for path in files:
        with pq.ParquetFile(path) as reader:
            current = reader.schema_arrow
            if schema is None:
                schema = current
                missing = set(SOURCE_COLUMNS + ['X']) - set(schema.names)
                if missing:
                    raise ValueError(f'{path}: missing columns {sorted(missing)}')
            if not current.equals(schema, check_metadata=False):
                raise ValueError(f'Schema mismatch: {path}')
            rows += reader.metadata.num_rows
    return files, schema, rows


def run_species(name, info, args):
    files, schema, expected_rows = info
    destination = args.output_dir / name
    if destination.exists():
        raise FileExistsError(f'Refusing to overwrite {destination}')
    # Stable per-species seed: selecting one species does not change its result.
    digest = hashlib.sha256(name.encode('utf-8')).digest()
    seed = np.random.SeedSequence([args.seed, int.from_bytes(digest[:8], 'little')])
    assign_seed, shuffle_seed = seed.spawn(2)
    rng = np.random.default_rng(assign_seed)
    # Small species do not need hundreds of tiny temporary buckets.
    buckets = min(args.num_buckets, max(1, math.ceil(expected_rows / 32768)))
    bucket_rngs = [np.random.default_rng(s) for s in shuffle_seed.spawn(buckets)]
    staging = Path(tempfile.mkdtemp(prefix=f'.{name}.partial-', dir=args.output_dir))
    temp = Path(tempfile.mkdtemp(prefix=f'{name}.buckets-', dir=args.temp_dir))
    log(f'{name}: {len(files)} input files, {expected_rows:,} cells, {buckets} buckets')
    try:
        pool = BucketWriterPool(temp, schema, buckets, args.bucket_buffer_rows, args.compression)
        before = Counter()
        input_rows = 0
        try:
            for i, path in enumerate(files, 1):
                with pq.ParquetFile(path) as reader:
                    for batch in reader.iter_batches(batch_size=args.read_batch_rows):
                        table = pa.Table.from_batches([batch]).replace_schema_metadata(schema.metadata)
                        before.update(source_counts(table))
                        ids = rng.integers(0, buckets, size=table.num_rows, dtype=np.uint32)
                        split_into_buckets(table, ids, pool)
                        input_rows += table.num_rows
                if i % 10 == 0 or i == len(files):
                    log(f'{name}: stage 1 {i}/{len(files)} files; {input_rows:,} cells')
        finally:
            pool.close()
        if input_rows != expected_rows or int(pool.bucket_sizes.sum()) != expected_rows:
            raise RuntimeError(f'{name}: input/bucket row-count mismatch')

        writer = FixedSizeOutputWriter(staging, args.rows_per_file, args.compression)
        after = Counter()
        for bucket_id, count in enumerate(pool.bucket_sizes):
            if not count:  # Empty buckets are valid, including empty input datasets.
                continue
            path = pool.path(bucket_id)
            with pq.ParquetFile(path) as reader:
                table = reader.read()
            if table.num_rows != int(count):
                raise RuntimeError(f'{name}: bucket {bucket_id} row-count mismatch')
            order = bucket_rngs[bucket_id].permutation(table.num_rows)
            shuffled = table.take(pa.array(order, type=pa.int64()))
            after.update(source_counts(shuffled))
            writer.append(shuffled)
            del shuffled, table, order
            path.unlink()
            if (bucket_id + 1) % 10 == 0 or bucket_id + 1 == buckets:
                log(f'{name}: stage 2 {bucket_id + 1}/{buckets} buckets')
        writer.finish('keep')
        if writer.rows_written != expected_rows or before != after:
            raise RuntimeError(f'{name}: cells or source-group counts changed')

        # Verify actual output footers and enforce exactly one possible short tail.
        outputs = sorted(staging.glob('shuffled_part_*.parquet'))
        if len(outputs) != math.ceil(expected_rows / args.rows_per_file):
            raise RuntimeError(f'{name}: output file-count mismatch')
        for i, path in enumerate(outputs):
            expected = min(args.rows_per_file, expected_rows - i * args.rows_per_file)
            with pq.ParquetFile(path) as reader:
                if reader.metadata.num_rows != expected:
                    raise RuntimeError(f'Wrong output size: {path}')
            path.rename(staging / f'macrogene_{i:06d}.parquet')

        # Preserve the macrogene axis and transformation provenance.
        for pattern in ('*_macrogene_lookup.parquet', '*.version'):
            for path in (args.input_path / name).glob(pattern):
                shutil.copy2(path, staging / path.name)
        metadata = {
            'species': name, 'algorithm': 'per_species_global_random_bucket_shuffle',
            'source_columns': SOURCE_COLUMNS, 'source_groups': len(before),
            'source_group_counts_verified': True,
            'input_path': str((args.input_path / name).resolve()),
            'input_files': len(files), 'input_rows': expected_rows,
            'output_files': len(outputs), 'output_rows': writer.rows_written,
            'rows_per_file': args.rows_per_file, 'dropped_rows': 0,
            'seed': args.seed, 'num_buckets': buckets,
            'read_batch_rows': args.read_batch_rows,
            'bucket_buffer_rows': args.bucket_buffer_rows,
            'compression': args.compression,
            'numpy_version': np.__version__, 'pyarrow_version': pa.__version__,
        }
        (staging / 'shuffle_metadata.json').write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        temp.rmdir()
        staging.rename(destination)
        log(f'{name}: complete; {writer.rows_written:,} cells in {len(outputs)} files')
    except BaseException:
        log(f'{name}: FAILED; unfinished output: {staging}; temporary buckets: {temp}')
        raise


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-path', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_INPUT.with_name(DEFAULT_INPUT.name + '_shuffle_within_species'))
    parser.add_argument('--species', nargs='+', help='Default: all species subdirectories')
    parser.add_argument('--rows-per-file', type=int, default=16384)
    parser.add_argument('--num-buckets', type=int, default=256, help='Maximum buckets per species; increase to reduce stage-2 memory')
    parser.add_argument('--read-batch-rows', type=int, default=4096)
    parser.add_argument('--bucket-buffer-rows', type=int, default=64)
    parser.add_argument('--temp-dir', type=Path, help='Temporary disk location; default: output directory')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--compression', default='snappy')
    parser.add_argument('--dry-run', action='store_true', help='Check all input schemas and print row/file counts without writing')
    args = parser.parse_args(argv)
    for key in ('rows_per_file', 'num_buckets', 'read_batch_rows', 'bucket_buffer_rows'):
        if getattr(args, key) <= 0:
            parser.error(f'--{key.replace("_", "-")} must be positive')
    if args.seed < 0:
        parser.error('--seed must be nonnegative')
    args.input_path = args.input_path.resolve()
    args.output_dir = args.output_dir.resolve()
    if not args.input_path.is_dir():
        parser.error(f'Input directory does not exist: {args.input_path}')
    if (args.output_dir == args.input_path or args.input_path in args.output_dir.parents
            or args.output_dir in args.input_path.parents):
        parser.error('Input and output directories must not overlap')
    args.temp_dir = args.temp_dir.resolve() if args.temp_dir else args.output_dir
    if args.temp_dir == args.input_path or args.input_path in args.temp_dir.parents:
        parser.error('Temporary directory must be outside input directory')
    return args


def main(argv=None):
    args = parse_args(argv)
    species = args.species or sorted(p.name for p in args.input_path.iterdir() if p.is_dir())
    if not species or len(species) != len(set(species)):
        raise ValueError('Species list must be nonempty and unique')
    plans = {}
    # Check all species before starting the expensive shuffle.
    for name in species:
        if name in ('.', '..') or Path(name).name != name:
            raise ValueError(f'Invalid species directory name: {name}')
        if (args.output_dir / name).exists():
            raise FileExistsError(f'Output already exists: {args.output_dir / name}; choose another output directory or only unfinished species')
        plans[name] = inspect_species(args.input_path / name)
        files, _, rows = plans[name]
        log(f'{name}: {len(files)} inputs, {rows:,} cells -> {math.ceil(rows / args.rows_per_file)} outputs; remainder {rows % args.rows_per_file}')
    if args.dry_run:
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.temp_dir.mkdir(parents=True, exist_ok=True)
    for name, info in plans.items():
        run_species(name, info, args)


if __name__ == '__main__':
    main()
