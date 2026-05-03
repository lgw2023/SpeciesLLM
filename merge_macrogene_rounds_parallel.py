#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Merge cross-species pretraining data from multiple batches by species,
with parallel transfer and progress bar.

Expected structure:
batch_root/
├── species_A/
│   ├── macrogene_0.parquet
│   ├── macrogene_1.parquet
│   └── ...
├── species_B/
│   ├── macrogene_0.parquet
│   └── ...
└── ...

Example:
python merge_macrogene_batches_parallel.py \
    --batch-dirs /path/to/1st /path/to/2nd /path/to/3rd \
    --batch-names 1st 2nd 3rd \
    --output-dir /path/to/merged_output \
    --mode copy \
    --workers 16
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


FILE_PATTERN = re.compile(r"^macrogene_(\d+)\.parquet$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge macrogene parquet files from multiple batches by species with parallel transfer."
    )
    parser.add_argument(
        "--batch-dirs",
        nargs="+",
        required=True,
        help="Input batch directories, in merge order, e.g. 1st 2nd 3rd",
    )
    parser.add_argument(
        "--batch-names",
        nargs="+",
        default=None,
        help="Optional batch names corresponding to --batch-dirs. "
             "If not provided, directory names will be used.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output root directory for merged species folders.",
    )
    parser.add_argument(
        "--mode",
        choices=["copy", "move"],
        default="copy",
        help="Transfer mode: copy or move. Default: copy",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Starting index for renamed macrogene files in each species folder. Default: 0",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(16, (os.cpu_count() or 8)),
        help="Number of parallel worker threads. Default: min(16, cpu_count)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Raise an error when encountering non-matching files inside species folders. "
             "Otherwise they are skipped.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip target files that already exist instead of raising an error.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only show what would be done, without actually copying/moving files.",
    )
    parser.add_argument(
        "--manifest-name",
        default="merge_manifest.csv",
        help="Filename of the output manifest CSV. Default: merge_manifest.csv",
    )
    return parser.parse_args()


def validate_inputs(batch_dirs: List[Path], batch_names: List[str] | None, output_dir: Path) -> List[str]:
    for batch_dir in batch_dirs:
        if not batch_dir.exists():
            raise FileNotFoundError(f"Batch directory does not exist: {batch_dir}")
        if not batch_dir.is_dir():
            raise NotADirectoryError(f"Batch path is not a directory: {batch_dir}")

    if batch_names is not None:
        if len(batch_names) != len(batch_dirs):
            raise ValueError(
                f"--batch-names must have the same length as --batch-dirs. "
                f"Got {len(batch_names)} names and {len(batch_dirs)} dirs."
            )
        final_batch_names = batch_names
    else:
        final_batch_names = [p.name for p in batch_dirs]

    output_dir.mkdir(parents=True, exist_ok=True)
    return final_batch_names


def collect_species_files(
    batch_dirs: List[Path],
    batch_names: List[str],
    strict: bool = False,
) -> Dict[str, List[Tuple[int, str, int, Path]]]:
    """
    Returns:
        species_to_records[species] = list of
        (batch_order, batch_name, original_idx, src_file_path)
    """
    species_to_records: Dict[str, List[Tuple[int, str, int, Path]]] = defaultdict(list)

    for batch_order, (batch_dir, batch_name) in enumerate(zip(batch_dirs, batch_names)):
        print(f"[INFO] Scanning batch: {batch_name} -> {batch_dir}")

        for species_dir in sorted(batch_dir.iterdir()):
            if not species_dir.is_dir():
                continue

            species_name = species_dir.name
            matched_files: List[Tuple[int, Path]] = []
            unmatched_files: List[Path] = []

            for f in species_dir.iterdir():
                if not f.is_file():
                    continue
                m = FILE_PATTERN.match(f.name)
                if m:
                    original_idx = int(m.group(1))
                    matched_files.append((original_idx, f))
                else:
                    unmatched_files.append(f)

            if unmatched_files:
                msg = (
                    f"[WARN] Found {len(unmatched_files)} non-matching files in "
                    f"{species_dir}. They will be skipped."
                )
                if strict:
                    raise ValueError(msg + f" Files: {[x.name for x in unmatched_files]}")
                else:
                    print(msg)

            matched_files.sort(key=lambda x: x[0])

            for original_idx, fpath in matched_files:
                species_to_records[species_name].append(
                    (batch_order, batch_name, original_idx, fpath)
                )

    return species_to_records


def build_transfer_plan(
    species_to_records: Dict[str, List[Tuple[int, str, int, Path]]],
    output_dir: Path,
    start_index: int = 0,
    skip_existing: bool = False,
) -> List[Dict]:
    """
    Build a complete transfer plan.
    """
    tasks: List[Dict] = []

    for species_name in sorted(species_to_records.keys()):
        records = species_to_records[species_name]
        if not records:
            continue

        species_out_dir = output_dir / species_name
        species_out_dir.mkdir(parents=True, exist_ok=True)

        # sort by batch order, then original idx, then filename
        records = sorted(records, key=lambda x: (x[0], x[2], x[3].name))

        for offset, (batch_order, batch_name, original_idx, src_path) in enumerate(records):
            new_index = start_index + offset
            new_name = f"macrogene_{new_index}.parquet"
            dst_path = species_out_dir / new_name

            if dst_path.exists() and not skip_existing:
                raise FileExistsError(
                    f"Target file already exists: {dst_path}\n"
                    f"Please use a fresh output directory or set --skip-existing."
                )

            tasks.append({
                "species": species_name,
                "new_index": new_index,
                "new_filename": new_name,
                "batch_order": batch_order,
                "batch_name": batch_name,
                "original_index": original_idx,
                "source_path": src_path,
                "target_path": dst_path,
                "skip": dst_path.exists() and skip_existing,
            })

    return tasks


def transfer_one_file(src: Path, dst: Path, mode: str, dry_run: bool = False) -> int:
    """
    Returns the file size in bytes (source size).
    """
    if dry_run:
        return src.stat().st_size

    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "move":
        shutil.move(str(src), str(dst))
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return src.stat().st_size if src.exists() else dst.stat().st_size


def write_manifest(manifest_path: Path, rows: List[Dict]) -> None:
    with manifest_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "species",
            "new_index",
            "new_filename",
            "batch_order",
            "batch_name",
            "original_index",
            "source_path",
            "target_path",
            "status",
            "size_bytes",
        ])
        for row in rows:
            writer.writerow([
                row["species"],
                row["new_index"],
                row["new_filename"],
                row["batch_order"],
                row["batch_name"],
                row["original_index"],
                row["source_path"],
                row["target_path"],
                row["status"],
                row["size_bytes"],
            ])


def human_readable_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{num_bytes} B"


def execute_transfer_plan(
    tasks: List[Dict],
    mode: str,
    workers: int,
    manifest_path: Path,
    dry_run: bool = False,
) -> None:
    done_rows: List[Dict] = []
    lock = threading.Lock()

    total_tasks = len(tasks)
    skipped_tasks = sum(1 for t in tasks if t["skip"])
    real_tasks = [t for t in tasks if not t["skip"]]

    total_bytes = 0

    if dry_run:
        for t in tasks:
            size_bytes = t["source_path"].stat().st_size
            total_bytes += size_bytes
            done_rows.append({
                **t,
                "source_path": str(t["source_path"].resolve()),
                "target_path": str(t["target_path"].resolve()),
                "status": "skipped_existing" if t["skip"] else "dry_run",
                "size_bytes": size_bytes,
            })
        write_manifest(manifest_path, done_rows)
        print(f"[DRY-RUN] Total tasks: {total_tasks}")
        print(f"[DRY-RUN] Skipped existing: {skipped_tasks}")
        print(f"[DRY-RUN] Planned data volume: {human_readable_size(total_bytes)}")
        print(f"[DRY-RUN] Manifest saved to: {manifest_path}")
        return

    pbar = None
    if tqdm is not None:
        pbar = tqdm(total=total_tasks, desc=f"{mode}", unit="file")
    else:
        print("[WARN] tqdm is not installed. No progress bar will be shown.")
        print("[WARN] Install with: pip install tqdm")

    # first record skipped
    for t in tasks:
        if t["skip"]:
            size_bytes = t["source_path"].stat().st_size
            total_bytes += size_bytes
            done_rows.append({
                **t,
                "source_path": str(t["source_path"].resolve()),
                "target_path": str(t["target_path"].resolve()),
                "status": "skipped_existing",
                "size_bytes": size_bytes,
            })
            if pbar is not None:
                pbar.update(1)

    future_to_task = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for task in real_tasks:
            future = executor.submit(
                transfer_one_file,
                task["source_path"],
                task["target_path"],
                mode,
                False,
            )
            future_to_task[future] = task

        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                size_bytes = future.result()
                with lock:
                    total_bytes += size_bytes
                    done_rows.append({
                        **task,
                        "source_path": str(task["source_path"].resolve()),
                        "target_path": str(task["target_path"].resolve()),
                        "status": "done",
                        "size_bytes": size_bytes,
                    })
            except Exception as e:
                if pbar is not None:
                    pbar.close()
                # write partial manifest before raising
                partial_manifest = manifest_path.with_name(manifest_path.stem + ".partial.csv")
                write_manifest(partial_manifest, done_rows)
                raise RuntimeError(
                    f"Failed while processing:\n"
                    f"  source: {task['source_path']}\n"
                    f"  target: {task['target_path']}\n"
                    f"  error : {repr(e)}\n"
                    f"Partial manifest saved to: {partial_manifest}"
                ) from e

            if pbar is not None:
                pbar.update(1)

    if pbar is not None:
        pbar.close()

    # sort rows in a stable, human-readable order before writing manifest
    done_rows.sort(key=lambda x: (x["species"], x["new_index"]))
    write_manifest(manifest_path, done_rows)

    print(f"[DONE] Total tasks: {total_tasks}")
    print(f"[DONE] Skipped existing: {skipped_tasks}")
    print(f"[DONE] Finished transfers: {sum(1 for r in done_rows if r['status'] == 'done')}")
    print(f"[DONE] Total data volume: {human_readable_size(total_bytes)}")
    print(f"[DONE] Manifest saved to: {manifest_path}")


def print_summary(tasks: List[Dict], manifest_path: Path, mode: str, workers: int, dry_run: bool) -> None:
    species_count = len(set(t["species"] for t in tasks))
    total_tasks = len(tasks)
    skipped = sum(1 for t in tasks if t["skip"])

    print("[INFO] ---------------- Summary ----------------")
    print(f"[INFO] Species count      : {species_count}")
    print(f"[INFO] Total file tasks   : {total_tasks}")
    print(f"[INFO] Skip-existing tasks: {skipped}")
    print(f"[INFO] Mode              : {mode}")
    print(f"[INFO] Workers           : {workers}")
    print(f"[INFO] Dry run           : {dry_run}")
    print(f"[INFO] Manifest path     : {manifest_path}")
    print("[INFO] ----------------------------------------")


def main() -> None:
    args = parse_args()

    batch_dirs = [Path(p).resolve() for p in args.batch_dirs]
    output_dir = Path(args.output_dir).resolve()
    manifest_path = output_dir / args.manifest_name

    batch_names = validate_inputs(batch_dirs, args.batch_names, output_dir)

    species_to_records = collect_species_files(
        batch_dirs=batch_dirs,
        batch_names=batch_names,
        strict=args.strict,
    )

    if not species_to_records:
        print("[WARN] No valid macrogene parquet files were found.")
        sys.exit(0)

    tasks = build_transfer_plan(
        species_to_records=species_to_records,
        output_dir=output_dir,
        start_index=args.start_index,
        skip_existing=args.skip_existing,
    )

    if not tasks:
        print("[WARN] No transfer tasks were generated.")
        sys.exit(0)

    print_summary(
        tasks=tasks,
        manifest_path=manifest_path,
        mode=args.mode,
        workers=args.workers,
        dry_run=args.dry_run,
    )

    execute_transfer_plan(
        tasks=tasks,
        mode=args.mode,
        workers=args.workers,
        manifest_path=manifest_path,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()