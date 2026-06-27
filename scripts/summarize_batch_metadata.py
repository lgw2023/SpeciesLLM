#!/usr/bin/env python3
"""Summarize per-batch metadata from SpeciesLLM metrics JSONL files."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional


LOOKUP_FILES = {
    "species": "species.parquet",
    "dataset_id": "dataset_id.parquet",
    "assay": "assay.parquet",
    "tissue": "tissue.parquet",
    "tech_sample": "tech_sample.parquet",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate batch_metadata fields from metrics.*.jsonl files."
    )
    parser.add_argument("run_dir", help="Training output directory containing metrics.*.jsonl")
    parser.add_argument("--epoch", type=int, default=None, help="Filter one 1-based epoch.")
    parser.add_argument("--batch-start", type=int, default=None, help="Inclusive 1-based batch_index lower bound.")
    parser.add_argument("--batch-end", type=int, default=None, help="Inclusive 1-based batch_index upper bound.")
    parser.add_argument("--rank", type=int, action="append", default=None, help="Filter rank. Repeat for multiple ranks.")
    parser.add_argument("--lookup-dir", default=None, help="Optional LOOKUP_categories_unified directory.")
    parser.add_argument("--source-manifest", default=None, help="Optional source_manifest.csv from the flatten output dir.")
    parser.add_argument("--out-csv", default=None, help="Optional CSV output path for full aggregated rows.")
    parser.add_argument("--limit", type=int, default=50, help="Max rows per field printed to stdout. CSV is not limited.")
    return parser.parse_args()


def load_lookup_dir(path: Optional[str]) -> Dict[str, Dict[int, str]]:
    if not path:
        return {}
    import pandas as pd

    lookup_dir = Path(path).expanduser()
    lookups: Dict[str, Dict[int, str]] = {}
    for field, filename in LOOKUP_FILES.items():
        parquet_path = lookup_dir / filename
        if not parquet_path.exists():
            continue
        df = pd.read_parquet(parquet_path)
        label_col = "label" if "label" in df.columns else df.columns[0]
        lookups[field] = {
            int(i): str(label)
            for i, label in enumerate(df[label_col].tolist())
        }
    return lookups


def load_source_manifest(path: Optional[str]) -> tuple[Dict[int, dict], Dict[int, str]]:
    if not path:
        return {}, {}
    source_rows: Dict[int, dict] = {}
    batch_names: Dict[int, str] = {}
    with Path(path).expanduser().open(newline="", encoding="utf-8") as csvfile:
        for row in csv.DictReader(csvfile):
            source_file_id = int(row["source_file_id"])
            source_batch_id = int(row["source_batch_id"])
            source_rows[source_file_id] = row
            batch_names.setdefault(source_batch_id, row.get("batch_name", ""))
    return source_rows, batch_names


def iter_metric_rows(run_dir: Path) -> Iterable[dict]:
    paths = sorted(run_dir.glob("metrics.*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"No metrics.*.jsonl files found under {run_dir}")
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path}:{line_number}") from exc


def row_matches(row: dict, args: argparse.Namespace) -> bool:
    if args.epoch is not None and row.get("epoch") != args.epoch:
        return False
    batch_index = row.get("batch_index")
    if batch_index is None and (args.batch_start is not None or args.batch_end is not None):
        return False
    if args.batch_start is not None and batch_index < args.batch_start:
        return False
    if args.batch_end is not None and batch_index > args.batch_end:
        return False
    if args.rank is not None and row.get("rank") not in set(args.rank):
        return False
    return True


def aggregate(
    rows: Iterable[dict],
    args: argparse.Namespace,
) -> tuple[Dict[str, Counter], Dict[str, int], Dict[str, int], Dict[str, int], int]:
    counts: Dict[str, Counter] = defaultdict(Counter)
    unique_max: Dict[str, int] = defaultdict(int)
    truncated_rows: Dict[str, int] = defaultdict(int)
    max_omitted: Dict[str, int] = defaultdict(int)
    metric_rows = 0
    for row in rows:
        if not row_matches(row, args):
            continue
        metadata = row.get("batch_metadata")
        if not isinstance(metadata, dict):
            continue
        metric_rows += 1
        for field, summary in metadata.items():
            if field == "batch_size" or not isinstance(summary, dict):
                continue
            unique_count = int(summary.get("unique_count") or 0)
            top_items = summary.get("top") or []
            unique_max[field] = max(unique_max[field], unique_count)
            omitted = max(0, unique_count - len(top_items))
            if omitted:
                truncated_rows[field] += 1
                max_omitted[field] = max(max_omitted[field], omitted)
            for item in top_items:
                counts[field][int(item["id"])] += int(item["count"])
    return counts, unique_max, truncated_rows, max_omitted, metric_rows


def decode_label(
    field: str,
    item_id: int,
    lookups: Dict[str, Dict[int, str]],
    source_rows: Dict[int, dict],
    source_batch_names: Dict[int, str],
) -> tuple[str, str, str]:
    source_path = ""
    source_batch = ""
    if field in lookups:
        return lookups[field].get(item_id, ""), source_batch, source_path
    if field == "source_batch_id":
        return source_batch_names.get(item_id, ""), source_batch_names.get(item_id, ""), source_path
    if field == "source_file_id":
        row = source_rows.get(item_id)
        if not row:
            return "", source_batch, source_path
        label = f"{row.get('batch_name', '')}:{row.get('new_filename', '')}"
        source_batch = row.get("batch_name", "")
        source_path = row.get("source_path", "")
        return label, source_batch, source_path
    return "", source_batch, source_path


def build_output_rows(
    counts: Dict[str, Counter],
    unique_max: Dict[str, int],
    truncated_rows: Dict[str, int],
    max_omitted: Dict[str, int],
    metric_rows: int,
    lookups: Dict[str, Dict[int, str]],
    source_rows: Dict[int, dict],
    source_batch_names: Dict[int, str],
) -> List[dict]:
    output_rows: List[dict] = []
    for field in sorted(counts):
        for item_id, count in counts[field].most_common():
            label, source_batch, source_path = decode_label(
                field,
                item_id,
                lookups,
                source_rows,
                source_batch_names,
            )
            output_rows.append({
                "field": field,
                "id": item_id,
                "label": label,
                "count": count,
                "metric_rows": metric_rows,
                "max_unique_count": unique_max.get(field, 0),
                "truncated_metric_rows": truncated_rows.get(field, 0),
                "max_omitted_unique_ids": max_omitted.get(field, 0),
                "source_batch": source_batch,
                "source_path": source_path,
            })
    return output_rows


def write_csv(path: str, rows: List[dict]) -> None:
    fieldnames = [
        "field",
        "id",
        "label",
        "count",
        "metric_rows",
        "max_unique_count",
        "truncated_metric_rows",
        "max_omitted_unique_ids",
        "source_batch",
        "source_path",
    ]
    with Path(path).expanduser().open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_table(rows: List[dict], limit: int) -> None:
    print(
        "field\tid\tlabel\tcount\tmetric_rows\tmax_unique_count\t"
        "truncated_metric_rows\tmax_omitted_unique_ids\tsource_batch\tsource_path"
    )
    printed_by_field: Dict[str, int] = defaultdict(int)
    for row in rows:
        field = row["field"]
        if printed_by_field[field] >= limit:
            continue
        printed_by_field[field] += 1
        print(
            f"{row['field']}\t{row['id']}\t{row['label']}\t{row['count']}\t"
            f"{row['metric_rows']}\t{row['max_unique_count']}\t"
            f"{row['truncated_metric_rows']}\t{row['max_omitted_unique_ids']}\t"
            f"{row['source_batch']}\t{row['source_path']}"
        )


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser()
    lookups = load_lookup_dir(args.lookup_dir)
    source_rows, source_batch_names = load_source_manifest(args.source_manifest)
    counts, unique_max, truncated_rows, max_omitted, metric_rows = aggregate(iter_metric_rows(run_dir), args)
    output_rows = build_output_rows(
        counts,
        unique_max,
        truncated_rows,
        max_omitted,
        metric_rows,
        lookups,
        source_rows,
        source_batch_names,
    )
    if args.out_csv:
        write_csv(args.out_csv, output_rows)
    print_table(output_rows, max(1, args.limit))


if __name__ == "__main__":
    main()
