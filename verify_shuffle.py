#!/usr/bin/env python3
"""
Verify shuffle quality of flatten data across species, batches, and files.

Usage:
  python verify_shuffle.py /path/to/all_flatten_data_full_no_1st_human_mouse_xxx
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter


def analyze_shuffle_quality(flat_dir: str, sample_files: int = 10):
    """Analyze species mixing quality in flatten parquet files."""
    flat_path = Path(flat_dir)
    parquet_files = sorted(flat_path.glob("all_flatten_part_*.parquet"))

    if not parquet_files:
        print(f"[ERROR] No all_flatten_part_*.parquet files found in {flat_dir}")
        return

    # Sample evenly across the file list
    n = len(parquet_files)
    if n <= sample_files:
        sampled = parquet_files
    else:
        indices = np.linspace(0, n - 1, sample_files, dtype=int)
        sampled = [parquet_files[i] for i in indices]

    print(f"Total output files: {n}")
    print(f"Sampling {len(sampled)} files for verification\n")

    global_species_counts = Counter()
    file_stats = []

    for fpath in sampled:
        df = pd.read_parquet(fpath, columns=["species"], engine="pyarrow")
        species_col = df["species"].values
        n_rows = len(species_col)

        if n_rows == 0:
            continue

        # Species distribution in this file
        species_counts = Counter(species_col.tolist())
        global_species_counts.update(species_counts)

        # Adjacency analysis: how often does species change between consecutive rows?
        changes = np.sum(species_col[1:] != species_col[:-1])
        change_ratio = changes / (n_rows - 1) if n_rows > 1 else 0

        # Run-length analysis: longest consecutive run of same species
        run_lengths = []
        current_run = 1
        for i in range(1, n_rows):
            if species_col[i] == species_col[i - 1]:
                current_run += 1
            else:
                run_lengths.append(current_run)
                current_run = 1
        run_lengths.append(current_run)
        max_run = max(run_lengths) if run_lengths else 0
        mean_run = np.mean(run_lengths) if run_lengths else 0

        # Entropy: higher = more even species mix
        total = sum(species_counts.values())
        probs = np.array(list(species_counts.values())) / total
        entropy = -np.sum(probs * np.log(probs)) if len(probs) > 0 else 0
        max_entropy = np.log(len(species_counts)) if len(species_counts) > 0 else 1
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

        file_stats.append({
            "file": fpath.name,
            "rows": n_rows,
            "unique_species": len(species_counts),
            "change_ratio": change_ratio,
            "max_run": max_run,
            "mean_run": mean_run,
            "normalized_entropy": normalized_entropy,
            "species_counts": species_counts,
        })

        print(f"--- {fpath.name} ---")
        print(f"  Rows: {n_rows:,}")
        print(f"  Unique species: {len(species_counts)}")
        print(f"  Normalized entropy: {normalized_entropy:.4f}  (1.0 = perfectly even)")
        print(f"  Adjacent-species change ratio: {change_ratio:.4f}  (higher = better mixed)")
        print(f"  Max same-species run: {max_run} rows")
        print(f"  Mean same-species run: {mean_run:.2f} rows")
        # Top 5 species
        top5 = species_counts.most_common(5)
        print(f"  Top species: {top5}")
        print()

    # Global summary
    print("=" * 60)
    print("GLOBAL SUMMARY")
    print("=" * 60)
    print(f"Total unique species across sampled files: {len(global_species_counts)}")
    print(f"Global species distribution (top 20):")
    for sp, cnt in global_species_counts.most_common(20):
        print(f"  species_id={sp}: {cnt:,} rows")

    # Aggregate metrics
    avg_change = np.mean([s["change_ratio"] for s in file_stats])
    avg_entropy = np.mean([s["normalized_entropy"] for s in file_stats])
    avg_max_run = np.mean([s["max_run"] for s in file_stats])
    avg_unique = np.mean([s["unique_species"] for s in file_stats])

    print(f"\nAggregate metrics across sampled files:")
    print(f"  Avg unique species per file: {avg_unique:.1f}")
    print(f"  Avg normalized entropy: {avg_entropy:.4f}")
    print(f"  Avg adjacent change ratio: {avg_change:.4f}")
    print(f"  Avg max same-species run length: {avg_max_run:.1f}")

    # Interpretation
    print(f"\nInterpretation:")
    if avg_change > 0.8 and avg_entropy > 0.7 and avg_max_run < 10:
        print("  Good shuffle: species are well-mixed within each output file.")
    elif avg_change > 0.5 and avg_entropy > 0.4:
        print("  Acceptable shuffle: reasonable mixing, some clustering present.")
        print("  Consider reducing BATCH_FILES or switching to --shuffle-mode external/in-memory.")
    else:
        print("  Poor shuffle: data appears clustered by species.")
        print("  Reduce BATCH_FILES or check input file organization.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <flatten_data_dir> [sample_files]")
        sys.exit(1)

    flat_dir = sys.argv[1]
    sample = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    analyze_shuffle_quality(flat_dir, sample)
