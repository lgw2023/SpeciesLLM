#!/usr/bin/env python3
"""
Verify shuffle quality of flatten data using statistical tests that account
for imbalanced species distributions.

Usage:
  python verify_shuffle.py /path/to/all_flatten_data_full_no_1st_human_mouse_xxx
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter, defaultdict
from scipy import stats as sp_stats


def expected_change_ratio(probs: np.ndarray) -> float:
    """Under perfect random shuffle, P(adjacent rows have different species)."""
    return float(1.0 - np.sum(probs ** 2))


def expected_mean_run(probs: np.ndarray) -> float:
    """Expected mean run length for a random sequence with given proportions.

    For species i with proportion p_i, the expected run length is 1/(1-p_i)
    in a random sequence. Overall mean = weighted average.
    """
    total = 0.0
    for p in probs:
        if p < 1.0:
            total += p / (1.0 - p)
    return float(total)


def runs_test_binary(seq: np.ndarray, species_a: int, species_b: int):
    """Wald-Wolfowitz runs test for two species. Returns z-score and p-value."""
    mask_a = (seq == species_a)
    mask_b = (seq == species_b)
    ab_seq = seq[mask_a | mask_b]

    n_a = np.sum(mask_a)
    n_b = np.sum(mask_b)

    if n_a < 2 or n_b < 2:
        return 0.0, 1.0  # too few samples

    # Count runs
    runs = 1 + np.sum(ab_seq[1:] != ab_seq[:-1])

    # Expected runs and std
    expected = (2.0 * n_a * n_b) / (n_a + n_b) + 1.0
    std = np.sqrt(
        (2.0 * n_a * n_b * (2.0 * n_a * n_b - n_a - n_b)) /
        ((n_a + n_b) ** 2 * (n_a + n_b - 1.0))
    )

    if std < 1e-10:
        return 0.0, 1.0

    z = (runs - expected) / std
    p_value = 2.0 * sp_stats.norm.sf(abs(z))
    return float(z), float(p_value)


def analyze_shuffle_quality(flat_dir: str, sample_files: int = 10):
    flat_path = Path(flat_dir)
    parquet_files = sorted(flat_path.glob("all_flatten_part_*.parquet"))

    if not parquet_files:
        print(f"[ERROR] No all_flatten_part_*.parquet files found in {flat_dir}")
        return

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

        species_counts = Counter(species_col.tolist())
        global_species_counts.update(species_counts)

        probs = np.array(list(species_counts.values())) / n_rows

        # Observed metrics
        changes = np.sum(species_col[1:] != species_col[:-1])
        obs_change = changes / (n_rows - 1) if n_rows > 1 else 0

        run_lengths = []
        current_run = 1
        for i in range(1, n_rows):
            if species_col[i] == species_col[i - 1]:
                current_run += 1
            else:
                run_lengths.append(current_run)
                current_run = 1
        run_lengths.append(current_run)
        obs_max_run = max(run_lengths) if run_lengths else 0
        obs_mean_run = np.mean(run_lengths) if run_lengths else 0

        # Expected metrics under random shuffle (given this file's distribution)
        exp_change = expected_change_ratio(probs)
        exp_mean_run = expected_mean_run(probs)

        # Runs test on top 2 species
        top2 = [sp for sp, _ in species_counts.most_common(2)]
        if len(top2) >= 2:
            z_score, p_value = runs_test_binary(species_col, top2[0], top2[1])
        else:
            z_score, p_value = 0.0, 1.0

        file_stats.append({
            "file": fpath.name,
            "rows": n_rows,
            "unique_species": len(species_counts),
            "obs_change": obs_change,
            "exp_change": exp_change,
            "obs_mean_run": obs_mean_run,
            "exp_mean_run": exp_mean_run,
            "obs_max_run": obs_max_run,
            "z_score": z_score,
            "p_value": p_value,
            "species_counts": species_counts,
        })

        print(f"--- {fpath.name} ---")
        print(f"  Rows: {n_rows:,}  |  Unique species: {len(species_counts)}")
        print(f"  Change ratio:  observed={obs_change:.4f}  expected={exp_change:.4f}  "
              f"delta={obs_change - exp_change:+.4f}")
        print(f"  Mean run len:  observed={obs_mean_run:.2f}  expected={exp_mean_run:.2f}  "
              f"delta={obs_mean_run - exp_mean_run:+.3f}")
        print(f"  Max run len:   {obs_max_run}")
        print(f"  Runs test (top2 species): z={z_score:+.3f}  p={p_value:.4f}  "
              f"({'significant clustering' if p_value < 0.01 else 'pass'})")
        print(f"  Top species: {species_counts.most_common(5)}")
        print()

    # --- Per-file summary ---
    deltas_change = [s["obs_change"] - s["exp_change"] for s in file_stats]
    p_values = [s["p_value"] for s in file_stats]
    mean_delta = np.mean(deltas_change)
    significant_count = sum(1 for p in p_values if p < 0.01)

    print("=" * 60)
    print("PER-FILE SHUFFLE QUALITY")
    print("=" * 60)
    print(f"  Mean (observed - expected) change ratio: {mean_delta:+.5f}")
    print(f"  Files with significant clustering (p<0.01): {significant_count}/{len(file_stats)}")

    # --- Rare species distribution across output files ---
    print(f"\n{'=' * 60}")
    print("RARE SPECIES DISTRIBUTION ACROSS FILES")
    print("=" * 60)
    print("Verifies that rare species aren't all clustered in a few output files.\n")

    rare_threshold = 5
    # Identify rare species (occur ≤rare_threshold times in sampled files)
    rare_species = [
        sp for sp, cnt in global_species_counts.items()
        if cnt <= rare_threshold * len(sampled)
    ]

    if rare_species:
        # For each rare species, count how many sampled files contain it
        for sp in sorted(rare_species, key=lambda x: global_species_counts[x], reverse=True):
            files_with_sp = sum(
                1 for s in file_stats if sp in s["species_counts"]
            )
            total_occurrences = global_species_counts[sp]
            print(f"  species_id={sp}: {total_occurrences} rows across "
                  f"{files_with_sp}/{len(file_stats)} files  "
                  f"({'suspicious: all rows in 1 file' if total_occurrences > 3 and files_with_sp == 1 else 'ok'})")
    else:
        print("  No rare species found in sampled files (all species appear frequently).")

    # --- Species spread across file index (detects batch-mode segment clustering) ---
    print(f"\n{'=' * 60}")
    print("SPECIES SPREAD ACROSS FILE INDEX")
    print("=" * 60)
    print("Detects batch-mode segment clustering: a species whose input files all")
    print("fall in a single shuffle batch will appear only in a contiguous range of")
    print("output files; its spread across sampled file indices will be small.")
    print("(Recommend sample_files >= 50 for meaningful spread resolution.)\n")

    species_to_sample_indices = defaultdict(list)
    for i, stat in enumerate(file_stats):
        for sp in stat["species_counts"]:
            species_to_sample_indices[sp].append(i)

    n_samples = len(file_stats)
    if n_samples >= 2:
        ranked = sorted(
            species_to_sample_indices.keys(),
            key=lambda x: global_species_counts[x],
            reverse=True,
        )
        for sp in ranked:
            indices = species_to_sample_indices[sp]
            spread = (max(indices) - min(indices)) / (n_samples - 1)
            n_files_with_sp = len(indices)
            if spread >= 0.85:
                verdict = "uniform"
            elif spread >= 0.5:
                verdict = "moderate"
            elif spread >= 0.05:
                verdict = "BATCH SEGMENT"
            else:
                verdict = "NEAR-SINGLE BATCH"
            print(
                f"  species_id={sp:>3d}: "
                f"{global_species_counts[sp]:>10,} rows, "
                f"in {n_files_with_sp:>3d}/{n_samples} sampled files, "
                f"spread={spread:.2f} [{verdict}]"
            )

    # --- Global interpretation ---
    print(f"\n{'=' * 60}")
    print("INTERPRETATION")
    print("=" * 60)
    print(f"  Species 5+8 make up {global_species_counts[5] + global_species_counts[8]:,} / "
          f"{sum(global_species_counts.values()):,} rows "
          f"({(global_species_counts[5] + global_species_counts[8]) / sum(global_species_counts.values()) * 100:.1f}%).")
    print(f"  With this imbalance, expected change_ratio ≈ {file_stats[0]['exp_change']:.4f} in a perfectly random shuffle.")

    if abs(mean_delta) < 0.02 and significant_count <= len(file_stats) * 0.2:
        print(f"\n  ✓ Shuffle looks correct.")
        print(f"    Observed change_ratio matches random expectation (delta={mean_delta:+.4f}).")
        print(f"    No evidence of species clustering within output files.")
    elif mean_delta < -0.05 or significant_count > len(file_stats) * 0.5:
        print(f"\n  ✗ Possible clustering detected.")
        print(f"    Observed change_ratio is lower than expected.")
        print(f"    Reduce BATCH_FILES or use --shuffle-mode external.")
    else:
        print(f"\n  ~ Marginal. Check rare species distribution above for clustering.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <flatten_data_dir> [sample_files]")
        sys.exit(1)

    flat_dir = sys.argv[1]
    sample = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    analyze_shuffle_quality(flat_dir, sample)
