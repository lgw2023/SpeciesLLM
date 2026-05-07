"""Reorder var/X columns for the 6 Mus_musculus h5ad files whose var_names
ordering differs from the canonical ordering used by ensure_same_var_for_species.

Canonical ordering: from the alphabetically first file in the directory
(ERX10016429.processed.h5ad), which the upstream code already treats as the
standard. Both orderings contain the SAME 22324 gene names, only the order
differs and var has no extra columns, so this is a pure reorder operation.

Steps per file:
  1. Back up original to <DIR>/_var_reorder_backup/<basename>
  2. Load fully (not backed): X (CSR), obs, var
  3. Compute permutation perm such that var_names[perm] == standard_names
  4. Reorder X columns and var rows
  5. Write to <path>.tmp then atomic rename
  6. Reload and verify var_names matches standard exactly
"""
import os
import shutil
import sys
import time
from os.path import basename, dirname, join

import anndata as ad
import numpy as np

try:
    ad.settings.allow_write_nullable_strings = True
except Exception:
    pass

DIR = "/data/disk1/SpeciesLLM_obs/scbasecount_demo/scbasecount_processed/Mus_musculus"
STANDARD_FILE = join(DIR, "ERX10016429.processed.h5ad")
BACKUP_DIR = join(DIR, "_var_reorder_backup")

MINORITY_FILES = [
    join(DIR, "SRX8994900.processed.h5ad"),
    join(DIR, "SRX9785584.processed.h5ad"),
    join(DIR, "SRX9785585.processed.h5ad"),
    join(DIR, "SRX9785586.processed.h5ad"),
    join(DIR, "SRX9785587.processed.h5ad"),
    join(DIR, "SRX9785588.processed.h5ad"),
]


def load_standard_var_names():
    a = ad.read_h5ad(STANDARD_FILE, backed="r")
    try:
        names = a.var_names.copy()
    finally:
        try:
            a.file.close()
        except Exception:
            pass
    return names


def fix_one(path: str, standard_names) -> None:
    base = basename(path)
    print(f"\n=== Fixing {base} ===")
    t0 = time.time()

    os.makedirs(BACKUP_DIR, exist_ok=True)
    backup_path = join(BACKUP_DIR, base)
    if not os.path.exists(backup_path):
        print(f"  backing up -> {backup_path}")
        shutil.copy2(path, backup_path)
    else:
        print(f"  backup already exists, skipping backup: {backup_path}")

    print("  loading (full)...")
    adata = ad.read_h5ad(path)
    print(f"  shape: {adata.shape}, X type: {type(adata.X).__name__}")

    cur_names = adata.var_names
    if cur_names.equals(standard_names):
        print("  already in standard order, nothing to do.")
        return

    if set(cur_names) != set(standard_names):
        raise RuntimeError(
            f"  gene name SETS differ for {base}; this script only handles pure reorder."
        )

    perm = cur_names.get_indexer(standard_names)
    if (perm < 0).any():
        raise RuntimeError(f"  perm has -1 entries for {base}; should not happen.")
    if not np.array_equal(np.sort(perm), np.arange(len(perm))):
        raise RuntimeError(f"  perm is not a valid permutation for {base}.")

    print(f"  reordering X columns (perm len={len(perm)})...")
    new_X = adata.X[:, perm]
    if hasattr(new_X, "tocsr"):
        new_X = new_X.tocsr()

    new_var = adata.var.iloc[perm].copy()
    new_var.index = standard_names.copy()

    new_adata = ad.AnnData(X=new_X, obs=adata.obs.copy(), var=new_var)

    tmp_path = path + ".tmp"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    print(f"  writing tmp: {tmp_path}")
    new_adata.write_h5ad(tmp_path, compression="gzip")

    os.replace(tmp_path, path)
    print(f"  replaced original: {path}")

    check = ad.read_h5ad(path, backed="r")
    try:
        ok = check.var_names.equals(standard_names)
        shape_ok = check.shape == adata.shape
    finally:
        try:
            check.file.close()
        except Exception:
            pass
    if not ok:
        raise RuntimeError(f"  POST-CHECK FAILED: var_names mismatch in {base}")
    if not shape_ok:
        raise RuntimeError(f"  POST-CHECK FAILED: shape changed in {base}")
    print(f"  OK. elapsed: {time.time() - t0:.2f}s")


def main():
    standard_names = load_standard_var_names()
    print(f"Standard var len: {len(standard_names)} (from {basename(STANDARD_FILE)})")

    targets = MINORITY_FILES
    if len(sys.argv) > 1 and sys.argv[1] == "--only-first":
        targets = MINORITY_FILES[:1]
        print("Mode: --only-first (dry-run on a single file)")

    for p in targets:
        if not os.path.exists(p):
            print(f"WARN: missing file {p}", file=sys.stderr)
            continue
        fix_one(p, standard_names)

    print("\nDone.")


if __name__ == "__main__":
    main()
