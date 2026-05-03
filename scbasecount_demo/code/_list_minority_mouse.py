"""List all Mus_musculus files belonging to the minority var ordering."""
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from os.path import join

import anndata as ad
from tqdm import tqdm

DIR = "/data/disk1/SpeciesLLM_obs/scbasecount_demo/scbasecount_processed/Mus_musculus"
OUT = "/data/disk1/SpeciesLLM_obs/scbasecount_demo/code/_mouse_minority_files.json"
TARGET_HASH = "9423ba8af683802592aab9a26ada0f86cc22c2fe"


def hash_var(path):
    a = ad.read_h5ad(path, backed="r")
    try:
        names = list(map(str, a.var_names))
    finally:
        try:
            a.file.close()
        except Exception:
            pass
    return path, hashlib.sha1("\n".join(names).encode("utf-8")).hexdigest()


def main():
    files = sorted(join(DIR, f) for f in os.listdir(DIR) if f.endswith(".h5ad"))
    minority = []
    with ProcessPoolExecutor(max_workers=16) as ex:
        futures = [ex.submit(hash_var, f) for f in files]
        for fut in tqdm(as_completed(futures), total=len(futures)):
            path, h = fut.result()
            if h == TARGET_HASH:
                minority.append(path)
    minority.sort()
    print("Minority files:")
    for p in minority:
        print(" ", p)
    with open(OUT, "w") as f:
        json.dump(minority, f, indent=2)
    print("Saved:", OUT, "count:", len(minority))


if __name__ == "__main__":
    main()
