"""Scan all Mus_musculus h5ad files and group by var_names ordering hash."""
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from os.path import join

import anndata as ad
from tqdm import tqdm

DIR = "/data/disk1/SpeciesLLM_obs/scbasecount_demo/scbasecount_processed/Mus_musculus"
OUT = "/data/disk1/SpeciesLLM_obs/scbasecount_demo/code/_mouse_var_groups.json"


def hash_var(path):
    try:
        a = ad.read_h5ad(path, backed="r")
        try:
            names = list(map(str, a.var_names))
        finally:
            try:
                a.file.close()
            except Exception:
                pass
        h = hashlib.sha1("\n".join(names).encode("utf-8")).hexdigest()
        return path, h, len(names), None
    except Exception as e:  # noqa: BLE001
        return path, None, 0, repr(e)


def main():
    files = sorted(join(DIR, f) for f in os.listdir(DIR) if f.endswith(".h5ad"))
    print(f"Total files: {len(files)}")
    groups = {}  # hash -> {"count": int, "n_var": int, "examples": [..]}
    errors = []

    with ProcessPoolExecutor(max_workers=16) as ex:
        futures = [ex.submit(hash_var, f) for f in files]
        for fut in tqdm(as_completed(futures), total=len(futures)):
            path, h, n, err = fut.result()
            if err is not None:
                errors.append({"path": path, "error": err})
                continue
            g = groups.setdefault(h, {"count": 0, "n_var": n, "examples": []})
            g["count"] += 1
            if len(g["examples"]) < 5:
                g["examples"].append(os.path.basename(path))

    summary = {"groups": groups, "errors": errors}
    with open(OUT, "w") as f:
        json.dump(summary, f, indent=2)
    print("Distinct orderings:", len(groups))
    for h, info in sorted(groups.items(), key=lambda kv: -kv[1]["count"]):
        print(f"  hash={h[:10]}  count={info['count']}  n_var={info['n_var']}  e.g. {info['examples']}")
    if errors:
        print("Errors:", len(errors))
    print("Saved:", OUT)


if __name__ == "__main__":
    main()
