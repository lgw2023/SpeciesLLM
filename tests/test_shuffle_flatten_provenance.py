import csv
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


ROOT = Path(__file__).resolve().parents[1]

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


def write_input_parquet(path, rows, species_id, dataset_id):
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pydict(
        {
            "X": [[float(i), float(i + 1)] for i in range(rows)],
            "soma_joinid": list(range(dataset_id * 100, dataset_id * 100 + rows)),
            "dataset_id": [dataset_id] * rows,
            "assay": [1] * rows,
            "cell_type": [2] * rows,
            "development_stage": [3] * rows,
            "disease": [4] * rows,
            "tissue": [5] * rows,
            "sex": [1] * rows,
            "tech_sample": [dataset_id * 10] * rows,
            "species": [species_id] * rows,
            "idx": list(range(rows)),
        },
        schema=SCHEMA,
    )
    pq.write_table(table, path)


def make_merged_tree(tmp_path):
    merged = tmp_path / "merged"
    files = [
        (merged / "Homo_sapiens" / "macrogene_0.parquet", 2, 5, 11, "1st", 1),
        (merged / "Homo_sapiens" / "macrogene_1.parquet", 3, 5, 12, "1st", 1),
        (merged / "Mus_musculus" / "macrogene_0.parquet", 1, 8, 21, "3scbasecount", 3),
    ]
    for path, rows, species_id, dataset_id, _batch_name, _batch_order in files:
        write_input_parquet(path, rows, species_id=species_id, dataset_id=dataset_id)

    manifest_path = merged / "merge_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
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
            ],
        )
        writer.writeheader()
        for original_index, (path, _rows, _species_id, _dataset_id, batch_name, batch_order) in enumerate(files):
            writer.writerow({
                "species": path.parent.name,
                "new_index": int(path.stem.split("_")[-1]),
                "new_filename": path.name,
                "batch_order": batch_order,
                "batch_name": batch_name,
                "original_index": original_index,
                "source_path": f"/source/{batch_name}/{path.parent.name}/{path.name}",
                "target_path": str(path.resolve()),
                "status": "done",
                "size_bytes": path.stat().st_size,
            })
    return merged, manifest_path


def run_flatten(tmp_path, mode, merge_manifest=True):
    merged, manifest_path = make_merged_tree(tmp_path)
    output = tmp_path / f"flat_{mode}_{int(merge_manifest)}"
    cmd = [
        sys.executable,
        str(ROOT / "shuffle_flatten_macrogene.py"),
        "--input-dir",
        str(merged),
        "--output-dir",
        str(output),
        "--rows-per-file",
        "2",
        "--seed",
        "7",
        "--workers",
        "1",
        "--compression",
        "snappy",
        "--manifest-name",
        "shuffle_manifest.csv",
        "--shuffle-mode",
        mode,
        "--batch-files",
        "1",
        "--shuffle-buckets",
        "2",
        "--temp-dir",
        str(output / "_shuffle_tmp"),
        "--overwrite",
        "--validate-all-schemas",
        "--keep-remainder",
    ]
    if merge_manifest:
        cmd.extend(["--merge-manifest", str(manifest_path)])
    subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True, text=True)
    return output


@pytest.mark.parametrize("mode", ["in-memory", "batch", "external"])
def test_flatten_adds_source_provenance_for_all_shuffle_modes(tmp_path, mode):
    output = run_flatten(tmp_path, mode, merge_manifest=True)
    parquet_files = sorted(output.glob("all_flatten_part_*.parquet"))
    assert parquet_files

    source_manifest = pd.read_csv(output / "source_manifest.csv")
    assert len(source_manifest) == 3
    assert source_manifest["source_file_id"].tolist() == [0, 1, 2]

    frames = [pd.read_parquet(path) for path in parquet_files]
    df = pd.concat(frames, ignore_index=True)
    assert "source_file_id" in df.columns
    assert "source_batch_id" in df.columns
    assert df.groupby("source_file_id").size().to_dict() == {0: 2, 1: 3, 2: 1}
    assert set(df["source_batch_id"].tolist()) == {1, 3}


def test_flatten_without_merge_manifest_keeps_legacy_schema(tmp_path):
    output = run_flatten(tmp_path, "in-memory", merge_manifest=False)
    df = pd.read_parquet(sorted(output.glob("all_flatten_part_*.parquet"))[0])
    assert "source_file_id" not in df.columns
    assert "source_batch_id" not in df.columns
    assert not (output / "source_manifest.csv").exists()
