"""Small offline checks for mapping, shuffling, and the torchrun training entry."""
import csv
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import pytest

ROOT = Path(__file__).resolve().parents[1]


def run(script, *args, cwd, ranks=1):
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    launcher = []
    if ranks > 1:
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        launcher = ["-m", "torch.distributed.run", f"--nproc-per-node={ranks}",
                    "--master-addr=127.0.0.1", f"--master-port={port}"]
        if sys.platform == "darwin":
            env["GLOO_SOCKET_IFNAME"] = "lo0"
    subprocess.run([sys.executable, *launcher, str(ROOT / script), *map(str, args)],
                   cwd=cwd, env=env, check=True, timeout=180)


def test_mapping_and_within_species_shuffle(tmp_path):
    source = tmp_path / "obs"
    source.mkdir()
    values = np.arange(10, dtype=np.float32).reshape(5, 2)
    obs = pd.DataFrame({"X": list(np.log1p(values)), "idx": range(5), "dataset_id": 1,
                        "tech_sample": 2, "tissue": 3})
    obs.to_parquet(source / "part.0.parquet")
    var = tmp_path / "var.parquet"
    pd.DataFrame({"feature_name": ["a", "b"]}).to_parquet(var)
    w = np.zeros((2, 1369), dtype=np.float32)
    w[:, 0] = [1., 0.5]
    w[1, 1] = 0.5
    weights = ad.AnnData(w)
    weights.obs_names = ["a", "b"]
    weights.var_names = [f"MacroGene-{i}" for i in range(1369)]
    weight_path = tmp_path / "weights.h5ad"
    weights.write_h5ad(weight_path)
    config = tmp_path / "tasks.csv"
    with config.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["species", "obs_parquet_path", "var_parquet_path", "weight_h5ad_path", "output_path"])
        writer.writerow(["species", source, var, weight_path, tmp_path / "mapped/species"])
    run("preprocess/batch_mapping_expm1_map_log1p.py", "--csv", config, "--scheduler", "synchronous", cwd=tmp_path)
    run("preprocess/shuffle_within_species.py", "--input-path", tmp_path / "mapped",
        "--rows-per-file", 2, "--read-batch-rows", 1, cwd=tmp_path)
    folder = tmp_path / "mapped_shuffle_within_species/species"
    parts = [pq.read_table(f).to_pandas() for f in sorted(folder.glob("macrogene_*.parquet"))]
    assert [len(p) for p in parts] == [2, 2, 1]
    frame = pd.concat(parts).sort_values("idx")
    np.testing.assert_array_equal(frame.idx, range(5))
    np.testing.assert_allclose(np.stack(frame.X), np.log1p(values @ w), rtol=1e-6)
    assert len(pd.read_parquet(folder / "species_macrogene_lookup.parquet")) == 1369


@pytest.mark.parametrize("ranks", [1, 2])
def test_tiny_joint_training(tmp_path, ranks):
    length = 12
    for source in ["batch1", "batch2"]:
        folder = tmp_path / source / "species"
        folder.mkdir(parents=True)
        for shard in range(5):
            rows = {"X": [np.arange(length) % 3] * 2}
            for column in ["species", "tissue", "assay", "disease", "sex", "development_stage"]:
                rows[column] = [0, 1]
            pd.DataFrame(rows).to_parquet(
                folder / f"macrogene_{shard:06d}.parquet")
        pd.DataFrame({"macrogene": ["test"]}).to_parquet(folder / "species_macrogene_lookup.parquet")
    (tmp_path / f"customized_gene{length}_vocab.json").write_text(
        json.dumps({f"gene_{i + 1}": i for i in range(length)}))
    for label, width in [("DNAseq", 2560), ("gene_desc", 1280), ("esm2", 1280)]:
        np.save(tmp_path / f"3rdrun_K1500_Eukaryota_macrogene_{label}_weighted_average.npy",
                np.ones((length, width), dtype=np.float32))
    preset = json.loads((ROOT / "configuration/350m_batch1_batch2.json").read_text())
    cfg = preset["model"]
    cfg.update(hidden_size=16, num_hidden_layers=1, num_attention_heads=4, intermediate_size=32)
    model_config = tmp_path / "tiny.json"
    model_config.write_text(json.dumps(preset))
    common = [
        "--device", os.environ.get("K1369_TEST_DEVICE", "cpu"), "--model_config", model_config,
        "--embedding_dir", tmp_path, "--seq_len", length,
        "--batch1_path", tmp_path / "batch1", "--batch2_path", tmp_path / "batch2",
        "--batch_size", 1, "--global_accumulation_steps", 2, "--epoch_cells", 4,
        "--epoch", 2, "--num_workers", 0, "--shard_rows", 2, "--log_interval", 1,
        "--diagnostic_stats_interval", 1,
    ]
    run("scripts/train_ascend.py", *common, "--train_mode", "scratch",
        "--out_path", tmp_path / "out", cwd=tmp_path, ranks=ranks)
    final = torch.load(tmp_path / "out/checkpoint_latest.pth", map_location="cpu", weights_only=False)
    assert final["optimizer_step"] == 4
    assert final["epoch_complete"] and np.isfinite(final["average_total_loss"])
    if ranks == 1:
        run("scripts/train_ascend.py", *common, "--train_mode", "resume",
            "--checkpoint_path", tmp_path / "out/checkpoint_epoch_0001.pth",
            "--out_path", tmp_path / "continued", cwd=tmp_path)
        resumed = torch.load(tmp_path / "continued/checkpoint_latest.pth", map_location="cpu", weights_only=False)
        assert resumed["optimizer_step"] == 4
        assert resumed["logits_branch"]["optimizer_parameters_restored"] > 0


def test_cross_bucket_shuffle_preserves_rows_and_seed(tmp_path):
    source = tmp_path / "input/species"
    source.mkdir(parents=True)
    # The upstream planner selects a second bucket above 32,768 rows.
    ids = np.arange(32769)
    frame = pd.DataFrame({"idx": ids, "X": ids[:, None].tolist(),
                          "dataset_id": ids % 7, "tissue": 1, "tech_sample": 2})
    frame.to_parquet(source / "macrogene_0.parquet")
    outputs = []
    for name in ["first", "second"]:
        root = tmp_path / name
        run("preprocess/shuffle_within_species.py", "--input-path", source.parent,
            "--output-dir", root, "--num-buckets", 2, "--rows-per-file", 19000, cwd=tmp_path)
        files = sorted((root / "species").glob("macrogene_*.parquet"))
        parts = [pq.read_table(f).to_pandas() for f in files]
        assert [len(p) for p in parts] == [19000, 13769]
        result = pd.concat(parts)
        np.testing.assert_array_equal(np.stack(result.X)[:, 0], result.idx)
        np.testing.assert_array_equal(result.dataset_id, result.idx % 7)
        outputs.append(result.idx.to_numpy())
    np.testing.assert_array_equal(np.sort(outputs[0]), ids)
    np.testing.assert_array_equal(*outputs)
    assert not np.array_equal(outputs[0], ids)
