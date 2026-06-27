import ast
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]


class SequenceDataset(Dataset):
    def __init__(self, values):
        self.values = list(values)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return self.values[index]


def load_make_parquet_data_loader():
    source = (ROOT / "train_MNodes_torchrun_mfu_preindexparquet.py").read_text()
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "make_parquet_data_loader"
    )
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "torch": torch,
        "DataLoader": DataLoader,
        "ParquetDataset": SequenceDataset,
        "log_or_print": lambda *args, **kwargs: None,
    }
    exec(compile(module, "<make_parquet_data_loader>", "exec"), namespace)
    return namespace["make_parquet_data_loader"]


def load_training_functions(*names):
    source = (ROOT / "train_MNodes_torchrun_mfu_preindexparquet.py").read_text()
    tree = ast.parse(source)
    selected_nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    module = ast.Module(body=selected_nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {}
    exec(compile(module, "<training_subset>", "exec"), namespace)
    return namespace


def test_shuffle_is_reproducible_and_changes_by_epoch_rank_and_chunk():
    make_loader = load_make_parquet_data_loader()
    args = SimpleNamespace(
        num_workers=0,
        shuffle_rows=True,
        shuffle_seed=42,
        batch_size=4,
        pin_memory=False,
        prefetch_factor=1,
        persistent_workers=False,
    )

    def sample_order(epoch, rank, chunk_index):
        loader = make_loader(
            list(range(32)),
            args,
            None,
            rank,
            epoch=epoch,
            chunk_index=chunk_index,
        )
        return torch.cat(list(loader)).tolist()

    baseline = sample_order(epoch=0, rank=0, chunk_index=1)
    assert baseline == sample_order(epoch=0, rank=0, chunk_index=1)
    assert baseline != sample_order(epoch=1, rank=0, chunk_index=1)
    assert baseline != sample_order(epoch=0, rank=1, chunk_index=1)
    assert baseline != sample_order(epoch=0, rank=0, chunk_index=2)
    assert sorted(baseline) == list(range(32))


def test_multinode_launcher_forwards_shuffle_arguments():
    env = os.environ.copy()
    env.update(
        {
            "TRAIN_SHUFFLE_ROWS": "true",
            "TRAIN_SHUFFLE_SEED": "123",
            "BATCH_METADATA_LOG_INTERVAL": "7",
            "BATCH_METADATA_TOP_K": "5",
            "MASKED_GEP_STATS_INTERVAL": "11",
            "DRY_RUN": "1",
            "NODE_RANK": "0",
            "NNODES": "1",
            "NPROC_PER_NODE": "1",
            "MASTER_ADDR": "127.0.0.1",
            "WORKDIR": str(ROOT),
            "PYTHON_BIN": sys.executable,
            "MODEL_CONFIG_JSON": str(
                ROOT / "Stage2_macrogene_embeddings" / "args_2nd_run.json"
            ),
            "DATA_PATH": "/tmp/speciesllm-shuffle-test",
            "EMB_PATH": str(ROOT / "Stage2_macrogene_embeddings"),
        }
    )
    result = subprocess.run(
        ["bash", "scripts/launch_multinode_torchrun.sh", "--worker"],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--shuffle_rows=true" in result.stdout
    assert "--shuffle_seed=123" in result.stdout
    assert "--batch_metadata_log_interval=7" in result.stdout
    assert "--batch_metadata_top_k=5" in result.stdout
    assert "--masked_gep_stats_interval=11" in result.stdout


def test_multinode_launcher_defaults_batch_metadata_interval_to_log_interval():
    env = os.environ.copy()
    env.update(
        {
            "LOG_INTERVAL": "17",
            "DRY_RUN": "1",
            "NODE_RANK": "0",
            "NNODES": "1",
            "NPROC_PER_NODE": "1",
            "MASTER_ADDR": "127.0.0.1",
            "WORKDIR": str(ROOT),
            "PYTHON_BIN": sys.executable,
            "MODEL_CONFIG_JSON": str(
                ROOT / "Stage2_macrogene_embeddings" / "args_2nd_run.json"
            ),
            "DATA_PATH": "/tmp/speciesllm-shuffle-test",
            "EMB_PATH": str(ROOT / "Stage2_macrogene_embeddings"),
        }
    )
    env.pop("BATCH_METADATA_LOG_INTERVAL", None)
    result = subprocess.run(
        ["bash", "scripts/launch_multinode_torchrun.sh", "--worker"],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--log_interval=17" in result.stdout
    assert "--batch_metadata_log_interval=17" in result.stdout


def test_multinode_launcher_keeps_explicit_zero_batch_metadata_interval():
    env = os.environ.copy()
    env.update(
        {
            "LOG_INTERVAL": "17",
            "BATCH_METADATA_LOG_INTERVAL": "0",
            "DRY_RUN": "1",
            "NODE_RANK": "0",
            "NNODES": "1",
            "NPROC_PER_NODE": "1",
            "MASTER_ADDR": "127.0.0.1",
            "WORKDIR": str(ROOT),
            "PYTHON_BIN": sys.executable,
            "MODEL_CONFIG_JSON": str(
                ROOT / "Stage2_macrogene_embeddings" / "args_2nd_run.json"
            ),
            "DATA_PATH": "/tmp/speciesllm-shuffle-test",
            "EMB_PATH": str(ROOT / "Stage2_macrogene_embeddings"),
        }
    )
    result = subprocess.run(
        ["bash", "scripts/launch_multinode_torchrun.sh", "--worker"],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--batch_metadata_log_interval=0" in result.stdout


def test_full_epoch_iterator_yields_parquet_context():
    namespace = load_training_functions("iter_parquet_data_loader_batches")
    batches = [
        ("batch0",),
        ("batch1",),
    ]

    rows = list(
        namespace["iter_parquet_data_loader_batches"](
            batches,
            resume_batch_offset=0,
            max_batch_index=2,
            data_load_s=1.25,
            batch_context={
                "parquet_chunk_index": 1,
                "parquet_chunk_total": 1,
                "parquet_file_start_index": 0,
                "parquet_file_end_index": 9,
                "parquet_file_count": 10,
            },
        )
    )

    assert rows[0] == (
        0,
        ("batch0",),
        1.25,
        {
            "parquet_chunk_index": 1,
            "parquet_chunk_total": 1,
            "parquet_file_start_index": 0,
            "parquet_file_end_index": 9,
            "parquet_file_count": 10,
        },
    )
    assert rows[1] == (
        1,
        ("batch1",),
        0.0,
        {
            "parquet_chunk_index": 1,
            "parquet_chunk_total": 1,
            "parquet_file_start_index": 0,
            "parquet_file_end_index": 9,
            "parquet_file_count": 10,
        },
    )


def test_generate_test_data_uses_custom_merge_manifest_for_flatten(tmp_path):
    batch_root = tmp_path / "Stage2_SpeciesLLMData"
    for dirname in (
        "1st_pretrain_data_preprocessed_step4",
        "2nd_pretrain_data_preprocessed_step4",
        "3scbasecount_pretrain_data_preprocessed_step4",
    ):
        (batch_root / dirname).mkdir(parents=True)

    output_dir = tmp_path / "merged"
    env = os.environ.copy()
    env.update(
        {
            "PYTHON_BIN": sys.executable,
            "BATCH_ROOT": str(batch_root),
            "OUTPUT_DIR": str(output_dir),
            "FLATTEN_OUTPUT_DIR": str(tmp_path / "flattened"),
            "MANIFEST_NAME": "custom_merge_manifest.csv",
            "DRY_RUN": "1",
        }
    )

    result = subprocess.run(
        ["bash", "scripts/generate_test_data.sh"],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert f"--merge-manifest {output_dir / 'custom_merge_manifest.csv'}" in result.stdout
