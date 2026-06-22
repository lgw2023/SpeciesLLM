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
