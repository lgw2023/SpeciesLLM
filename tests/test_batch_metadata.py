import ast
import csv
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]


def load_training_functions(*names):
    source = (ROOT / "train_MNodes_torchrun_mfu_preindexparquet.py").read_text()
    tree = ast.parse(source)
    selected_nodes = [
        node
        for node in tree.body
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "MASKED_GEP_DIAGNOSTIC_FIELDS"
                for target in node.targets
            )
        )
        or (isinstance(node, ast.FunctionDef) and node.name in names)
    ]
    module = ast.Module(body=selected_nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"np": np, "torch": torch}
    exec(compile(module, "<training_subset>", "exec"), namespace)
    return namespace


def load_custom_collate_3geneemb():
    source = (ROOT / "nanoBERT" / "utils" / "data_collator_3GeneEmb.py").read_text()
    tree = ast.parse(source)
    selected_nodes = [
        node
        for node in tree.body
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "DIAGNOSTIC_META_KEYS" for target in node.targets)
        )
        or (isinstance(node, ast.ClassDef) and node.name == "CustomCollate_3GeneEmb")
    ]
    module = ast.Module(body=selected_nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "dataclass": dataclass,
        "Any": Any,
        "Dict": Dict,
        "List": List,
        "Mapping": Mapping,
        "Optional": Optional,
        "Tuple": Tuple,
        "Union": Union,
        "np": np,
        "torch": torch,
        "GeneVocab": object,
        "BERTConfig": object,
    }
    exec(compile(module, "<data_collator_3GeneEmb_subset>", "exec"), namespace)
    return namespace["CustomCollate_3GeneEmb"]


def test_collator_keeps_diagnostic_metadata_without_model_labels():
    CustomCollate_3GeneEmb = load_custom_collate_3geneemb()
    config = SimpleNamespace(
        vocab={"<cls>": 0},
        use_batch_labels=False,
        use_species_labels=False,
        use_tissue_labels=False,
        use_seqmethod_labels=False,
        use_disease_labels=False,
        use_sex_labels=False,
        use_age_labels=False,
        do_cls=False,
    )
    collate = CustomCollate_3GeneEmb(
        config=config,
        genes=np.array([1, 2], dtype=np.int64),
        esm_embeddings=np.zeros((2, 1), dtype=np.float32),
        desc_embeddings=np.zeros((2, 1), dtype=np.float32),
        dna_embeddings=np.zeros((2, 1), dtype=np.float32),
        return_static_inputs=False,
    )
    batch = [
        {
            "values": np.array([1.0, 0.0], dtype=np.float32),
            "batch_labels": 10,
            "species_labels": 5,
            "meta_species": 5,
            "meta_dataset_id": 101,
            "meta_assay": 2,
            "meta_tissue": 21,
            "meta_tech_sample": 1001,
            "meta_source_file_id": 0,
            "meta_source_batch_id": 1,
        },
        {
            "values": np.array([0.0, 2.0], dtype=np.float32),
            "batch_labels": 11,
            "species_labels": 8,
            "meta_species": 8,
            "meta_dataset_id": 102,
            "meta_assay": 4,
            "meta_tissue": 34,
            "meta_tech_sample": 1002,
            "meta_source_file_id": 1,
            "meta_source_batch_id": 3,
        },
    ]

    output = collate(batch)

    assert "batch_labels" not in output
    assert "species_labels" not in output
    assert torch.equal(output["meta_species"], torch.tensor([5, 8]))
    assert torch.equal(output["meta_dataset_id"], torch.tensor([101, 102]))
    assert torch.equal(output["meta_assay"], torch.tensor([2, 4]))
    assert torch.equal(output["meta_tissue"], torch.tensor([21, 34]))
    assert torch.equal(output["meta_tech_sample"], torch.tensor([1001, 1002]))
    assert torch.equal(output["meta_source_file_id"], torch.tensor([0, 1]))
    assert torch.equal(output["meta_source_batch_id"], torch.tensor([1, 3]))


def test_masked_gep_diagnostic_stats_uses_only_masked_positions():
    namespace = load_training_functions("empty_masked_gep_diagnostic_stats", "masked_gep_diagnostic_stats")
    stats = namespace["masked_gep_diagnostic_stats"](
        target_values=torch.tensor([[0.0, 1.0, 5.0], [2.0, 0.0, 9.0]]),
        pred_values=torch.tensor([[0.5, 1.5, 1.0], [2.5, 3.0, 13.0]]),
        mask_positions=torch.tensor([[False, True, True], [True, False, True]]),
    )

    assert stats["target_mean_masked"] == 4.25
    assert np.isclose(stats["target_std_masked"], np.std([1.0, 5.0, 2.0, 9.0]))
    assert np.isclose(stats["target_p95_masked"], np.percentile([1.0, 5.0, 2.0, 9.0], 95))
    assert np.isclose(stats["target_p99_masked"], np.percentile([1.0, 5.0, 2.0, 9.0], 99))
    assert stats["target_nonzero_ratio_masked"] == 1.0
    assert stats["pred_mean_masked"] == 4.5
    assert np.isclose(stats["pred_std_masked"], np.std([1.5, 1.0, 2.5, 13.0]))
    assert np.isclose(stats["abs_error_p95"], np.percentile([0.5, 4.0, 0.5, 4.0], 95))
    assert np.isclose(stats["abs_error_p99"], np.percentile([0.5, 4.0, 0.5, 4.0], 99))


def test_masked_gep_diagnostic_stats_returns_none_for_empty_mask():
    namespace = load_training_functions("empty_masked_gep_diagnostic_stats", "masked_gep_diagnostic_stats")
    stats = namespace["masked_gep_diagnostic_stats"](
        target_values=torch.tensor([[0.0, 1.0]]),
        pred_values=torch.tensor([[0.0, 1.0]]),
        mask_positions=torch.zeros((1, 2), dtype=torch.bool),
    )

    assert set(stats) == {
        "target_mean_masked",
        "target_std_masked",
        "target_p95_masked",
        "target_p99_masked",
        "target_nonzero_ratio_masked",
        "pred_mean_masked",
        "pred_std_masked",
        "abs_error_p95",
        "abs_error_p99",
    }
    assert all(value is None for value in stats.values())


def test_update_clip_fraction_rolling_tracks_recent_optimizer_steps_only():
    namespace = load_training_functions("update_clip_fraction_rolling")
    history = []

    assert namespace["update_clip_fraction_rolling"](history, "no_step", window=3) is None
    assert history == []
    assert namespace["update_clip_fraction_rolling"](history, "pass", window=3) == 0.0
    assert namespace["update_clip_fraction_rolling"](history, "clip", window=3) == 0.5
    assert namespace["update_clip_fraction_rolling"](history, "skip_norm", window=3) == 1 / 3
    assert namespace["update_clip_fraction_rolling"](history, "clip", window=3) == 2 / 3
    assert history == [True, False, True]


def test_batch_metadata_interval_ignores_optimizer_step_trigger():
    namespace = load_training_functions("should_trace_step", "should_log_batch_metadata_step")
    should_trace_step = namespace["should_trace_step"]
    should_log_batch_metadata_step = namespace["should_log_batch_metadata_step"]

    assert should_trace_step(7, 1, should_step=True) is True
    assert should_log_batch_metadata_step(7, 1) is False
    assert should_log_batch_metadata_step(7, 0) is True
    assert should_log_batch_metadata_step(7, 6) is True
    assert should_log_batch_metadata_step(7, 1, final_step=True) is True


def test_masked_gep_diagnostic_stats_interval_is_opt_in():
    namespace = load_training_functions("should_trace_step", "should_log_masked_gep_stats_step")
    should_trace_step = namespace["should_trace_step"]
    should_log_masked_gep_stats_step = namespace["should_log_masked_gep_stats_step"]

    assert should_trace_step(10, 0) is True
    assert should_log_masked_gep_stats_step(0, 0) is False
    assert should_log_masked_gep_stats_step(None, 0, final_step=True) is False
    assert should_log_masked_gep_stats_step(5, 0) is True
    assert should_log_masked_gep_stats_step(5, 1) is False
    assert should_log_masked_gep_stats_step(5, 4) is True
    assert should_log_masked_gep_stats_step(5, 1, final_step=True) is True


def test_summarize_batch_metadata_script_aggregates_and_decodes_source(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    metrics_path = run_dir / "metrics.0-0.jsonl"
    rows = [
        {
            "epoch": 1,
            "batch_index": 10,
            "rank": 0,
            "batch_metadata": {
                "species": {"unique_count": 2, "top": [{"id": 5, "count": 3}, {"id": 8, "count": 1}]},
                "assay": {"unique_count": 1, "top": [{"id": 2, "count": 4}]},
                "tissue": {"unique_count": 1, "top": [{"id": 21, "count": 4}]},
                "source_file_id": {"unique_count": 1, "top": [{"id": 0, "count": 4}]},
                "source_batch_id": {"unique_count": 1, "top": [{"id": 1, "count": 4}]},
            },
        },
        {
            "epoch": 1,
            "batch_index": 11,
            "rank": 0,
            "batch_metadata": {
                "species": {"unique_count": 1, "top": [{"id": 5, "count": 4}]},
                "assay": {"unique_count": 1, "top": [{"id": 4, "count": 4}]},
                "tissue": {"unique_count": 1, "top": [{"id": 34, "count": 4}]},
                "source_file_id": {"unique_count": 1, "top": [{"id": 1, "count": 4}]},
                "source_batch_id": {"unique_count": 1, "top": [{"id": 3, "count": 4}]},
            },
        },
    ]
    with metrics_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    source_manifest = tmp_path / "source_manifest.csv"
    with source_manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source_file_id",
                "source_batch_id",
                "batch_name",
                "species",
                "new_index",
                "new_filename",
                "original_index",
                "source_path",
                "target_path",
                "status",
                "size_bytes",
            ],
        )
        writer.writeheader()
        writer.writerow({
            "source_file_id": 0,
            "source_batch_id": 1,
            "batch_name": "1st",
            "species": "Homo_sapiens",
            "new_index": 0,
            "new_filename": "macrogene_0.parquet",
            "original_index": 0,
            "source_path": "/source/1st/Homo_sapiens/macrogene_0.parquet",
            "target_path": "/merged/Homo_sapiens/macrogene_0.parquet",
            "status": "done",
            "size_bytes": 123,
        })
        writer.writerow({
            "source_file_id": 1,
            "source_batch_id": 3,
            "batch_name": "3scbasecount",
            "species": "Mus_musculus",
            "new_index": 0,
            "new_filename": "macrogene_0.parquet",
            "original_index": 0,
            "source_path": "/source/3sc/Mus_musculus/macrogene_0.parquet",
            "target_path": "/merged/Mus_musculus/macrogene_0.parquet",
            "status": "done",
            "size_bytes": 456,
        })

    lookup_dir = tmp_path / "LOOKUP_categories_unified"
    lookup_dir.mkdir()
    pd = __import__("pandas")
    pd.DataFrame({"label": ["assay_0", "assay_1", "assay_2", "assay_3", "assay_4"]}).to_parquet(
        lookup_dir / "assay.parquet"
    )
    pd.DataFrame({"label": [f"tissue_{i}" for i in range(40)]}).to_parquet(
        lookup_dir / "tissue.parquet"
    )

    out_csv = tmp_path / "summary.csv"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "summarize_batch_metadata.py"),
            str(run_dir),
            "--epoch",
            "1",
            "--batch-start",
            "10",
            "--batch-end",
            "11",
            "--source-manifest",
            str(source_manifest),
            "--lookup-dir",
            str(lookup_dir),
            "--out-csv",
            str(out_csv),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "source_file_id" in result.stdout
    summary_rows = list(csv.DictReader(out_csv.open(encoding="utf-8")))
    species_5 = next(row for row in summary_rows if row["field"] == "species" and row["id"] == "5")
    assay_4 = next(row for row in summary_rows if row["field"] == "assay" and row["id"] == "4")
    tissue_34 = next(row for row in summary_rows if row["field"] == "tissue" and row["id"] == "34")
    source_1 = next(row for row in summary_rows if row["field"] == "source_file_id" and row["id"] == "1")
    batch_3 = next(row for row in summary_rows if row["field"] == "source_batch_id" and row["id"] == "3")
    assert species_5["count"] == "7"
    assert assay_4["label"] == "assay_4"
    assert tissue_34["label"] == "tissue_34"
    assert source_1["label"] == "3scbasecount:macrogene_0.parquet"
    assert batch_3["label"] == "3scbasecount"


def test_summarize_batch_metadata_script_marks_truncated_top_k(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    metrics_path = run_dir / "metrics.0-0.jsonl"
    with metrics_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "epoch": 1,
            "batch_index": 1,
            "rank": 0,
            "batch_metadata": {
                "species": {
                    "unique_count": 3,
                    "top": [{"id": 5, "count": 4}, {"id": 8, "count": 2}],
                },
            },
        }) + "\n")

    out_csv = tmp_path / "summary.csv"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "summarize_batch_metadata.py"),
            str(run_dir),
            "--out-csv",
            str(out_csv),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "truncated_metric_rows" in result.stdout
    summary_rows = list(csv.DictReader(out_csv.open(encoding="utf-8")))
    species_5 = next(row for row in summary_rows if row["field"] == "species" and row["id"] == "5")
    assert species_5["truncated_metric_rows"] == "1"
    assert species_5["max_omitted_unique_ids"] == "1"
