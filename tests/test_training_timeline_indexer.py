from __future__ import annotations

import sqlite3
from pathlib import Path

from tests.training_timeline_fixtures import make_run_dir
from training_timeline.db import list_runs
from training_timeline.indexer import index_source_roots


def test_indexer_writes_run_configs_metrics_and_artifacts(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    make_run_dir(
        root,
        "training_output_100m_data_1_2_3_stable_from_scratch_20260514_011839",
        run_record={"created_at_utc": "2026-05-13T17:18:39Z", "argv": ["train.py", "--lr", "1e-6"]},
        summary="# Stable control\n\nLoss dropped cleanly across the full epoch.\n",
        metrics_rows=[
            {"update_step": 1, "epoch": 0.0, "loss_total": 100.0, "lr": 1e-6},
            {"update_step": 2, "epoch": 0.1, "loss_total": 50.0, "lr": 9e-7},
        ],
    )
    db_path = tmp_path / "timeline.sqlite"

    result = index_source_roots(db_path, [root])

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    runs = list_runs(conn)
    assert result["indexed"] == 1
    assert runs[0]["model_size"] == "100m"
    assert runs[0]["data_recipe"] == "data_1_2_3"
    assert runs[0]["summary_title"] == "Stable control"
    assert conn.execute("SELECT COUNT(*) FROM run_configs").fetchone()[0] > 0
    assert conn.execute("SELECT COUNT(*) FROM metric_series").fetchone()[0] > 0
    assert conn.execute("SELECT best_loss FROM metric_summaries").fetchone()[0] == 50.0
    assert conn.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0] >= 3


def test_indexer_skips_unchanged_runs_and_reindexes_changed_runs(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    run = make_run_dir(
        root,
        "training_output_100m_data_1_2_3_stable_from_scratch_20260514_011839",
        metrics_rows=[{"update_step": 1, "loss_total": 100.0}],
    )
    db_path = tmp_path / "timeline.sqlite"

    first = index_source_roots(db_path, [root])
    second = index_source_roots(db_path, [root])
    with (run / "metrics.0-0.jsonl").open("a", encoding="utf-8") as handle:
        handle.write('{"update_step": 2, "loss_total": 40.0}\n')
    changed = index_source_roots(db_path, [root])

    assert first == {"discovered": 1, "indexed": 1, "skipped": 0, "warnings": 0}
    assert second["indexed"] == 0
    assert second["skipped"] == 1
    assert changed["indexed"] == 1
