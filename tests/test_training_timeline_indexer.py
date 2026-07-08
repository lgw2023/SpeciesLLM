from __future__ import annotations

import sqlite3
import shutil
import json
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


def test_indexer_infers_relationship_edges_with_script_evidence(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    work_record = root / "work_record"
    work_record.mkdir()
    (work_record / "step1_data_1_3.sh").write_text(
        "python train.py --data_recipe data_1_3 --lr 5e-4 --shuffle_rows\n",
        encoding="utf-8",
    )
    make_run_dir(
        root,
        "training_output_100m_data_1_from_scratch_20260501_000000",
        run_record={"argv": ["train.py"]},
        metrics_rows=[{"update_step": 1, "loss_total": 100.0}],
    )
    make_run_dir(
        root,
        "training_output_100m_data_1_3_stable_lr5em4_shuffleall_from_scratch_20260502_000000",
        run_record={"argv": ["train.py", "--data_recipe", "data_1_3", "--lr", "5e-4", "--shuffle_rows"]},
        metrics_rows=[{"update_step": 1, "loss_total": 80.0}],
    )
    db_path = tmp_path / "timeline.sqlite"

    index_source_roots(db_path, [root])

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    edge = conn.execute("SELECT * FROM run_relationships").fetchone()
    assert edge is not None
    assert "training data recipe: data_1 -> data_1_3" in edge["change_summary"]
    assert "learning rate: default -> 5e-4" in edge["change_summary"]
    evidence = json.loads(edge["evidence_json"])
    assert {item["kind"] for item in evidence} >= {"run_config", "script"}
    assert any("work_record/step1_data_1_3.sh" in item["ref"] for item in evidence)


def test_indexer_writes_auto_context_notes_from_records(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    work_record = root / "work_record"
    work_record.mkdir()
    (work_record / "stability_experiments.sh").write_text(
        "stable_lr5em4_from_scratch uses data_1_3 with lr5em4 as a learning-rate sweep.\n",
        encoding="utf-8",
    )
    memory_root = tmp_path / "memory"
    memory_root.mkdir()
    (memory_root / "MEMORY.md").write_text(
        "SpeciesLLM training timeline: data_1_3 stable_lr5em4 is a low-learning-rate follow-up.\n",
        encoding="utf-8",
    )
    sessions_root = tmp_path / "sessions"
    sessions_root.mkdir()
    (sessions_root / "conversation.jsonl").write_text(
        '{"role":"user","content":"stable_lr5em4 和 data_1_3 的关系需要保留在训练时间线里"}\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("TRAINING_TIMELINE_MEMORY_ROOT", str(memory_root))
    monkeypatch.setenv("TRAINING_TIMELINE_SESSIONS_ROOT", str(sessions_root))
    make_run_dir(
        root,
        "training_output_100m_data_1_from_scratch_20260501_000000",
        metrics_rows=[{"update_step": 1, "loss_total": 10.0}],
    )
    make_run_dir(
        root,
        "training_output_100m_data_1_3_stable_lr5em4_from_scratch_20260502_000000",
        metrics_rows=[{"update_step": 1, "loss_total": 8.0}],
    )
    db_path = tmp_path / "timeline.sqlite"

    index_source_roots(db_path, [root])

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    child_id = conn.execute("SELECT id FROM runs WHERE data_recipe = 'data_1_3'").fetchone()["id"]
    note = conn.execute(
        "SELECT * FROM analysis_notes WHERE run_id = ? AND author = 'auto-context'",
        (child_id,),
    ).fetchone()
    assert note is not None
    assert note["title"] == "Auto context inference"
    assert "实验定位" in note["body"]
    assert "training data recipe: data_1 -> data_1_3" in note["body"]
    evidence_refs = json.loads(note["evidence_refs"])
    assert any(ref.startswith("script:work_record/stability_experiments.sh") for ref in evidence_refs)
    assert any(ref.startswith("memory:MEMORY.md") for ref in evidence_refs)
    assert any(ref.startswith("conversation:conversation.jsonl") for ref in evidence_refs)


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

    assert first == {"discovered": 1, "indexed": 1, "skipped": 0, "warnings": 0, "removed": 0}
    assert second["indexed"] == 0
    assert second["skipped"] == 1
    assert changed["indexed"] == 1


def test_indexer_removes_stale_runs_from_same_source_root(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    run = make_run_dir(
        root,
        "training_output_100m_data_1_2_3_stable_from_scratch_20260514_011839",
        metrics_rows=[{"update_step": 1, "loss_total": 100.0}],
    )
    db_path = tmp_path / "timeline.sqlite"

    index_source_roots(db_path, [root])
    shutil.rmtree(run)
    result = index_source_roots(db_path, [root])

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    assert result["removed"] == 1
    assert list_runs(conn) == []
