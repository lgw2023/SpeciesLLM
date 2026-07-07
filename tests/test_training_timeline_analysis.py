from __future__ import annotations

from pathlib import Path

from tests.training_timeline_fixtures import make_run_dir
from training_timeline.analysis import create_analysis_note, get_evidence_bundle, list_analysis_notes, update_analysis_note
from training_timeline.db import connect
from training_timeline.indexer import index_source_roots


def test_analysis_note_create_update_and_list(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    make_run_dir(root, "training_output_100m_data_1_2_3_stable_from_scratch_20260514_011839")
    db_path = tmp_path / "timeline.sqlite"
    index_source_roots(db_path, [root])
    conn = connect(db_path)
    run_id = conn.execute("SELECT id FROM runs").fetchone()["id"]

    note = create_analysis_note(
        conn,
        run_id,
        {
            "note_type": "deep_review",
            "title": "Manual interpretation",
            "body": "Manual read confirms this is a healthy control run.",
            "confidence": "high",
            "supersedes_diagnostic_ids": [],
            "evidence_refs": ["summary.md"],
            "author": "local",
        },
    )
    updated = update_analysis_note(conn, note["id"], {"confidence": "medium", "body": "Manual read is useful but still preliminary."})
    notes = list_analysis_notes(conn, run_id)

    assert updated["confidence"] == "medium"
    assert notes[0]["title"] == "Manual interpretation"


def test_evidence_bundle_is_bounded_and_uses_artifacts(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    run = make_run_dir(
        root,
        "training_output_100m_data_1_2_3_stable_from_scratch_20260514_011839",
        summary="# Summary\n\n" + ("A" * 100),
    )
    (run / "log.0-0.txt").write_text("B" * 100, encoding="utf-8")
    db_path = tmp_path / "timeline.sqlite"
    index_source_roots(db_path, [root])
    conn = connect(db_path)
    run_id = conn.execute("SELECT id FROM runs").fetchone()["id"]

    bundle = get_evidence_bundle(conn, run_id, max_chars_per_file=20)

    assert bundle["run_id"] == run_id
    assert all(len(item["snippet"]) <= 20 for item in bundle["snippets"])
    assert {item["kind"] for item in bundle["snippets"]} >= {"summary_md", "log"}
