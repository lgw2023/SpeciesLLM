from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from training_timeline.config import is_inside_source


NOTE_TYPES = {"curated_note", "deep_review"}
CONFIDENCE_VALUES = {"low", "medium", "high"}
EVIDENCE_KINDS = {"summary_md", "run_record_json", "loss_log", "log"}


def list_analysis_notes(conn: sqlite3.Connection, run_id: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT *
        FROM analysis_notes
        WHERE run_id = ?
        ORDER BY CASE WHEN author = 'auto-context' THEN 1 ELSE 0 END, created_at, id
        """,
        (run_id,),
    ).fetchall()
    return [_decode_note(dict(row)) for row in rows]


def create_analysis_note(conn: sqlite3.Connection, run_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    note = _validated_payload(run_id, payload)
    now = _utc_now()
    note_id = uuid.uuid4().hex
    with conn:
        conn.execute(
            """
            INSERT INTO analysis_notes (
              id, run_id, note_type, title, body, confidence,
              supersedes_diagnostic_ids, evidence_refs, author, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                note_id,
                run_id,
                note["note_type"],
                note["title"],
                note["body"],
                note["confidence"],
                json.dumps(note["supersedes_diagnostic_ids"], ensure_ascii=False),
                json.dumps(note["evidence_refs"], ensure_ascii=False),
                note["author"],
                now,
                now,
            ),
        )
    return _load_note(conn, note_id)


def update_analysis_note(conn: sqlite3.Connection, analysis_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    current = _load_note(conn, analysis_id)
    merged = {**current, **payload}
    note = _validated_payload(str(current["run_id"]), merged)
    updated_at = _utc_now()
    with conn:
        conn.execute(
            """
            UPDATE analysis_notes
            SET note_type = ?, title = ?, body = ?, confidence = ?,
                supersedes_diagnostic_ids = ?, evidence_refs = ?, author = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                note["note_type"],
                note["title"],
                note["body"],
                note["confidence"],
                json.dumps(note["supersedes_diagnostic_ids"], ensure_ascii=False),
                json.dumps(note["evidence_refs"], ensure_ascii=False),
                note["author"],
                updated_at,
                analysis_id,
            ),
        )
    return _load_note(conn, analysis_id)


def get_evidence_bundle(conn: sqlite3.Connection, run_id: str, *, max_chars_per_file: int = 8000) -> dict[str, Any]:
    run = conn.execute("SELECT id, source_root FROM runs WHERE id = ?", (run_id,)).fetchone()
    if run is None:
        raise KeyError(f"Unknown run_id: {run_id}")

    source_roots = [Path(run["source_root"])]
    artifact_rows = conn.execute(
        """
        SELECT kind, path, size_bytes, mtime
        FROM artifacts
        WHERE run_id = ?
        ORDER BY kind, path
        """,
        (run_id,),
    ).fetchall()
    max_chars = max(0, max_chars_per_file)
    snippets: list[dict[str, Any]] = []
    for artifact in artifact_rows:
        kind = artifact["kind"]
        if kind not in EVIDENCE_KINDS:
            continue
        path = Path(artifact["path"]).resolve()
        if not is_inside_source(path, source_roots) or not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        snippet = text[:max_chars]
        snippets.append(
            {
                "kind": kind,
                "path": str(path),
                "size_bytes": artifact["size_bytes"],
                "mtime": artifact["mtime"],
                "snippet": snippet,
                "truncated": len(text) > max_chars,
            }
        )
    return {"run_id": run_id, "snippets": snippets}


def _load_note(conn: sqlite3.Connection, analysis_id: str) -> dict[str, Any]:
    row = conn.execute("SELECT * FROM analysis_notes WHERE id = ?", (analysis_id,)).fetchone()
    if row is None:
        raise KeyError(f"Unknown analysis_id: {analysis_id}")
    return _decode_note(dict(row))


def _decode_note(row: dict[str, Any]) -> dict[str, Any]:
    row["supersedes_diagnostic_ids"] = json.loads(row["supersedes_diagnostic_ids"])
    row["evidence_refs"] = json.loads(row["evidence_refs"])
    return row


def _validated_payload(run_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    note_type = str(payload.get("note_type", "curated_note"))
    confidence = str(payload.get("confidence", "medium"))
    if note_type not in NOTE_TYPES:
        raise ValueError(f"Unsupported note_type: {note_type}")
    if confidence not in CONFIDENCE_VALUES:
        raise ValueError(f"Unsupported confidence: {confidence}")
    return {
        "run_id": run_id,
        "note_type": note_type,
        "title": str(payload.get("title", "")).strip(),
        "body": str(payload.get("body", "")).strip(),
        "confidence": confidence,
        "supersedes_diagnostic_ids": _string_list(payload.get("supersedes_diagnostic_ids", [])),
        "evidence_refs": _string_list(payload.get("evidence_refs", [])),
        "author": str(payload.get("author", "local")).strip() or "local",
    }


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
