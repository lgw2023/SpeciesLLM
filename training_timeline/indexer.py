from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from training_timeline.db import connect, init_db
from training_timeline.context_notes import refresh_auto_context_notes
from training_timeline.diagnostics import diagnose_run
from training_timeline.metrics import choose_metrics_file, dedupe_by_update_step, downsample_series, iter_metric_rows, summarize_metrics
from training_timeline.models import ArtifactRef, DiagnosticEvent, MetricPoint, MetricSummary, ParsedRunName, RunDiscovery, SummaryInfo
from training_timeline.parsers import collect_artifacts, parse_run_directory_name, parse_run_record, parse_summary
from training_timeline.relationships import refresh_run_relationships
from training_timeline.scanner import discover_runs


INDEX_VERSION = 1


def index_source_roots(db_path: Path, source_roots: list[Path], *, force: bool = False) -> dict[str, int]:
    conn = connect(db_path)
    init_db(conn)
    runs = discover_runs(source_roots)
    result = {"discovered": len(runs), "indexed": 0, "skipped": 0, "warnings": 0, "removed": 0}
    discovered_ids = {run.id for run in runs}
    result["removed"] = _remove_stale_runs(conn, source_roots, discovered_ids)

    for discovery in runs:
        fingerprint = _fingerprint_run(discovery.path)
        if not force and _existing_fingerprint(conn, discovery.id) == fingerprint:
            result["skipped"] += 1
            continue
        try:
            _index_one_run(conn, discovery, fingerprint)
            result["indexed"] += 1
        except Exception as exc:  # noqa: BLE001 - per-run warnings must not abort a full index pass.
            _write_warning_run(conn, discovery, fingerprint, exc)
            result["warnings"] += 1
            result["indexed"] += 1

    refresh_run_relationships(conn, source_roots)
    refresh_auto_context_notes(conn, source_roots)
    conn.close()
    return result


def _index_one_run(conn: sqlite3.Connection, discovery: RunDiscovery, fingerprint: str) -> None:
    parsed = parse_run_directory_name(discovery.name)
    run_record = parse_run_record(discovery.path)
    summary = parse_summary(discovery.path)
    artifacts = collect_artifacts(discovery.path)
    metrics_path = choose_metrics_file(discovery.path)
    rows = dedupe_by_update_step(iter_metric_rows(metrics_path)) if metrics_path is not None else []
    metric_summary = summarize_metrics(discovery.id, rows)
    metric_points = downsample_series(discovery.id, rows)
    diagnostics = diagnose_run(discovery.id, rows, metric_summary)

    with conn:
        _delete_derived_rows(conn, discovery.id)
        _insert_run(conn, discovery, parsed, run_record, summary, metric_summary)
        _insert_configs(conn, discovery.id, parsed, run_record)
        _insert_metric_summary(conn, metric_summary)
        _insert_metric_points(conn, discovery.id, metric_points)
        _insert_diagnostics(conn, diagnostics)
        _insert_artifacts(conn, discovery.id, artifacts)
        _upsert_index_state(conn, discovery.id, fingerprint)


def _remove_stale_runs(conn: sqlite3.Connection, source_roots: list[Path], discovered_ids: set[str]) -> int:
    roots = [str(root.resolve()) for root in source_roots]
    if not roots:
        return 0
    rows = conn.execute(
        f"SELECT id FROM runs WHERE source_root IN ({','.join('?' for _ in roots)})",
        tuple(roots),
    ).fetchall()
    stale_ids = [row["id"] for row in rows if row["id"] not in discovered_ids]
    if not stale_ids:
        return 0
    with conn:
        conn.executemany("DELETE FROM runs WHERE id = ?", [(run_id,) for run_id in stale_ids])
        conn.executemany("DELETE FROM index_state WHERE run_id = ?", [(run_id,) for run_id in stale_ids])
    return len(stale_ids)


def _write_warning_run(conn: sqlite3.Connection, discovery: RunDiscovery, fingerprint: str, exc: Exception) -> None:
    parsed = parse_run_directory_name(discovery.name)
    reason = f"{exc.__class__.__name__}: {exc}"
    with conn:
        _delete_derived_rows(conn, discovery.id)
        conn.execute(
            """
            INSERT OR REPLACE INTO runs (
              id, path, real_path, name, source_root, started_at, created_at_utc, mtime,
              model_size, experiment_name, data_recipe, status, status_reason,
              summary_title, summary_one_liner, git_commit, git_subject, git_dirty,
              indexed_at, index_version, parse_warnings_count
            ) VALUES (
              :id, :path, :real_path, :name, :source_root, :started_at, NULL, :mtime,
              :model_size, :experiment_name, :data_recipe, 'needs_review', :status_reason,
              '', '', NULL, NULL, NULL, :indexed_at, :index_version, 1
            )
            """,
            {
                "id": discovery.id,
                "path": str(discovery.path),
                "real_path": str(discovery.real_path),
                "name": discovery.name,
                "source_root": str(discovery.source_root),
                "started_at": parsed.started_at,
                "mtime": discovery.mtime,
                "model_size": parsed.model_size,
                "experiment_name": parsed.experiment_name,
                "data_recipe": parsed.data_recipe,
                "status_reason": reason[:500],
                "indexed_at": _utc_now(),
                "index_version": INDEX_VERSION,
            },
        )
        _upsert_index_state(conn, discovery.id, fingerprint)


def _delete_derived_rows(conn: sqlite3.Connection, run_id: str) -> None:
    for table in ["run_configs", "metric_series", "metric_summaries", "diagnostic_events", "artifacts"]:
        conn.execute(f"DELETE FROM {table} WHERE run_id = ?", (run_id,))


def _insert_run(
    conn: sqlite3.Connection,
    discovery: RunDiscovery,
    parsed: ParsedRunName,
    run_record: dict[str, Any],
    summary: SummaryInfo,
    metric_summary: MetricSummary,
) -> None:
    git_info = run_record.get("git") if isinstance(run_record.get("git"), dict) else {}
    status = "needs_review"
    status_reason = "indexed"
    if metric_summary.final_step is not None and metric_summary.best_loss is not None:
        status_reason = "metrics_indexed"

    conn.execute(
        """
        INSERT OR REPLACE INTO runs (
          id, path, real_path, name, source_root, started_at, created_at_utc, mtime,
          model_size, experiment_name, data_recipe, status, status_reason,
          summary_title, summary_one_liner, git_commit, git_subject, git_dirty,
          indexed_at, index_version, parse_warnings_count
        ) VALUES (
          :id, :path, :real_path, :name, :source_root, :started_at, :created_at_utc, :mtime,
          :model_size, :experiment_name, :data_recipe, :status, :status_reason,
          :summary_title, :summary_one_liner, :git_commit, :git_subject, :git_dirty,
          :indexed_at, :index_version, 0
        )
        """,
        {
            "id": discovery.id,
            "path": str(discovery.path),
            "real_path": str(discovery.real_path),
            "name": discovery.name,
            "source_root": str(discovery.source_root),
            "started_at": parsed.started_at,
            "created_at_utc": _as_optional_str(run_record.get("created_at_utc")),
            "mtime": discovery.mtime,
            "model_size": parsed.model_size,
            "experiment_name": parsed.experiment_name,
            "data_recipe": parsed.data_recipe,
            "status": status,
            "status_reason": status_reason,
            "summary_title": summary.title,
            "summary_one_liner": summary.one_liner,
            "git_commit": _as_optional_str(git_info.get("commit") or run_record.get("git_commit")),
            "git_subject": _as_optional_str(git_info.get("subject") or run_record.get("git_subject")),
            "git_dirty": _as_optional_bool_int(git_info.get("dirty") or run_record.get("git_dirty")),
            "indexed_at": _utc_now(),
            "index_version": INDEX_VERSION,
        },
    )


def _insert_configs(conn: sqlite3.Connection, run_id: str, parsed: ParsedRunName, run_record: dict[str, Any]) -> None:
    records: list[tuple[str, str, Any]] = [
        ("directory_name", "model_size", parsed.model_size),
        ("directory_name", "data_recipe", parsed.data_recipe),
        ("directory_name", "experiment_name", parsed.experiment_name),
        ("directory_name", "started_at", parsed.started_at),
        ("directory_name", "tags", parsed.tags),
    ]
    records.extend(("run_record", key, value) for key, value in sorted(run_record.items()))

    conn.executemany(
        """
        INSERT OR REPLACE INTO run_configs (run_id, source, key, value, value_type)
        VALUES (?, ?, ?, ?, ?)
        """,
        [(run_id, source, key, _json_value(value), _value_type(value)) for source, key, value in records],
    )


def _insert_metric_summary(conn: sqlite3.Connection, summary: MetricSummary) -> None:
    data = asdict(summary)
    conn.execute(
        """
        INSERT OR REPLACE INTO metric_summaries (
          run_id, best_loss, best_loss_step, final_loss, final_step, early_loss_mean,
          tail_loss_mean, grad_norm_p50, grad_norm_p95, grad_norm_p99, grad_norm_max,
          clip_count, clip_fraction, skip_count, skip_fraction, row_count
        ) VALUES (
          :run_id, :best_loss, :best_loss_step, :final_loss, :final_step, :early_loss_mean,
          :tail_loss_mean, :grad_norm_p50, :grad_norm_p95, :grad_norm_p99, :grad_norm_max,
          :clip_count, :clip_fraction, :skip_count, :skip_fraction, :row_count
        )
        """,
        data,
    )


def _insert_metric_points(conn: sqlite3.Connection, run_id: str, points: list[MetricPoint]) -> None:
    conn.executemany(
        """
        INSERT INTO metric_series (run_id, series_name, step, epoch, value, sample_count, aggregation)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [(run_id, point.series_name, point.step, point.epoch, point.value, point.sample_count, point.aggregation) for point in points],
    )


def _insert_diagnostics(conn: sqlite3.Connection, diagnostics: list[DiagnosticEvent]) -> None:
    conn.executemany(
        """
        INSERT OR REPLACE INTO diagnostic_events (
          id, run_id, event_type, severity, title, description,
          start_step, end_step, evidence_json, source_file, created_by
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                event.id,
                event.run_id,
                event.event_type,
                event.severity,
                event.title,
                event.description,
                event.start_step,
                event.end_step,
                json.dumps(event.evidence, ensure_ascii=False, sort_keys=True),
                event.source_file,
                event.created_by,
            )
            for event in diagnostics
        ],
    )


def _insert_artifacts(conn: sqlite3.Connection, run_id: str, artifacts: list[ArtifactRef]) -> None:
    conn.executemany(
        """
        INSERT OR REPLACE INTO artifacts (run_id, kind, path, size_bytes, mtime)
        VALUES (?, ?, ?, ?, ?)
        """,
        [(run_id, artifact.kind, str(artifact.path), artifact.size_bytes, artifact.mtime) for artifact in artifacts],
    )


def _upsert_index_state(conn: sqlite3.Connection, run_id: str, fingerprint: str) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO index_state (run_id, fingerprint, indexed_at, index_version)
        VALUES (?, ?, ?, ?)
        """,
        (run_id, fingerprint, _utc_now(), INDEX_VERSION),
    )


def _existing_fingerprint(conn: sqlite3.Connection, run_id: str) -> str | None:
    row = conn.execute("SELECT fingerprint FROM index_state WHERE run_id = ? AND index_version = ?", (run_id, INDEX_VERSION)).fetchone()
    return row["fingerprint"] if row is not None else None


def _fingerprint_run(run_dir: Path) -> str:
    entries: list[tuple[str, int, float]] = [(".", 0, run_dir.stat().st_mtime)]
    for artifact in collect_artifacts(run_dir):
        try:
            relative = artifact.path.relative_to(run_dir)
        except ValueError:
            relative = artifact.path
        entries.append((str(relative), artifact.size_bytes, artifact.mtime))
    payload = json.dumps({"version": INDEX_VERSION, "entries": entries}, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_value(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _value_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int) and not isinstance(value, bool):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "dict"
    return "str"


def _as_optional_str(value: Any) -> str | None:
    return str(value) if value is not None else None


def _as_optional_bool_int(value: Any) -> int | None:
    if value is None:
        return None
    return 1 if bool(value) else 0
