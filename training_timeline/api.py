from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from training_timeline.analysis import create_analysis_note, get_evidence_bundle, list_analysis_notes, update_analysis_note
from training_timeline.db import connect, init_db, list_runs, load_run
from training_timeline.indexer import index_source_roots


def create_app(db_path: Path, source_roots: list[Path] | None = None) -> FastAPI:
    sources = [root.resolve() for root in (source_roots or [Path.cwd()])]
    app = FastAPI(title="SpeciesLLM Training Timeline", version="0.1.0")
    app.state.db_path = db_path
    app.state.source_roots = sources
    _init_database(db_path)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "db_path": str(app.state.db_path), "source_roots": [str(root) for root in app.state.source_roots]}

    @app.get("/api/sources")
    def sources_endpoint() -> dict[str, Any]:
        with _conn(app) as conn:
            runs = list_runs(conn)
        return {"sources": [{"path": str(root)} for root in app.state.source_roots], "run_count": len(runs)}

    @app.post("/api/index/rebuild")
    def rebuild_index() -> dict[str, Any]:
        result = index_source_roots(app.state.db_path, app.state.source_roots, force=False)
        return {"result": result}

    @app.post("/api/index/runs/{run_id}/refresh")
    def refresh_run(run_id: str) -> dict[str, Any]:
        result = index_source_roots(app.state.db_path, app.state.source_roots, force=True)
        return {"run_id": run_id, "result": result}

    @app.get("/api/runs")
    def runs_endpoint() -> dict[str, Any]:
        with _conn(app) as conn:
            runs = [_with_tags(conn, run) for run in list_runs(conn)]
        return {"runs": runs}

    @app.get("/api/runs/{run_id}")
    def run_detail(run_id: str) -> dict[str, Any]:
        with _conn(app) as conn:
            run = load_run(conn, run_id)
            if run is None:
                raise HTTPException(status_code=404, detail="run not found")
            return {"run": _with_tags(conn, run)}

    @app.get("/api/runs/{run_id}/metrics")
    def run_metrics(run_id: str, series: str | None = None) -> dict[str, Any]:
        names = [item.strip() for item in series.split(",")] if series else []
        with _conn(app) as conn:
            _require_run(conn, run_id)
            if names:
                rows = conn.execute(
                    f"""
                    SELECT series_name, step, epoch, value, sample_count, aggregation
                    FROM metric_series
                    WHERE run_id = ? AND series_name IN ({','.join('?' for _ in names)})
                    ORDER BY series_name, step
                    """,
                    (run_id, *names),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT series_name, step, epoch, value, sample_count, aggregation
                    FROM metric_series
                    WHERE run_id = ?
                    ORDER BY series_name, step
                    """,
                    (run_id,),
                ).fetchall()
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            grouped.setdefault(row["series_name"], []).append(
                {
                    "step": row["step"],
                    "epoch": row["epoch"],
                    "value": row["value"],
                    "sample_count": row["sample_count"],
                    "aggregation": row["aggregation"],
                }
            )
        return {"run_id": run_id, "series": grouped}

    @app.get("/api/runs/{run_id}/configs")
    def run_configs(run_id: str) -> dict[str, Any]:
        with _conn(app) as conn:
            _require_run(conn, run_id)
            rows = conn.execute(
                "SELECT source, key, value, value_type FROM run_configs WHERE run_id = ? ORDER BY source, key",
                (run_id,),
            ).fetchall()
        return {"run_id": run_id, "configs": [dict(row) for row in rows]}

    @app.get("/api/runs/{run_id}/diagnostics")
    def run_diagnostics(run_id: str) -> dict[str, Any]:
        with _conn(app) as conn:
            _require_run(conn, run_id)
            rows = conn.execute(
                """
                SELECT *
                FROM diagnostic_events
                WHERE run_id = ?
                ORDER BY start_step, event_type
                """,
                (run_id,),
            ).fetchall()
        diagnostics = []
        for row in rows:
            item = dict(row)
            item["evidence"] = json.loads(item.pop("evidence_json"))
            diagnostics.append(item)
        return {"run_id": run_id, "diagnostics": diagnostics}

    @app.get("/api/runs/{run_id}/artifacts")
    def run_artifacts(run_id: str) -> dict[str, Any]:
        with _conn(app) as conn:
            _require_run(conn, run_id)
            rows = conn.execute(
                "SELECT kind, path, size_bytes, mtime FROM artifacts WHERE run_id = ? ORDER BY kind, path",
                (run_id,),
            ).fetchall()
        return {"run_id": run_id, "artifacts": [dict(row) for row in rows]}

    @app.get("/api/runs/{run_id}/evidence")
    def run_evidence(run_id: str) -> dict[str, Any]:
        with _conn(app) as conn:
            _require_run(conn, run_id)
            return get_evidence_bundle(conn, run_id)

    @app.post("/api/compare")
    def compare_runs(payload: dict[str, Any]) -> dict[str, Any]:
        run_ids = [str(item) for item in payload.get("run_ids", [])]
        with _conn(app) as conn:
            runs = [_with_tags(conn, run) for run_id in run_ids if (run := load_run(conn, run_id)) is not None]
            summaries = _metric_summaries(conn, run_ids)
            diagnostics = _diagnostics_for_runs(conn, run_ids)
            config_diffs = _config_diffs(conn, run_ids)
        return {"runs": runs, "metric_summaries": summaries, "diagnostics": diagnostics, "config_diffs": config_diffs}

    @app.get("/api/report/timeline")
    def report_timeline() -> dict[str, Any]:
        with _conn(app) as conn:
            runs = [_with_tags(conn, run) for run in list_runs(conn)]
        return {"runs": runs}

    @app.get("/api/report/stages")
    def report_stages() -> dict[str, Any]:
        with _conn(app) as conn:
            runs = [_with_tags(conn, run) for run in list_runs(conn)]
        stages: dict[str, list[dict[str, Any]]] = {}
        for run in runs:
            stages.setdefault(_stage_name(run), []).append(run)
        return {"stages": [{"name": name, "runs": items} for name, items in stages.items()]}

    @app.get("/api/runs/{run_id}/analysis")
    def run_analysis(run_id: str) -> dict[str, Any]:
        with _conn(app) as conn:
            _require_run(conn, run_id)
            return {"run_id": run_id, "notes": list_analysis_notes(conn, run_id)}

    @app.post("/api/runs/{run_id}/analysis")
    def create_analysis(run_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        with _conn(app) as conn:
            _require_run(conn, run_id)
            return create_analysis_note(conn, run_id, payload)

    @app.patch("/api/runs/{run_id}/analysis/{analysis_id}")
    def patch_analysis(run_id: str, analysis_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        with _conn(app) as conn:
            _require_run(conn, run_id)
            note = update_analysis_note(conn, analysis_id, payload)
            if note["run_id"] != run_id:
                raise HTTPException(status_code=404, detail="analysis note not found for run")
            return note

    return app


def _init_database(db_path: Path) -> None:
    conn = connect(db_path)
    try:
        init_db(conn)
    finally:
        conn.close()


@contextmanager
def _conn(app: FastAPI) -> Iterator[Any]:
    conn = connect(app.state.db_path)
    init_db(conn)
    try:
        yield conn
    finally:
        conn.close()


def _require_run(conn, run_id: str) -> None:
    if load_run(conn, run_id) is None:
        raise HTTPException(status_code=404, detail="run not found")


def _with_tags(conn, run: dict[str, Any]) -> dict[str, Any]:
    row = conn.execute(
        """
        SELECT value
        FROM run_configs
        WHERE run_id = ? AND source = 'directory_name' AND key = 'tags'
        """,
        (run["id"],),
    ).fetchone()
    run = dict(run)
    run["tags"] = json.loads(row["value"]) if row is not None else []
    return run


def _metric_summaries(conn, run_ids: list[str]) -> list[dict[str, Any]]:
    if not run_ids:
        return []
    rows = conn.execute(
        f"SELECT * FROM metric_summaries WHERE run_id IN ({','.join('?' for _ in run_ids)})",
        tuple(run_ids),
    ).fetchall()
    return [dict(row) for row in rows]


def _diagnostics_for_runs(conn, run_ids: list[str]) -> list[dict[str, Any]]:
    if not run_ids:
        return []
    rows = conn.execute(
        f"SELECT * FROM diagnostic_events WHERE run_id IN ({','.join('?' for _ in run_ids)}) ORDER BY run_id, event_type",
        tuple(run_ids),
    ).fetchall()
    diagnostics = []
    for row in rows:
        item = dict(row)
        item["evidence"] = json.loads(item.pop("evidence_json"))
        diagnostics.append(item)
    return diagnostics


def _config_diffs(conn, run_ids: list[str]) -> list[dict[str, Any]]:
    if not run_ids:
        return []
    rows = conn.execute(
        f"""
        SELECT run_id, source, key, value, value_type
        FROM run_configs
        WHERE run_id IN ({','.join('?' for _ in run_ids)})
        ORDER BY key, run_id
        """,
        tuple(run_ids),
    ).fetchall()
    by_key: dict[str, dict[str, str]] = {}
    for row in rows:
        key = f"{row['source']}:{row['key']}"
        by_key.setdefault(key, {})[row["run_id"]] = row["value"]
    return [{"key": key, "values": values} for key, values in by_key.items() if len(set(values.values())) > 1]


def _stage_name(run: dict[str, Any]) -> str:
    name = run.get("name", "")
    data_recipe = run.get("data_recipe", "")
    if "stab" in name or "500m_data_1_2_3_stable" in name:
        return "500M stability incident and fixes"
    if "lr" in name:
        return "100M learning-rate sweep"
    if "5epoch" in name or "epoch5" in name:
        return "Multi-epoch and resume experiments"
    if "huber" in name or "fp32" in name or "lossw" in name or "shuffleall" in name:
        return "fp32 + Huber, modality, loss-weight, and shuffle experiments"
    if data_recipe in {"data_1", "data_1_3", "data_1_2_3"}:
        return "Data 1 / 1+3 / 1+2+3 comparisons"
    return "Initial smoke and model-scale tests"
