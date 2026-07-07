# Training Timeline Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local FastAPI + React/Vite + SQLite application that reconstructs and presents the SpeciesLLM training history from existing `training_output*` directories.

**Architecture:** The backend scans configured local source roots, extracts run metadata and bounded metric summaries, stores derived records in a rebuildable SQLite index, and serves FastAPI endpoints. The frontend consumes only API responses and renders timeline, run detail, comparison, report, source, and curated-analysis views.

**Tech Stack:** Python 3, FastAPI, SQLite, pytest, React, TypeScript, Vite, Vitest, React Testing Library, Recharts, lucide-react.

## Global Constraints

- This is an offline retrospective tool; do not run training, evaluation, checkpoint loading, full indexing over datasets, or remote sync locally.
- Large datasets, generated parquet shards, checkpoints, and server-only artifacts may be absent on this workstation and must not be treated as broken state.
- The real training stack is Huawei Ascend NPU with CANN `8.2.1.RC1` and `torch_npu`; do not introduce CUDA-only assumptions or `bf16` assumptions.
- Raw training output directories are read-only inputs; indexing must not modify files under `training_output*`.
- Default source root is `/Volumes/SSD1/SpeciesLLM`; extra source roots are configurable.
- Candidate run directories match `training_output*`; `*_text_split` archive mirrors are skipped in the first version.
- All file reads must stay inside configured source roots.
- Frontend data requests must be bounded and use downsampled curves by default.
- Automatic diagnostics are preliminary, evidence-bound triage, not final scientific judgment.
- Curated and deep-review analysis notes must be stored separately from automatic diagnostics and can supplement or supersede them.

---

## Scope Check

The spec covers one product surface: a local training timeline dashboard. Backend indexing and frontend presentation are coupled by the API contract, so this plan keeps them together while sequencing work as independently testable vertical slices.

## File Structure

All paths are relative to `/Volumes/SSD1/SpeciesLLM`.

Backend package:

- Create `training_timeline/__init__.py`: package marker and version.
- Create `training_timeline/models.py`: dataclasses and type aliases shared by scanner, parser, indexer, diagnostics, and API layers.
- Create `training_timeline/schema.sql`: SQLite tables and indexes.
- Create `training_timeline/config.py`: source-root config loading and path containment checks.
- Create `training_timeline/scanner.py`: discovery of candidate run directories, symlink resolution, skip rules, and duplicate prevention.
- Create `training_timeline/parsers.py`: directory-name, `run_record.json`, `summary.md`, log, and artifact parsing.
- Create `training_timeline/metrics.py`: streaming metrics JSONL parsing, rank-file selection, resume deduplication, summary statistics, and downsampling.
- Create `training_timeline/db.py`: SQLite connection, schema initialization, and small persistence helpers.
- Create `training_timeline/indexer.py`: incremental indexing orchestration and transaction boundaries.
- Create `training_timeline/diagnostics.py`: conservative automatic diagnostic rules and sweep summaries.
- Create `training_timeline/analysis.py`: curated/deep-review notes and bounded evidence retrieval.
- Create `training_timeline/api.py`: FastAPI app factory and routes.
- Create `training_timeline/cli.py`: local rebuild and serve entrypoints.
- Create `requirements-training-timeline.txt`: backend dependencies.

Backend tests:

- Create `tests/training_timeline_fixtures.py`: synthetic run-directory fixture builders.
- Create `tests/test_training_timeline_models.py`
- Create `tests/test_training_timeline_scanner_parsers.py`
- Create `tests/test_training_timeline_metrics.py`
- Create `tests/test_training_timeline_indexer.py`
- Create `tests/test_training_timeline_diagnostics.py`
- Create `tests/test_training_timeline_analysis.py`
- Create `tests/test_training_timeline_api.py`

Frontend package:

- Create `training_timeline_ui/package.json`
- Create `training_timeline_ui/package-lock.json`
- Create `training_timeline_ui/index.html`
- Create `training_timeline_ui/vite.config.ts`
- Create `training_timeline_ui/tsconfig.json`
- Create `training_timeline_ui/src/main.tsx`
- Create `training_timeline_ui/src/App.tsx`
- Create `training_timeline_ui/src/api.ts`
- Create `training_timeline_ui/src/types.ts`
- Create `training_timeline_ui/src/styles.css`
- Create `training_timeline_ui/src/pages/TimelinePage.tsx`
- Create `training_timeline_ui/src/pages/SourcesPage.tsx`
- Create `training_timeline_ui/src/pages/RunDetailPage.tsx`
- Create `training_timeline_ui/src/pages/ComparePage.tsx`
- Create `training_timeline_ui/src/pages/ReportPage.tsx`
- Create `training_timeline_ui/src/components/AnalysisPanel.tsx`
- Create `training_timeline_ui/src/components/MetricChart.tsx`
- Create `training_timeline_ui/src/components/RunStatusBadge.tsx`
- Create `training_timeline_ui/src/test/App.test.tsx`
- Create `training_timeline_ui/src/test/TimelinePage.test.tsx`
- Create `training_timeline_ui/src/test/RunDetailPage.test.tsx`
- Create `training_timeline_ui/src/test/CompareReportAnalysis.test.tsx`

Operational files:

- Create `scripts/run_training_timeline_backend.sh`
- Create `scripts/run_training_timeline_frontend.sh`
- Create `docs/training_timeline_dashboard.md`

---

### Task 1: Backend Package Skeleton, Types, Schema, and Fixtures

**Files:**
- Create: `requirements-training-timeline.txt`
- Create: `training_timeline/__init__.py`
- Create: `training_timeline/models.py`
- Create: `training_timeline/schema.sql`
- Create: `tests/training_timeline_fixtures.py`
- Create: `tests/test_training_timeline_models.py`

**Interfaces:**
- Produces: `RunDiscovery`, `ParsedRunName`, `SummaryInfo`, `MetricPoint`, `MetricSummary`, `DiagnosticEvent`, `AnalysisNote`, and `ArtifactRef` dataclasses from `training_timeline.models`.
- Produces: `tests.training_timeline_fixtures.make_run_dir(root: Path, name: str, *, run_record: dict | None = None, summary: str | None = None, metrics_rows: list[dict] | None = None) -> Path`.
- Later tasks use the table names from `training_timeline/schema.sql`: `runs`, `run_configs`, `metric_series`, `metric_summaries`, `diagnostic_events`, `analysis_notes`, `artifacts`, `index_state`.

- [ ] **Step 1: Write the failing model and schema tests**

Create `tests/test_training_timeline_models.py` with:

```python
from __future__ import annotations

from pathlib import Path

from training_timeline.models import (
    AnalysisNote,
    ArtifactRef,
    DiagnosticEvent,
    MetricPoint,
    MetricSummary,
    ParsedRunName,
    RunDiscovery,
    SummaryInfo,
)


def test_core_models_are_small_serializable_records() -> None:
    run = RunDiscovery(
        id="run-1",
        path=Path("/repo/training_output_demo_20260514_011839"),
        real_path=Path("/repo/training_output_demo_20260514_011839"),
        name="training_output_demo_20260514_011839",
        source_root=Path("/repo"),
        mtime=1.0,
    )
    parsed = ParsedRunName(
        model_size="100m",
        data_recipe="data_1_2_3",
        experiment_name="stable_from_scratch",
        started_at="2026-05-14T01:18:39",
        tags=["stable"],
    )
    summary = SummaryInfo(title="stable run", one_liner="loss improved", text="full summary")
    point = MetricPoint(series_name="loss_total", step=10, epoch=0.0, value=12.5, sample_count=1, aggregation="raw")
    metric_summary = MetricSummary(run_id="run-1", best_loss=10.0, best_loss_step=20, final_loss=12.0, final_step=30)
    event = DiagnosticEvent(
        id="evt-1",
        run_id="run-1",
        event_type="converged",
        severity="info",
        title="Converged",
        description="tail loss is stable",
        evidence={"tail_loss": 10.0},
        created_by="auto",
    )
    note = AnalysisNote(
        id="note-1",
        run_id="run-1",
        note_type="deep_review",
        title="Manual read",
        body="Manual review agrees with the automatic convergence event.",
        confidence="high",
        supersedes_diagnostic_ids=["evt-1"],
        evidence_refs=["summary.md"],
        author="local",
        created_at="2026-07-07T00:00:00",
        updated_at="2026-07-07T00:00:00",
    )
    artifact = ArtifactRef(run_id="run-1", kind="summary_md", path=Path("/repo/run/summary.md"), size_bytes=12, mtime=2.0)

    assert run.name.startswith("training_output")
    assert parsed.model_size == "100m"
    assert summary.one_liner == "loss improved"
    assert point.series_name == "loss_total"
    assert metric_summary.best_loss_step == 20
    assert event.evidence["tail_loss"] == 10.0
    assert note.supersedes_diagnostic_ids == ["evt-1"]
    assert artifact.kind == "summary_md"


def test_schema_declares_expected_tables() -> None:
    schema = Path("training_timeline/schema.sql").read_text(encoding="utf-8")
    for table in [
        "runs",
        "run_configs",
        "metric_series",
        "metric_summaries",
        "diagnostic_events",
        "analysis_notes",
        "artifacts",
        "index_state",
    ]:
        assert f"CREATE TABLE IF NOT EXISTS {table}" in schema
```

- [ ] **Step 2: Run the new test to verify it fails**

Run: `python -m pytest tests/test_training_timeline_models.py -v`

Expected: FAIL because `training_timeline.models` does not exist.

- [ ] **Step 3: Create backend dependency file**

Create `requirements-training-timeline.txt` with:

```text
fastapi>=0.111
uvicorn[standard]>=0.30
pydantic>=2.7
python-multipart>=0.0.9
httpx>=0.27
```

- [ ] **Step 4: Create the package marker**

Create `training_timeline/__init__.py` with:

```python
from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
```

- [ ] **Step 5: Create the shared models**

Create `training_timeline/models.py` with dataclasses that match the test imports and field names exactly. Use `Path` for local file paths and plain `dict[str, object]` for diagnostic evidence.

- [ ] **Step 6: Create the SQLite schema**

Create `training_timeline/schema.sql` with all eight tables named in Step 1. Add these required constraints:

```sql
CREATE TABLE IF NOT EXISTS runs (
  id TEXT PRIMARY KEY,
  path TEXT NOT NULL,
  real_path TEXT NOT NULL UNIQUE,
  name TEXT NOT NULL,
  source_root TEXT NOT NULL,
  started_at TEXT,
  created_at_utc TEXT,
  mtime REAL NOT NULL,
  model_size TEXT NOT NULL DEFAULT 'unknown',
  experiment_name TEXT NOT NULL DEFAULT '',
  data_recipe TEXT NOT NULL DEFAULT '',
  status TEXT NOT NULL DEFAULT 'needs_review',
  status_reason TEXT NOT NULL DEFAULT '',
  summary_title TEXT NOT NULL DEFAULT '',
  summary_one_liner TEXT NOT NULL DEFAULT '',
  git_commit TEXT,
  git_subject TEXT,
  git_dirty INTEGER,
  indexed_at TEXT NOT NULL,
  index_version INTEGER NOT NULL,
  parse_warnings_count INTEGER NOT NULL DEFAULT 0
);
```

Also create indexes for `runs(started_at)`, `run_configs(run_id, key)`, `metric_series(run_id, series_name, step)`, `diagnostic_events(run_id, event_type)`, and `analysis_notes(run_id, note_type)`.

- [ ] **Step 7: Create fixture helpers**

Create `tests/training_timeline_fixtures.py` with `make_run_dir`. It must create optional `run_record.json`, `summary.md`, and `metrics.0-0.jsonl` files using small synthetic data only.

- [ ] **Step 8: Run the model test to verify it passes**

Run: `python -m pytest tests/test_training_timeline_models.py -v`

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add requirements-training-timeline.txt training_timeline/__init__.py training_timeline/models.py training_timeline/schema.sql tests/training_timeline_fixtures.py tests/test_training_timeline_models.py
git commit -m "feat: add training timeline backend skeleton"
```

---

### Task 2: Source Configuration, Directory Scanner, and Lightweight Parsers

**Files:**
- Create: `training_timeline/config.py`
- Create: `training_timeline/scanner.py`
- Create: `training_timeline/parsers.py`
- Create: `tests/test_training_timeline_scanner_parsers.py`
- Modify: `tests/training_timeline_fixtures.py`

**Interfaces:**
- Consumes: `RunDiscovery`, `ParsedRunName`, `SummaryInfo`, and `ArtifactRef` from `training_timeline.models`.
- Produces: `load_source_roots(config_path: Path | None, repo_root: Path) -> list[Path]`.
- Produces: `is_inside_source(path: Path, source_roots: list[Path]) -> bool`.
- Produces: `discover_runs(source_roots: list[Path]) -> list[RunDiscovery]`.
- Produces: `parse_run_directory_name(name: str) -> ParsedRunName`.
- Produces: `parse_run_record(run_dir: Path) -> dict[str, object]`.
- Produces: `parse_summary(run_dir: Path) -> SummaryInfo`.
- Produces: `collect_artifacts(run_dir: Path) -> list[ArtifactRef]`.

- [ ] **Step 1: Write failing scanner and parser tests**

Create `tests/test_training_timeline_scanner_parsers.py` with tests for:

```python
from __future__ import annotations

import json
from pathlib import Path

from tests.training_timeline_fixtures import make_run_dir
from training_timeline.config import is_inside_source, load_source_roots
from training_timeline.parsers import collect_artifacts, parse_run_directory_name, parse_run_record, parse_summary
from training_timeline.scanner import discover_runs


def test_load_source_roots_uses_repo_and_extra_roots(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    extra = tmp_path / "extra"
    repo.mkdir()
    extra.mkdir()
    config = tmp_path / "sources.json"
    config.write_text(json.dumps({"extra_source_roots": [str(extra)]}), encoding="utf-8")

    roots = load_source_roots(config, repo)

    assert roots == [repo.resolve(), extra.resolve()]


def test_is_inside_source_blocks_outside_paths(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    inside = root / "training_output_demo"
    outside = tmp_path / "outside"

    assert is_inside_source(inside, [root]) is True
    assert is_inside_source(outside, [root]) is False


def test_discover_runs_skips_text_split_and_deduplicates_symlinks(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    run = make_run_dir(root, "training_output_100m_data_1_2_3_stable_from_scratch_20260514_011839")
    make_run_dir(root, "training_output_100m_data_1_2_3_stable_from_scratch_20260514_011839_text_split")
    symlink = root / "training_output_duplicate_link"
    symlink.symlink_to(run, target_is_directory=True)

    runs = discover_runs([root])

    assert [item.name for item in runs] == ["training_output_100m_data_1_2_3_stable_from_scratch_20260514_011839"]


def test_parse_run_directory_name_extracts_main_tags() -> None:
    parsed = parse_run_directory_name(
        "training_output_100m_data_1_3_E2_huber5_fp32_esm2_dnaseq_lossw_gepc01_shuffleall_from_scratch_20260622_162826"
    )

    assert parsed.model_size == "100m"
    assert parsed.data_recipe == "data_1_3"
    assert parsed.started_at == "2026-06-22T16:28:26"
    assert "huber5" in parsed.tags
    assert "fp32" in parsed.tags
    assert "shuffleall" in parsed.tags
    assert "lossw_gepc01" in parsed.tags


def test_parse_run_record_summary_and_artifacts(tmp_path: Path) -> None:
    run = make_run_dir(
        tmp_path,
        "training_output_500m_data_1_2_3_stable_from_scratch_20260515_195223",
        run_record={"created_at_utc": "2026-05-15T11:52:23Z", "argv": ["train.py"], "git": {"commit": "abc"}},
        summary="# Failure summary\n\nThe primary head failed while skip behavior repeated.\n",
        metrics_rows=[{"update_step": 1, "loss_total": 10.0}],
    )
    (run / "loss_detail.png").write_bytes(b"png")

    record = parse_run_record(run)
    summary = parse_summary(run)
    artifacts = collect_artifacts(run)

    assert record["created_at_utc"] == "2026-05-15T11:52:23Z"
    assert summary.title == "Failure summary"
    assert "primary head" in summary.one_liner
    assert {item.kind for item in artifacts} >= {"summary_md", "run_record_json", "metrics_jsonl", "training_curve_png"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_training_timeline_scanner_parsers.py -v`

Expected: FAIL because scanner and parser modules do not exist.

- [ ] **Step 3: Implement config loading and containment**

Create `training_timeline/config.py`. Read JSON config with shape `{"extra_source_roots": ["/path"]}`. Return resolved paths. `is_inside_source` must compare resolved paths and accept a path equal to the root or under it.

- [ ] **Step 4: Implement scanner**

Create `training_timeline/scanner.py`. Discovery must:

- Iterate direct children of each source root.
- Keep names that start with `training_output`.
- Skip names ending in `_text_split`.
- Skip non-directories.
- Resolve real paths.
- Deduplicate by real path.
- Build stable IDs from `sha1(f"{real_path}|{name}")[:16]`.
- Sort by directory name for deterministic output.

- [ ] **Step 5: Implement parsers**

Create `training_timeline/parsers.py`. Directory-name parsing must detect model sizes `100m`, `500m`, `1b`; data recipes such as `data_1`, `data_1_3`, and `data_1_2_3`; timestamps matching `YYYYMMDD_HHMMSS`; exact tags `stable`, `fp32`, `esm2`, `dnaseq`, `shuffleall`, `resume`, and `from_scratch`; and prefix tags beginning with `lr`, `epoch`, `huber`, `lossw`, or `clip`.

- [ ] **Step 6: Verify fixture helper writes metrics JSONL**

Update `tests/training_timeline_fixtures.py` so metric rows are written as one JSON object per line and optional summary/run-record files use UTF-8.

- [ ] **Step 7: Run scanner and parser tests**

Run: `python -m pytest tests/test_training_timeline_scanner_parsers.py -v`

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add training_timeline/config.py training_timeline/scanner.py training_timeline/parsers.py tests/training_timeline_fixtures.py tests/test_training_timeline_scanner_parsers.py
git commit -m "feat: discover and parse training timeline runs"
```

---

### Task 3: Streaming Metrics, Resume Deduplication, Summaries, and Downsampling

**Files:**
- Create: `training_timeline/metrics.py`
- Create: `tests/test_training_timeline_metrics.py`
- Modify: `training_timeline/models.py`

**Interfaces:**
- Consumes: `MetricPoint` and `MetricSummary`.
- Produces: `choose_metrics_file(run_dir: Path) -> Path | None`.
- Produces: `iter_metric_rows(metrics_path: Path) -> Iterator[dict[str, object]]`.
- Produces: `dedupe_by_update_step(rows: Iterable[dict[str, object]]) -> list[dict[str, object]]`.
- Produces: `summarize_metrics(run_id: str, rows: list[dict[str, object]]) -> MetricSummary`.
- Produces: `downsample_series(run_id: str, rows: list[dict[str, object]], max_points: int = 1200) -> list[MetricPoint]`.

- [ ] **Step 1: Write failing metrics tests**

Create `tests/test_training_timeline_metrics.py` with:

```python
from __future__ import annotations

from pathlib import Path

from tests.training_timeline_fixtures import make_run_dir
from training_timeline.metrics import (
    choose_metrics_file,
    dedupe_by_update_step,
    downsample_series,
    iter_metric_rows,
    summarize_metrics,
)


def test_choose_metrics_file_prefers_rank_zero(tmp_path: Path) -> None:
    run = make_run_dir(tmp_path, "training_output_demo_20260514_011839")
    (run / "metrics.0-2.jsonl").write_text('{"update_step": 1, "loss_total": 9.0}\n', encoding="utf-8")
    (run / "metrics.0-0.jsonl").write_text('{"update_step": 1, "loss_total": 8.0}\n', encoding="utf-8")

    assert choose_metrics_file(run) == run / "metrics.0-0.jsonl"


def test_choose_metrics_file_falls_back_to_any_rank(tmp_path: Path) -> None:
    run = make_run_dir(tmp_path, "training_output_demo_20260514_011839")
    (run / "metrics.0-3.jsonl").write_text('{"update_step": 1, "loss_total": 7.0}\n', encoding="utf-8")

    assert choose_metrics_file(run) == run / "metrics.0-3.jsonl"


def test_stream_rows_dedupes_resume_steps_and_summarizes(tmp_path: Path) -> None:
    run = make_run_dir(
        tmp_path,
        "training_output_demo_20260514_011839",
        metrics_rows=[
            {"update_step": 1, "epoch": 0.0, "loss_total": 100.0, "lr": 1e-6},
            {"update_step": 2, "epoch": 0.0, "loss_total": 80.0, "lr": 1e-6},
            {"update_step": 2, "epoch": 0.1, "loss_total": 70.0, "lr": 9e-7},
            {"update_step": 3, "epoch": 0.2, "loss_total": 60.0, "lr": 8e-7, "grad_norm_raw": 10.0},
        ],
    )

    rows = list(iter_metric_rows(run / "metrics.0-0.jsonl"))
    deduped = dedupe_by_update_step(rows)
    summary = summarize_metrics("run-1", deduped)
    points = downsample_series("run-1", deduped, max_points=2)

    assert [row["loss_total"] for row in deduped] == [100.0, 70.0, 60.0]
    assert summary.best_loss == 60.0
    assert summary.best_loss_step == 3
    assert summary.final_loss == 60.0
    assert summary.final_step == 3
    assert len([p for p in points if p.series_name == "loss_total"]) <= 2
    assert {p.series_name for p in points} >= {"loss_total", "lr", "grad_norm_raw"}
```

- [ ] **Step 2: Run metrics tests to verify they fail**

Run: `python -m pytest tests/test_training_timeline_metrics.py -v`

Expected: FAIL because `training_timeline.metrics` does not exist.

- [ ] **Step 3: Implement streaming JSONL parser**

Create `training_timeline/metrics.py`. Use file iteration and `json.loads(line)` so a metrics file is processed line by line. Ignore blank lines. Raise `ValueError` with file path and line number for malformed JSON.

- [ ] **Step 4: Implement rank preference and resume dedupe**

`choose_metrics_file` must prefer `metrics.0-0.jsonl`; if absent, return the lexicographically first `metrics.*.jsonl`. `dedupe_by_update_step` must keep the last row for each `update_step` and return rows sorted by numeric step.

- [ ] **Step 5: Implement summaries and downsampling**

Recognize series keys:

```python
SERIES_KEYS = {
    "loss_total": ["loss_total", "loss"],
    "loss_gep": ["loss_gep"],
    "loss_zero_prob": ["loss_zero_prob"],
    "loss_gepc": ["loss_gepc"],
    "lr": ["lr", "learning_rate"],
    "grad_norm_raw": ["grad_norm_raw"],
    "clip_fraction": ["clip_fraction", "clip_fraction_rolling"],
    "skip_fraction": ["skip_fraction", "skip_fraction_rolling"],
    "samples_per_s": ["samples_per_s"],
    "mfu": ["mfu"],
}
```

For downsampling, use deterministic stride selection per series: if a series has `n <= max_points`, return all points; otherwise keep every `ceil(n / max_points)` row and append the final point if the stride missed it.

- [ ] **Step 6: Run metrics tests**

Run: `python -m pytest tests/test_training_timeline_metrics.py -v`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add training_timeline/models.py training_timeline/metrics.py tests/test_training_timeline_metrics.py
git commit -m "feat: parse and summarize training metrics"
```

---

### Task 4: SQLite Persistence and Incremental Indexing

**Files:**
- Create: `training_timeline/db.py`
- Create: `training_timeline/indexer.py`
- Create: `tests/test_training_timeline_indexer.py`
- Modify: `training_timeline/schema.sql`

**Interfaces:**
- Consumes: scanner, parsers, metrics, and schema from Tasks 1-3.
- Produces: `connect(db_path: Path) -> sqlite3.Connection`.
- Produces: `init_db(conn: sqlite3.Connection) -> None`.
- Produces: `index_source_roots(db_path: Path, source_roots: list[Path], *, force: bool = False) -> dict[str, int]`.
- Produces: `load_run(conn: sqlite3.Connection, run_id: str) -> dict[str, object] | None`.
- Produces: `list_runs(conn: sqlite3.Connection) -> list[dict[str, object]]`.

- [ ] **Step 1: Write failing indexer tests**

Create `tests/test_training_timeline_indexer.py` with:

```python
from __future__ import annotations

import sqlite3
from pathlib import Path

from tests.training_timeline_fixtures import make_run_dir
from training_timeline.db import connect, init_db, list_runs
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
        handle.write('{"update_step": 2, "loss_total": 40.0}\\n')
    changed = index_source_roots(db_path, [root])

    assert first == {"discovered": 1, "indexed": 1, "skipped": 0, "warnings": 0}
    assert second["indexed"] == 0
    assert second["skipped"] == 1
    assert changed["indexed"] == 1
```

- [ ] **Step 2: Run indexer tests to verify they fail**

Run: `python -m pytest tests/test_training_timeline_indexer.py -v`

Expected: FAIL because `training_timeline.db` and `training_timeline.indexer` do not exist.

- [ ] **Step 3: Implement SQLite helpers**

Create `training_timeline/db.py`. `connect` sets `row_factory = sqlite3.Row`. `init_db` executes `training_timeline/schema.sql`. Persistence helpers should accept simple dictionaries and perform `INSERT OR REPLACE` or delete-and-insert per run inside one transaction.

- [ ] **Step 4: Implement incremental indexer**

Create `training_timeline/indexer.py`. Use file size plus mtime for `summary.md`, `run_record.json`, selected metrics file, and artifact list as the change detector. Store the fingerprint in `index_state(run_id, fingerprint, indexed_at, index_version)`. On reindex, delete derived rows for the run from `run_configs`, `metric_series`, `metric_summaries`, `diagnostic_events`, and `artifacts`, then insert fresh derived rows.

- [ ] **Step 5: Record parse warnings without aborting the full build**

In `index_source_roots`, catch per-run parse exceptions, increment `warnings`, write a `runs` row with `status='needs_review'` and `status_reason` containing the exception class name, and continue to the next run.

- [ ] **Step 6: Run indexer tests**

Run: `python -m pytest tests/test_training_timeline_indexer.py -v`

Expected: PASS.

- [ ] **Step 7: Run backend regression subset**

Run: `python -m pytest tests/test_training_timeline_models.py tests/test_training_timeline_scanner_parsers.py tests/test_training_timeline_metrics.py tests/test_training_timeline_indexer.py -v`

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add training_timeline/db.py training_timeline/indexer.py training_timeline/schema.sql tests/test_training_timeline_indexer.py
git commit -m "feat: index training timeline data in sqlite"
```

---

### Task 5: Automatic Diagnostics and Sweep Summaries

**Files:**
- Create: `training_timeline/diagnostics.py`
- Create: `tests/test_training_timeline_diagnostics.py`
- Modify: `training_timeline/indexer.py`
- Modify: `training_timeline/db.py`

**Interfaces:**
- Consumes: `MetricSummary`, metric rows, and `DiagnosticEvent`.
- Produces: `diagnose_run(run_id: str, rows: list[dict[str, object]], summary: MetricSummary) -> list[DiagnosticEvent]`.
- Produces: `diagnose_sweep(runs: list[dict[str, object]]) -> list[dict[str, object]]`.
- Indexer stores `diagnostic_events` generated by `diagnose_run`.

- [ ] **Step 1: Write failing diagnostic tests**

Create `tests/test_training_timeline_diagnostics.py` with:

```python
from __future__ import annotations

from training_timeline.diagnostics import diagnose_run
from training_timeline.metrics import summarize_metrics


def event_types(rows: list[dict[str, object]]) -> set[str]:
    summary = summarize_metrics("run-1", rows)
    return {event.event_type for event in diagnose_run("run-1", rows, summary)}


def test_converged_detection_is_conservative() -> None:
    rows = [{"update_step": i, "loss_total": 100.0 - i} for i in range(1, 80)]

    assert "converged" in event_types(rows)


def test_bad_plateau_detection_requires_rebound() -> None:
    rows = [
        {"update_step": 1, "loss_total": 100.0},
        {"update_step": 2, "loss_total": 20.0},
        {"update_step": 3, "loss_total": 55.0},
        {"update_step": 4, "loss_total": 58.0},
        {"update_step": 5, "loss_total": 60.0},
    ]

    assert "bad_plateau" in event_types(rows)


def test_clip_storm_and_skip_loop_detection() -> None:
    rows = [
        {"update_step": i, "loss_total": 10.0 + i, "clip_fraction": 0.75, "skip_fraction": 0.35}
        for i in range(1, 8)
    ]

    types = event_types(rows)
    assert "clip_storm" in types
    assert "skip_loop" in types


def test_primary_head_failure_and_lr_floor_freeze_detection() -> None:
    rows = [
        {"update_step": 1, "loss_total": 100.0, "loss_gepc": 100.0, "loss_gep": 50.0, "loss_zero_prob": 40.0, "lr": 1e-6},
        {"update_step": 2, "loss_total": 80.0, "loss_gepc": 20.0, "loss_gep": 52.0, "loss_zero_prob": 41.0, "lr": 1e-7},
        {"update_step": 3, "loss_total": 82.0, "loss_gepc": 18.0, "loss_gep": 53.0, "loss_zero_prob": 42.0, "lr": 1e-7},
    ]

    types = event_types(rows)
    assert "primary_head_failure" in types
    assert "lr_floor_freeze" in types
```

- [ ] **Step 2: Run diagnostic tests to verify they fail**

Run: `python -m pytest tests/test_training_timeline_diagnostics.py -v`

Expected: FAIL because `training_timeline.diagnostics` does not exist.

- [ ] **Step 3: Implement diagnostic rules**

Create `training_timeline/diagnostics.py` with the thresholds from the spec:

- Convergence: tail loss below 30 percent of early loss and low tail volatility.
- Bad plateau: tail loss more than 2 times the best-window loss for multiple windows.
- Clip storm: clip fraction above 50 percent with simultaneous or following loss increase.
- Skip loop: more than 50 consecutive skips or skip fraction above 30 percent in a window.
- Primary head failure: GEPC-style loss improves while GEP or zero-prob stays flat or worsens.
- LR floor freeze: LR is at its minimum observed value while loss is flat or worsening.
- Resume boundary: duplicate `update_step` values were seen before dedupe or resume markers exist in configs or artifacts.

Each `DiagnosticEvent.evidence` must include the numeric threshold values used by the rule.

- [ ] **Step 4: Store diagnostics during indexing**

Modify `training_timeline/indexer.py` so each indexed run calls `diagnose_run` after metrics summarization and writes diagnostic events into `diagnostic_events`. Keep `created_by='auto'`.

- [ ] **Step 5: Run diagnostic and indexer tests**

Run: `python -m pytest tests/test_training_timeline_diagnostics.py tests/test_training_timeline_indexer.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add training_timeline/diagnostics.py training_timeline/indexer.py training_timeline/db.py tests/test_training_timeline_diagnostics.py
git commit -m "feat: add preliminary training diagnostics"
```

---

### Task 6: Curated Analysis Notes and Bounded Evidence Retrieval

**Files:**
- Create: `training_timeline/analysis.py`
- Create: `tests/test_training_timeline_analysis.py`
- Modify: `training_timeline/db.py`

**Interfaces:**
- Consumes: `AnalysisNote`.
- Produces: `list_analysis_notes(conn: sqlite3.Connection, run_id: str) -> list[dict[str, object]]`.
- Produces: `create_analysis_note(conn: sqlite3.Connection, run_id: str, payload: dict[str, object]) -> dict[str, object]`.
- Produces: `update_analysis_note(conn: sqlite3.Connection, analysis_id: str, payload: dict[str, object]) -> dict[str, object]`.
- Produces: `get_evidence_bundle(conn: sqlite3.Connection, run_id: str, *, max_chars_per_file: int = 8000) -> dict[str, object]`.

- [ ] **Step 1: Write failing analysis tests**

Create `tests/test_training_timeline_analysis.py` with:

```python
from __future__ import annotations

from pathlib import Path

from tests.training_timeline_fixtures import make_run_dir
from training_timeline.analysis import create_analysis_note, get_evidence_bundle, list_analysis_notes, update_analysis_note
from training_timeline.db import connect, init_db
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
```

- [ ] **Step 2: Run analysis tests to verify they fail**

Run: `python -m pytest tests/test_training_timeline_analysis.py -v`

Expected: FAIL because `training_timeline.analysis` does not exist.

- [ ] **Step 3: Implement analysis note persistence**

Create `training_timeline/analysis.py`. Validate `note_type` as `curated_note` or `deep_review`; validate `confidence` as `low`, `medium`, or `high`; store list fields as JSON strings in SQLite; return decoded lists to callers.

- [ ] **Step 4: Implement bounded evidence retrieval**

`get_evidence_bundle` reads only artifact kinds `summary_md`, `run_record_json`, `loss_log`, and `log`. It must:

- Use artifact paths from the database.
- Resolve paths and confirm they remain under one of the run source roots.
- Return snippets capped by `max_chars_per_file`.
- Include file kind, path, size, mtime, and whether the snippet was truncated.

- [ ] **Step 5: Run analysis tests**

Run: `python -m pytest tests/test_training_timeline_analysis.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add training_timeline/analysis.py training_timeline/db.py tests/test_training_timeline_analysis.py
git commit -m "feat: add curated training analysis notes"
```

---

### Task 7: FastAPI App, CLI, and Backend Endpoint Tests

**Files:**
- Create: `training_timeline/api.py`
- Create: `training_timeline/cli.py`
- Create: `tests/test_training_timeline_api.py`
- Modify: `training_timeline/db.py`
- Modify: `training_timeline/indexer.py`

**Interfaces:**
- Consumes: all backend modules from Tasks 1-6.
- Produces: `create_app(db_path: Path, source_roots: list[Path] | None = None) -> fastapi.FastAPI`.
- Produces CLI commands:
  - `python -m training_timeline.cli rebuild --db .training_timeline/timeline.sqlite --source /Volumes/SSD1/SpeciesLLM`
  - `python -m training_timeline.cli serve --db .training_timeline/timeline.sqlite --source /Volumes/SSD1/SpeciesLLM --host 127.0.0.1 --port 8765`

- [ ] **Step 1: Write failing API tests**

Create `tests/test_training_timeline_api.py` with:

```python
from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from tests.training_timeline_fixtures import make_run_dir
from training_timeline.api import create_app


def test_api_lists_runs_and_serves_run_details(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    make_run_dir(
        root,
        "training_output_100m_data_1_2_3_stable_from_scratch_20260514_011839",
        summary="# Stable\n\nLoss improved.\n",
        metrics_rows=[{"update_step": 1, "loss_total": 100.0}, {"update_step": 2, "loss_total": 50.0}],
    )
    app = create_app(tmp_path / "timeline.sqlite", [root])
    client = TestClient(app)

    rebuild = client.post("/api/index/rebuild")
    runs = client.get("/api/runs").json()
    run_id = runs["runs"][0]["id"]
    detail = client.get(f"/api/runs/{run_id}").json()
    metrics = client.get(f"/api/runs/{run_id}/metrics?series=loss_total").json()

    assert rebuild.status_code == 200
    assert runs["runs"][0]["model_size"] == "100m"
    assert detail["run"]["summary_title"] == "Stable"
    assert metrics["series"]["loss_total"][0]["step"] == 1


def test_api_analysis_crud_and_evidence(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    make_run_dir(root, "training_output_100m_data_1_2_3_stable_from_scratch_20260514_011839", summary="# Stable\n\nLoss improved.\n")
    app = create_app(tmp_path / "timeline.sqlite", [root])
    client = TestClient(app)
    client.post("/api/index/rebuild")
    run_id = client.get("/api/runs").json()["runs"][0]["id"]

    created = client.post(
        f"/api/runs/{run_id}/analysis",
        json={
            "note_type": "deep_review",
            "title": "Manual read",
            "body": "This run is a healthy control.",
            "confidence": "high",
            "supersedes_diagnostic_ids": [],
            "evidence_refs": ["summary.md"],
            "author": "local",
        },
    ).json()
    patched = client.patch(f"/api/runs/{run_id}/analysis/{created['id']}", json={"confidence": "medium"}).json()
    evidence = client.get(f"/api/runs/{run_id}/evidence").json()

    assert patched["confidence"] == "medium"
    assert evidence["run_id"] == run_id
```

- [ ] **Step 2: Run API tests to verify they fail**

Run: `python -m pytest tests/test_training_timeline_api.py -v`

Expected: FAIL because `training_timeline.api` does not exist.

- [ ] **Step 3: Implement FastAPI routes**

Create `training_timeline/api.py` with these routes:

```text
GET  /api/health
GET  /api/sources
POST /api/index/rebuild
POST /api/index/runs/{run_id}/refresh
GET  /api/runs
GET  /api/runs/{run_id}
GET  /api/runs/{run_id}/metrics
GET  /api/runs/{run_id}/configs
GET  /api/runs/{run_id}/diagnostics
GET  /api/runs/{run_id}/artifacts
GET  /api/runs/{run_id}/evidence
POST /api/compare
GET  /api/report/timeline
GET  /api/report/stages
GET  /api/runs/{run_id}/analysis
POST /api/runs/{run_id}/analysis
PATCH /api/runs/{run_id}/analysis/{analysis_id}
```

Return JSON objects with stable top-level keys such as `runs`, `run`, `series`, `diagnostics`, `artifacts`, `notes`, and `stages`.

- [ ] **Step 4: Implement compare and report endpoints**

`POST /api/compare` accepts `{"run_ids": ["run-1", "run-2"]}` and returns selected run rows, config diffs, metric summaries, and diagnostics. `GET /api/report/stages` groups runs into the phase names from the spec using directory-name tags and start time.

- [ ] **Step 5: Implement CLI**

Create `training_timeline/cli.py` using `argparse`. `rebuild` calls `index_source_roots`. `serve` creates the app and starts `uvicorn.run`.

- [ ] **Step 6: Run API tests**

Run: `python -m pytest tests/test_training_timeline_api.py -v`

Expected: PASS.

- [ ] **Step 7: Run complete backend test set**

Run: `python -m pytest tests/test_training_timeline_*.py -v`

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add training_timeline/api.py training_timeline/cli.py training_timeline/db.py training_timeline/indexer.py tests/test_training_timeline_api.py
git commit -m "feat: serve training timeline api"
```

---

### Task 8: Frontend Scaffold, API Client, Timeline, and Sources

**Files:**
- Create: `training_timeline_ui/package.json`
- Create: `training_timeline_ui/package-lock.json`
- Create: `training_timeline_ui/index.html`
- Create: `training_timeline_ui/vite.config.ts`
- Create: `training_timeline_ui/tsconfig.json`
- Create: `training_timeline_ui/src/main.tsx`
- Create: `training_timeline_ui/src/App.tsx`
- Create: `training_timeline_ui/src/api.ts`
- Create: `training_timeline_ui/src/types.ts`
- Create: `training_timeline_ui/src/styles.css`
- Create: `training_timeline_ui/src/pages/TimelinePage.tsx`
- Create: `training_timeline_ui/src/pages/SourcesPage.tsx`
- Create: `training_timeline_ui/src/components/RunStatusBadge.tsx`
- Create: `training_timeline_ui/src/test/App.test.tsx`
- Create: `training_timeline_ui/src/test/TimelinePage.test.tsx`

**Interfaces:**
- Consumes: API response keys from Task 7.
- Produces: `fetchRuns(): Promise<RunsResponse>`.
- Produces: `fetchSources(): Promise<SourcesResponse>`.
- Produces: `rebuildIndex(): Promise<IndexRebuildResponse>`.
- Produces: UI routes held in React state: `timeline`, `sources`, `run-detail`, `compare`, `report`.

- [ ] **Step 1: Create frontend package files**

Create `training_timeline_ui/package.json` with scripts:

```json
{
  "name": "speciesllm-training-timeline-ui",
  "version": "0.1.0",
  "private": true,
  "type": "module",
  "scripts": {
    "dev": "vite --host 127.0.0.1 --port 5173",
    "build": "tsc && vite build",
    "test": "vitest run"
  },
  "dependencies": {
    "lucide-react": "^0.468.0",
    "react": "^18.3.1",
    "react-dom": "^18.3.1",
    "recharts": "^2.12.7"
  },
  "devDependencies": {
    "@testing-library/jest-dom": "^6.4.0",
    "@testing-library/react": "^15.0.0",
    "@vitejs/plugin-react": "^4.3.0",
    "@types/react": "^18.3.0",
    "@types/react-dom": "^18.3.0",
    "jsdom": "^24.1.0",
    "typescript": "^5.5.0",
    "vite": "^5.4.0",
    "vitest": "^2.0.0"
  }
}
```

- [ ] **Step 2: Install frontend dependencies**

Run: `cd training_timeline_ui && npm install`

Expected: PASS and `training_timeline_ui/package-lock.json` is created.

- [ ] **Step 3: Write failing frontend tests**

Create `training_timeline_ui/src/test/App.test.tsx`:

```tsx
import "@testing-library/jest-dom/vitest";
import { render, screen } from "@testing-library/react";
import App from "../App";

test("renders the timeline dashboard shell", async () => {
  vi.stubGlobal("fetch", vi.fn(async () => ({
    ok: true,
    json: async () => ({ runs: [] }),
  })));

  render(<App />);

  expect(await screen.findByText("Training Timeline")).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Sources" })).toBeInTheDocument();
});
```

Create `training_timeline_ui/src/test/TimelinePage.test.tsx`:

```tsx
import "@testing-library/jest-dom/vitest";
import { render, screen } from "@testing-library/react";
import { TimelinePage } from "../pages/TimelinePage";

test("timeline shows run metadata and preliminary labels", () => {
  render(
    <TimelinePage
      runs={[
        {
          id: "run-1",
          name: "training_output_100m_data_1_2_3_stable_from_scratch_20260514_011839",
          started_at: "2026-05-14T01:18:39",
          model_size: "100m",
          data_recipe: "data_1_2_3",
          experiment_name: "stable_from_scratch",
          status: "success",
          status_reason: "converged",
          summary_one_liner: "Loss improved across the full epoch.",
          tags: ["stable"],
        },
      ]}
      loading={false}
      onOpenRun={() => undefined}
    />
  );

  expect(screen.getByText("100m")).toBeInTheDocument();
  expect(screen.getByText("data_1_2_3")).toBeInTheDocument();
  expect(screen.getByText("Preliminary")).toBeInTheDocument();
  expect(screen.getByText("Loss improved across the full epoch.")).toBeInTheDocument();
});
```

- [ ] **Step 4: Run frontend tests to verify they fail**

Run: `cd training_timeline_ui && npm test`

Expected: FAIL because frontend source files do not exist.

- [ ] **Step 5: Implement Vite config and API client**

Create `vite.config.ts` with proxy `/api -> http://127.0.0.1:8765`. Create `src/types.ts` matching backend response shapes used by the UI. Create `src/api.ts` with a small `requestJson<T>(path: string)` helper and the typed functions listed in Interfaces.

- [ ] **Step 6: Implement shell, timeline, sources, badge, and CSS**

Create `App.tsx` with a left navigation rail and a work-focused dashboard layout. Create `TimelinePage.tsx`, `SourcesPage.tsx`, and `RunStatusBadge.tsx`. Keep cards at 8px border radius or less, use lucide icons for navigation and rebuild actions, and label automatic diagnostics as `Preliminary`.

- [ ] **Step 7: Run frontend tests**

Run: `cd training_timeline_ui && npm test`

Expected: PASS.

- [ ] **Step 8: Build frontend**

Run: `cd training_timeline_ui && npm run build`

Expected: PASS and `training_timeline_ui/dist` is created.

- [ ] **Step 9: Commit**

```bash
git add training_timeline_ui/package.json training_timeline_ui/package-lock.json training_timeline_ui/index.html training_timeline_ui/vite.config.ts training_timeline_ui/tsconfig.json training_timeline_ui/src
git commit -m "feat: add training timeline frontend shell"
```

---

### Task 9: Run Detail, Compare, Report, Charts, and Analysis UI

**Files:**
- Create: `training_timeline_ui/src/pages/RunDetailPage.tsx`
- Create: `training_timeline_ui/src/pages/ComparePage.tsx`
- Create: `training_timeline_ui/src/pages/ReportPage.tsx`
- Create: `training_timeline_ui/src/components/AnalysisPanel.tsx`
- Create: `training_timeline_ui/src/components/MetricChart.tsx`
- Create: `training_timeline_ui/src/test/RunDetailPage.test.tsx`
- Create: `training_timeline_ui/src/test/CompareReportAnalysis.test.tsx`
- Modify: `training_timeline_ui/src/App.tsx`
- Modify: `training_timeline_ui/src/api.ts`
- Modify: `training_timeline_ui/src/types.ts`
- Modify: `training_timeline_ui/src/styles.css`

**Interfaces:**
- Consumes: `/api/runs/{run_id}`, `/metrics`, `/diagnostics`, `/artifacts`, `/evidence`, `/analysis`, `/compare`, `/report/timeline`, and `/report/stages`.
- Produces: `RunDetailPage`, `ComparePage`, `ReportPage`, `AnalysisPanel`, and `MetricChart`.

- [ ] **Step 1: Write failing UI tests for detail, compare, report, and analysis**

Create `training_timeline_ui/src/test/RunDetailPage.test.tsx`:

```tsx
import "@testing-library/jest-dom/vitest";
import { render, screen } from "@testing-library/react";
import { RunDetailPage } from "../pages/RunDetailPage";

test("run detail separates automatic diagnostics and deep review notes", () => {
  render(
    <RunDetailPage
      run={{
        id: "run-1",
        name: "training_output_demo",
        status: "needs_review",
        summary_title: "Clip storm candidate",
        summary_one_liner: "GEP rebounded after clipping increased.",
      }}
      metrics={{ loss_total: [{ step: 1, value: 10 }, { step: 2, value: 12 }] }}
      diagnostics={[{ id: "evt-1", event_type: "clip_storm", severity: "warning", title: "Clip storm", description: "Clip fraction exceeded threshold." }]}
      artifacts={[{ kind: "summary_md", path: "/repo/run/summary.md" }]}
      notes={[{ id: "note-1", note_type: "deep_review", title: "Manual read", body: "Needs per-run interpretation.", confidence: "medium" }]}
    />
  );

  expect(screen.getByText("Automatic diagnostics")).toBeInTheDocument();
  expect(screen.getByText("Deep review")).toBeInTheDocument();
  expect(screen.getByText("Manual read")).toBeInTheDocument();
});
```

Create `training_timeline_ui/src/test/CompareReportAnalysis.test.tsx` with tests that render `ComparePage`, `ReportPage`, and `AnalysisPanel` using in-memory props and assert that config diffs, phase names, evidence links, and note confidence are visible.

- [ ] **Step 2: Run UI tests to verify they fail**

Run: `cd training_timeline_ui && npm test`

Expected: FAIL because the new pages and components do not exist.

- [ ] **Step 3: Implement MetricChart**

Use Recharts `ResponsiveContainer`, `LineChart`, `Line`, `XAxis`, `YAxis`, `Tooltip`, and `ReferenceArea` for event bands. The component accepts:

```ts
type MetricChartProps = {
  title: string;
  series: Record<string, Array<{ step: number; value: number }>>;
  events?: Array<{ start_step?: number; end_step?: number; title: string }>;
};
```

- [ ] **Step 4: Implement RunDetailPage**

Show top-level status, best/final loss if present, summary, charts, configs, diagnostics, analysis notes, and artifacts. Place automatic diagnostics under a visibly labeled `Preliminary` section and curated/deep-review notes in a separate section.

- [ ] **Step 5: Implement ComparePage**

Support multi-run selection from already loaded runs. Fetch `/api/compare` when selection changes. Render config diffs, summary matrix, diagnostic chips, and overlay curves.

- [ ] **Step 6: Implement ReportPage**

Fetch report timeline and stages. Render phase groups in chronological order:

- Initial smoke and model-scale tests.
- Data 1 / 1+3 / 1+2+3 comparisons.
- 500M stability incident and fixes.
- 100M learning-rate sweep.
- Multi-epoch and resume experiments.
- fp32 + Huber, modality, loss-weight, and shuffle experiments.

Each conclusion link opens the run detail page.

- [ ] **Step 7: Implement AnalysisPanel**

Display existing notes and provide fields for note type, title, body, confidence, superseded diagnostic IDs, evidence refs, and author. Submit to `POST /api/runs/{run_id}/analysis`; updates use `PATCH /api/runs/{run_id}/analysis/{analysis_id}`.

- [ ] **Step 8: Wire pages into App**

Update `App.tsx` navigation so users can move among Timeline, Sources, Compare, Report, and Run Detail. Keep current selection in React state. Do not introduce a router unless needed by implementation complexity.

- [ ] **Step 9: Run UI tests and build**

Run:

```bash
cd training_timeline_ui && npm test
cd training_timeline_ui && npm run build
```

Expected: both commands PASS.

- [ ] **Step 10: Commit**

```bash
git add training_timeline_ui/src
git commit -m "feat: add training timeline detail and report views"
```

---

### Task 10: Run Scripts, Documentation, and End-to-End Smoke

**Files:**
- Create: `scripts/run_training_timeline_backend.sh`
- Create: `scripts/run_training_timeline_frontend.sh`
- Create: `docs/training_timeline_dashboard.md`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: backend CLI and frontend scripts from earlier tasks.
- Produces: local backend command on `127.0.0.1:8765`.
- Produces: local frontend command on `127.0.0.1:5173`.
- Produces: documentation for rebuild, serve, source-root config, and acceptance checks.

- [ ] **Step 1: Write a lightweight script test**

Create a shell-check style pytest in `tests/test_training_timeline_scripts.py`:

```python
from __future__ import annotations

from pathlib import Path


def test_training_timeline_scripts_are_safe_local_entrypoints() -> None:
    backend = Path("scripts/run_training_timeline_backend.sh").read_text(encoding="utf-8")
    frontend = Path("scripts/run_training_timeline_frontend.sh").read_text(encoding="utf-8")

    assert "python -m training_timeline.cli serve" in backend
    assert "--host 127.0.0.1" in backend
    assert "--port 8765" in backend
    assert "npm run dev" in frontend
    assert "rm -rf" not in backend
    assert "rm -rf" not in frontend
```

- [ ] **Step 2: Run script test to verify it fails**

Run: `python -m pytest tests/test_training_timeline_scripts.py -v`

Expected: FAIL because scripts do not exist.

- [ ] **Step 3: Create backend run script**

Create `scripts/run_training_timeline_backend.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p .training_timeline

python -m training_timeline.cli serve \
  --db .training_timeline/timeline.sqlite \
  --source "$ROOT" \
  --host 127.0.0.1 \
  --port 8765
```

- [ ] **Step 4: Create frontend run script**

Create `scripts/run_training_timeline_frontend.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/training_timeline_ui"
npm run dev
```

Run: `chmod +x scripts/run_training_timeline_backend.sh scripts/run_training_timeline_frontend.sh`

- [ ] **Step 5: Update `.gitignore`**

Add local generated paths:

```gitignore
.training_timeline/
training_timeline_ui/node_modules/
training_timeline_ui/dist/
```

- [ ] **Step 6: Write user documentation**

Create `docs/training_timeline_dashboard.md` with:

- What the dashboard does.
- How to install backend dependencies.
- How to install frontend dependencies.
- How to rebuild the index.
- How to start backend and frontend.
- How to add extra source roots through JSON config.
- What automatic diagnostics mean.
- How curated and deep-review notes should be used.
- Local safety constraints from this plan.

- [ ] **Step 7: Run script test**

Run: `python -m pytest tests/test_training_timeline_scripts.py -v`

Expected: PASS.

- [ ] **Step 8: Run complete backend and frontend checks**

Run:

```bash
python -m pytest tests/test_training_timeline_*.py -v
cd training_timeline_ui && npm test
cd training_timeline_ui && npm run build
```

Expected: all commands PASS.

- [ ] **Step 9: Run local backend smoke against current repo**

Run:

```bash
python -m training_timeline.cli rebuild --db .training_timeline/timeline.sqlite --source /Volumes/SSD1/SpeciesLLM
python -m training_timeline.cli serve --db .training_timeline/timeline.sqlite --source /Volumes/SSD1/SpeciesLLM --host 127.0.0.1 --port 8765
```

In a second terminal, run:

```bash
curl -s http://127.0.0.1:8765/api/health
curl -s http://127.0.0.1:8765/api/runs
curl -s http://127.0.0.1:8765/api/report/stages
```

Expected:

- `/api/health` returns a JSON object with `ok` set to `true`.
- `/api/runs` returns existing May-June 2026 `training_output*` runs.
- `/api/report/stages` includes the main experiment phases from the spec.

- [ ] **Step 10: Run local frontend smoke**

Run:

```bash
scripts/run_training_timeline_backend.sh
scripts/run_training_timeline_frontend.sh
```

Open `http://127.0.0.1:5173`. Verify:

- Timeline renders existing training runs sorted by inferred start time.
- Sources page shows `/Volumes/SSD1/SpeciesLLM`.
- Run detail opens from a timeline run.
- Compare page can select at least two runs.
- Report page shows phase grouping.
- Automatic diagnostics are labeled `Preliminary`.
- Analysis panel can create and display a `deep_review` note.

- [ ] **Step 11: Commit**

```bash
git add .gitignore scripts/run_training_timeline_backend.sh scripts/run_training_timeline_frontend.sh docs/training_timeline_dashboard.md tests/test_training_timeline_scripts.py
git commit -m "docs: add training timeline dashboard runbook"
```

---

## Final Acceptance Checklist

- [ ] Backend tests pass: `python -m pytest tests/test_training_timeline_*.py -v`.
- [ ] Frontend tests pass: `cd training_timeline_ui && npm test`.
- [ ] Frontend build passes: `cd training_timeline_ui && npm run build`.
- [ ] Local rebuild succeeds against `/Volumes/SSD1/SpeciesLLM` without modifying `training_output*` directories.
- [ ] `/api/runs` includes existing May-June 2026 training output directories.
- [ ] `/api/report/stages` shows the experiment arc required by the spec.
- [ ] UI visibly separates raw evidence, automatic diagnostics, curated notes, and deep-review conclusions.
- [ ] Every automatic conclusion exposed in UI links to supporting evidence or a bounded evidence bundle.
- [ ] Deep-review notes can supplement or supersede automatic diagnostics.

## Self-Review

**Spec coverage:** This plan covers source scanning, extra roots, SQLite indexing, parsers, metrics summaries, preliminary diagnostics, deep-review analysis notes, API routes, timeline/detail/compare/report/source pages, tests, scripts, and local acceptance.

**Red-flag scan:** A scan for unfinished markers from the writing-plans skill should return no matches.

**Type consistency:** The task interfaces consistently use `RunDiscovery`, `ParsedRunName`, `SummaryInfo`, `MetricPoint`, `MetricSummary`, `DiagnosticEvent`, `AnalysisNote`, `ArtifactRef`, `create_app`, `index_source_roots`, and the API response keys defined in Tasks 1-7.
