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
