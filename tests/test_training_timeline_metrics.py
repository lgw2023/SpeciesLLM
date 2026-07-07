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
