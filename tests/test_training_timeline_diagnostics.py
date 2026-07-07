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
