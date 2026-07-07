from __future__ import annotations

import hashlib
from typing import Any

from training_timeline.metrics import SERIES_KEYS
from training_timeline.models import DiagnosticEvent, MetricSummary


def diagnose_run(run_id: str, rows: list[dict[str, Any]], summary: MetricSummary) -> list[DiagnosticEvent]:
    events: list[DiagnosticEvent] = []
    loss_points = _points(rows, "loss_total")
    if not loss_points:
        return events

    if _is_converged(loss_points):
        events.append(
            _event(
                run_id,
                "converged",
                "info",
                "Converged",
                "Loss fell substantially from the early window and the tail stayed stable.",
                loss_points,
                {"tail_below_early_fraction": 0.30, "max_tail_volatility": 0.50},
            )
        )
    if _is_bad_plateau(loss_points, summary):
        events.append(
            _event(
                run_id,
                "bad_plateau",
                "warning",
                "Bad plateau",
                "Loss reached a lower point, rebounded, and remained above the best window.",
                loss_points,
                {"tail_over_best_multiplier": 2.0},
            )
        )
    if _is_clip_storm(rows, loss_points):
        events.append(
            _event(
                run_id,
                "clip_storm",
                "warning",
                "Clip storm",
                "Clip fraction exceeded threshold while loss worsened.",
                loss_points,
                {"clip_fraction_threshold": 0.50},
            )
        )
    if _is_skip_loop(rows):
        events.append(
            _event(
                run_id,
                "skip_loop",
                "warning",
                "Skip loop",
                "Skip fraction exceeded threshold or skip behavior repeated continuously.",
                loss_points,
                {"skip_fraction_threshold": 0.30, "consecutive_skip_threshold": 50},
            )
        )
    if _is_primary_head_failure(rows):
        events.append(
            _event(
                run_id,
                "primary_head_failure",
                "warning",
                "Primary head failure",
                "GEPC-style loss improved while GEP or zero-probability losses stayed flat or worsened.",
                loss_points,
                {"gepc_improvement_fraction": 0.50, "primary_flat_fraction": 0.95},
            )
        )
    if _is_lr_floor_freeze(rows, loss_points, summary):
        events.append(
            _event(
                run_id,
                "lr_floor_freeze",
                "warning",
                "LR floor freeze",
                "Learning rate reached its observed floor while loss stopped improving.",
                loss_points,
                {"final_loss_over_best_multiplier": 1.02},
            )
        )
    if _has_resume_boundary(rows):
        events.append(
            _event(
                run_id,
                "resume_boundary",
                "info",
                "Resume boundary",
                "Repeated update steps or resume markers indicate a run boundary that needs continuity review.",
                loss_points,
                {"duplicate_update_step_count": _duplicate_step_count(rows)},
            )
        )
    return events


def diagnose_sweep(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return []


def _is_converged(loss_points: list[tuple[int, float]]) -> bool:
    window = _window_size(loss_points)
    early_values = [value for _, value in loss_points[:window]]
    tail_values = [value for _, value in loss_points[-window:]]
    early_mean = _mean(early_values)
    tail_mean = _mean(tail_values)
    if early_mean is None or tail_mean is None or early_mean <= 0:
        return False
    tail_volatility = (max(tail_values) - min(tail_values)) / max(abs(tail_mean), 1e-12)
    return tail_mean < early_mean * 0.30 and tail_volatility <= 0.50


def _is_bad_plateau(loss_points: list[tuple[int, float]], summary: MetricSummary) -> bool:
    if summary.best_loss is None or len(loss_points) < 3:
        return False
    best_index = min(range(len(loss_points)), key=lambda index: loss_points[index][1])
    if best_index >= len(loss_points) - 1:
        return False
    window = _window_size(loss_points)
    tail_mean = _mean([value for _, value in loss_points[-window:]])
    final_loss = loss_points[-1][1]
    return tail_mean is not None and tail_mean > summary.best_loss * 2.0 and final_loss > summary.best_loss * 2.0


def _is_clip_storm(rows: list[dict[str, Any]], loss_points: list[tuple[int, float]]) -> bool:
    clip_values = _series(rows, "clip_fraction")
    if not clip_values or max(clip_values) <= 0.50:
        return False
    return loss_points[-1][1] > loss_points[0][1]


def _is_skip_loop(rows: list[dict[str, Any]]) -> bool:
    skip_values = _series(rows, "skip_fraction")
    if skip_values and max(skip_values) > 0.30:
        return True
    return _max_consecutive_positive(skip_values) > 50


def _is_primary_head_failure(rows: list[dict[str, Any]]) -> bool:
    gepc = _series(rows, "loss_gepc")
    gep = _series(rows, "loss_gep")
    zero_prob = _series(rows, "loss_zero_prob")
    if len(gepc) < 2 or not (gep or zero_prob):
        return False
    gepc_improved = gepc[-1] <= gepc[0] * 0.50
    gep_flat = bool(gep) and gep[-1] >= gep[0] * 0.95
    zero_prob_flat = bool(zero_prob) and zero_prob[-1] >= zero_prob[0] * 0.95
    return gepc_improved and (gep_flat or zero_prob_flat)


def _is_lr_floor_freeze(rows: list[dict[str, Any]], loss_points: list[tuple[int, float]], summary: MetricSummary) -> bool:
    lrs = _series(rows, "lr")
    if not lrs or summary.best_loss is None:
        return False
    floor = min(lrs)
    final_lr = lrs[-1]
    final_loss = loss_points[-1][1]
    return final_lr <= floor * (1 + 1e-9) and final_loss >= summary.best_loss * 1.02


def _has_resume_boundary(rows: list[dict[str, Any]]) -> bool:
    if _duplicate_step_count(rows) > 0:
        return True
    return any(bool(row.get("resume") or row.get("resumed")) for row in rows)


def _duplicate_step_count(rows: list[dict[str, Any]]) -> int:
    seen: set[int] = set()
    duplicates = 0
    for row in rows:
        step = _to_int(row.get("update_step"))
        if step is None:
            continue
        if step in seen:
            duplicates += 1
        seen.add(step)
    return duplicates


def _event(
    run_id: str,
    event_type: str,
    severity: str,
    title: str,
    description: str,
    loss_points: list[tuple[int, float]],
    evidence: dict[str, Any],
) -> DiagnosticEvent:
    start_step = loss_points[0][0]
    end_step = loss_points[-1][0]
    event_id = hashlib.sha1(f"{run_id}:{event_type}:{start_step}:{end_step}".encode("utf-8")).hexdigest()[:16]
    return DiagnosticEvent(
        id=event_id,
        run_id=run_id,
        event_type=event_type,
        severity=severity,
        title=title,
        description=description,
        start_step=start_step,
        end_step=end_step,
        evidence=evidence,
        created_by="auto",
    )


def _points(rows: list[dict[str, Any]], series_name: str) -> list[tuple[int, float]]:
    points: list[tuple[int, float]] = []
    for row in rows:
        step = _to_int(row.get("update_step"))
        value = _value(row, series_name)
        if step is not None and value is not None:
            points.append((step, value))
    return points


def _series(rows: list[dict[str, Any]], series_name: str) -> list[float]:
    return [value for row in rows if (value := _value(row, series_name)) is not None]


def _value(row: dict[str, Any], series_name: str) -> float | None:
    for key in SERIES_KEYS[series_name]:
        value = _to_float(row.get(key))
        if value is not None:
            return value
    return None


def _window_size(points: list[tuple[int, float]]) -> int:
    return min(100, max(3, len(points) // 10 or 1))


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _max_consecutive_positive(values: list[float]) -> int:
    longest = 0
    current = 0
    for value in values:
        if value > 0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _to_int(value: Any) -> int | None:
    numeric = _to_float(value)
    return int(numeric) if numeric is not None else None


def _to_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None
