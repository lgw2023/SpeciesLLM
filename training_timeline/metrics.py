from __future__ import annotations

import json
import math
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from training_timeline.models import MetricPoint, MetricSummary


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


def choose_metrics_file(run_dir: Path) -> Path | None:
    rank_zero = run_dir / "metrics.0-0.jsonl"
    if rank_zero.exists():
        return rank_zero
    candidates = sorted(run_dir.glob("metrics.*.jsonl"), key=lambda item: item.name)
    return candidates[0] if candidates else None


def iter_metric_rows(metrics_path: Path) -> Iterator[dict[str, Any]]:
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSON in {metrics_path}:{line_number}") from exc
            if isinstance(row, dict):
                yield row


def dedupe_by_update_step(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    by_step: dict[int, dict[str, Any]] = {}
    for row in rows:
        step = _numeric_int(row.get("update_step"))
        if step is None:
            continue
        copied = dict(row)
        copied["update_step"] = step
        by_step[step] = copied
    return [by_step[step] for step in sorted(by_step)]


def summarize_metrics(run_id: str, rows: list[dict[str, Any]]) -> MetricSummary:
    loss_rows = [(int(row["update_step"]), value) for row in rows if (value := _series_value(row, "loss_total")) is not None]
    grad_values = [value for row in rows if (value := _series_value(row, "grad_norm_raw")) is not None]
    clip_values = [value for row in rows if (value := _series_value(row, "clip_fraction")) is not None]
    skip_values = [value for row in rows if (value := _series_value(row, "skip_fraction")) is not None]

    best_loss = None
    best_loss_step = None
    final_loss = None
    final_step = None
    early_loss_mean = None
    tail_loss_mean = None
    if loss_rows:
        best_loss_step, best_loss = min(loss_rows, key=lambda item: item[1])
        final_step, final_loss = loss_rows[-1]
        window = min(10, len(loss_rows))
        early_loss_mean = _mean([value for _, value in loss_rows[:window]])
        tail_loss_mean = _mean([value for _, value in loss_rows[-window:]])

    return MetricSummary(
        run_id=run_id,
        best_loss=best_loss,
        best_loss_step=best_loss_step,
        final_loss=final_loss,
        final_step=final_step,
        early_loss_mean=early_loss_mean,
        tail_loss_mean=tail_loss_mean,
        grad_norm_p50=_percentile(grad_values, 50),
        grad_norm_p95=_percentile(grad_values, 95),
        grad_norm_p99=_percentile(grad_values, 99),
        grad_norm_max=max(grad_values) if grad_values else None,
        clip_count=sum(1 for value in clip_values if value > 0),
        clip_fraction=_mean(clip_values),
        skip_count=sum(1 for value in skip_values if value > 0),
        skip_fraction=_mean(skip_values),
        row_count=len(rows),
    )


def downsample_series(run_id: str, rows: list[dict[str, Any]], max_points: int = 1200) -> list[MetricPoint]:
    points: list[MetricPoint] = []
    for series_name in SERIES_KEYS:
        raw_points = _series_points(series_name, rows)
        selected_points = _stride_select(raw_points, max_points)
        points.extend(
            MetricPoint(
                series_name=series_name,
                step=step,
                epoch=epoch,
                value=value,
                sample_count=1,
                aggregation="raw" if len(raw_points) <= max_points else "stride",
            )
            for step, epoch, value in selected_points
        )
    return points


def _series_points(series_name: str, rows: list[dict[str, Any]]) -> list[tuple[int, float | None, float]]:
    points: list[tuple[int, float | None, float]] = []
    for row in rows:
        step = _numeric_int(row.get("update_step"))
        value = _series_value(row, series_name)
        if step is None or value is None:
            continue
        epoch = _numeric_float(row.get("epoch"))
        points.append((step, epoch, value))
    return points


def _stride_select(points: list[tuple[int, float | None, float]], max_points: int) -> list[tuple[int, float | None, float]]:
    if max_points <= 0 or len(points) <= max_points:
        return points
    stride = math.ceil(len(points) / max_points)
    selected = points[::stride]
    if selected[-1] != points[-1]:
        selected.append(points[-1])
    return selected


def _series_value(row: dict[str, Any], series_name: str) -> float | None:
    for key in SERIES_KEYS[series_name]:
        value = _numeric_float(row.get(key))
        if value is not None:
            return value
    return None


def _numeric_int(value: Any) -> int | None:
    numeric = _numeric_float(value)
    return int(numeric) if numeric is not None else None


def _numeric_float(value: Any) -> float | None:
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


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _percentile(values: list[float], percentile: int) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * (percentile / 100)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return sorted_values[lower]
    lower_value = sorted_values[lower] * (upper - rank)
    upper_value = sorted_values[upper] * (rank - lower)
    return lower_value + upper_value
