from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunDiscovery:
    id: str
    path: Path
    real_path: Path
    name: str
    source_root: Path
    mtime: float


@dataclass(frozen=True)
class ParsedRunName:
    model_size: str = "unknown"
    data_recipe: str = ""
    experiment_name: str = ""
    started_at: str | None = None
    tags: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SummaryInfo:
    title: str = ""
    one_liner: str = ""
    text: str = ""


@dataclass(frozen=True)
class MetricPoint:
    series_name: str
    step: int
    epoch: float | None
    value: float
    sample_count: int = 1
    aggregation: str = "raw"


@dataclass(frozen=True)
class MetricSummary:
    run_id: str
    best_loss: float | None = None
    best_loss_step: int | None = None
    final_loss: float | None = None
    final_step: int | None = None
    early_loss_mean: float | None = None
    tail_loss_mean: float | None = None
    grad_norm_p50: float | None = None
    grad_norm_p95: float | None = None
    grad_norm_p99: float | None = None
    grad_norm_max: float | None = None
    clip_count: int = 0
    clip_fraction: float | None = None
    skip_count: int = 0
    skip_fraction: float | None = None
    row_count: int = 0


@dataclass(frozen=True)
class DiagnosticEvent:
    id: str
    run_id: str
    event_type: str
    severity: str
    title: str
    description: str
    evidence: dict[str, Any]
    start_step: int | None = None
    end_step: int | None = None
    source_file: str | None = None
    created_by: str = "auto"


@dataclass(frozen=True)
class AnalysisNote:
    id: str
    run_id: str
    note_type: str
    title: str
    body: str
    confidence: str
    supersedes_diagnostic_ids: list[str]
    evidence_refs: list[str]
    author: str
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class ArtifactRef:
    run_id: str
    kind: str
    path: Path
    size_bytes: int
    mtime: float
