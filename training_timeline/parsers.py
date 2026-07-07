from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from training_timeline.models import ArtifactRef, ParsedRunName, SummaryInfo


_TIMESTAMP_RE = re.compile(r"(?P<date>\d{8})_(?P<time>\d{6})$")
_DATA_RE = re.compile(r"data(?:_\d+)+")


def parse_run_directory_name(name: str) -> ParsedRunName:
    body = name.removeprefix("training_output_")
    timestamp = _parse_trailing_timestamp(body)
    body_without_timestamp = _TIMESTAMP_RE.sub("", body).strip("_")
    model_size = _parse_model_size(body_without_timestamp)
    data_recipe = _parse_data_recipe(body_without_timestamp)
    tags = _parse_tags(body_without_timestamp)
    experiment_name = _parse_experiment_name(body_without_timestamp, model_size, data_recipe)
    return ParsedRunName(
        model_size=model_size,
        data_recipe=data_recipe,
        experiment_name=experiment_name,
        started_at=timestamp,
        tags=tags,
    )


def parse_run_record(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "run_record.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def parse_summary(run_dir: Path) -> SummaryInfo:
    path = run_dir / "summary.md"
    if not path.exists():
        return SummaryInfo()

    text = path.read_text(encoding="utf-8", errors="replace")
    title = ""
    one_liner = ""
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#") and not title:
            title = stripped.lstrip("#").strip()
            continue
        if not stripped.startswith("#") and not one_liner:
            one_liner = stripped[:240]
        if title and one_liner:
            break
    return SummaryInfo(title=title, one_liner=one_liner, text=text)


def collect_artifacts(run_dir: Path) -> list[ArtifactRef]:
    artifacts: list[ArtifactRef] = []
    for path in sorted(run_dir.iterdir(), key=lambda item: item.name):
        if not path.is_file():
            continue
        kind = _artifact_kind(path)
        if kind is None:
            continue
        stat = path.stat()
        artifacts.append(ArtifactRef(run_id="", kind=kind, path=path, size_bytes=stat.st_size, mtime=stat.st_mtime))
    return artifacts


def _parse_trailing_timestamp(body: str) -> str | None:
    match = _TIMESTAMP_RE.search(body)
    if match is None:
        return None
    parsed = datetime.strptime(f"{match.group('date')}_{match.group('time')}", "%Y%m%d_%H%M%S")
    return parsed.isoformat()


def _parse_model_size(body: str) -> str:
    for model_size in ("100m", "500m", "1b"):
        if re.search(rf"(^|_){re.escape(model_size)}(_|$)", body):
            return model_size
    return "unknown"


def _parse_data_recipe(body: str) -> str:
    match = _DATA_RE.search(body)
    return match.group(0) if match is not None else ""


def _parse_experiment_name(body: str, model_size: str, data_recipe: str) -> str:
    parts = body
    if model_size != "unknown":
        parts = re.sub(rf"(^|_){re.escape(model_size)}(_|$)", "_", parts, count=1).strip("_")
    if data_recipe:
        parts = parts.replace(data_recipe, "", 1).strip("_")
    return re.sub(r"_+", "_", parts).strip("_")


def _parse_tags(body: str) -> list[str]:
    tokens = [token for token in body.split("_") if token]
    tags: list[str] = []
    exact_tags = {"stable", "fp32", "esm2", "dnaseq", "shuffleall", "resume", "from_scratch"}

    for index, token in enumerate(tokens):
        if token in exact_tags:
            tags.append(token)
        if token == "from" and index + 1 < len(tokens) and tokens[index + 1] == "scratch":
            tags.append("from_scratch")
        if token == "lossw" and index + 1 < len(tokens):
            tags.append(f"lossw_{tokens[index + 1]}")
        if token.startswith(("lr", "epoch", "huber", "clip", "lossw")):
            tags.append(token)

    seen: set[str] = set()
    ordered: list[str] = []
    for tag in tags:
        if tag not in seen:
            seen.add(tag)
            ordered.append(tag)
    return ordered


def _artifact_kind(path: Path) -> str | None:
    name = path.name
    suffix = path.suffix.lower()
    if name == "summary.md":
        return "summary_md"
    if name == "run_record.json":
        return "run_record_json"
    if name.startswith("metrics.") and suffix == ".jsonl":
        return "metrics_jsonl"
    if name.startswith("loss_to_log") and suffix == ".txt":
        return "loss_log"
    if name.startswith("log.") and suffix == ".txt":
        return "log"
    if suffix == ".png":
        if "grad" in name and "clip" in name:
            return "grad_clip_png"
        return "training_curve_png"
    if suffix == ".pt":
        return "checkpoint"
    return None
