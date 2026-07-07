from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def make_run_dir(
    root: Path,
    name: str,
    *,
    run_record: dict[str, Any] | None = None,
    summary: str | None = None,
    metrics_rows: list[dict[str, Any]] | None = None,
) -> Path:
    run_dir = root / name
    run_dir.mkdir(parents=True, exist_ok=True)

    if run_record is not None:
        (run_dir / "run_record.json").write_text(
            json.dumps(run_record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    if summary is not None:
        (run_dir / "summary.md").write_text(summary, encoding="utf-8")

    if metrics_rows is not None:
        with (run_dir / "metrics.0-0.jsonl").open("w", encoding="utf-8") as handle:
            for row in metrics_rows:
                handle.write(json.dumps(row, ensure_ascii=False))
                handle.write("\n")

    return run_dir
