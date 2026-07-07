from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_source_roots(config_path: Path | None, repo_root: Path) -> list[Path]:
    roots = [repo_root.resolve()]
    if config_path is None or not config_path.exists():
        return roots

    data = json.loads(config_path.read_text(encoding="utf-8"))
    for raw_path in _list_extra_roots(data):
        root = Path(raw_path).expanduser().resolve()
        if root not in roots:
            roots.append(root)
    return roots


def is_inside_source(path: Path, source_roots: list[Path]) -> bool:
    resolved = path.resolve()
    for root in source_roots:
        resolved_root = root.resolve()
        if resolved == resolved_root or resolved_root in resolved.parents:
            return True
    return False


def _list_extra_roots(data: Any) -> list[str]:
    if not isinstance(data, dict):
        return []
    values = data.get("extra_source_roots", [])
    if not isinstance(values, list):
        return []
    return [str(value) for value in values]
