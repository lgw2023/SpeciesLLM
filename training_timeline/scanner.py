from __future__ import annotations

import hashlib
from pathlib import Path

from training_timeline.models import RunDiscovery


def discover_runs(source_roots: list[Path]) -> list[RunDiscovery]:
    discoveries: list[RunDiscovery] = []
    seen_real_paths: set[Path] = set()

    for source_root in source_roots:
        root = source_root.resolve()
        if not root.exists():
            continue
        for child in sorted(root.iterdir(), key=lambda item: item.name):
            if not _is_candidate_run_dir(child):
                continue
            real_path = child.resolve()
            if real_path in seen_real_paths:
                continue
            seen_real_paths.add(real_path)
            stat = child.stat()
            discoveries.append(
                RunDiscovery(
                    id=_stable_run_id(real_path, child.name),
                    path=child,
                    real_path=real_path,
                    name=child.name,
                    source_root=root,
                    mtime=stat.st_mtime,
                )
            )

    return sorted(discoveries, key=lambda item: item.name)


def _is_candidate_run_dir(path: Path) -> bool:
    if not path.name.startswith("training_output"):
        return False
    if path.name.endswith("_text_split"):
        return False
    return path.is_dir()


def _stable_run_id(real_path: Path, name: str) -> str:
    payload = f"{real_path}|{name}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:16]
