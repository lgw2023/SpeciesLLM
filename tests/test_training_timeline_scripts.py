from __future__ import annotations

from pathlib import Path


def test_training_timeline_scripts_are_safe_local_entrypoints() -> None:
    backend = Path("scripts/run_training_timeline_backend.sh").read_text(encoding="utf-8")
    frontend = Path("scripts/run_training_timeline_frontend.sh").read_text(encoding="utf-8")

    assert "python -m training_timeline.cli serve" in backend
    assert "python -m training_timeline.cli rebuild" in backend
    assert "--host 127.0.0.1" in backend
    assert 'TRAINING_TIMELINE_BACKEND_PORT:-8766' in backend
    assert '--port "$PORT"' in backend
    assert "TRAINING_TIMELINE_BACKEND_URL" in frontend
    assert "127.0.0.1:${TRAINING_TIMELINE_BACKEND_PORT:-8766}" in frontend
    assert "npm run dev" in frontend
    assert "rm -rf" not in backend
    assert "rm -rf" not in frontend
