#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p .training_timeline

PORT="${TRAINING_TIMELINE_BACKEND_PORT:-8766}"

python -m training_timeline.cli rebuild \
  --db .training_timeline/timeline.sqlite \
  --source "$ROOT"

python -m training_timeline.cli serve \
  --db .training_timeline/timeline.sqlite \
  --source "$ROOT" \
  --host 127.0.0.1 \
  --port "$PORT"
