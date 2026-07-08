#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/training_timeline_ui"
export TRAINING_TIMELINE_BACKEND_URL="${TRAINING_TIMELINE_BACKEND_URL:-http://127.0.0.1:${TRAINING_TIMELINE_BACKEND_PORT:-8766}}"
npm run dev
