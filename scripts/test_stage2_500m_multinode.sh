#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
echo "[WARN] scripts/test_stage2_500m_multinode.sh is deprecated; use scripts/pretrain_pipeline.sh." >&2
exec bash "${SCRIPT_DIR}/pretrain_pipeline.sh" "$@"
