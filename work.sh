#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
echo "[WARN] work.sh is deprecated; use scripts/smoke_500m_3node.sh." >&2
exec bash "${SCRIPT_DIR}/scripts/smoke_500m_3node.sh" "$@"
