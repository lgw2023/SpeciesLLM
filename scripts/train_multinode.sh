#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
echo "[WARN] scripts/train_multinode.sh is deprecated; use scripts/launch_multinode_torchrun.sh." >&2
exec bash "${SCRIPT_DIR}/launch_multinode_torchrun.sh" "$@"
