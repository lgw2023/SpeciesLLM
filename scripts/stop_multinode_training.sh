#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/stop_multinode_training.sh OUT_DIR PATTERN

Arguments:
  OUT_DIR   Training output directory to inspect for recent checkpoints.
  PATTERN   Fixed string used to identify the target training command.

Environment:
  ENV_FILE       Defaults to .env.
  HOSTS          Comma- or space-separated hosts. Defaults to .env HOSTS.
  MASTER_ADDR    Host treated as local. Defaults to first HOSTS entry.
  SSH_USER       Optional SSH user.
  SSH_PASSWORD   Optional password used via sshpass -e.
  SSH_EXTRA_OPTS Optional SSH options.
  WAIT_SECONDS   Seconds to wait after SIGTERM before reporting. Default: 30.
  FORCE_KILL     Set to 1 to send SIGKILL to remaining matching processes.
USAGE
}

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || "${1:-}" == "help" ]]; then
  usage
  exit 0
fi

[[ "$#" -eq 2 ]] || {
  usage >&2
  exit 1
}

ARG_OUT_DIR="$1"
ARG_PATTERN="$2"
ENV_FILE="${ENV_FILE:-.env}"

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "$ENV_FILE"
  set +a
fi

OUT_DIR="$ARG_OUT_DIR"
PATTERN="$ARG_PATTERN"
WAIT_SECONDS="${WAIT_SECONDS:-30}"
FORCE_KILL="${FORCE_KILL:-0}"

[[ -n "$OUT_DIR" ]] || die "OUT_DIR is required"
[[ -n "$PATTERN" ]] || die "PATTERN is required"
[[ "$WAIT_SECONDS" =~ ^[0-9]+$ ]] || die "WAIT_SECONDS must be a non-negative integer"

HOSTS="${HOSTS:-}"
[[ -n "$HOSTS" ]] || die "HOSTS is required in ${ENV_FILE} or the environment"

shell_quote() {
  local value="$1"
  printf "'%s'" "${value//\'/\'\\\'\'}"
}

split_hosts() {
  HOSTS_ARR=()
  local normalized raw host
  normalized="${HOSTS//,/ }"
  for raw in $normalized; do
    host="${raw#"${raw%%[![:space:]]*}"}"
    host="${host%"${host##*[![:space:]]}"}"
    [[ -n "$host" ]] && HOSTS_ARR+=("$host")
  done
  [[ "${#HOSTS_ARR[@]}" -gt 0 ]] || die "HOSTS did not contain any hosts"
}

split_hosts
MASTER_ADDR="${MASTER_ADDR:-${HOSTS_ARR[0]}}"

read -r -a SSH_OPTS <<< "${SSH_EXTRA_OPTS:-}"

if [[ -n "${SSH_PASSWORD:-}" ]] && ! command -v sshpass >/dev/null 2>&1; then
  die "SSH_PASSWORD is set, but sshpass is not available"
fi

remote_target() {
  local host="$1"
  if [[ -n "${SSH_USER:-}" ]]; then
    printf "%s@%s" "$SSH_USER" "$host"
  else
    printf "%s" "$host"
  fi
}

is_local_host() {
  local host="$1"
  [[ "$host" == "$MASTER_ADDR" || "$host" == "localhost" || "$host" == "127.0.0.1" ]]
}

run_on_host() {
  local host="$1"
  local script="$2"
  if is_local_host "$host"; then
    bash -lc "$script"
  elif [[ -n "${SSH_PASSWORD:-}" ]]; then
    SSHPASS="$SSH_PASSWORD" sshpass -e ssh "${SSH_OPTS[@]}" "$(remote_target "$host")" "bash -lc $(shell_quote "$script")"
  else
    ssh "${SSH_OPTS[@]}" "$(remote_target "$host")" "bash -lc $(shell_quote "$script")"
  fi
}

process_filter_script() {
  local pattern_q
  pattern_q="$(shell_quote "$PATTERN")"
  cat <<SCRIPT
pgrep -af 'torch.distributed.run|train_MNodes_torchrun_mfu_preindexparquet.py' | grep -F -- ${pattern_q} || true
SCRIPT
}

kill_script() {
  local signal="$1"
  local pattern_q
  pattern_q="$(shell_quote "$PATTERN")"
  cat <<SCRIPT
pids=\$(pgrep -af 'torch.distributed.run|train_MNodes_torchrun_mfu_preindexparquet.py' | grep -F -- ${pattern_q} | awk '{print \$1}' || true)
if [[ -n "\$pids" ]]; then
  kill -${signal} \$pids
  printf '%s\n' "\$pids"
fi
SCRIPT
}

echo "OUT_DIR=${OUT_DIR}"
echo "PATTERN=${PATTERN}"
echo "HOSTS=${HOSTS_ARR[*]}"
echo "MASTER_ADDR=${MASTER_ADDR}"

if [[ -d "$OUT_DIR" ]]; then
  echo "===== local latest checkpoints ====="
  find "$OUT_DIR" -maxdepth 1 -type f \( -name 'SC-node-*-rank-*-epoch-*-step-*.pt' -o -name 'SC-node-*-rank-*-epoch-*-step-*.optimizer.pt' \) \
    -print | sort | tail -10 || true
else
  echo "[WARN] local OUT_DIR does not exist: ${OUT_DIR}"
fi

for host in "${HOSTS_ARR[@]}"; do
  echo "===== ${host}: matching processes before SIGTERM ====="
  if ! run_on_host "$host" "$(process_filter_script)"; then
    echo "[WARN] failed to inspect ${host}" >&2
  fi

  echo "===== ${host}: send SIGTERM ====="
  run_on_host "$host" "$(kill_script TERM)" || true
done

sleep "$WAIT_SECONDS"

remaining=0
for host in "${HOSTS_ARR[@]}"; do
  echo "===== ${host}: remaining after ${WAIT_SECONDS}s ====="
  if ! output="$(run_on_host "$host" "$(process_filter_script)" 2>&1)"; then
    remaining=1
    echo "[WARN] failed to inspect ${host}: ${output}" >&2
    continue
  fi
  if [[ -n "$output" ]]; then
    remaining=1
    printf '%s\n' "$output"
  fi
done

if [[ "$remaining" -eq 0 ]]; then
  echo "[OK] no matching training processes remain"
  exit 0
fi

if [[ "$FORCE_KILL" == "1" ]]; then
  for host in "${HOSTS_ARR[@]}"; do
    echo "===== ${host}: send SIGKILL ====="
    run_on_host "$host" "$(kill_script KILL)" || true
  done
  echo "[WARN] SIGKILL was sent to remaining matching processes"
else
  echo "[WARN] matching processes remain; rerun with FORCE_KILL=1 if they do not exit"
fi
