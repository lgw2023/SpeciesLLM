#!/usr/bin/env bash
# 500M gradient-clip stability validation suite.
#
# Three experiments wrap work_record/step3_model_500M.sh with the parameter
# combinations recommended after the data_1_2_3_stable post-mortem
# (see ../work_record/step3_model_500M.sh and ../train_MNodes_...py for context):
#
#   smoke - 500 step canary, original LR, fixed adaptive clip
#           Verifies (a) the EMA-cap fix works end-to-end, (b) the controller
#           doesn't fail-fast on this model's natural gradient scale.
#
#   A     - 4000 step run, original LR=1e-6, GRAD_CLIP_MAX=1e6
#           "Is the EMA-cap fix alone enough to prevent the runaway we hit?"
#           If A stays healthy past warmup completion (~step 7500) we know
#           the bug was purely adaptive-clip misconfiguration.
#
#   B     - 4000 step run, halved LR=5e-7, tighter adaptive
#           Insurance: if A diverges, B is the next thing to try.
#           Smaller LR + faster EMA decay + tighter clip ratios.
#
# Usage:
#   bash work_record/stability_experiments.sh smoke
#   bash work_record/stability_experiments.sh A
#   bash work_record/stability_experiments.sh B
#   bash work_record/stability_experiments.sh all          # smoke → A → B in order
#
# Override DATA_PATH if needed:
#   DATA_PATH=/some/other/dir bash work_record/stability_experiments.sh smoke
#
# After each experiment, run:
#   python3 work_record/check_stability_health.py <out_dir>/metrics.0-0.jsonl
# to verify gradient health (clip≫skip, EMA<max_clip, no consec_skips runs).
#
# NOTE: this wrapper passes vars to step3_model_500M.sh as KEY=VALUE positional
# arguments. step3 re-execs with `env -i` so any exported envs are dropped --
# do not rely on the calling environment beyond DATA_PATH below.

set -euo pipefail

usage() {
  sed -n '4,30p' "$0" | sed 's/^# \{0,1\}//'
}

EXP="${1:-}"
if [[ -z "$EXP" || "$EXP" == "-h" || "$EXP" == "--help" || "$EXP" == "help" ]]; then
  usage
  exit 0
fi

cd /data/disk1/SpeciesLLM

DATA_PATH="${DATA_PATH:-/data/disk1/SpeciesLLM_obs/Stage2_SpeciesLLMData/all_flatten_data_full_no_1st_human_mouse_20260506_165244_external}"

# ── Common args shared by every experiment ─────────────────────────────────
# Safety nets stay on for ALL experiments. If any of these fires we want to
# see it immediately rather than silently waste compute (the failure mode of
# the data_1_2_3_stable run).
COMMON_ARGS=(
  GRAD_CLIP=1.0                           # static fallback during warmup
  ADAPTIVE_GRAD_CLIP=true
  GRAD_CLIP_MIN=0.5
  GRAD_CLIP_WARMUP_STEPS=200
  GRAD_CLIP_MAX_CONSECUTIVE_SKIPS=50      # abort after 50 skipped optimizer steps
  GRAD_CLIP_EMA_RUNAWAY_FACTOR=2.0        # defensive: EMA cap should make this unreachable
  GRAD_CLIP_HARD_RAW_NORM_LIMIT=1.0e+12   # physical fuse far above normal regime
  BETA2=0.98
  WARMUP_RATIO=0.10
  NAN_CHECK_INTERVAL=10
  # Crucial for this model: natural grad_norm ~1e4-1e6 so the adaptive ceiling
  # has to be much higher than the controller's stock default of 1000.
  GRAD_CLIP_MAX=1000000.0                 # 1e6 — see post-mortem
)

run_exp() {
  local name="$1"; shift
  echo
  echo "============================================================"
  echo "▶ Launching $name"
  echo "  DATA_PATH=$DATA_PATH"
  for kv in "$@"; do echo "  $kv"; done
  echo "============================================================"
  bash work_record/step3_model_500M.sh \
    ACTION=launch \
    EXPERIMENT_NAME="$name" \
    DATA_PATH="$DATA_PATH" \
    "${COMMON_ARGS[@]}" \
    "$@"
}

# ── Experiment definitions ─────────────────────────────────────────────────

run_smoke() {
  # 500 step canary: original LR, baseline adaptive ratios, large ceiling.
  # ~50 min wall. Just checks the pipeline launches and the EMA cap holds.
  run_exp stab_smoke_500 \
    LEARNING_RATE=0.000001 \
    MIN_LR=0.0000001 \
    GRAD_CLIP_RATIO=3.0 \
    GRAD_SKIP_RATIO=10.0 \
    GRAD_CLIP_EMA_BETA=0.98 \
    MAX_TRAIN_STEPS=500
}

run_A() {
  # 4000 step run, ORIGINAL LR + baseline adaptive ratios + EMA-cap fix.
  # ~7h wall. Tests the hypothesis "the EMA-cap code fix alone fixes the bug".
  # Watch for: EMA staying well below 1e6, low consecutive_skips, loss
  # decreasing past warmup (~step 7500).
  run_exp stab_A_lr1e-6_4k \
    LEARNING_RATE=0.000001 \
    MIN_LR=0.0000001 \
    GRAD_CLIP_RATIO=3.0 \
    GRAD_SKIP_RATIO=10.0 \
    GRAD_CLIP_EMA_BETA=0.98 \
    MAX_TRAIN_STEPS=4000
}

run_B() {
  # 4000 step run, HALVED LR + tighter adaptive + faster-decaying EMA.
  # ~7h wall. The "safer" baseline if A turns out to still diverge.
  #   * LR  1e-6 → 5e-7  (half — gives the optimizer more headroom)
  #   * clip ratio 3 → 2 (spike caught one step earlier)
  #   * skip ratio 10 → 5 (don't tolerate huge outliers)
  #   * EMA beta 0.98 → 0.95 (half-life 35 → 14 step, recovers faster)
  run_exp stab_B_lr5e-7_4k \
    LEARNING_RATE=0.0000005 \
    MIN_LR=0.00000005 \
    GRAD_CLIP_RATIO=2.0 \
    GRAD_SKIP_RATIO=5.0 \
    GRAD_CLIP_EMA_BETA=0.95 \
    MAX_TRAIN_STEPS=4000
}

case "$EXP" in
  smoke|0)
    run_smoke
    ;;
  A|a)
    run_A
    ;;
  B|b)
    run_B
    ;;
  all)
    run_smoke
    run_A
    run_B
    ;;
  *)
    echo "[ERROR] Unknown experiment: $EXP" >&2
    echo "Use one of: smoke | A | B | all" >&2
    exit 1
    ;;
esac
