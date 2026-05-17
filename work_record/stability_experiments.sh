#!/usr/bin/env bash
# 500M gradient-clip stability validation suite.
#
# Experiments wrap work_record/step3_model_500M.sh with the parameter
# combinations recommended after the data_1_2_3_stable post-mortem
# (see ../work_record/step3_model_500M.sh and ../train_MNodes_...py for context):
#
#   smoke - 500 step launch canary, original LR, adaptive clip
#           Verifies the controller does not false-trip on the natural 1e8 raw
#           grad-norm scale seen during early 500M training.
#
#   A     - 13000 step canary, original LR=1e-6
#           Reaches beyond the old first skip at step 11895. Uses a decoupled
#           raw-norm skip fuse so 1e8 early norms are clipped, not skipped, while
#           1e11-class excursions still abort quickly.
#
#   B     - 13000 step canary, halved LR=5e-7, tighter adaptive
#           Insurance: if A diverges, B is the next thing to try.
#           Smaller LR + faster EMA decay + tighter clip ratios.
#
#   static - 13000 step conservative fallback
#           Disables adaptive skip entirely and uses static GRAD_CLIP=0.5.
#
# Usage:
#   bash work_record/stability_experiments.sh smoke
#   bash work_record/stability_experiments.sh A
#   bash work_record/stability_experiments.sh B
#   bash work_record/stability_experiments.sh static
#   bash work_record/stability_experiments.sh all          # smoke → A → B in order
#
# Override DATA_PATH if needed:
#   DATA_PATH=/some/other/dir bash work_record/stability_experiments.sh smoke
#
# After each experiment, run:
#   python3 work_record/check_stability_health.py <out_dir>/metrics.0-0.jsonl
# to verify gradient health (low skip rate, no consec_skips runs, raw norm below
# the configured skip/hard fuse).
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
  GRAD_CLIP=0.5                           # static fallback during warmup
  ADAPTIVE_GRAD_CLIP=true
  GRAD_CLIP_MIN=0.5
  GRAD_CLIP_WARMUP_STEPS=200
  GRAD_CLIP_MAX_CONSECUTIVE_SKIPS=50      # abort after 50 skipped optimizer steps
  GRAD_CLIP_EMA_RUNAWAY_FACTOR=2.0        # defensive: relative to GRAD_SKIP_MAX / GRAD_SKIP_RATIO
  GRAD_CLIP_HARD_RAW_NORM_LIMIT=1.0e+11   # physical fuse above old 4.3e10 peak, below 1e11-class blowups
  BETA2=0.98
  WARMUP_RATIO=0.10
  NAN_CHECK_INTERVAL=10
  GRAD_CLIP_MAX=1000.0                    # actual optimizer update remains bounded
  GRAD_SKIP_MAX=100000000000.0            # raw-norm skip fuse independent from GRAD_CLIP_MAX
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
  # 500 step canary: original LR, adaptive clip, absolute raw-norm fuse.
  # ~50 min wall. Checks launch + no false skip at the natural early 1e8 scale.
  run_exp stab_smoke_500 \
    LEARNING_RATE=0.000001 \
    MIN_LR=0.0000001 \
    GRAD_CLIP_RATIO=3.0 \
    GRAD_SKIP_RATIO=100.0 \
    GRAD_CLIP_EMA_BETA=0.98 \
    MAX_TRAIN_STEPS=500
}

run_A() {
  # 13000 step run, ORIGINAL LR + decoupled adaptive skip fuse.
  # Reaches beyond the old first skip at step 11895.
  # Watch for: low skip rate, raw_norm below 1e11, loss still moving down.
  run_exp stab_A_lr1e-6_13k \
    LEARNING_RATE=0.000001 \
    MIN_LR=0.0000001 \
    GRAD_CLIP_RATIO=3.0 \
    GRAD_SKIP_RATIO=100.0 \
    GRAD_CLIP_EMA_BETA=0.98 \
    MAX_TRAIN_STEPS=13000
}

run_B() {
  # 13000 step run, HALVED LR + tighter adaptive + faster-decaying EMA.
  # The safer baseline if A turns out to still diverge.
  #   * LR  1e-6 → 5e-7  (half — gives the optimizer more headroom)
  #   * clip ratio 3 → 2 (spike caught one step earlier)
  #   * skip ratio 100 → 50, skip fuse 1e11 → 5e10
  #   * EMA beta 0.98 → 0.95 (half-life 35 → 14 step, recovers faster)
  run_exp stab_B_lr5e-7_13k \
    LEARNING_RATE=0.0000005 \
    MIN_LR=0.00000005 \
    GRAD_CLIP_RATIO=2.0 \
    GRAD_SKIP_RATIO=50.0 \
    GRAD_SKIP_MAX=50000000000.0 \
    GRAD_CLIP_HARD_RAW_NORM_LIMIT=80000000000.0 \
    GRAD_CLIP_MAX=500.0 \
    GRAD_CLIP_EMA_BETA=0.95 \
    MAX_TRAIN_STEPS=13000
}

run_static() {
  # Conservative fallback: no adaptive skip/EMA; every finite update is clipped
  # to the original 0.5 global norm.
  run_exp stab_static_clip0p5_13k \
    ADAPTIVE_GRAD_CLIP=false \
    LEARNING_RATE=0.0000005 \
    MIN_LR=0.00000005 \
    GRAD_CLIP=0.5 \
    MAX_TRAIN_STEPS=13000
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
  static|S|s)
    run_static
    ;;
  all)
    run_smoke
    run_A
    run_B
    ;;
  *)
    echo "[ERROR] Unknown experiment: $EXP" >&2
    echo "Use one of: smoke | A | B | static | all" >&2
    exit 1
    ;;
esac
