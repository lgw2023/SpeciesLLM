#!/usr/bin/env python3
"""
Quick health-check for a stability-validation training run.

Reads a per-rank metrics.0-0.jsonl produced by train_MNodes_torchrun_mfu_preindexparquet.py
and reports the four diagnostics that matter after the data_1_2_3_stable
post-mortem:

  1. grad_action distribution   — clip should dominate, skip should be small.
  2. raw/EMA vs skip fuses      — raw_norm should stay below the configured
                                  skip/hard raw-norm fuse; EMA is no longer
                                  expected to be comparable to max_clip.
  3. consecutive_skips peak     — if it ever approaches max_consecutive_skips
                                  the safety net is at risk of firing.
  4. loss trajectory            — quick sanity print of where the run ended.
  5. primary-task health        — GEP and zero-prob should move, not only GEPC.

Returns exit code 0 if healthy, 1 if any tripwire-class flag is set.

Usage:
  python3 work_record/check_stability_health.py <out_dir>/metrics.0-0.jsonl
  python3 work_record/check_stability_health.py <out_dir>/metrics.0-0.jsonl --skip-max 1e11
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional


LOSS_KEYS = ("loss_gep", "loss_zero_prob", "loss_gepc", "loss_gepc_zero_prob")


def window_stats(series: list[tuple[int | None, float]], window: int) -> Optional[tuple[float, float, float]]:
    if not series:
        return None
    n = max(1, min(int(window), len(series)))
    first = sum(v for _, v in series[:n]) / n
    last = sum(v for _, v in series[-n:]) / n
    drop_frac = (first - last) / max(abs(first), 1e-12)
    return first, last, drop_frac


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path, help="path to metrics.0-0.jsonl")
    ap.add_argument("--max-clip", type=float, default=None,
                    help="GRAD_CLIP_MAX used for the run (auto-detected if "
                         "grad_clip_threshold is present)")
    ap.add_argument("--max-consec-skips", type=int, default=50,
                    help="GRAD_CLIP_MAX_CONSECUTIVE_SKIPS used for the run")
    ap.add_argument("--skip-max", type=float, default=None,
                    help="GRAD_SKIP_MAX used for the run (auto-detected if present)")
    ap.add_argument("--hard-raw-norm-limit", type=float, default=None,
                    help="GRAD_CLIP_HARD_RAW_NORM_LIMIT used for the run (auto-detected if present)")
    ap.add_argument("--primary-min-steps", type=int, default=4000,
                    help="Only run GEP/zero-prob health checks after this many observed steps")
    ap.add_argument("--primary-window", type=int, default=25,
                    help="Number of logged loss rows in the first/last window for primary-task checks")
    ap.add_argument("--max-final-zero-prob", type=float, default=0.65,
                    help="Fail if final-window loss_zero_prob remains above this value")
    ap.add_argument("--min-gep-drop-frac", type=float, default=0.50,
                    help="Fail if loss_gep drops by less than this fraction across the run")
    ap.add_argument("--disable-primary-loss-check", action="store_true",
                    help="Only evaluate gradient-controller health; skip GEP/zero-prob checks")
    args = ap.parse_args()

    if not args.path.exists():
        print(f"[ERROR] file not found: {args.path}", file=sys.stderr)
        return 2

    n = 0
    actions: dict[str, int] = {}
    ema_max = raw_max = 0.0
    loss_first: Optional[float] = None
    loss_last: Optional[float] = None
    row_step_first: Optional[int] = None
    row_step_last: Optional[int] = None
    step_first: Optional[int] = None
    step_last: Optional[int] = None
    consec_skip_max = 0
    consec_clip_max = 0
    derived_consec_skip = 0
    derived_consec_clip = 0
    clip_thr_seen: Optional[float] = None
    inferred_max_clip: Optional[float] = None
    skip_thr_min: Optional[float] = None
    skip_thr_max: Optional[float] = None
    inferred_skip_max: Optional[float] = args.skip_max
    inferred_hard_raw_norm_limit: Optional[float] = args.hard_raw_norm_limit
    sub_loss_first: dict[str, Optional[float]] = {}
    sub_loss_last: dict[str, Optional[float]] = {}
    loss_series: dict[str, list[tuple[int | None, float]]] = {k: [] for k in LOSS_KEYS}

    with args.path.open() as f:
        for line in f:
            row = json.loads(line)
            n += 1
            step = row.get("update_step")
            if step is not None:
                if row_step_first is None:
                    row_step_first = step
                row_step_last = step
            act = row.get("grad_action")
            if act:
                actions[act] = actions.get(act, 0) + 1
            ema = row.get("grad_norm_ema")
            raw = row.get("grad_norm_raw")
            if ema is not None and ema > ema_max:
                ema_max = ema
            if raw is not None and raw > raw_max:
                raw_max = raw
            cs = row.get("consecutive_skips")
            if cs is not None and cs > consec_skip_max:
                consec_skip_max = cs
            cc = row.get("consecutive_clips")
            if cc is not None and cc > consec_clip_max:
                consec_clip_max = cc
            if raw is not None:
                if row.get("skipped") or act in ("skip_norm", "skip_nan"):
                    derived_consec_skip += 1
                else:
                    derived_consec_skip = 0
                if act == "clip":
                    derived_consec_clip += 1
                else:
                    derived_consec_clip = 0
                consec_skip_max = max(consec_skip_max, derived_consec_skip)
                consec_clip_max = max(consec_clip_max, derived_consec_clip)
            # grad_clip_threshold capped at max_clip; track its observed ceiling
            ct = row.get("grad_clip_threshold")
            if ct is not None:
                if clip_thr_seen is None or ct > clip_thr_seen:
                    clip_thr_seen = ct
            st = row.get("grad_skip_threshold")
            if st is not None:
                if skip_thr_min is None or st < skip_thr_min:
                    skip_thr_min = st
                if skip_thr_max is None or st > skip_thr_max:
                    skip_thr_max = st
            sm = row.get("grad_skip_max")
            if sm is not None and inferred_skip_max is None:
                inferred_skip_max = sm
            hl = row.get("grad_hard_raw_norm_limit")
            if hl is not None and inferred_hard_raw_norm_limit is None:
                inferred_hard_raw_norm_limit = hl
            lt = row.get("loss_total")
            if lt is not None:
                if loss_first is None:
                    loss_first = lt
                    step_first = step
                    for k in LOSS_KEYS:
                        sub_loss_first[k] = row.get(k)
                loss_last = lt
                step_last = step
                for k in LOSS_KEYS:
                    sub_loss_last[k] = row.get(k)
                    if row.get(k) is not None:
                        loss_series[k].append((step, row[k]))

    if args.max_clip is not None:
        inferred_max_clip = args.max_clip
    elif clip_thr_seen is not None:
        # max_clip is the ceiling on clip_threshold; observed max is a good proxy
        inferred_max_clip = clip_thr_seen

    print(f"=== Stability health for {args.path} ===")
    print(f"rows: {n}, step range: {row_step_first} → {row_step_last}")
    if inferred_max_clip is not None:
        print(f"observed clip_threshold ceiling: {inferred_max_clip:.3e}")
    if inferred_skip_max is not None:
        print(f"configured skip_max: {inferred_skip_max:.3e}")
    if inferred_hard_raw_norm_limit is not None:
        print(f"configured hard_raw_norm_limit: {inferred_hard_raw_norm_limit:.3e}")

    print(f"\n[1/5] grad_action distribution:")
    action_total = sum(actions.values())
    total = action_total or 1
    if action_total == 0:
        print("  grad_action not present in this metrics file")
    else:
        for k in ("pass", "clip", "skip_norm", "skip_nan", "no_step"):
            v = actions.get(k, 0)
            pct = 100.0 * v / total
            marker = ""
            if k == "skip_norm" and pct > 10.0:
                marker = "  ⚠ skip rate > 10%"
            if k == "skip_norm" and pct > 50.0:
                marker = "  ❌ skip rate > 50% — run is mostly skipping"
            if k == "pass" and v == 0:
                marker = "  ℹ all steps clipped; healthy if natural grad > clip_threshold"
            print(f"  {k:>10} : {v:>8}  ({pct:5.1f}%)  {marker}")

    print(f"\n[2/5] raw/EMA vs fuses:")
    print(f"      EMA peak:    {ema_max:.3e}")
    print(f"      raw peak:    {raw_max:.3e}")
    if skip_thr_min is not None and skip_thr_max is not None:
        print(f"      skip_thr:    {skip_thr_min:.3e} → {skip_thr_max:.3e}")
    if inferred_max_clip is not None:
        print(f"      clip_thr max:{inferred_max_clip:.3e}")
    if inferred_skip_max is not None and inferred_skip_max > 0:
        ratio = raw_max / inferred_skip_max
        marker = "✅" if ratio < 0.8 else ("⚠" if ratio < 1.0 else "❌")
        print(f"  {marker} raw peak / skip_max = {ratio:.2f}")
    if inferred_hard_raw_norm_limit is not None and inferred_hard_raw_norm_limit > 0:
        ratio = raw_max / inferred_hard_raw_norm_limit
        marker = "✅" if ratio < 0.8 else ("⚠" if ratio < 1.0 else "❌")
        print(f"  {marker} raw peak / hard_raw_norm_limit = {ratio:.2f}")

    print(f"\n[3/5] consecutive_skips max seen: {consec_skip_max}")
    print(f"      consecutive_clips max seen: {consec_clip_max}")
    if consec_skip_max >= args.max_consec_skips:
        print(f"  ❌ hit GRAD_CLIP_MAX_CONSECUTIVE_SKIPS ({args.max_consec_skips}) — "
              f"training would have aborted")
    elif consec_skip_max >= args.max_consec_skips * 0.5:
        print(f"  ⚠ consec_skips at >50% of safety threshold — close to abort")
    else:
        print(f"  ✅ well below safety threshold ({args.max_consec_skips})")

    print(f"\n[4/5] loss trajectory:")
    if loss_first is not None and loss_last is not None:
        delta = loss_last - loss_first
        print(f"  step {step_first} → {step_last}")
        print(f"  loss_total: {loss_first:.2f}  →  {loss_last:.2f}  (Δ {delta:+.2f})")
        for k, v0 in sub_loss_first.items():
            v1 = sub_loss_last.get(k)
            if v0 is not None and v1 is not None:
                print(f"  {k:>20}: {v0:>8.3f}  →  {v1:>8.3f}")
        if delta > 0:
            print(f"  ⚠ loss INCREASED — investigate before continuing")
    else:
        print(f"  no loss_total rows found")

    primary_bad = False
    primary_span = 0
    if row_step_first is not None and row_step_last is not None:
        primary_span = int(row_step_last) - int(row_step_first) + 1

    print(f"\n[5/5] primary-task health:")
    if args.disable_primary_loss_check:
        print("  skipped (--disable-primary-loss-check)")
    elif primary_span < args.primary_min_steps:
        print(f"  skipped: observed {primary_span} steps < primary_min_steps={args.primary_min_steps}")
    else:
        stats = {k: window_stats(v, args.primary_window) for k, v in loss_series.items()}
        gep_stats = stats.get("loss_gep")
        zero_stats = stats.get("loss_zero_prob")
        gepc_stats = stats.get("loss_gepc")

        if gep_stats is not None:
            first, last, drop_frac = gep_stats
            marker = "✅" if drop_frac >= args.min_gep_drop_frac else "❌"
            print(
                f"  {marker} loss_gep window avg: {first:.3f} → {last:.3f} "
                f"(drop {drop_frac * 100:.1f}%, required {args.min_gep_drop_frac * 100:.1f}%)"
            )
            if drop_frac < args.min_gep_drop_frac:
                primary_bad = True
        else:
            print("  ⚠ loss_gep not found; cannot evaluate GEP learning")

        if zero_stats is not None:
            first, last, drop_frac = zero_stats
            marker = "✅" if last <= args.max_final_zero_prob else "❌"
            print(
                f"  {marker} loss_zero_prob window avg: {first:.3f} → {last:.3f} "
                f"(final max {args.max_final_zero_prob:.3f})"
            )
            if last > args.max_final_zero_prob:
                primary_bad = True
        else:
            print("  ⚠ loss_zero_prob not found; cannot evaluate zero-prob learning")

        if gepc_stats is not None:
            first, last, drop_frac = gepc_stats
            print(f"  info: loss_gepc window avg: {first:.3f} → {last:.3f} (drop {drop_frac * 100:.1f}%)")
            if primary_bad and drop_frac > 0.5:
                print("  ❌ GEPC is learning while GEP/zero-prob is stuck; this is task-unhealthy")

    # Final verdict
    print()
    bad = False
    if (actions.get("skip_norm", 0) / total) > 0.5:
        bad = True
    if consec_skip_max >= args.max_consec_skips:
        bad = True
    if inferred_skip_max is not None and inferred_skip_max > 0 and raw_max >= inferred_skip_max:
        bad = True
    if (inferred_hard_raw_norm_limit is not None
            and inferred_hard_raw_norm_limit > 0
            and raw_max >= inferred_hard_raw_norm_limit):
        bad = True
    if loss_first is not None and loss_last is not None and loss_last > loss_first * 1.5:
        bad = True
    if primary_bad:
        bad = True

    if bad:
        print("=== VERDICT: ❌ UNHEALTHY — see flags above ===")
        return 1
    else:
        print("=== VERDICT: ✅ HEALTHY ===")
        return 0


if __name__ == "__main__":
    sys.exit(main())
