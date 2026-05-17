#!/usr/bin/env python3
"""
Quick health-check for a stability-validation training run.

Reads a per-rank metrics.0-0.jsonl produced by train_MNodes_torchrun_mfu_preindexparquet.py
and reports the four diagnostics that matter after the data_1_2_3_stable
post-mortem:

  1. grad_action distribution   — clip should dominate, skip should be small.
  2. EMA peak vs max_clip       — EMA pinned at max_clip ⇒ controller is at
                                  its ceiling for the whole run; raw_norm is
                                  fundamentally larger than what max_clip
                                  expects ⇒ raise max_clip OR lower LR.
  3. consecutive_skips peak     — if it ever approaches max_consecutive_skips
                                  the safety net is at risk of firing.
  4. loss + grad trajectory     — quick sanity print of where the run ended.

Returns exit code 0 if healthy, 1 if any tripwire-class flag is set.

Usage:
  python3 work_record/check_stability_health.py <out_dir>/metrics.0-0.jsonl
  python3 work_record/check_stability_health.py <out_dir>/metrics.0-0.jsonl --max-clip 1e6
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path, help="path to metrics.0-0.jsonl")
    ap.add_argument("--max-clip", type=float, default=None,
                    help="GRAD_CLIP_MAX used for the run (auto-detected if "
                         "grad_clip_threshold is present)")
    ap.add_argument("--max-consec-skips", type=int, default=50,
                    help="GRAD_CLIP_MAX_CONSECUTIVE_SKIPS used for the run")
    args = ap.parse_args()

    if not args.path.exists():
        print(f"[ERROR] file not found: {args.path}", file=sys.stderr)
        return 2

    n = 0
    actions: dict[str, int] = {}
    ema_max = raw_max = 0.0
    loss_first: Optional[float] = None
    loss_last: Optional[float] = None
    step_first: Optional[int] = None
    step_last: Optional[int] = None
    consec_skip_max = 0
    consec_clip_max = 0
    clip_thr_seen: Optional[float] = None
    inferred_max_clip: Optional[float] = None
    sub_loss_first: dict[str, Optional[float]] = {}
    sub_loss_last: dict[str, Optional[float]] = {}

    with args.path.open() as f:
        for line in f:
            row = json.loads(line)
            n += 1
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
            # grad_clip_threshold capped at max_clip; track its observed ceiling
            ct = row.get("grad_clip_threshold")
            if ct is not None:
                if clip_thr_seen is None or ct > clip_thr_seen:
                    clip_thr_seen = ct
            lt = row.get("loss_total")
            if lt is not None:
                if loss_first is None:
                    loss_first = lt
                    step_first = row.get("update_step")
                    for k in ("loss_gep", "loss_zero_prob", "loss_gepc", "loss_gepc_zero_prob"):
                        sub_loss_first[k] = row.get(k)
                loss_last = lt
                step_last = row.get("update_step")
                for k in ("loss_gep", "loss_zero_prob", "loss_gepc", "loss_gepc_zero_prob"):
                    sub_loss_last[k] = row.get(k)

    if args.max_clip is not None:
        inferred_max_clip = args.max_clip
    elif clip_thr_seen is not None:
        # max_clip is the ceiling on clip_threshold; observed max is a good proxy
        inferred_max_clip = clip_thr_seen

    print(f"=== Stability health for {args.path} ===")
    print(f"rows: {n}, step range: {step_first} → {step_last}")
    if inferred_max_clip is not None:
        print(f"detected max_clip (from clip_threshold ceiling): {inferred_max_clip:.3e}")

    print(f"\n[1/4] grad_action distribution:")
    total = sum(actions.values()) or 1
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

    print(f"\n[2/4] EMA peak:    {ema_max:.3e}")
    print(f"      raw peak:    {raw_max:.3e}")
    if inferred_max_clip is not None and ema_max > 0:
        ratio = ema_max / inferred_max_clip
        if ratio > 0.95:
            print(f"  ⚠ EMA peak / max_clip = {ratio:.2f} — controller at ceiling. "
                  f"Either raise GRAD_CLIP_MAX or lower LEARNING_RATE.")
        else:
            print(f"  ✅ EMA peak / max_clip = {ratio:.2f}  (controller has headroom)")

    print(f"\n[3/4] consecutive_skips max seen: {consec_skip_max}")
    print(f"      consecutive_clips max seen: {consec_clip_max}")
    if consec_skip_max >= args.max_consec_skips:
        print(f"  ❌ hit GRAD_CLIP_MAX_CONSECUTIVE_SKIPS ({args.max_consec_skips}) — "
              f"training would have aborted")
    elif consec_skip_max >= args.max_consec_skips * 0.5:
        print(f"  ⚠ consec_skips at >50% of safety threshold — close to abort")
    else:
        print(f"  ✅ well below safety threshold ({args.max_consec_skips})")

    print(f"\n[4/4] loss trajectory:")
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

    # Final verdict
    print()
    bad = False
    if (actions.get("skip_norm", 0) / total) > 0.5:
        bad = True
    if consec_skip_max >= args.max_consec_skips:
        bad = True
    if inferred_max_clip is not None and ema_max > 0 and ema_max / inferred_max_clip > 0.999:
        bad = True
    if loss_first is not None and loss_last is not None and loss_last > loss_first * 1.5:
        bad = True

    if bad:
        print("=== VERDICT: ❌ UNHEALTHY — see flags above ===")
        return 1
    else:
        print("=== VERDICT: ✅ HEALTHY ===")
        return 0


if __name__ == "__main__":
    sys.exit(main())
