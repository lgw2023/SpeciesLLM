#!/usr/bin/env python3
"""
Replay adaptive grad-clip decisions against an existing metrics JSONL file.

This is a lightweight diagnostics tool for workstation use. It reads only the
logged raw gradient norms and simulates the controller thresholds; it does not
load training data, checkpoints, torch, or torch_npu.

Example:
  python3 work_record/replay_adaptive_grad_clip.py \
    training_output_500m_data_1_2_3_stable_from_scratch_20260515_195223/metrics.0-0.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


class ReplayAdaptiveGradClip:
    def __init__(
        self,
        *,
        beta: float,
        clip_ratio: float,
        skip_ratio: float,
        min_clip: float,
        max_clip: float,
        warmup_steps: int,
        static_clip: float,
        skip_max: float,
        max_consecutive_skips: int,
        ema_runaway_factor: float,
        hard_raw_norm_limit: float,
    ) -> None:
        self.beta = float(beta)
        self.clip_ratio = float(clip_ratio)
        self.skip_ratio = float(skip_ratio)
        self.min_clip = float(min_clip)
        self.max_clip = float(max_clip)
        self.warmup_steps = max(0, int(warmup_steps))
        self.static_clip = float(static_clip) if static_clip > 0 else 1.0
        self.skip_max = max(0.0, float(skip_max))
        self.max_consecutive_skips = max(0, int(max_consecutive_skips))
        self.ema_runaway_factor = max(0.0, float(ema_runaway_factor))
        self.hard_raw_norm_limit = max(0.0, float(hard_raw_norm_limit))
        if self.skip_max > 0.0 and self.skip_ratio > 0.0 and math.isfinite(self.skip_ratio):
            self.ema_contribution_cap = self.skip_max / self.skip_ratio
        else:
            self.ema_contribution_cap = 0.0
        self.ema: float | None = None
        self.observed = 0
        self.skipped = 0
        self.clipped = 0
        self.consecutive_skips = 0
        self.consecutive_clips = 0
        self.max_consecutive_skips_seen = 0
        self.max_consecutive_clips_seen = 0

    def warmup_done(self) -> bool:
        return self.ema is not None and self.observed >= self.warmup_steps

    def thresholds(self) -> tuple[float, float]:
        if not self.warmup_done():
            clip = self.static_clip
            skip = float("inf")
        else:
            assert self.ema is not None
            clip = max(self.min_clip, min(self.max_clip, self.ema * self.clip_ratio))
            if self.skip_ratio > 0.0 and math.isfinite(self.skip_ratio):
                skip = self.ema * self.skip_ratio
            else:
                skip = float("inf")
            if self.skip_max > 0.0:
                skip = min(skip, self.skip_max)
        if self.hard_raw_norm_limit > 0.0:
            skip = min(skip, self.hard_raw_norm_limit)
        return clip, skip

    def update(self, raw_norm: float) -> None:
        if not math.isfinite(raw_norm):
            return
        if self.ema is None:
            contribution = raw_norm
        else:
            contribution = min(raw_norm, self.ema * self.clip_ratio)
        if self.ema_contribution_cap > 0.0:
            contribution = min(contribution, self.ema_contribution_cap)
        if self.ema is None:
            self.ema = contribution
        else:
            self.ema = self.beta * self.ema + (1.0 - self.beta) * contribution
        self.observed += 1

    def note_action(self, action: str) -> None:
        if action in ("skip_nan", "skip_norm"):
            self.skipped += 1
            self.consecutive_skips += 1
            self.consecutive_clips = 0
            self.max_consecutive_skips_seen = max(
                self.max_consecutive_skips_seen, self.consecutive_skips
            )
        elif action == "clip":
            self.clipped += 1
            self.consecutive_clips += 1
            self.consecutive_skips = 0
            self.max_consecutive_clips_seen = max(
                self.max_consecutive_clips_seen, self.consecutive_clips
            )
        else:
            self.consecutive_skips = 0
            self.consecutive_clips = 0

    def safety_check(self, raw_norm: float) -> str | None:
        if (
            self.hard_raw_norm_limit > 0.0
            and math.isfinite(raw_norm)
            and raw_norm > self.hard_raw_norm_limit
        ):
            return f"raw_norm={raw_norm:.4e} exceeds hard_raw_norm_limit={self.hard_raw_norm_limit:.4e}"
        if (
            self.max_consecutive_skips > 0
            and self.consecutive_skips >= self.max_consecutive_skips
        ):
            return (
                f"consecutive_skips={self.consecutive_skips} "
                f">= max_consecutive_skips={self.max_consecutive_skips}"
            )
        if (
            self.ema_runaway_factor > 0.0
            and self.ema is not None
            and self.ema_contribution_cap > 0.0
            and self.ema > self.ema_contribution_cap * self.ema_runaway_factor
        ):
            return (
                f"ema={self.ema:.4e} > ema_runaway_factor={self.ema_runaway_factor:.2f} "
                f"* ema_contribution_cap={self.ema_contribution_cap:.4e}"
            )
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path, help="metrics.<node>-<rank>.jsonl to replay")
    ap.add_argument("--grad-clip", type=float, default=0.5)
    ap.add_argument("--grad-clip-ema-beta", type=float, default=0.98)
    ap.add_argument("--grad-clip-ratio", type=float, default=3.0)
    ap.add_argument("--grad-skip-ratio", type=float, default=100.0)
    ap.add_argument("--grad-skip-max", type=float, default=100000000000.0)
    ap.add_argument("--grad-clip-min", type=float, default=0.5)
    ap.add_argument("--grad-clip-max", type=float, default=1000.0)
    ap.add_argument("--grad-clip-warmup-steps", type=int, default=200)
    ap.add_argument("--grad-clip-max-consecutive-skips", type=int, default=50)
    ap.add_argument("--grad-clip-ema-runaway-factor", type=float, default=2.0)
    ap.add_argument("--grad-clip-hard-raw-norm-limit", type=float, default=100000000000.0)
    args = ap.parse_args()

    if not args.path.exists():
        print(f"[ERROR] file not found: {args.path}", file=sys.stderr)
        return 2

    state = ReplayAdaptiveGradClip(
        beta=args.grad_clip_ema_beta,
        clip_ratio=args.grad_clip_ratio,
        skip_ratio=args.grad_skip_ratio,
        min_clip=args.grad_clip_min,
        max_clip=args.grad_clip_max,
        warmup_steps=args.grad_clip_warmup_steps,
        static_clip=args.grad_clip,
        skip_max=args.grad_skip_max,
        max_consecutive_skips=args.grad_clip_max_consecutive_skips,
        ema_runaway_factor=args.grad_clip_ema_runaway_factor,
        hard_raw_norm_limit=args.grad_clip_hard_raw_norm_limit,
    )

    rows = 0
    raw_rows = 0
    first_skip: tuple[int | None, float, float] | None = None
    abort: tuple[int | None, str] | None = None
    actions: dict[str, int] = {}
    raw_max = 0.0
    ema_max = 0.0
    clip_thr_max = 0.0
    skip_thr_min = float("inf")
    skip_thr_max = 0.0

    with args.path.open(encoding="utf-8") as f:
        for line in f:
            rows += 1
            row = json.loads(line)
            raw = row.get("grad_norm_raw")
            if raw is None:
                continue
            raw_rows += 1
            raw = float(raw)
            step = row.get("update_step")
            clip_thr, skip_thr = state.thresholds()
            raw_max = max(raw_max, raw)
            clip_thr_max = max(clip_thr_max, clip_thr)
            if math.isfinite(skip_thr):
                skip_thr_min = min(skip_thr_min, skip_thr)
                skip_thr_max = max(skip_thr_max, skip_thr)

            if not math.isfinite(raw):
                action = "skip_nan"
            elif math.isfinite(skip_thr) and raw > skip_thr:
                action = "skip_norm"
            elif raw > clip_thr:
                action = "clip"
            else:
                action = "pass"
            actions[action] = actions.get(action, 0) + 1
            if action in ("skip_nan", "skip_norm"):
                if first_skip is None:
                    first_skip = (step, raw, skip_thr)
            else:
                state.update(raw)
                if state.ema is not None:
                    ema_max = max(ema_max, state.ema)
            state.note_action(action)

            reason = state.safety_check(raw)
            if reason is not None:
                abort = (step, reason)
                break

    print(f"=== Replay: {args.path} ===")
    print(f"rows={rows} raw_rows={raw_rows}")
    print(f"actions={actions}")
    print(f"raw_max={raw_max:.4e} ema_max={ema_max:.4e} clip_thr_max={clip_thr_max:.4e}")
    if skip_thr_min < float("inf"):
        print(f"skip_thr_range={skip_thr_min:.4e} -> {skip_thr_max:.4e}")
    print(
        "config="
        f"clip={args.grad_clip} clip_max={args.grad_clip_max} "
        f"skip_ratio={args.grad_skip_ratio} skip_max={args.grad_skip_max:.4e} "
        f"hard={args.grad_clip_hard_raw_norm_limit:.4e}"
    )
    if first_skip is None:
        print("first_skip=None")
    else:
        step, raw, thr = first_skip
        print(f"first_skip=step {step}, raw={raw:.4e}, threshold={thr:.4e}")
    if abort is None:
        print("abort=None")
        return 0
    step, reason = abort
    print(f"abort=step {step}: {reason}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
