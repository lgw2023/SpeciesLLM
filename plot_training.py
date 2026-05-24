#!/usr/bin/env python3
"""
Training metrics visualizer for SpeciesLLM.

Log structure (from train_MNodes_torchrun_mfu_preindexparquet.py):
  - metrics.{node}-{rank}.jsonl  — one row per local batch iteration
      · step_ms / samples_per_s / tokens_per_s / lr → every batch (always present)
      · loss_total / loss_gep / loss_gepc / cluster_* → every log_interval batches (null otherwise)
      · mfu → every 100 batches (null otherwise)
  - log.{node}-{rank}.txt — human-readable; first block contains JSON args dump

Scans the project root for training_output_* directories, loads rank-0 metrics,
parses training args from the matching log file, and writes per-run figures
back into each training output directory by default.

Usage:
    python plot_training.py [training_output_dir ...]

    --smooth N        box-car window for loss curves (default 1 = off)
    --rank R          which GPU rank file to read (default 0)
    --max-steps N     only plot rows up to update_step N (default: all)
    --no-detail       skip the loss-components breakdown figure
    --no-timing       skip the per-step timing breakdown figure
    --no-grad-clip    skip the adaptive-grad-clip diagnostics figure
    --grad-clip-window N
                      rolling window for skip/clip RATE panel (default 100)
    --compare         when 2+ dirs are given, also produce a single overlay
                      figure under figures/compare_<keys>/ by default
    --no-individual   skip per-run figures (useful together with --compare)
    --out DIR         base output directory. When set, per-run figures are
                      saved under DIR/<run_key>/ instead of the run directory.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

matplotlib.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
        "lines.linewidth": 1.8,
    }
)

PALETTE = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800", "#00BCD4"]


# ── helpers ───────────────────────────────────────────────────────────────────

def _dir_label(path: Path) -> str:
    """training_output_100m_test_from_scratch_20260506_220944 → '100M · 2026-05-06 22:09'"""
    name = path.name
    size_m = re.search(r"_(\d+(?:\.\d+)?[mMbBkK](?:b|B)?)_", name)
    size = size_m.group(1).upper() if size_m else ""
    ts_m = re.search(r"_(\d{8})_(\d{6})$", name)
    if ts_m:
        d, t = ts_m.group(1), ts_m.group(2)
        ts = f"{d[:4]}-{d[4:6]}-{d[6:8]} {t[:2]}:{t[2:4]}"
    else:
        ts = ""
    parts = [p for p in [size, ts] if p]
    return " · ".join(parts) if parts else name


def _run_short_key(path: Path) -> str:
    """Compact filesystem-safe key for a run dir, used to name the output folder.

    training_output_100m_test_from_scratch_20260506_220944 → '100m_220944'
    training_output_1b_test_from_scratch_20260507_010521   → '1b_010521'
    """
    name = path.name
    size_m = re.search(r"_(\d+(?:\.\d+)?[mMbBkK](?:b|B)?)_", name)
    size = size_m.group(1).lower() if size_m else ""
    ts_m = re.search(r"_(\d{8})_(\d{6})$", name)
    ts_suffix = ts_m.group(2) if ts_m else ""          # last 6 digits = HHMMSS
    parts = [p for p in [size, ts_suffix] if p]
    return "_".join(parts) if parts else name


def _parse_args_from_log(log_path: Path) -> dict:
    """Extract the JSON args block written by the trainer at startup."""
    if not log_path.exists():
        return {}
    text = log_path.read_text(encoding="utf-8", errors="replace")
    # The trainer logs: 'args={\n  "key": value, ...\n}'
    m = re.search(r"args=(\{.*?\n\})", text, re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return {}


def _make_subtitle(train_args: dict) -> str:
    """Compact one-line subtitle from the parsed training args."""
    fields = [
        ("lr", "lr"),
        ("min_lr", "min_lr"),
        ("warmup_ratio", "warmup"),
        ("batch_size", "bs"),
        ("gradient_accumulation_steps", "gas"),
        ("epoch", "ep"),
        ("seq_len", "seq"),
    ]
    parts = []
    for key, short in fields:
        v = train_args.get(key)
        if v is not None:
            parts.append(f"{short}={v}")
    return "  ".join(parts)


# ── data loading ──────────────────────────────────────────────────────────────

def load_run(run_dir: Path, prefer_rank: int = 0, max_steps: Optional[int] = None) -> Optional[dict]:
    """Load all metric rows from one training-output directory.

    Returns a dict with:
        'all'   — DataFrame with every row (one per batch iteration)
        'loss'  — subset where loss_total is not null (logged every log_interval)
        'args'  — dict of training hyper-parameters (from log.0-0.txt)
        'label' — human-readable run label

    If ``max_steps`` is given, rows with ``update_step > max_steps`` are dropped.
    """
    # Find epoch-0 (or any epoch) metrics file for the requested rank
    pattern = re.compile(rf"metrics\.(\d+)-{prefer_rank}\.jsonl")
    epoch_files = sorted(
        (int(m.group(1)), p)
        for p in run_dir.iterdir()
        if (m := pattern.match(p.name))
    )
    if not epoch_files:
        # fall back to any rank
        any_pat = re.compile(r"metrics\.(\d+)-(\d+)\.jsonl")
        epoch_files = sorted(
            (int(m.group(1)), p)
            for p in run_dir.iterdir()
            if (m := any_pat.match(p.name))
        )
    if not epoch_files:
        print(f"  [warn] no metrics files in {run_dir.name}", file=sys.stderr)
        return None

    records = []
    for _ep, fpath in epoch_files:
        with open(fpath) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    if not records:
        return None

    df_all = pd.DataFrame(records).sort_values("batch_index").reset_index(drop=True)

    if max_steps is not None and "update_step" in df_all.columns:
        df_all = df_all[df_all["update_step"] <= max_steps].reset_index(drop=True)

    # loss rows: logged every log_interval steps (loss_total is non-null)
    if "loss_total" in df_all.columns:
        df_loss = df_all[df_all["loss_total"].notna()].copy()
    else:
        df_loss = pd.DataFrame()

    # parse training args
    log_candidates = sorted(run_dir.glob(f"log.0-{prefer_rank}.txt"))
    log_path = log_candidates[0] if log_candidates else run_dir / f"log.0-{prefer_rank}.txt"
    train_args = _parse_args_from_log(log_path)

    return {
        "all": df_all,
        "loss": df_loss,
        "args": train_args,
        "label": _dir_label(run_dir),
    }


# ── smoothing ─────────────────────────────────────────────────────────────────

def smooth(y: np.ndarray, window: int = 1) -> np.ndarray:
    if len(y) < window or window <= 1:
        return y.astype(float)
    kernel = np.ones(window) / window
    pad = window // 2
    padded = np.pad(y.astype(float), pad, mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(y)]


# ── main figure ───────────────────────────────────────────────────────────────

def make_figure(runs: dict[str, dict], smooth_window: int = 1) -> plt.Figure:
    """3×3 overview figure.

    Top row    — loss components (x = update_step, only on log_interval rows)
    Middle row — learning rate + zero-prob losses + MFU (x = update_step)
    Bottom row — throughput & timing (x = batch_index, every row)
    """
    colours = {label: PALETTE[i % len(PALETTE)] for i, label in enumerate(runs)}

    fig, axes = plt.subplots(3, 3, figsize=(15, 11))

    # Build a combined subtitle from first run's args
    first_args = next(iter(runs.values()))["args"]
    subtitle = _make_subtitle(first_args) if first_args else ""
    fig.suptitle(
        f"SpeciesLLM Training Curves\n{subtitle}" if subtitle else "SpeciesLLM Training Curves",
        fontsize=12,
        fontweight="bold",
        y=0.99,
    )

    legend_handles = []

    def _plot(ax, runs_data, df_key, x_col, y_col, y_label, title, log_y=False, sw=1):
        ax.set_title(title)
        ax.set_xlabel("Update Step" if x_col == "update_step" else "Batch Index")
        ax.set_ylabel(y_label)
        if log_y:
            ax.set_yscale("log")
        any_plotted = False
        for label, run in runs_data.items():
            df = run[df_key]
            if df.empty or y_col not in df.columns or x_col not in df.columns:
                continue
            sub = df[[x_col, y_col]].dropna()
            if sub.empty:
                continue
            x = sub[x_col].values
            y = smooth(sub[y_col].values, sw)
            line, = ax.plot(x, y, color=colours[label], label=label, alpha=0.9)
            any_plotted = True
            if not legend_handles or label not in [h.get_label() for h in legend_handles]:
                legend_handles.append(line)
        if not any_plotted:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, color="gray", fontsize=9)

    sw = smooth_window

    # Row 0: loss components (x = update_step, from loss-only rows)
    _plot(axes[0, 0], runs, "loss", "update_step", "loss_total",    "Loss",  "Total Loss",        log_y=True,  sw=sw)
    _plot(axes[0, 1], runs, "loss", "update_step", "loss_gep",      "Loss",  "GEP Loss",          log_y=True,  sw=sw)
    _plot(axes[0, 2], runs, "loss", "update_step", "loss_gepc",     "Loss",  "GEPC Loss",         log_y=True,  sw=sw)

    # cluster loss band on total loss panel
    ax_loss = axes[0, 0]
    for label, run in runs.items():
        df = run["loss"]
        if {"cluster_loss_total_min", "cluster_loss_total_max", "update_step"}.issubset(df.columns):
            sub = df[["update_step", "cluster_loss_total_min", "cluster_loss_total_max"]].dropna()
            if not sub.empty:
                ax_loss.fill_between(
                    sub["update_step"],
                    sub["cluster_loss_total_min"],
                    sub["cluster_loss_total_max"],
                    alpha=0.12,
                    color=colours[label],
                )

    # Row 1: lr, zero-prob losses, MFU
    _plot(axes[1, 0], runs, "loss", "update_step", "lr",                   "Learning Rate",     "LR Schedule",           sw=1)
    _plot(axes[1, 1], runs, "loss", "update_step", "loss_zero_prob",        "Loss",              "Zero-Prob Loss (GEP)",  sw=sw)
    _plot(axes[1, 2], runs, "loss", "update_step", "loss_gepc_zero_prob",   "Loss",              "Zero-Prob Loss (GEPC)", sw=sw)

    # Row 2: throughput and timing — every batch row is valid
    _plot(axes[2, 0], runs, "all",  "batch_index", "step_ms",              "Step Time (ms)",    "Step Duration",          sw=sw)
    _plot(axes[2, 1], runs, "all",  "batch_index", "tokens_per_s",         "Tokens / s (rank)", "Per-Rank Throughput",    sw=sw)
    _plot(axes[2, 2], runs, "loss", "update_step", "cluster_tokens_per_s_sum", "Tokens / s (cluster)", "Cluster Throughput", sw=sw)

    if legend_handles:
        fig.legend(
            legend_handles,
            [h.get_label() for h in legend_handles],
            loc="lower center",
            ncol=min(len(legend_handles), 4),
            bbox_to_anchor=(0.5, 0.005),
            frameon=True,
            framealpha=0.9,
        )

    fig.tight_layout(rect=[0, 0.06, 1, 0.97])
    return fig


def make_loss_detail_figure(runs: dict[str, dict], smooth_window: int = 1) -> plt.Figure:
    """All five loss terms side-by-side (x = update_step)."""
    loss_cols = [
        ("loss_total",          "Total Loss"),
        ("loss_gep",            "GEP Loss"),
        ("loss_gepc",           "GEPC Loss"),
        ("loss_zero_prob",      "Zero-Prob (GEP)"),
        ("loss_gepc_zero_prob", "Zero-Prob (GEPC)"),
    ]
    n = len(loss_cols)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=False)
    fig.suptitle("Loss Components Breakdown", fontsize=12, fontweight="bold")
    colours = {label: PALETTE[i % len(PALETTE)] for i, label in enumerate(runs)}

    for ax, (col, title) in zip(axes, loss_cols):
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Update Step")
        ax.set_ylabel("Loss")
        ax.set_yscale("log")
        for label, run in runs.items():
            df = run["loss"]
            if col not in df.columns or "update_step" not in df.columns:
                continue
            sub = df[["update_step", col]].dropna()
            if sub.empty:
                continue
            y = smooth(sub[col].values, smooth_window)
            ax.plot(sub["update_step"].values, y, label=label, color=colours[label])
        ax.legend(fontsize=8)

    fig.tight_layout()
    return fig


def make_timing_figure(runs: dict[str, dict], smooth_window: int = 1) -> plt.Figure:
    """Detailed timing breakdown per step (x = batch_index, every row)."""
    timing_cols = [
        ("data_load_s",  "Data Load (s)"),
        ("to_device_s",  "To Device (s)"),
        ("forward_s",    "Forward (s)"),
        ("loss_s",       "Loss Compute (s)"),
        ("backward_s",   "Backward (s)"),
        ("optimizer_s",  "Optimizer Step (s)"),
        ("grad_sync_s",  "Grad Sync (s)"),
    ]
    n = len(timing_cols)
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    axes_flat = axes.flatten()
    fig.suptitle("Step Timing Breakdown (per batch)", fontsize=12, fontweight="bold")
    colours = {label: PALETTE[i % len(PALETTE)] for i, label in enumerate(runs)}

    for ax, (col, title) in zip(axes_flat, timing_cols):
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Batch Index")
        ax.set_ylabel("Seconds")
        for label, run in runs.items():
            df = run["all"]
            if col not in df.columns or "batch_index" not in df.columns:
                continue
            sub = df[["batch_index", col]].dropna()
            if sub.empty:
                continue
            y = smooth(sub[col].values, smooth_window)
            ax.plot(sub["batch_index"].values, y, label=label, color=colours[label], alpha=0.85)
        ax.legend(fontsize=8)

    # hide unused panel
    for ax in axes_flat[len(timing_cols):]:
        ax.set_visible(False)

    fig.tight_layout()
    return fig


def _derive_consecutive(skipped_series: pd.Series) -> pd.Series:
    """Run-length-encode a boolean series into "consecutive True so far".

    Used to back-fill ``consecutive_skips`` for runs from BEFORE the safety-net
    patch landed, where only the per-row ``skipped`` flag is recorded.
    """
    skipped = skipped_series.fillna(False).astype(bool).values
    out = np.zeros(len(skipped), dtype=np.int64)
    run = 0
    for i, s in enumerate(skipped):
        run = run + 1 if s else 0
        out[i] = run
    return pd.Series(out, index=skipped_series.index)


def _has_grad_clip_data(df: pd.DataFrame) -> bool:
    """A run is plottable if at least one optimizer step recorded a raw norm."""
    if df.empty or "grad_norm_raw" not in df.columns:
        return False
    return df["grad_norm_raw"].notna().any()


def make_grad_clip_figure(runs: dict[str, dict], window: int = 100, smooth_window: int = 1) -> Optional[plt.Figure]:
    """2×2 adaptive-grad-clip diagnostic figure.

    Reads per-optimizer-step rows (where ``grad_norm_raw`` is non-null) from
    ``runs[*]["all"]`` and produces:

      • Top-left   raw grad norm vs EMA vs clip / skip thresholds/fuses (log y).
                   The prod failure pattern is raw_norm climbing into the clip
                   band and dragging EMA up with it; both lines ratcheting
                   together is the leading indicator of an upcoming skip
                   cascade.
      • Top-right  consecutive_skips / consecutive_clips streaks (linear y).
                   Bigger streaks ⇒ the safety net is closer to firing.
      • Bottom-left  rolling skip/clip RATE over the last ``window`` steps
                   (% of optimizer steps).
      • Bottom-right cumulative skip / clip counts (linear y).

    Backward-compatible: if ``consecutive_skips`` / ``consecutive_clips`` are
    missing (older runs from before the safety-net patch), they are derived
    on-the-fly from the per-row boolean ``skipped`` flag.

    Returns ``None`` if no run has any usable gradient-step rows.
    """
    plottable = {label: run for label, run in runs.items() if _has_grad_clip_data(run["all"])}
    if not plottable:
        return None

    colours = {label: PALETTE[i % len(PALETTE)] for i, label in enumerate(plottable)}

    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5))
    first_args = next(iter(plottable.values()))["args"]
    subtitle = _make_subtitle(first_args) if first_args else ""
    fig.suptitle(
        f"Adaptive Grad-Clip Diagnostics\n{subtitle}" if subtitle else "Adaptive Grad-Clip Diagnostics",
        fontsize=12,
        fontweight="bold",
        y=0.99,
    )

    ax_norm, ax_streak = axes[0]
    ax_rate, ax_cum    = axes[1]

    # ── (0,0) raw_norm + EMA + clip/skip thresholds ─────────────────────────
    ax_norm.set_title(f"Grad Norm vs EMA / Thresholds")
    ax_norm.set_xlabel("Update Step")
    ax_norm.set_ylabel("L2 Norm (log)")
    ax_norm.set_yscale("log")

    for label, run in plottable.items():
        df = run["all"]
        sub = df.dropna(subset=["grad_norm_raw"])
        if sub.empty:
            continue
        x = sub["update_step"].values if "update_step" in sub.columns else np.arange(len(sub))
        col = colours[label]
        # Raw norm: thin alpha line — every step is plotted; clip with .clip()
        # to avoid log-scale crashes on the rare 0-norm.
        raw = np.clip(smooth(sub["grad_norm_raw"].values, smooth_window), 1e-12, None)
        ax_norm.plot(x, raw, color=col, alpha=0.45, linewidth=1.0,
                     label=f"{label} raw")
        if "grad_norm_ema" in sub.columns:
            ema = sub["grad_norm_ema"].dropna()
            if not ema.empty:
                xe = sub.loc[ema.index, "update_step"].values if "update_step" in sub.columns else ema.index.values
                ax_norm.plot(xe, np.clip(ema.values, 1e-12, None),
                             color=col, linewidth=2.0, label=f"{label} ema")
        if "grad_clip_threshold" in sub.columns:
            clip_t = sub["grad_clip_threshold"].dropna()
            if not clip_t.empty:
                xc = sub.loc[clip_t.index, "update_step"].values if "update_step" in sub.columns else clip_t.index.values
                ax_norm.plot(xc, np.clip(clip_t.values, 1e-12, None),
                             color=col, linewidth=1.0, linestyle="--",
                             alpha=0.85, label=f"{label} clip_thr")
        if "grad_skip_threshold" in sub.columns:
            skip_t = sub["grad_skip_threshold"].dropna()
            if not skip_t.empty:
                xs = sub.loc[skip_t.index, "update_step"].values if "update_step" in sub.columns else skip_t.index.values
                ax_norm.plot(xs, np.clip(skip_t.values, 1e-12, None),
                             color=col, linewidth=1.0, linestyle=":",
                             alpha=0.85, label=f"{label} skip_thr")
        for field, linestyle, suffix in (
            ("grad_skip_max", "-.", "skip_max"),
            ("grad_hard_raw_norm_limit", (0, (3, 1, 1, 1)), "hard_raw"),
        ):
            if field in sub.columns:
                fuse = sub[field].dropna()
                if not fuse.empty:
                    val = float(fuse.iloc[-1])
                    if val > 0:
                        ax_norm.hlines(
                            val,
                            x[0],
                            x[-1],
                            color=col,
                            linewidth=0.9,
                            linestyle=linestyle,
                            alpha=0.55,
                            label=f"{label} {suffix}",
                        )
        # Mark skipped steps as red ticks along the top of the panel.
        if "skipped" in sub.columns:
            skipped_x = sub.loc[sub["skipped"].fillna(False).astype(bool), "update_step"]
            if not skipped_x.empty:
                ax_norm.scatter(skipped_x.values,
                                np.full(len(skipped_x), raw.max() if len(raw) else 1.0),
                                marker="|", s=22, color=col, alpha=0.6, zorder=5)
    ax_norm.legend(fontsize=7, ncol=2, loc="best")

    # ── (0,1) consecutive_skips / consecutive_clips streaks ─────────────────
    ax_streak.set_title("Consecutive Skip / Clip Streak")
    ax_streak.set_xlabel("Update Step")
    ax_streak.set_ylabel("Streak Length")
    for label, run in plottable.items():
        df = run["all"]
        sub = df.dropna(subset=["grad_norm_raw"]).copy()
        if sub.empty:
            continue
        x = sub["update_step"].values if "update_step" in sub.columns else np.arange(len(sub))
        col = colours[label]
        # consecutive_skips: prefer logged column, fall back to derived from `skipped`.
        if "consecutive_skips" in sub.columns and sub["consecutive_skips"].notna().any():
            cs = sub["consecutive_skips"].fillna(0).values
        elif "skipped" in sub.columns:
            cs = _derive_consecutive(sub["skipped"]).values
        else:
            cs = None
        if cs is not None and cs.max() > 0:
            ax_streak.plot(x, cs, color=col, linewidth=1.4,
                           label=f"{label} skip streak")
        if "consecutive_clips" in sub.columns and sub["consecutive_clips"].notna().any():
            cc = sub["consecutive_clips"].fillna(0).values
            if cc.max() > 0:
                ax_streak.plot(x, cc, color=col, linewidth=1.0, linestyle="--",
                               alpha=0.8, label=f"{label} clip streak")
    if ax_streak.has_data():
        ax_streak.legend(fontsize=7, ncol=2, loc="best")
    else:
        ax_streak.text(0.5, 0.5, "no skip / clip events",
                       ha="center", va="center", transform=ax_streak.transAxes,
                       color="gray", fontsize=9)

    # ── (1,0) rolling skip/clip RATE over `window` optimizer steps ──────────
    ax_rate.set_title(f"Skip / Clip Rate (rolling window={window})")
    ax_rate.set_xlabel("Update Step")
    ax_rate.set_ylabel("% of steps in window")
    ax_rate.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    any_rate = False
    for label, run in plottable.items():
        df = run["all"]
        sub = df.dropna(subset=["grad_norm_raw"]).copy()
        if sub.empty:
            continue
        x = sub["update_step"].values if "update_step" in sub.columns else np.arange(len(sub))
        col = colours[label]
        if "skipped" in sub.columns:
            skip_flag = sub["skipped"].fillna(False).astype(bool)
            rolling = skip_flag.rolling(window=window, min_periods=1).mean().values
            ax_rate.plot(x, rolling, color=col, linewidth=1.6,
                         label=f"{label} skip rate")
            any_rate = True
        if "grad_action" in sub.columns:
            clip_flag = (sub["grad_action"] == "clip")
            if clip_flag.any():
                rolling_c = clip_flag.rolling(window=window, min_periods=1).mean().values
                ax_rate.plot(x, rolling_c, color=col, linewidth=1.0, linestyle="--",
                             alpha=0.75, label=f"{label} clip rate")
                any_rate = True
    if any_rate:
        ax_rate.legend(fontsize=7, ncol=2, loc="best")
        ax_rate.set_ylim(bottom=0)
    else:
        ax_rate.text(0.5, 0.5, "no skip / clip events",
                     ha="center", va="center", transform=ax_rate.transAxes,
                     color="gray", fontsize=9)

    # ── (1,1) cumulative skip / clip counts ─────────────────────────────────
    ax_cum.set_title("Cumulative Skipped / Clipped Counts")
    ax_cum.set_xlabel("Update Step")
    ax_cum.set_ylabel("Count")
    any_cum = False
    for label, run in plottable.items():
        df = run["all"]
        sub = df.dropna(subset=["grad_norm_raw"]).copy()
        if sub.empty:
            continue
        x = sub["update_step"].values if "update_step" in sub.columns else np.arange(len(sub))
        col = colours[label]
        # Prefer the logged cumulative; fall back to cumulative sum of `skipped`.
        if "skipped_count_cum" in sub.columns and sub["skipped_count_cum"].notna().any():
            sk_cum = sub["skipped_count_cum"].ffill().fillna(0).values
        elif "skipped" in sub.columns:
            sk_cum = sub["skipped"].fillna(False).astype(int).cumsum().values
        else:
            sk_cum = None
        if sk_cum is not None and (len(sk_cum) and sk_cum[-1] > 0):
            ax_cum.plot(x, sk_cum, color=col, linewidth=1.6, label=f"{label} skipped")
            any_cum = True
        if "clipped_count_cum" in sub.columns and sub["clipped_count_cum"].notna().any():
            cl_cum = sub["clipped_count_cum"].ffill().fillna(0).values
            if len(cl_cum) and cl_cum[-1] > 0:
                ax_cum.plot(x, cl_cum, color=col, linewidth=1.0, linestyle="--",
                            alpha=0.8, label=f"{label} clipped")
                any_cum = True
        elif "grad_action" in sub.columns:
            cl_cum = (sub["grad_action"] == "clip").astype(int).cumsum().values
            if len(cl_cum) and cl_cum[-1] > 0:
                ax_cum.plot(x, cl_cum, color=col, linewidth=1.0, linestyle="--",
                            alpha=0.8, label=f"{label} clipped (derived)")
                any_cum = True
    if any_cum:
        ax_cum.legend(fontsize=7, ncol=2, loc="best")
    else:
        ax_cum.text(0.5, 0.5, "no skip / clip events",
                    ha="center", va="center", transform=ax_cum.transAxes,
                    color="gray", fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


# ── discovery & CLI ───────────────────────────────────────────────────────────

def discover_runs(root: Path) -> list[Path]:
    return sorted(
        p for p in root.iterdir()
        if p.is_dir() and p.name.startswith("training_output_")
    )


def _individual_out_dir(run_dir: Path, base_out: Optional[Path]) -> Path:
    """Resolve where plots for one training run should be written."""
    if base_out is None:
        return run_dir
    return base_out / _run_short_key(run_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot SpeciesLLM training metrics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("dirs", nargs="*", type=Path,
                        help="training_output_* dirs (auto-discovered if omitted)")
    parser.add_argument("--root", type=Path, default=Path(__file__).parent,
                        help="project root to search (default: script dir)")
    parser.add_argument("--out", type=Path, default=None,
                        help="base output directory for figures; default writes "
                             "per-run figures into each training_output_* dir")
    parser.add_argument("--smooth", type=int, default=1, metavar="N",
                        help="box-car smoothing window for loss/lr (default 1 = off)")
    parser.add_argument("--rank", type=int, default=0,
                        help="GPU rank metrics file to load (default 0)")
    parser.add_argument("--no-detail", action="store_true",
                        help="skip loss-components detail figure")
    parser.add_argument("--no-timing", action="store_true",
                        help="skip per-step timing breakdown figure")
    parser.add_argument("--no-grad-clip", action="store_true",
                        help="skip the adaptive-grad-clip diagnostics figure")
    parser.add_argument("--grad-clip-window", type=int, default=100, metavar="N",
                        help="rolling window for the skip/clip RATE panel (default 100)")
    parser.add_argument("--max-steps", type=int, default=None, metavar="N",
                        help="only plot rows up to update_step N (default: all)")
    parser.add_argument("--compare", action="store_true",
                        help="when 2+ dirs are given, also overlay all runs in a single "
                             "comparison figure (default output: <root>/figures/compare_<keys>/)")
    parser.add_argument("--no-individual", action="store_true",
                        help="skip per-run figures (only useful together with --compare)")
    args = parser.parse_args()

    # resolve run directories
    if args.dirs:
        run_dirs = [Path(d).resolve() for d in args.dirs]
    else:
        run_dirs = discover_runs(args.root.resolve())
        if not run_dirs:
            sys.exit(f"No training_output_* directories found under {args.root}")

    print(f"Found {len(run_dirs)} run(s):")
    for d in run_dirs:
        print(f"  {d.name}")

    runs: dict[str, dict] = {}
    for run_dir in run_dirs:
        result = load_run(run_dir, prefer_rank=args.rank, max_steps=args.max_steps)
        if result is None:
            print(f"  [skip] no usable data in {run_dir.name}", file=sys.stderr)
            continue
        label = result["label"]
        n_all = len(result["all"])
        n_loss = len(result["loss"])
        step_range = (
            f"batch {result['all']['batch_index'].iloc[0]}–{result['all']['batch_index'].iloc[-1]}"
            if not result["all"].empty else "?"
        )
        subtitle = _make_subtitle(result["args"])
        print(f"  Loaded '{label}': {n_all} rows total, {n_loss} with loss, {step_range}")
        if subtitle:
            print(f"    args: {subtitle}")
        runs[label] = result

    if not runs:
        sys.exit("No data could be loaded.")

    def _save_figures(runs_dict: dict[str, dict], out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        fig = make_figure(runs_dict, smooth_window=args.smooth)
        for ext in (".pdf", ".png"):
            p = out_dir / f"training_curves{ext}"
            fig.savefig(p, bbox_inches="tight")
            print(f"  Saved: {p}")
        plt.close(fig)

        if not args.no_detail:
            fig2 = make_loss_detail_figure(runs_dict, smooth_window=args.smooth)
            for ext in (".pdf", ".png"):
                p = out_dir / f"loss_detail{ext}"
                fig2.savefig(p, bbox_inches="tight")
                print(f"  Saved: {p}")
            plt.close(fig2)

        if not args.no_timing:
            fig3 = make_timing_figure(runs_dict, smooth_window=args.smooth)
            for ext in (".pdf", ".png"):
                p = out_dir / f"step_timing{ext}"
                fig3.savefig(p, bbox_inches="tight")
                print(f"  Saved: {p}")
            plt.close(fig3)

        if not args.no_grad_clip:
            fig4 = make_grad_clip_figure(
                runs_dict,
                window=args.grad_clip_window,
                smooth_window=args.smooth,
            )
            if fig4 is None:
                print("  [skip] grad_clip figure: no grad_norm_raw rows in any run")
            else:
                for ext in (".pdf", ".png"):
                    p = out_dir / f"grad_clip{ext}"
                    fig4.savefig(p, bbox_inches="tight")
                    print(f"  Saved: {p}")
                plt.close(fig4)

    base_out = args.out.resolve() if args.out else None

    # Per-run figures
    if not args.no_individual:
        for run_dir in run_dirs:
            label = _dir_label(run_dir)
            if label not in runs:
                continue
            out_dir = _individual_out_dir(run_dir, base_out)
            print(f"\n[{label}] → {out_dir}")
            _save_figures({label: runs[label]}, out_dir)

    # Overlay comparison figure across all loaded runs
    if args.compare and len(runs) >= 2:
        compare_key = "_vs_".join(
            _run_short_key(d) for d in run_dirs if _dir_label(d) in runs
        )
        compare_base_out = base_out if base_out is not None else (args.root.resolve() / "figures")
        out_dir = compare_base_out / f"compare_{compare_key}"
        print(f"\n[compare {len(runs)} runs] → {out_dir}")
        _save_figures(runs, out_dir)
    elif args.compare and len(runs) < 2:
        print("\n[compare] skipped: need at least 2 runs to overlay", file=sys.stderr)


if __name__ == "__main__":
    main()
