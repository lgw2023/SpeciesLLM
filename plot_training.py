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
parses training args from the matching log file, and produces figures saved to
./figures/.

Usage:
    python plot_training.py [training_output_dir ...]

    --smooth N        box-car window for loss curves (default 1 = off)
    --rank R          which GPU rank file to read (default 0)
    --no-detail       skip the loss-components breakdown figure
    --out DIR         output directory (default <root>/figures/)
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


# ── discovery & CLI ───────────────────────────────────────────────────────────

def discover_runs(root: Path) -> list[Path]:
    return sorted(
        p for p in root.iterdir()
        if p.is_dir() and p.name.startswith("training_output_")
    )


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
                        help="output directory for figures (default: <root>/figures/<run_keys>/)")
    parser.add_argument("--smooth", type=int, default=1, metavar="N",
                        help="box-car smoothing window for loss/lr (default 1 = off)")
    parser.add_argument("--rank", type=int, default=0,
                        help="GPU rank metrics file to load (default 0)")
    parser.add_argument("--no-detail", action="store_true",
                        help="skip loss-components detail figure")
    parser.add_argument("--no-timing", action="store_true",
                        help="skip per-step timing breakdown figure")
    parser.add_argument("--max-steps", type=int, default=None, metavar="N",
                        help="only plot rows up to update_step N (default: all)")
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

    # Each run gets its own output folder, regardless of how many runs were loaded.
    for run_dir in run_dirs:
        label = _dir_label(run_dir)
        if label not in runs:
            continue

        single_run = {label: runs[label]}

        if args.out:
            out_dir = args.out / _run_short_key(run_dir)
        else:
            out_dir = args.root.resolve() / "figures" / _run_short_key(run_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[{label}] → {out_dir}")

        fig = make_figure(single_run, smooth_window=args.smooth)
        for ext in (".pdf", ".png"):
            p = out_dir / f"training_curves{ext}"
            fig.savefig(p, bbox_inches="tight")
            print(f"  Saved: {p}")
        plt.close(fig)

        if not args.no_detail:
            fig2 = make_loss_detail_figure(single_run, smooth_window=args.smooth)
            for ext in (".pdf", ".png"):
                p = out_dir / f"loss_detail{ext}"
                fig2.savefig(p, bbox_inches="tight")
                print(f"  Saved: {p}")
            plt.close(fig2)

        if not args.no_timing:
            fig3 = make_timing_figure(single_run, smooth_window=args.smooth)
            for ext in (".pdf", ".png"):
                p = out_dir / f"step_timing{ext}"
                fig3.savefig(p, bbox_inches="tight")
                print(f"  Saved: {p}")
            plt.close(fig3)


if __name__ == "__main__":
    main()
