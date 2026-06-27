from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import matplotlib
import pandas as pd


matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "plot_training.py"


def load_plot_module():
    spec = importlib.util.spec_from_file_location("plot_training", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_masked_gep_figure_uses_new_diagnostic_fields():
    plot_training = load_plot_module()
    df = pd.DataFrame(
        [
            {
                "update_step": 10,
                "target_mean_masked": 1.0,
                "target_std_masked": 0.5,
                "target_p95_masked": 2.0,
                "target_p99_masked": 3.0,
                "target_nonzero_ratio_masked": 0.75,
                "pred_mean_masked": 1.1,
                "pred_std_masked": 0.6,
                "abs_error_p95": 0.4,
                "abs_error_p99": 0.7,
                "clip_fraction_rolling": 0.2,
            },
            {
                "update_step": 20,
                "target_mean_masked": 1.2,
                "target_std_masked": 0.7,
                "target_p95_masked": 2.5,
                "target_p99_masked": 3.5,
                "target_nonzero_ratio_masked": 0.8,
                "pred_mean_masked": 1.3,
                "pred_std_masked": 0.8,
                "abs_error_p95": 0.5,
                "abs_error_p99": 0.9,
                "clip_fraction_rolling": 0.1,
            },
        ]
    )
    runs = {
        "demo": {
            "all": df,
            "loss": df,
            "args": {"lr": 1e-6, "batch_size": 512},
            "label": "demo",
        }
    }

    fig = plot_training.make_masked_gep_figure(runs)
    assert fig is not None
    assert len(fig.axes) == 4
    titles = [ax.get_title() for ax in fig.axes]
    assert "Masked Target / Prediction Mean" in titles
    assert "Data Sparsity / Clip Fraction" in titles
    fig.canvas.draw()
    plot_training.plt.close(fig)


def test_masked_gep_figure_returns_none_without_diagnostics():
    plot_training = load_plot_module()
    df = pd.DataFrame([{"update_step": 1, "loss_total": 1.0}])
    runs = {"demo": {"all": df, "loss": df, "args": {}, "label": "demo"}}

    assert plot_training.make_masked_gep_figure(runs) is None
