#!/usr/bin/env python3
"""Summarize a checkpoint interval from metrics JSONL and model checkpoints.

The default target is the late epoch2 S8320 -> S9152 window. Metrics windows use
open-closed bounds: S8320~S8736 means S8320 < update_step <= S8736 after adding
the resume offset.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shlex
from pathlib import Path
from typing import Any


METRIC_FIELDNAMES = [
    "window",
    "start_step",
    "end_step",
    "start_update_step",
    "end_update_step",
    "num_rows",
    "num_loss_rows",
    "num_grad_rows",
    "loss_gep",
    "loss_zero_prob",
    "loss_gepc",
    "loss_gepc_zero_prob",
    "raw_grad_norm",
    "clip_rate",
    "clip_coef_mean_when_clipped",
    "lr",
]

PARAM_FIELDNAMES = [
    "ckpt",
    "step",
    "checkpoint_path",
    "gep_head_weight_norm",
    "gep_head_bias_mean",
    "gep_head_bias_std",
    "zero_head_weight_norm",
    "zero_head_bias_mean",
    "zero_head_bias_std",
    "shared_projection_norm",
    "last_shared_encoder_norm",
    "last_shared_encoder_layer",
    "gep_head_weight_tensors",
    "gep_head_bias_tensors",
    "zero_head_weight_tensors",
    "zero_head_bias_tensors",
    "shared_projection_tensors",
    "last_shared_encoder_tensors",
]


def parse_steps(value: str) -> list[int]:
    steps = [int(item.strip()) for item in value.split(",") if item.strip()]
    if len(steps) < 2:
        raise argparse.ArgumentTypeError("--steps must contain at least two comma-separated steps")
    if steps != sorted(steps):
        raise argparse.ArgumentTypeError("--steps must be sorted ascending")
    return steps


def parse_checkpoint_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"--checkpoint must use NAME=/path/file.pt: {value}")
    name, path = value.split("=", 1)
    name = name.strip()
    path = path.strip()
    if not name or not path:
        raise argparse.ArgumentTypeError(f"--checkpoint must use non-empty NAME=/path/file.pt: {value}")
    return name, Path(path).expanduser()


def clean_csv_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: ("" if value is None else value) for key, value in row.items()}


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def parse_run_record_resume_step(path: Path) -> int | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    command = data.get("command")
    if not isinstance(command, str):
        return None
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()
    for token in tokens:
        if token.startswith("RESUME_UPDATE_STEP="):
            _, value = token.split("=", 1)
            try:
                return int(value)
            except ValueError:
                return None
    return None


def resolve_resume_update_step(args: argparse.Namespace) -> int:
    if args.steps_are_global:
        return 0
    if args.resume_update_step != "auto":
        return int(args.resume_update_step)
    run_record_step = parse_run_record_resume_step(args.run_record)
    if run_record_step is not None:
        return run_record_step
    return 0


def build_windows(steps: list[int], resume_update_step: int) -> list[dict[str, int | str]]:
    windows = []
    for start, end in zip(steps, steps[1:]):
        windows.append(
            {
                "window": f"S{start}~S{end}",
                "start_step": start,
                "end_step": end,
                "start_update_step": resume_update_step + start,
                "end_update_step": resume_update_step + end,
            }
        )
    return windows


def aggregate_metrics(metrics_path: Path, windows: list[dict[str, int | str]]) -> list[dict[str, Any]]:
    accum: list[dict[str, Any]] = []
    for window in windows:
        accum.append(
            {
                **window,
                "num_rows": 0,
                "num_loss_rows": 0,
                "num_grad_rows": 0,
                "_loss_gep": [],
                "_loss_zero_prob": [],
                "_loss_gepc": [],
                "_loss_gepc_zero_prob": [],
                "_raw_grad_norm": [],
                "_clip_coef_when_clipped": [],
                "_lr": [],
                "_clip_count": 0,
            }
        )

    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            update_step = row.get("update_step")
            if update_step is None:
                continue
            update_step = int(update_step)
            for item in accum:
                if not (int(item["start_update_step"]) < update_step <= int(item["end_update_step"])):
                    continue

                item["num_rows"] += 1
                for field in ("loss_gep", "loss_zero_prob", "loss_gepc", "loss_gepc_zero_prob"):
                    value = row.get(field)
                    if value is not None:
                        item[f"_{field}"].append(float(value))
                if row.get("loss_gep") is not None:
                    item["num_loss_rows"] += 1

                raw_norm = row.get("grad_norm_raw")
                if raw_norm is not None:
                    raw_norm = float(raw_norm)
                    item["_raw_grad_norm"].append(raw_norm)
                    item["num_grad_rows"] += 1
                    if row.get("grad_action") == "clip":
                        item["_clip_count"] += 1
                        threshold = row.get("grad_clip_threshold")
                        if threshold is not None and raw_norm > 0:
                            item["_clip_coef_when_clipped"].append(float(threshold) / raw_norm)

                lr = row.get("lr")
                if lr is not None:
                    item["_lr"].append(float(lr))

    rows = []
    for item in accum:
        grad_rows = int(item["num_grad_rows"])
        row = {field: item[field] for field in METRIC_FIELDNAMES if field in item}
        row.update(
            {
                "loss_gep": mean(item["_loss_gep"]),
                "loss_zero_prob": mean(item["_loss_zero_prob"]),
                "loss_gepc": mean(item["_loss_gepc"]),
                "loss_gepc_zero_prob": mean(item["_loss_gepc_zero_prob"]),
                "raw_grad_norm": mean(item["_raw_grad_norm"]),
                "clip_rate": (item["_clip_count"] / grad_rows) if grad_rows else None,
                "clip_coef_mean_when_clipped": mean(item["_clip_coef_when_clipped"]),
                "lr": mean(item["_lr"]),
            }
        )
        rows.append(row)
    return rows


def unwrap_checkpoint_state(checkpoint: Any) -> dict[str, Any]:
    if not isinstance(checkpoint, dict):
        return checkpoint
    for key in ("state_dict", "model_state_dict", "model"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    return checkpoint


def strip_module_prefix(state_dict: dict[str, Any]) -> dict[str, Any]:
    if not state_dict:
        return state_dict
    if all(isinstance(key, str) and key.startswith("module.") for key in state_dict):
        return {key[len("module."):]: value for key, value in state_dict.items()}
    return state_dict


def tensor_l2_norm(items: list[tuple[str, Any]]) -> float | None:
    if not items:
        return None
    total = 0.0
    for _name, tensor in items:
        t = tensor.detach().float().cpu()
        total += float(torch_sum_square(t))
    return math.sqrt(total)


def torch_sum_square(tensor: Any) -> float:
    return tensor.pow(2).sum().item()


def tensor_mean_std(items: list[tuple[str, Any]]) -> tuple[float | None, float | None]:
    if not items:
        return None, None
    import torch

    values = torch.cat([tensor.detach().float().cpu().flatten() for _name, tensor in items])
    return float(values.mean()), float(values.std(unbiased=False))


def select_tensors(state_dict: dict[str, Any], predicate) -> list[tuple[str, Any]]:
    selected = []
    for name, tensor in state_dict.items():
        if not hasattr(tensor, "detach"):
            continue
        if not getattr(tensor, "is_floating_point", lambda: False)():
            continue
        if predicate(name):
            selected.append((name, tensor))
    return selected


def detect_last_encoder_layer(state_dict: dict[str, Any]) -> tuple[str | None, int | None]:
    patterns = [
        re.compile(r"^bert\.h\.(\d+)\."),
        re.compile(r"^bert\.encoder\.layer\.(\d+)\."),
    ]
    for pattern in patterns:
        layer_ids = []
        for name in state_dict:
            match = pattern.match(name)
            if match:
                layer_ids.append(int(match.group(1)))
        if layer_ids:
            last = max(layer_ids)
            prefix = pattern.pattern.split(r"(\d+)")[0].replace("^", "").replace("\\.", ".")
            return f"{prefix}{last}.", last
    return None, None


def summarize_checkpoint(checkpoint_name: str, checkpoint_path: Path) -> dict[str, Any]:
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = strip_module_prefix(unwrap_checkpoint_state(checkpoint))

    gep_weight = select_tensors(state_dict, lambda name: name.startswith("decoder.fc.") and name.endswith(".weight"))
    gep_bias = select_tensors(state_dict, lambda name: name.startswith("decoder.fc.") and name.endswith(".bias"))
    zero_weight = select_tensors(
        state_dict,
        lambda name: name.startswith("decoder.zero_logit.") and name.endswith(".weight"),
    )
    zero_bias = select_tensors(
        state_dict,
        lambda name: name.startswith("decoder.zero_logit.") and name.endswith(".bias"),
    )
    shared_projection = select_tensors(
        state_dict,
        lambda name: name.startswith("bert.value_encoder.") or name.startswith("bert.enhanced_fusion."),
    )
    last_prefix, last_layer = detect_last_encoder_layer(state_dict)
    last_encoder = select_tensors(state_dict, lambda name: last_prefix is not None and name.startswith(last_prefix))

    gep_bias_mean, gep_bias_std = tensor_mean_std(gep_bias)
    zero_bias_mean, zero_bias_std = tensor_mean_std(zero_bias)
    step_match = re.search(r"step-(\d+)-", checkpoint_path.name)

    return {
        "ckpt": checkpoint_name,
        "step": int(step_match.group(1)) if step_match else None,
        "checkpoint_path": str(checkpoint_path),
        "gep_head_weight_norm": tensor_l2_norm(gep_weight),
        "gep_head_bias_mean": gep_bias_mean,
        "gep_head_bias_std": gep_bias_std,
        "zero_head_weight_norm": tensor_l2_norm(zero_weight),
        "zero_head_bias_mean": zero_bias_mean,
        "zero_head_bias_std": zero_bias_std,
        "shared_projection_norm": tensor_l2_norm(shared_projection),
        "last_shared_encoder_norm": tensor_l2_norm(last_encoder),
        "last_shared_encoder_layer": last_layer,
        "gep_head_weight_tensors": len(gep_weight),
        "gep_head_bias_tensors": len(gep_bias),
        "zero_head_weight_tensors": len(zero_weight),
        "zero_head_bias_tensors": len(zero_bias),
        "shared_projection_tensors": len(shared_projection),
        "last_shared_encoder_tensors": len(last_encoder),
    }


def find_checkpoints(run_dir: Path, steps: list[int]) -> list[tuple[str, Path]]:
    checkpoints = []
    for step in steps:
        pattern = f"SC-node-00-rank-00-epoch-01-step-{step}-loss-*.pt"
        matches = [
            path for path in sorted(run_dir.glob(pattern))
            if not path.name.endswith(".optimizer.pt")
        ]
        if not matches:
            raise FileNotFoundError(f"No checkpoint found for S{step} under {run_dir} with pattern {pattern}")
        if len(matches) > 1:
            raise RuntimeError(f"Multiple checkpoints found for S{step}: {matches}")
        checkpoints.append((f"S{step}", matches[0]))
    return checkpoints


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(clean_csv_row({field: row.get(field) for field in fieldnames}))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--metrics", type=Path, default=None)
    parser.add_argument("--run-record", type=Path, default=None)
    parser.add_argument("--steps", type=parse_steps, default=parse_steps("8320,8736,9152"))
    parser.add_argument("--resume-update-step", default="auto")
    parser.add_argument("--steps-are-global", action="store_true")
    parser.add_argument("--checkpoint", action="append", type=parse_checkpoint_spec, default=[])
    parser.add_argument("--skip-param-norms", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--output-prefix", default="s8320_s9152")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser()
    metrics_path = (args.metrics or (run_dir / "metrics.0-0.jsonl")).expanduser()
    args.run_record = (args.run_record or (run_dir / "run_record.json")).expanduser()
    output_dir = (args.output_dir or (run_dir / "diagnostics" / args.output_prefix)).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not metrics_path.is_file():
        raise FileNotFoundError(f"Missing metrics JSONL: {metrics_path}")

    resume_update_step = resolve_resume_update_step(args)
    windows = build_windows(args.steps, resume_update_step)
    metric_rows = aggregate_metrics(metrics_path, windows)

    metrics_csv = output_dir / f"{args.output_prefix}_metrics_windows.csv"
    metrics_jsonl = output_dir / f"{args.output_prefix}_metrics_windows.jsonl"
    write_rows(metrics_csv, METRIC_FIELDNAMES, metric_rows)
    write_jsonl(metrics_jsonl, metric_rows)

    print(f"resume_update_step={resume_update_step}")
    print(f"metrics_csv={metrics_csv}")
    print(f"metrics_jsonl={metrics_jsonl}")

    param_rows = []
    if not args.skip_param_norms:
        checkpoints = args.checkpoint or find_checkpoints(run_dir, args.steps)
        for checkpoint_name, checkpoint_path in checkpoints:
            checkpoint_path = checkpoint_path.expanduser()
            if not checkpoint_path.is_file():
                raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
            param_rows.append(summarize_checkpoint(checkpoint_name, checkpoint_path))

        params_csv = output_dir / f"{args.output_prefix}_checkpoint_params.csv"
        params_jsonl = output_dir / f"{args.output_prefix}_checkpoint_params.jsonl"
        write_rows(params_csv, PARAM_FIELDNAMES, param_rows)
        write_jsonl(params_jsonl, param_rows)
        print(f"params_csv={params_csv}")
        print(f"params_jsonl={params_jsonl}")

    print("\nwindow,loss_gep,loss_zero_prob,loss_gepc,loss_gepc_zero_prob,raw_grad_norm,clip_rate,clip_coef_mean_when_clipped,lr")
    for row in metric_rows:
        print(
            f"{row['window']},{row.get('loss_gep')},{row.get('loss_zero_prob')},"
            f"{row.get('loss_gepc')},{row.get('loss_gepc_zero_prob')},"
            f"{row.get('raw_grad_norm')},{row.get('clip_rate')},"
            f"{row.get('clip_coef_mean_when_clipped')},{row.get('lr')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
