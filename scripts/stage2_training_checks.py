#!/usr/bin/env python3
"""Reusable Stage 2 training configuration, data, and artifact checks."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
import shlex
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

try:
    from stage2_training_config import (
        MODEL_CONFIG_FIELD_MAP,
        label_limits_from_config,
        load_model_config,
        shell_value,
    )
except ModuleNotFoundError:  # pragma: no cover - used when imported as a package.
    from scripts.stage2_training_config import (
        MODEL_CONFIG_FIELD_MAP,
        label_limits_from_config,
        load_model_config,
        shell_value,
    )


REQUIRED_PARQUET_COLUMNS = [
    "X",
    "soma_joinid",
    "dataset_id",
    "assay",
    "cell_type",
    "development_stage",
    "disease",
    "tissue",
    "sex",
    "tech_sample",
    "species",
    "idx",
]

EMBEDDING_FILES = [
    "2nd_run_macrogene_features_sum_esm2.npy",
    "2nd_run_macrogene_features_sum_gene_desc.npy",
    "2nd_run_macrogene_features_sum_dnaseq.npy",
]


def fail(title: str, errors: Iterable[str]) -> None:
    print(f"[ERROR] {title}:")
    for item in errors:
        print(f"  - {item}")
    raise SystemExit(1)


def print_warnings(title: str, warnings: Iterable[str]) -> None:
    warnings = list(warnings)
    if not warnings:
        return
    print(f"[WARN] {title}:")
    for item in warnings:
        print(f"  - {item}")


def emit_config_env(args: argparse.Namespace) -> None:
    config = load_model_config(args.config_json)
    print(f"SEQ_LEN={shlex.quote(str(config['seq_len']))}")
    for env_name, json_key in MODEL_CONFIG_FIELD_MAP.items():
        print(f"{env_name}={shlex.quote(shell_value(config[json_key]))}")


def preflight(args: argparse.Namespace) -> None:
    required = set(REQUIRED_PARQUET_COLUMNS)
    errors: list[str] = []

    for batch_dir in args.batch_dirs:
        if not batch_dir.exists():
            errors.append(f"missing input directory: {batch_dir}")
            continue

        files = sorted(batch_dir.glob("*/macrogene_*.parquet"))
        print(f"[INFO] {batch_dir}: matched macrogene parquet files={len(files)}")
        if not files:
            errors.append(f"no macrogene parquet files under {batch_dir}")
            continue

        checked = 0
        empty = 0
        bad: list[str] = []
        for scanned, path in enumerate(files, start=1):
            if scanned > args.max_scan and checked == 0:
                break
            if path.stat().st_size == 0:
                empty += 1
                continue
            try:
                parquet = pq.ParquetFile(path)
                schema_names = set(parquet.schema_arrow.names)
                missing = sorted(required - schema_names)
                if missing:
                    bad.append(f"{path}: missing columns {missing}")
                    continue
                if parquet.metadata.num_rows <= 0:
                    bad.append(f"{path}: no rows")
                    continue
                table = parquet.read_row_group(0, columns=["X"]).slice(0, 1)
                first_x = table.column("X")[0].as_py()
                if len(first_x) != args.seq_len:
                    bad.append(f"{path}: X length {len(first_x)} != seq_len {args.seq_len}")
                    continue
            except Exception as exc:  # noqa: BLE001 - report any parquet failure.
                bad.append(f"{path}: {type(exc).__name__}: {exc}")
                continue

            checked += 1
            if checked >= args.files_per_batch:
                break

        print(f"[INFO] {batch_dir}: checked valid files={checked}, empty placeholders seen={empty}")
        if checked == 0:
            hint = " This is expected on a workstation with placeholder data; run this step on the data server."
            errors.append(f"no valid non-empty parquet files found in sampled scan for {batch_dir}.{hint}")
        if bad:
            errors.extend(bad[:5])

    if errors:
        fail("source preflight failed", errors)
    print("[OK] source preflight passed")


def distributed_samples_per_rank(num_files: int, world_size: int) -> int:
    if num_files % world_size == 0:
        return math.ceil(num_files / world_size)
    return math.ceil((num_files - world_size) / world_size)


def require_arg(value: int | float | str | None, message: str):
    if value is None:
        raise SystemExit(message)
    return value


def validate_flat_file(
    path: Path,
    seq_len: int,
    sample_rows: int,
    label_limits: dict[str, int],
) -> list[str]:
    errors: list[str] = []
    try:
        parquet = pq.ParquetFile(path)
        names = parquet.schema_arrow.names
        missing = [col for col in REQUIRED_PARQUET_COLUMNS if col not in names]
        if missing:
            return [f"{path}: missing columns {missing}; found {names}"]

        table = parquet.read(columns=REQUIRED_PARQUET_COLUMNS)
        if sample_rows > 0 and table.num_rows > sample_rows:
            table = table.slice(0, sample_rows)
        df = table.to_pandas()
        if df.empty:
            return [f"{path}: validation sample is empty"]

        for idx, x in enumerate(df["X"]):
            arr = np.asarray(x, dtype=np.float64)
            if arr.shape != (seq_len,):
                errors.append(f"{path}: row {idx} X shape {arr.shape} != ({seq_len},)")
                break
            if not np.isfinite(arr).all():
                errors.append(f"{path}: row {idx} X contains non-finite values")
                break

        for col, limit in label_limits.items():
            vals = pd.to_numeric(df[col], errors="raise")
            if vals.isnull().any():
                errors.append(f"{path}: {col} has null values")
            bad = vals[(vals < 0) | (vals >= limit)]
            if len(bad) > 0:
                errors.append(
                    f"{path}: {col} values outside [0, {limit}): "
                    f"min={int(vals.min())}, max={int(vals.max())}"
                )
    except Exception as exc:  # noqa: BLE001 - report validation context.
        errors.append(f"{path}: validation failed: {type(exc).__name__}: {exc}")
    return errors


def validate_data(args: argparse.Namespace) -> None:
    args.command_dir.mkdir(parents=True, exist_ok=True)
    if args.config_json is not None:
        config = load_model_config(args.config_json)
        seq_len = int(config["seq_len"])
        label_limits = label_limits_from_config(config)
    else:
        seq_len = require_arg(args.seq_len, "--seq-len or --config-json is required")
        label_limits = {
            "assay": require_arg(args.num_seqmethod_labels, "--num-seqmethod-labels or --config-json is required"),
            "tech_sample": require_arg(args.num_batch_labels, "--num-batch-labels or --config-json is required"),
            "species": require_arg(args.num_species_labels, "--num-species-labels or --config-json is required"),
            "tissue": require_arg(args.num_tissue_labels, "--num-tissue-labels or --config-json is required"),
            "disease": require_arg(args.num_disease_labels, "--num-disease-labels or --config-json is required"),
            "development_stage": require_arg(args.num_age_labels, "--num-age-labels or --config-json is required"),
            "sex": require_arg(args.num_sex_labels, "--num-sex-labels or --config-json is required"),
        }

    errors: list[str] = []
    warnings: list[str] = []
    files = sorted(args.flat_dir.glob("all_flatten_part_*.parquet"))
    if not files:
        errors.append(f"no all_flatten_part_*.parquet files found in {args.flat_dir}")

    file_rows: list[int] = []
    species_counter: Counter[int] = Counter()
    validate_files = files if args.max_validate_files == 0 else files[: args.max_validate_files]

    for path in files:
        try:
            parquet = pq.ParquetFile(path)
            file_rows.append(parquet.metadata.num_rows)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{path}: cannot read parquet metadata: {type(exc).__name__}: {exc}")
            file_rows.append(0)

    for path in validate_files:
        errors.extend(
            validate_flat_file(
                path=path,
                seq_len=seq_len,
                sample_rows=args.max_validate_rows_per_file,
                label_limits=label_limits,
            )
        )

    for path in files:
        try:
            table = pq.read_table(path, columns=["species"])
            species_counter.update(int(x) for x in table.column("species").to_pylist())
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"{path}: could not read species distribution: {exc}")

    manifest_path = args.flat_dir / "shuffle_manifest.csv"
    if manifest_path.exists():
        with manifest_path.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        manifest_total = sum(int(row["num_rows"]) for row in rows)
        metadata_total = sum(file_rows)
        manifest_files = {row["output_file"] for row in rows}
        actual_files = {path.name for path in files}
        if manifest_total != metadata_total:
            errors.append(f"manifest row total {manifest_total} != parquet metadata rows {metadata_total}")
        missing_manifest_files = actual_files - manifest_files
        if missing_manifest_files:
            errors.append(f"manifest missing output files: {sorted(missing_manifest_files)[:5]}")
    else:
        warnings.append(f"missing shuffle manifest: {manifest_path}")

    if files and len(files) < args.world_size:
        errors.append(
            f"flat file count {len(files)} < world_size {args.world_size}; "
            "DistributedFileSampler(drop_last=True) would give empty/invalid rank shards"
        )

    if files:
        if len(files) % args.world_size != 0:
            dropped = len(files) - distributed_samples_per_rank(len(files), args.world_size) * args.world_size
            warnings.append(
                f"flat file count {len(files)} is not divisible by world_size {args.world_size}; "
                f"current drop_last=True sampler will drop {dropped} shuffled file(s) per epoch"
            )

        samples_per_rank = distributed_samples_per_rank(len(files), args.world_size)
        total_size = samples_per_rank * args.world_size
        if samples_per_rank <= 0:
            errors.append(f"samples_per_rank computed as {samples_per_rank}; need more flat files")
        else:
            indices = list(range(len(files)))[:total_size]
            plan_path = args.command_dir / "distributed_file_plan_epoch0.csv"
            with plan_path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(
                    fh,
                    fieldnames=[
                        "node_rank",
                        "local_rank",
                        "global_rank",
                        "num_files",
                        "num_rows",
                        "files",
                    ],
                )
                writer.writeheader()
                for rank in range(args.world_size):
                    rank_indices = indices[rank:total_size:args.world_size]
                    rank_files = [files[i] for i in rank_indices]
                    writer.writerow(
                        {
                            "node_rank": rank // args.nproc_per_node,
                            "local_rank": rank % args.nproc_per_node,
                            "global_rank": rank,
                            "num_files": len(rank_files),
                            "num_rows": sum(file_rows[i] for i in rank_indices),
                            "files": ";".join(path.name for path in rank_files),
                        }
                    )
            print(f"[INFO] wrote distributed file plan: {plan_path}")

    if args.emb_path.exists():
        for name in EMBEDDING_FILES:
            path = args.emb_path / name
            if not path.exists():
                errors.append(f"missing embedding file: {path}")
                continue
            arr = np.load(path, mmap_mode="r")
            if arr.ndim != 2:
                errors.append(f"{path}: expected 2D embedding array, got shape={arr.shape}")
            if arr.shape[0] != seq_len:
                errors.append(f"{path}: first dimension {arr.shape[0]} != seq_len {seq_len}")
            print(f"[INFO] embedding {name}: shape={arr.shape}, dtype={arr.dtype}")
    elif args.require_embeddings:
        errors.append(f"embedding directory does not exist: {args.emb_path}")
    else:
        warnings.append(f"embedding directory does not exist: {args.emb_path}")

    summary = {
        "flat_dir": str(args.flat_dir),
        "num_files": len(files),
        "total_rows": int(sum(file_rows)),
        "world_size": args.world_size,
        "seq_len": seq_len,
        "species_distribution": dict(sorted(species_counter.items())),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))

    print_warnings("validation warnings", warnings)
    if errors:
        fail("validation failed", errors)
    print("[OK] flat data validation passed")


def resolve_out_dir(args: argparse.Namespace) -> None:
    if args.config_json is not None:
        config = load_model_config(args.config_json)
        hidden_size = int(config["hidden_size"])
        num_hidden_layers = int(config["num_hidden_layers"])
        num_attention_heads = int(config["num_attention_heads"])
        hidden_dropout_prob = float(config["hidden_dropout_prob"])
    else:
        hidden_size = require_arg(args.hidden_size, "--hidden-size or --config-json is required")
        num_hidden_layers = require_arg(args.num_hidden_layers, "--num-hidden-layers or --config-json is required")
        num_attention_heads = require_arg(
            args.num_attention_heads,
            "--num-attention-heads or --config-json is required",
        )
        hidden_dropout_prob = require_arg(
            args.hidden_dropout_prob,
            "--hidden-dropout-prob or --config-json is required",
        )

    out_path = args.out_path.format(
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        hidden_dropout_prob=hidden_dropout_prob,
        learning_rate=args.learning_rate,
        min_lr=args.min_lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
    )
    path = Path(out_path)
    if not path.is_absolute():
        path = args.workdir / path
    print(path.resolve())


def check_training(args: argparse.Namespace) -> None:
    errors: list[str] = []
    warnings: list[str] = []

    if not args.train_out_dir.exists():
        errors.append(f"training output directory does not exist: {args.train_out_dir}")
    else:
        rank_logs = sorted(args.train_out_dir.glob("log.*.txt"))
        loss_logs = sorted(args.train_out_dir.glob("loss_to_log.*.txt"))
        metrics_logs = sorted(args.train_out_dir.glob("metrics.*.jsonl"))
        all_pt_files = sorted(args.train_out_dir.glob("SC-node-*-rank-*-epoch-*-step-*-loss-*.pt"))
        weights = [p for p in all_pt_files if not p.name.endswith(".optimizer.pt")]
        optimizer_weights = sorted(
            args.train_out_dir.glob("SC-node-*-rank-*-epoch-*-step-*-loss-*.optimizer.pt")
        )

        print(f"[INFO] output dir: {args.train_out_dir}")
        print(
            f"[INFO] rank logs={len(rank_logs)}, loss logs={len(loss_logs)}, metrics logs={len(metrics_logs)}, "
            f"weights={len(weights)}, optimizer states={len(optimizer_weights)}"
        )

        if len(rank_logs) < args.world_size:
            errors.append(f"expected at least {args.world_size} rank logs, found {len(rank_logs)}")
        if len(loss_logs) < args.world_size:
            errors.append(f"expected at least {args.world_size} loss logs, found {len(loss_logs)}")
        if len(metrics_logs) < args.world_size:
            warnings.append(f"expected metrics jsonl logs for {args.world_size} ranks, found {len(metrics_logs)}")

        final_epoch = args.epoch + 1
        final_pattern = re.compile(
            rf"SC-node-\d+-rank-\d+-epoch-{final_epoch:02d}-step-0-loss-0\.000000\.pt$"
        )
        final_weights = [p for p in weights if final_pattern.search(p.name)]
        if len(final_weights) < args.world_size:
            errors.append(
                f"expected at least {args.world_size} final model weights for "
                f"epoch {final_epoch:02d}, found {len(final_weights)}"
            )

        bad_pattern = re.compile(r"(Traceback|RuntimeError|ValueError|out of memory|\bnan\b)", re.IGNORECASE)
        data_pattern = re.compile(r"Node:\s*([^,]+),\s*Rank:\s*([^,]+),\s*Epoch:\s*(\d+),\s*Data:\s*(\[.*\])")
        data_by_epoch: defaultdict[int, list[tuple[str, str, list[str], str]]] = defaultdict(list)
        ranks_with_loss = set()

        for log_path in rank_logs:
            text = log_path.read_text(encoding="utf-8", errors="replace")
            bad = bad_pattern.search(text)
            if bad:
                errors.append(f"{log_path}: found failure marker {bad.group(1)!r}")
            if "loss:" in text:
                ranks_with_loss.add(log_path.name)
            for match in data_pattern.finditer(text):
                node = match.group(1).strip()
                rank = match.group(2).strip()
                epoch = int(match.group(3))
                try:
                    files = ast.literal_eval(match.group(4))
                except Exception:  # noqa: BLE001
                    files = []
                    warnings.append(f"{log_path}: could not parse Data list for epoch {epoch}")
                data_by_epoch[epoch].append((node, rank, files, log_path.name))

        if len(ranks_with_loss) < args.world_size:
            errors.append(f"expected loss lines in {args.world_size} rank logs, found {len(ranks_with_loss)}")

        for epoch in range(args.epoch):
            records = data_by_epoch.get(epoch, [])
            if len(records) < args.world_size:
                errors.append(
                    f"epoch {epoch}: expected data assignment records for "
                    f"{args.world_size} ranks, found {len(records)}"
                )
                continue
            all_files: list[str] = []
            empty_ranks = []
            for _node, _rank, files, log_name in records:
                if not files:
                    empty_ranks.append(log_name)
                all_files.extend(files)
            if empty_ranks:
                errors.append(f"epoch {epoch}: ranks with empty data assignments: {empty_ranks[:5]}")
            duplicates = [name for name, count in Counter(all_files).items() if count > 1]
            if duplicates:
                errors.append(f"epoch {epoch}: duplicate file assignment across ranks: {duplicates[:10]}")
            print(f"[INFO] epoch {epoch}: assigned files={len(all_files)}, unique files={len(set(all_files))}")

        for loss_path in loss_logs:
            try:
                with loss_path.open(newline="", encoding="utf-8") as fh:
                    reader = csv.DictReader(fh)
                    first_row = next(reader, None)
                if first_row is None:
                    errors.append(f"{loss_path}: empty loss csv")
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{loss_path}: cannot parse loss csv: {exc}")

        for metrics_path in metrics_logs:
            try:
                with metrics_path.open(encoding="utf-8", errors="replace") as fh:
                    first_line = fh.readline()
                if not first_line:
                    raise IndexError
                json.loads(first_line)
            except IndexError:
                warnings.append(f"{metrics_path}: empty metrics jsonl")
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"{metrics_path}: cannot parse first metrics jsonl row: {exc}")

    if args.node_log_dir.exists():
        node_logs = sorted(args.node_log_dir.glob("node_rank*.log"))
        print(f"[INFO] node launcher logs={len(node_logs)} in {args.node_log_dir}")
        bad_pattern = re.compile(r"(Traceback|RuntimeError|ValueError|out of memory|\bnan\b)", re.IGNORECASE)
        for path in node_logs:
            text = path.read_text(encoding="utf-8", errors="replace")
            bad = bad_pattern.search(text)
            if bad:
                errors.append(f"{path}: found failure marker {bad.group(1)!r}")
            if "Complete pretraining!" not in text:
                warnings.append(f"{path}: missing 'Complete pretraining!' marker; job may still be running")
    else:
        warnings.append(f"node log directory does not exist: {args.node_log_dir}")

    print_warnings("training check warnings", warnings)
    if errors:
        fail("training check failed", errors)
    print("[OK] training artifacts and logs passed checks")


def add_common_label_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--num-batch-labels", type=int)
    parser.add_argument("--num-species-labels", type=int)
    parser.add_argument("--num-tissue-labels", type=int)
    parser.add_argument("--num-seqmethod-labels", type=int)
    parser.add_argument("--num-disease-labels", type=int)
    parser.add_argument("--num-age-labels", type=int)
    parser.add_argument("--num-sex-labels", type=int)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    emit_parser = subparsers.add_parser("emit-config-env", help="Print shell assignments from model JSON.")
    emit_parser.add_argument("--config-json", type=Path, required=True)
    emit_parser.set_defaults(func=emit_config_env)

    preflight_parser = subparsers.add_parser("preflight", help="Check source Stage 2 parquet inputs.")
    preflight_parser.add_argument("--seq-len", type=int, required=True)
    preflight_parser.add_argument("--files-per-batch", type=int, required=True)
    preflight_parser.add_argument("--max-scan", type=int, required=True)
    preflight_parser.add_argument("batch_dirs", type=Path, nargs="+")
    preflight_parser.set_defaults(func=preflight)

    validate_parser = subparsers.add_parser("validate-data", help="Validate flattened training parquet data.")
    validate_parser.add_argument("--flat-dir", type=Path, required=True)
    validate_parser.add_argument("--command-dir", type=Path, required=True)
    validate_parser.add_argument("--emb-path", type=Path, required=True)
    validate_parser.add_argument("--config-json", type=Path)
    validate_parser.add_argument("--seq-len", type=int)
    validate_parser.add_argument("--world-size", type=int, required=True)
    validate_parser.add_argument("--nnodes", type=int, required=True)
    validate_parser.add_argument("--nproc-per-node", type=int, required=True)
    validate_parser.add_argument("--max-validate-rows-per-file", type=int, required=True)
    validate_parser.add_argument("--max-validate-files", type=int, required=True)
    validate_parser.add_argument("--require-embeddings", action="store_true")
    add_common_label_args(validate_parser)
    validate_parser.set_defaults(func=validate_data)

    resolve_parser = subparsers.add_parser("resolve-out-dir", help="Resolve formatted training output path.")
    resolve_parser.add_argument("--out-path", required=True)
    resolve_parser.add_argument("--workdir", type=Path, required=True)
    resolve_parser.add_argument("--config-json", type=Path)
    resolve_parser.add_argument("--hidden-size", type=int)
    resolve_parser.add_argument("--num-hidden-layers", type=int)
    resolve_parser.add_argument("--num-attention-heads", type=int)
    resolve_parser.add_argument("--hidden-dropout-prob", type=float)
    resolve_parser.add_argument("--learning-rate", type=float, required=True)
    resolve_parser.add_argument("--min-lr", type=float, required=True)
    resolve_parser.add_argument("--weight-decay", type=float, required=True)
    resolve_parser.add_argument("--warmup-ratio", type=float, required=True)
    resolve_parser.set_defaults(func=resolve_out_dir)

    check_parser = subparsers.add_parser("check-training", help="Check distributed training logs and artifacts.")
    check_parser.add_argument("--train-out-dir", type=Path, required=True)
    check_parser.add_argument("--node-log-dir", type=Path, required=True)
    check_parser.add_argument("--world-size", type=int, required=True)
    check_parser.add_argument("--epoch", type=int, required=True)
    check_parser.set_defaults(func=check_training)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
