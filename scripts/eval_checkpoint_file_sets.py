#!/usr/bin/env python3
"""No-grad checkpoint x parquet-file-set probe for SpeciesLLM pretraining.

This is intentionally separate from the training loop: it loads one model
checkpoint at a time, evaluates named parquet file sets with fixed MLM masks,
and writes aggregate loss/residual, activation-variance, and parameter-norm
summaries. Activation ``feature_std`` fields measure variation across evaluated
samples for each feature, averaged over features. The probe does not load
optimizer state or update model parameters.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import logging
import math
import os
import random
import re
import sys
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch_npu

from nanoBERT.model.nanoBERTmodel_cellmeta2_plusEncode_adbc import BERTForPreTraining
from nanoBERT.utils import CustomCollate_3GeneEmb
from nanoBERT.utils import criterion_neg_log_bernoulli, masked_huber_loss, masked_mse_loss
from train_MNodes_torchrun_mfu_preindexparquet import (
    apply_config_json,
    apply_runtime_overrides,
    build_bertconfig,
    iter_chunked_parquet_batches,
    load_gene_embeddings,
    load_model_checkpoint,
    parquet_row_counts,
    parse_gene_embedding_modalities,
    resolve_amp_dtype,
    resolve_torch_dtype,
)


RESULT_FIELDNAMES = [
    "time",
    "checkpoint",
    "checkpoint_path",
    "file_set",
    "num_files",
    "num_batches",
    "num_samples",
    "mask_count",
    "seconds",
    "loss_total",
    "loss_gep",
    "loss_zero_prob",
    "loss_gepc",
    "loss_gepc_zero_prob",
    "gep_target_mean",
    "gep_target_std",
    "gep_target_p50",
    "gep_target_p90",
    "gep_target_p95",
    "gep_target_p99",
    "gep_pred_mean",
    "gep_pred_std",
    "gep_pred_p50",
    "gep_pred_p90",
    "gep_pred_p95",
    "gep_pred_p99",
    "abs_err_mean",
    "abs_err_p50",
    "abs_err_p90",
    "abs_err_p95",
    "abs_err_p99",
    "frac_abs_err_gt_5",
    "frac_abs_err_gt_10",
    "frac_abs_err_gt_20",
    "signed_err_mean",
    "signed_err_p50",
    "signed_err_p90",
    "signed_err_p95",
    "zero_target_rate",
    "nonzero_target_rate",
    "zero_prob_mean",
    "zero_prob_p50",
    "zero_prob_p90",
    "zero_prob_p95",
    "zero_prob_p99",
    "zero_prob_on_zero_target_mean",
    "zero_prob_on_nonzero_target_mean",
    "masked_sequence_output_feature_std",
    "masked_decoder_input_feature_std",
    "cell_emb_feature_std",
    "gep_head_hidden_feature_std",
    "zero_head_hidden_feature_std",
    "zero_logits_std",
    "gep_head_weight_norm",
    "gep_head_bias_mean",
    "gep_head_bias_std",
    "zero_head_weight_norm",
    "zero_head_bias_mean",
    "zero_head_bias_std",
    "shared_projection_norm",
    "last_shared_encoder_norm",
    "last_shared_encoder_layer",
]

ACTIVATION_METRICS = [
    "masked_sequence_output_feature_std",
    "masked_decoder_input_feature_std",
    "cell_emb_feature_std",
    "gep_head_hidden_feature_std",
    "zero_head_hidden_feature_std",
    "zero_logits_std",
]

PARAMETER_METRICS = [
    "gep_head_weight_norm",
    "gep_head_bias_mean",
    "gep_head_bias_std",
    "zero_head_weight_norm",
    "zero_head_bias_mean",
    "zero_head_bias_std",
    "shared_projection_norm",
    "last_shared_encoder_norm",
    "last_shared_encoder_layer",
]

TRAJECTORY_METRICS = [
    "loss_gep",
    "loss_zero_prob",
    "loss_gepc",
    "gep_pred_mean",
    "gep_pred_std",
    "gep_pred_p95",
    "gep_pred_p99",
    "signed_err_mean",
    "abs_err_p95",
    "abs_err_p99",
    "frac_abs_err_gt_20",
    "zero_prob_on_zero_target_mean",
    "zero_prob_on_nonzero_target_mean",
    *ACTIVATION_METRICS,
    *PARAMETER_METRICS,
]

TRAJECTORY_FIELDNAMES = ["ckpt", *TRAJECTORY_METRICS]


class FeatureMoments:
    """Track variability across samples for every feature without retaining activations."""

    def __init__(self) -> None:
        self.count = 0
        self.sum: torch.Tensor | None = None
        self.sum_sq: torch.Tensor | None = None

    def update(self, tensor: torch.Tensor) -> None:
        if tensor.numel() == 0:
            return
        values = tensor.detach().float()
        if values.ndim == 0:
            values = values.reshape(1, 1)
        elif values.ndim == 1:
            values = values.reshape(-1, 1)
        else:
            values = values.reshape(-1, values.shape[-1])

        batch_sum = values.sum(dim=0).cpu().double()
        batch_sum_sq = values.square().sum(dim=0).cpu().double()
        if self.sum is None:
            self.sum = batch_sum
            self.sum_sq = batch_sum_sq
        else:
            if self.sum.shape != batch_sum.shape:
                raise ValueError(f"Activation feature shape changed: {self.sum.shape} -> {batch_sum.shape}")
            self.sum += batch_sum
            self.sum_sq += batch_sum_sq
        self.count += int(values.shape[0])

    def feature_std_mean(self) -> float | None:
        if self.count == 0 or self.sum is None or self.sum_sq is None:
            return None
        mean = self.sum / self.count
        variance = (self.sum_sq / self.count - mean.square()).clamp_min(0.0)
        return float(variance.sqrt().mean())


class ForwardActivationProbe:
    """Capture the tensors that distinguish the token decoder path from MVC."""

    def __init__(self, model) -> None:
        self.tensors: dict[str, torch.Tensor] = {}
        self.handles = []
        self.handles.append(model.bert.register_forward_hook(self._output_hook("sequence_output")))
        self.handles.append(model.decoder.register_forward_pre_hook(self._input_hook("decoder_input")))
        self.handles.append(model.decoder.fc[-2].register_forward_hook(self._output_hook("gep_head_hidden")))
        if hasattr(model.decoder, "zero_logit"):
            self.handles.append(
                model.decoder.zero_logit[-2].register_forward_hook(self._output_hook("zero_head_hidden"))
            )
            self.handles.append(
                model.decoder.zero_logit[-1].register_forward_hook(self._output_hook("zero_logits"))
            )
        if hasattr(model, "mvc_decoder"):
            self.handles.append(model.mvc_decoder.register_forward_pre_hook(self._input_hook("cell_emb")))

    @staticmethod
    def _first_tensor(value) -> torch.Tensor | None:
        if torch.is_tensor(value):
            return value
        if isinstance(value, (tuple, list)):
            for item in value:
                tensor = ForwardActivationProbe._first_tensor(item)
                if tensor is not None:
                    return tensor
        return None

    def _output_hook(self, name: str):
        def hook(_module, _inputs, output) -> None:
            tensor = self._first_tensor(output)
            if tensor is not None:
                self.tensors[name] = tensor

        return hook

    def _input_hook(self, name: str):
        def hook(_module, inputs) -> None:
            tensor = self._first_tensor(inputs)
            if tensor is not None:
                self.tensors[name] = tensor

        return hook

    def clear(self) -> None:
        self.tensors.clear()

    def validate(self, *, explicit_zero_prob: bool, run_mvc: bool) -> None:
        expected = {"sequence_output", "decoder_input", "gep_head_hidden"}
        if explicit_zero_prob:
            expected.update({"zero_head_hidden", "zero_logits"})
        if run_mvc:
            expected.add("cell_emb")
        missing = sorted(expected.difference(self.tensors))
        if missing:
            raise RuntimeError(f"Activation hooks did not capture expected tensors: {missing}")

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self.close()


class ActivationAccumulator:
    MASKED_TENSORS = {
        "sequence_output": "masked_sequence_output_feature_std",
        "decoder_input": "masked_decoder_input_feature_std",
        "gep_head_hidden": "gep_head_hidden_feature_std",
        "zero_head_hidden": "zero_head_hidden_feature_std",
        "zero_logits": "zero_logits_std",
    }

    def __init__(self) -> None:
        self.moments = {metric: FeatureMoments() for metric in ACTIVATION_METRICS}

    def update(self, tensors: dict[str, torch.Tensor], mask_positions: torch.Tensor) -> None:
        for tensor_name, metric_name in self.MASKED_TENSORS.items():
            tensor = tensors.get(tensor_name)
            if tensor is not None:
                self.moments[metric_name].update(tensor[mask_positions])
        cell_emb = tensors.get("cell_emb")
        if cell_emb is not None:
            self.moments["cell_emb_feature_std"].update(cell_emb)

    def to_row(self) -> dict[str, float | None]:
        return {name: moments.feature_std_mean() for name, moments in self.moments.items()}


class EvalAccumulator:
    def __init__(self, collect_distributions: bool, collect_activations: bool) -> None:
        self.collect_distributions = collect_distributions
        self.activation_accumulator = ActivationAccumulator() if collect_activations else None
        self.loss_sums = defaultdict(float)
        self.mask_count = 0
        self.batch_count = 0
        self.sample_count = 0
        self.values = defaultdict(list)
        self.zero_target_count = 0
        self.nonzero_target_count = 0
        self.zero_prob_on_zero_sum = 0.0
        self.zero_prob_on_nonzero_sum = 0.0

    def update_losses(self, losses: dict[str, torch.Tensor | None], mask_count: int, batch_size: int) -> None:
        if mask_count <= 0:
            return
        self.mask_count += int(mask_count)
        self.batch_count += 1
        self.sample_count += int(batch_size)

        total = 0.0
        for name, value in losses.items():
            if value is None:
                continue
            scalar = float(value.detach().float().cpu())
            self.loss_sums[name] += scalar * mask_count
            total += scalar
        self.loss_sums["loss_total"] += total * mask_count

    def update_distributions(
        self,
        target_values: torch.Tensor,
        pred_values: torch.Tensor,
        zero_prob_values: torch.Tensor | None,
    ) -> None:
        if target_values.numel() == 0:
            return

        target = target_values.detach().float().cpu()
        pred = pred_values.detach().float().cpu()
        signed_err = pred - target
        abs_err = signed_err.abs()

        zero_target_mask = target <= 0
        nonzero_target_mask = target > 0
        self.zero_target_count += int(zero_target_mask.sum())
        self.nonzero_target_count += int(nonzero_target_mask.sum())

        if self.collect_distributions:
            self.values["gep_target"].append(target)
            self.values["gep_pred"].append(pred)
            self.values["abs_err"].append(abs_err)
            self.values["signed_err"].append(signed_err)

        if zero_prob_values is None:
            return

        zero_prob = zero_prob_values.detach().float().cpu()
        if self.collect_distributions:
            self.values["zero_prob"].append(zero_prob)
        if zero_target_mask.any():
            self.zero_prob_on_zero_sum += float(zero_prob[zero_target_mask].sum())
        if nonzero_target_mask.any():
            self.zero_prob_on_nonzero_sum += float(zero_prob[nonzero_target_mask].sum())

    def update_activations(
        self,
        tensors: dict[str, torch.Tensor],
        mask_positions: torch.Tensor,
    ) -> None:
        if self.activation_accumulator is not None:
            self.activation_accumulator.update(tensors, mask_positions)

    def _cat(self, name: str) -> torch.Tensor | None:
        parts = self.values.get(name)
        if not parts:
            return None
        return torch.cat(parts)

    @staticmethod
    def _quantiles(tensor: torch.Tensor, probs: tuple[float, ...]) -> list[float | None]:
        if tensor.numel() == 0:
            return [None for _ in probs]
        q = torch.quantile(tensor, torch.tensor(probs, dtype=torch.float32))
        return [float(v) for v in q.tolist()]

    def _basic_stats(self, prefix: str, tensor: torch.Tensor | None) -> dict[str, float | None]:
        if tensor is None or tensor.numel() == 0:
            return {
                f"{prefix}_mean": None,
                f"{prefix}_std": None,
                f"{prefix}_p50": None,
                f"{prefix}_p90": None,
                f"{prefix}_p95": None,
                f"{prefix}_p99": None,
            }
        p50, p90, p95, p99 = self._quantiles(tensor, (0.50, 0.90, 0.95, 0.99))
        return {
            f"{prefix}_mean": float(tensor.mean()),
            f"{prefix}_std": float(tensor.std(unbiased=False)),
            f"{prefix}_p50": p50,
            f"{prefix}_p90": p90,
            f"{prefix}_p95": p95,
            f"{prefix}_p99": p99,
        }

    def to_row(self) -> dict[str, float | int | None]:
        denom = max(1, self.mask_count)
        row: dict[str, float | int | None] = {
            "num_batches": self.batch_count,
            "num_samples": self.sample_count,
            "mask_count": self.mask_count,
        }
        for name in ("loss_total", "loss_gep", "loss_zero_prob", "loss_gepc", "loss_gepc_zero_prob"):
            row[name] = self.loss_sums[name] / denom if self.mask_count else None

        target = self._cat("gep_target")
        pred = self._cat("gep_pred")
        abs_err = self._cat("abs_err")
        signed_err = self._cat("signed_err")
        zero_prob = self._cat("zero_prob")

        row.update(self._basic_stats("gep_target", target))
        row.update(self._basic_stats("gep_pred", pred))
        row.update(self._basic_stats("abs_err", abs_err))

        if signed_err is None or signed_err.numel() == 0:
            row.update({
                "signed_err_mean": None,
                "signed_err_p50": None,
                "signed_err_p90": None,
                "signed_err_p95": None,
            })
        else:
            p50, p90, p95 = self._quantiles(signed_err, (0.50, 0.90, 0.95))
            row.update({
                "signed_err_mean": float(signed_err.mean()),
                "signed_err_p50": p50,
                "signed_err_p90": p90,
                "signed_err_p95": p95,
            })

        if abs_err is None or abs_err.numel() == 0:
            row.update({
                "frac_abs_err_gt_5": None,
                "frac_abs_err_gt_10": None,
                "frac_abs_err_gt_20": None,
            })
        else:
            row.update({
                "frac_abs_err_gt_5": float((abs_err > 5).float().mean()),
                "frac_abs_err_gt_10": float((abs_err > 10).float().mean()),
                "frac_abs_err_gt_20": float((abs_err > 20).float().mean()),
            })

        total_targets = self.zero_target_count + self.nonzero_target_count
        row["zero_target_rate"] = self.zero_target_count / total_targets if total_targets else None
        row["nonzero_target_rate"] = self.nonzero_target_count / total_targets if total_targets else None
        row.update(self._basic_stats("zero_prob", zero_prob))
        row["zero_prob_on_zero_target_mean"] = (
            self.zero_prob_on_zero_sum / self.zero_target_count if self.zero_target_count else None
        )
        row["zero_prob_on_nonzero_target_mean"] = (
            self.zero_prob_on_nonzero_sum / self.nonzero_target_count if self.nonzero_target_count else None
        )
        if self.activation_accumulator is None:
            row.update({name: None for name in ACTIVATION_METRICS})
        else:
            row.update(self.activation_accumulator.to_row())
        return row


def parse_name_value(spec: str, kind: str) -> tuple[str, str]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError(f"{kind} must use NAME=VALUE: {spec}")
    name, value = spec.split("=", 1)
    name = name.strip()
    value = value.strip()
    if not name or not value:
        raise argparse.ArgumentTypeError(f"{kind} must use non-empty NAME=VALUE: {spec}")
    return name, value


def read_file_entries(value: str) -> list[str]:
    if value.startswith("@"):
        path = Path(value[1:]).expanduser()
        entries = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                entries.append(line)
        return entries
    return [item.strip() for item in value.split(",") if item.strip()]


def resolve_parquet_files(data_path: Path, entries: list[str]) -> list[str]:
    paths = []
    for entry in entries:
        path = Path(entry).expanduser()
        if not path.is_absolute():
            path = data_path / path
        paths.append(str(path))
    missing = [path for path in paths if not Path(path).is_file()]
    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(f"Missing parquet files ({len(missing)}): {preview}")
    return paths


def set_eval_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    manual_seed_all = getattr(torch_npu.npu, "manual_seed_all", None)
    if manual_seed_all is not None:
        manual_seed_all(seed)


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("speciesllm.checkpoint_file_sets")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def clean_csv_row(row: dict[str, object]) -> dict[str, object]:
    return {key: ("" if value is None else value) for key, value in row.items()}


def mean_metric(rows: list[dict[str, object]], metric: str) -> float | None:
    values = []
    for row in rows:
        value = row.get(metric)
        if value is None:
            continue
        value = float(value)
        if not math.isnan(value):
            values.append(value)
    return sum(values) / len(values) if values else None


def build_trajectory_row(
    checkpoint_name: str,
    checkpoint_rows: list[dict[str, object]],
    parameter_stats: dict[str, float | int | None],
) -> dict[str, object]:
    row: dict[str, object] = {"ckpt": checkpoint_name}
    for metric in TRAJECTORY_METRICS:
        if metric in parameter_stats:
            row[metric] = parameter_stats[metric]
        else:
            row[metric] = mean_metric(checkpoint_rows, metric)
    return row


def select_named_parameters(model, predicate) -> list[tuple[str, torch.Tensor]]:
    return [(name, parameter) for name, parameter in model.named_parameters() if predicate(name)]


def parameter_l2_norm(items: list[tuple[str, torch.Tensor]]) -> float | None:
    if not items:
        return None
    sum_sq = 0.0
    for _name, parameter in items:
        sum_sq += float(parameter.detach().float().square().sum().cpu())
    return math.sqrt(sum_sq)


def parameter_mean_std(items: list[tuple[str, torch.Tensor]]) -> tuple[float | None, float | None]:
    if not items:
        return None, None
    count = 0
    total = 0.0
    total_sq = 0.0
    for _name, parameter in items:
        values = parameter.detach().float()
        count += values.numel()
        total += float(values.sum().cpu())
        total_sq += float(values.square().sum().cpu())
    mean = total / count
    variance = max(0.0, total_sq / count - mean * mean)
    return mean, math.sqrt(variance)


def summarize_model_parameters(model, *, explicit_zero_prob: bool) -> dict[str, float | int | None]:
    gep_weight = select_named_parameters(
        model,
        lambda name: name.startswith("decoder.fc.") and name.endswith(".weight"),
    )
    gep_bias = select_named_parameters(
        model,
        lambda name: name.startswith("decoder.fc.") and name.endswith(".bias"),
    )
    zero_weight = select_named_parameters(
        model,
        lambda name: name.startswith("decoder.zero_logit.") and name.endswith(".weight"),
    )
    zero_bias = select_named_parameters(
        model,
        lambda name: name.startswith("decoder.zero_logit.") and name.endswith(".bias"),
    )
    shared_projection = select_named_parameters(
        model,
        lambda name: name.startswith("bert.value_encoder.") or name.startswith("bert.enhanced_fusion."),
    )

    layer_ids = []
    for name, _parameter in model.named_parameters():
        match = re.match(r"^bert\.h\.(\d+)\.", name)
        if match:
            layer_ids.append(int(match.group(1)))
    last_layer = max(layer_ids) if layer_ids else None
    last_encoder = select_named_parameters(
        model,
        lambda name: last_layer is not None and name.startswith(f"bert.h.{last_layer}."),
    )

    gep_bias_mean, gep_bias_std = parameter_mean_std(gep_bias)
    zero_bias_mean, zero_bias_std = parameter_mean_std(zero_bias)
    stats = {
        "gep_head_weight_norm": parameter_l2_norm(gep_weight),
        "gep_head_bias_mean": gep_bias_mean,
        "gep_head_bias_std": gep_bias_std,
        "zero_head_weight_norm": parameter_l2_norm(zero_weight),
        "zero_head_bias_mean": zero_bias_mean,
        "zero_head_bias_std": zero_bias_std,
        "shared_projection_norm": parameter_l2_norm(shared_projection),
        "last_shared_encoder_norm": parameter_l2_norm(last_encoder),
        "last_shared_encoder_layer": last_layer,
    }
    required = ["gep_head_weight_norm", "shared_projection_norm", "last_shared_encoder_norm"]
    if explicit_zero_prob:
        required.append("zero_head_weight_norm")
    missing = [name for name in required if stats[name] is None]
    if missing:
        raise RuntimeError(f"Could not match expected checkpoint parameters: {missing}")
    return stats


def build_model(args: argparse.Namespace, device: str, logger: logging.Logger):
    apply_config_json(args)
    args.gene_embedding_modalities = parse_gene_embedding_modalities(args.gene_embedding_modalities)
    dtype, ptdtype = resolve_amp_dtype(args)

    src = np.arange(1, args.seq_len + 1)
    gene_embeddings = load_gene_embeddings(args.emb_path, args.gene_embedding_modalities, args.seq_len)
    esm_embeddings = gene_embeddings.get("esm2")
    desc_embeddings = gene_embeddings.get("gene_desc")
    dna_embeddings = gene_embeddings.get("dnaseq")

    gene_ids = ["gene_" + str(idx) for idx in range(1, args.seq_len + 1)]
    cls_token = ["<cls>"]
    special_tokens = ["<cls>", "<sep>", "<pad>", "<mask>"]
    vocab = {token: idx for idx, token in enumerate(cls_token + gene_ids + special_tokens)}

    config = build_bertconfig(vocab, args.seq_len, args)
    apply_runtime_overrides(config, args, logger=logger)

    collate_fn = CustomCollate_3GeneEmb(
        config=config,
        genes=src,
        esm_embeddings=esm_embeddings,
        desc_embeddings=desc_embeddings,
        dna_embeddings=dna_embeddings,
        return_static_inputs=False,
    )

    model = BERTForPreTraining(config).to(device)
    model.gradient_checkpointing = False
    model.bert.gradient_checkpointing = False
    model.set_static_gene_inputs(
        src,
        esm_embeddings,
        desc_embeddings,
        dna_embeddings,
        cls_id=vocab["<cls>"],
        append_cls=True,
        dtype=resolve_torch_dtype(args.static_gene_dtype, ptdtype),
    )
    return model, config, collate_fn, dtype, ptdtype


def regression_loss_fn(args: argparse.Namespace):
    if args.gep_loss == "huber":
        delta = float(args.huber_delta)

        def _loss(pred, target, mask):
            return masked_huber_loss(pred, target, mask, delta=delta)

        return _loss
    return masked_mse_loss


def move_batch_to_device(batch_data: dict[str, torch.Tensor], config, device: str) -> dict[str, torch.Tensor | None]:
    moved = {
        "values": batch_data["values"].to(device, non_blocking=True),
        "target_values": batch_data["target_values"].to(device, non_blocking=True),
        "batch_labels": None,
        "species_labels": None,
        "tissue_labels": None,
        "seqmethod_labels": None,
        "disease_labels": None,
        "sex_labels": None,
        "age_labels": None,
    }
    if config.use_batch_labels:
        moved["batch_labels"] = batch_data["batch_labels"].to(device, non_blocking=True)
    if config.use_species_labels:
        moved["species_labels"] = batch_data["species_labels"].to(device, non_blocking=True)
    if config.use_tissue_labels:
        moved["tissue_labels"] = batch_data["tissue_labels"].to(device, non_blocking=True)
    if config.use_seqmethod_labels:
        moved["seqmethod_labels"] = batch_data["seqmethod_labels"].to(device, non_blocking=True)
    if config.use_disease_labels:
        moved["disease_labels"] = batch_data["disease_labels"].to(device, non_blocking=True)
    if config.use_sex_labels:
        moved["sex_labels"] = batch_data["sex_labels"].to(device, non_blocking=True)
    if config.use_age_labels:
        moved["age_labels"] = batch_data["age_labels"].to(device, non_blocking=True)
    return moved


def evaluate_file_set(
    args: argparse.Namespace,
    model,
    config,
    collate_fn,
    ctx,
    regression_loss,
    file_paths: list[str],
    device: str,
    logger: logging.Logger,
    activation_probe: ForwardActivationProbe | None,
) -> EvalAccumulator:
    rows_by_file = parquet_row_counts(file_paths)
    max_batch_index = int(math.ceil(sum(rows_by_file) / args.batch_size))
    if args.max_batches_per_file_set > 0:
        max_batch_index = min(max_batch_index, int(args.max_batches_per_file_set))

    batch_args = SimpleNamespace(
        parquet_chunk_files=args.parquet_chunk_files,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
        pin_memory=args.pin_memory,
        shuffle_rows=False,
        shuffle_seed=42,
    )
    batch_iter = iter_chunked_parquet_batches(
        batch_args,
        file_paths,
        rows_by_file,
        rank=0,
        NODE_RANK="0",
        collate_fn=collate_fn,
        logger=logger,
        epoch=0,
        resume_batch_offset=0,
        max_batch_index=max_batch_index,
    )

    accum = EvalAccumulator(
        collect_distributions=args.collect_distributions,
        collect_activations=activation_probe is not None,
    )
    model.eval()
    run_mvc = bool(args.train_mvc and config.do_mvc)

    with torch.no_grad():
        for batch_index, batch_data, _data_load_s, _batch_context in batch_iter:
            moved = move_batch_to_device(batch_data, config, device)
            if activation_probe is not None:
                activation_probe.clear()
            with ctx:
                outputs = model(
                    values=moved["values"],
                    batch_labels=moved["batch_labels"],
                    species_labels=moved["species_labels"],
                    tissue_labels=moved["tissue_labels"],
                    seqmethod_labels=moved["seqmethod_labels"],
                    disease_labels=moved["disease_labels"],
                    sex_labels=moved["sex_labels"],
                    age_labels=moved["age_labels"],
                    CLS=False,
                    MVC=run_mvc,
                    output_hidden_states=False,
                    output_attentions=False,
                )
                mask_positions = moved["values"].eq(-1)
                mask_count = int(mask_positions.sum().detach().cpu())
                loss_gep = regression_loss(outputs["model_output"], moved["target_values"], mask_positions)
                loss_zero_prob = None
                loss_gepc = None
                loss_gepc_zero_prob = None
                if config.explicit_zero_prob:
                    loss_zero_prob = criterion_neg_log_bernoulli(
                        outputs["model_zero_prob"],
                        moved["target_values"],
                        mask_positions,
                    )
                if "mvc_output" in outputs:
                    loss_gepc = regression_loss(outputs["mvc_output"], moved["target_values"], mask_positions)
                if "mvc_output" in outputs and config.explicit_zero_prob:
                    loss_gepc_zero_prob = criterion_neg_log_bernoulli(
                        outputs["mvc_zero_probs"],
                        moved["target_values"],
                        mask_positions,
                    )

            accum.update_losses(
                {
                    "loss_gep": loss_gep,
                    "loss_zero_prob": loss_zero_prob,
                    "loss_gepc": loss_gepc,
                    "loss_gepc_zero_prob": loss_gepc_zero_prob,
                },
                mask_count=mask_count,
                batch_size=int(moved["target_values"].shape[0]),
            )
            accum.update_distributions(
                moved["target_values"][mask_positions],
                outputs["model_output"][mask_positions],
                outputs["model_zero_prob"][mask_positions] if config.explicit_zero_prob else None,
            )
            if activation_probe is not None:
                activation_probe.validate(
                    explicit_zero_prob=bool(config.explicit_zero_prob),
                    run_mvc=run_mvc,
                )
                accum.update_activations(activation_probe.tensors, mask_positions)

            if args.log_every_batches > 0 and accum.batch_count % args.log_every_batches == 0:
                logger.info(
                    "eval_progress batch=%s/%s seen_batches=%s mask_count=%s",
                    batch_index + 1,
                    max_batch_index,
                    accum.batch_count,
                    accum.mask_count,
                )

    return accum


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--emb_path", required=True)
    parser.add_argument("--gene_embedding_modalities", default="esm2,dnaseq")
    parser.add_argument("--config_json", required=True)
    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument("--checkpoint", action="append", required=True, help="Repeat NAME=/path/checkpoint.pt")
    parser.add_argument("--file-set", action="append", required=True, help="Repeat NAME=file.parquet,... or NAME=@filelist.txt")
    parser.add_argument("--output_dir", default="diagnostics/checkpoint_file_sets")
    parser.add_argument("--output_prefix", default="checkpoint_file_sets")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--parquet_chunk_files", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--prefetch_factor", type=int, default=1)
    parser.add_argument("--persistent_workers", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max_batches_per_file_set", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--log_every_batches", type=int, default=50)
    parser.add_argument("--collect_distributions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--collect_activations", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp_dtype", choices=["auto", "float16", "bfloat16", "float32"], default="float32")
    parser.add_argument("--static_gene_dtype", choices=["float32", "float16", "bfloat16", "amp"], default="float32")
    parser.add_argument("--gep_loss", choices=["mse", "huber"], default="huber")
    parser.add_argument("--huber_delta", type=float, default=5.0)
    parser.add_argument("--device_type", choices=["npu", "cpu"], default="npu")
    parser.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", "0")))
    parser.add_argument("--runtime_attn_implementation", choices=["eager", "sdpa"], default="sdpa")
    parser.add_argument("--runtime_explicit_zero_prob", default=None)
    parser.add_argument("--runtime_do_mvc", default=None)
    parser.add_argument("--runtime_chunk_size_feed_forward", type=int, default=None)
    parser.add_argument("--train_mvc", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hidden_size", type=int, default=None)
    parser.add_argument("--num_hidden_layers", type=int, default=None)
    parser.add_argument("--num_attention_heads", type=int, default=None)
    parser.add_argument("--intermediate_size", type=int, default=None)
    parser.add_argument("--hidden_act", type=str, default=None)
    parser.add_argument("--hidden_dropout_prob", type=float, default=None)
    parser.add_argument("--cell_hidden_size", type=int, default=None)
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=None)
    parser.add_argument("--type_vocab_size", type=int, default=None)
    parser.add_argument("--initializer_range", type=float, default=None)
    parser.add_argument("--layer_norm_eps", type=float, default=None)
    parser.add_argument("--_attn_implementation", type=str, default=None)
    parser.add_argument("--use_batch_labels", type=str, default=None)
    parser.add_argument("--num_batch_labels", type=int, default=None)
    parser.add_argument("--use_species_labels", type=str, default=None)
    parser.add_argument("--num_species_labels", type=int, default=None)
    parser.add_argument("--use_tissue_labels", type=str, default=None)
    parser.add_argument("--num_tissue_labels", type=int, default=None)
    parser.add_argument("--use_seqmethod_labels", type=str, default=None)
    parser.add_argument("--num_seqmethod_labels", type=int, default=None)
    parser.add_argument("--use_disease_labels", type=str, default=None)
    parser.add_argument("--num_disease_labels", type=int, default=None)
    parser.add_argument("--use_age_labels", type=str, default=None)
    parser.add_argument("--num_age_labels", type=int, default=None)
    parser.add_argument("--use_sex_labels", type=str, default=None)
    parser.add_argument("--num_sex_labels", type=int, default=None)
    parser.add_argument("--cell_emb_style", type=str, default=None)
    parser.add_argument("--chunk_size_feed_forward", type=int, default=None)
    parser.add_argument("--explicit_zero_prob", type=str, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger = setup_logger()

    checkpoints = [parse_name_value(spec, "--checkpoint") for spec in args.checkpoint]
    file_sets_raw = [parse_name_value(spec, "--file-set") for spec in args.file_set]
    data_path = Path(args.data_path).expanduser()
    if not data_path.is_dir():
        raise FileNotFoundError(f"Missing data_path: {data_path}")

    file_sets = []
    for name, value in file_sets_raw:
        entries = read_file_entries(value)
        if not entries:
            raise ValueError(f"Empty file set: {name}")
        file_sets.append((name, resolve_parquet_files(data_path, entries)))

    for _name, checkpoint_path in checkpoints:
        if not Path(checkpoint_path).expanduser().is_file():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    if args.device_type == "npu":
        torch_npu.npu.set_device(args.local_rank)
        device = f"npu:{args.local_rank}"
    else:
        device = "cpu"

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{args.output_prefix}.csv"
    jsonl_path = output_dir / f"{args.output_prefix}.jsonl"
    trajectory_csv_path = output_dir / f"{args.output_prefix}_trajectory.csv"
    trajectory_jsonl_path = output_dir / f"{args.output_prefix}_trajectory.jsonl"

    logger.info("device=%s checkpoints=%s file_sets=%s", device, len(checkpoints), len(file_sets))
    logger.info("csv=%s jsonl=%s", csv_path, jsonl_path)
    logger.info("trajectory_csv=%s trajectory_jsonl=%s", trajectory_csv_path, trajectory_jsonl_path)

    model, config, collate_fn, dtype, ptdtype = build_model(args, device, logger)
    ctx = nullcontext() if args.device_type == "cpu" or dtype == "float32" else torch.autocast(
        device_type=args.device_type,
        dtype=ptdtype,
    )
    regression_loss = regression_loss_fn(args)

    rows = []
    trajectory_rows = []
    activation_probe_context = ForwardActivationProbe(model) if args.collect_activations else nullcontext(None)
    with (
        activation_probe_context as activation_probe,
        csv_path.open("w", newline="", encoding="utf-8") as csv_file,
        jsonl_path.open("w", encoding="utf-8") as jsonl_file,
        trajectory_csv_path.open("w", newline="", encoding="utf-8") as trajectory_csv_file,
        trajectory_jsonl_path.open("w", encoding="utf-8") as trajectory_jsonl_file,
    ):
        writer = csv.DictWriter(csv_file, fieldnames=RESULT_FIELDNAMES)
        writer.writeheader()
        trajectory_writer = csv.DictWriter(trajectory_csv_file, fieldnames=TRAJECTORY_FIELDNAMES)
        trajectory_writer.writeheader()

        for checkpoint_name, checkpoint_path in checkpoints:
            checkpoint_path = str(Path(checkpoint_path).expanduser())
            logger.info("loading_checkpoint name=%s path=%s", checkpoint_name, checkpoint_path)
            load_model_checkpoint(model, checkpoint_path, device, logger, ddp=False, rank=0)
            model.eval()
            parameter_stats = summarize_model_parameters(
                model,
                explicit_zero_prob=bool(config.explicit_zero_prob),
            )
            logger.info(
                "PARAMS checkpoint=%s gep_head_weight_norm=%s zero_head_weight_norm=%s "
                "shared_projection_norm=%s last_shared_encoder_norm=%s",
                checkpoint_name,
                parameter_stats["gep_head_weight_norm"],
                parameter_stats["zero_head_weight_norm"],
                parameter_stats["shared_projection_norm"],
                parameter_stats["last_shared_encoder_norm"],
            )
            checkpoint_rows = []

            for file_set_index, (file_set_name, file_paths) in enumerate(file_sets):
                seed = args.seed + file_set_index
                set_eval_seed(seed)
                logger.info(
                    "eval_start checkpoint=%s file_set=%s seed=%s num_files=%s",
                    checkpoint_name,
                    file_set_name,
                    seed,
                    len(file_paths),
                )
                start = time.time()
                accum = evaluate_file_set(
                    args,
                    model,
                    config,
                    collate_fn,
                    ctx,
                    regression_loss,
                    file_paths,
                    device,
                    logger,
                    activation_probe,
                )
                elapsed = time.time() - start
                row = {
                    "time": _dt.datetime.now().isoformat(timespec="seconds"),
                    "checkpoint": checkpoint_name,
                    "checkpoint_path": checkpoint_path,
                    "file_set": file_set_name,
                    "num_files": len(file_paths),
                    "seconds": elapsed,
                    **accum.to_row(),
                    **parameter_stats,
                }
                rows.append(row)
                checkpoint_rows.append(row)
                writer.writerow(clean_csv_row({field: row.get(field) for field in RESULT_FIELDNAMES}))
                csv_file.flush()
                jsonl_file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                jsonl_file.flush()
                logger.info(
                    "RESULT checkpoint=%s file_set=%s loss_gep=%.6f loss_zero_prob=%.6f "
                    "loss_gepc=%.6f abs_err_p95=%s frac_abs_err_gt_5=%s",
                    checkpoint_name,
                    file_set_name,
                    row["loss_gep"] if row["loss_gep"] is not None else float("nan"),
                    row["loss_zero_prob"] if row["loss_zero_prob"] is not None else float("nan"),
                    row["loss_gepc"] if row["loss_gepc"] is not None else float("nan"),
                    row.get("abs_err_p95"),
                    row.get("frac_abs_err_gt_5"),
                )

            trajectory_row = build_trajectory_row(checkpoint_name, checkpoint_rows, parameter_stats)
            trajectory_rows.append(trajectory_row)
            trajectory_writer.writerow(
                clean_csv_row({field: trajectory_row.get(field) for field in TRAJECTORY_FIELDNAMES})
            )
            trajectory_csv_file.flush()
            trajectory_jsonl_file.write(json.dumps(trajectory_row, ensure_ascii=False, sort_keys=True) + "\n")
            trajectory_jsonl_file.flush()
            logger.info(
                "TRAJECTORY checkpoint=%s loss_gep=%s loss_zero_prob=%s loss_gepc=%s "
                "gep_pred_std=%s sequence_output_std=%s decoder_input_std=%s "
                "cell_emb_std=%s zero_logits_std=%s",
                checkpoint_name,
                trajectory_row.get("loss_gep"),
                trajectory_row.get("loss_zero_prob"),
                trajectory_row.get("loss_gepc"),
                trajectory_row.get("gep_pred_std"),
                trajectory_row.get("masked_sequence_output_feature_std"),
                trajectory_row.get("masked_decoder_input_feature_std"),
                trajectory_row.get("cell_emb_feature_std"),
                trajectory_row.get("zero_logits_std"),
            )

    print("\n" + ",".join(TRAJECTORY_FIELDNAMES))
    for row in trajectory_rows:
        print(",".join("" if row.get(field) is None else str(row.get(field)) for field in TRAJECTORY_FIELDNAMES))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
