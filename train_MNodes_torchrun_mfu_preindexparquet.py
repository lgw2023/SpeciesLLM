import os
import sys
import dask
import json
import time
import math
import glob
import gc
import csv
import logging
import argparse
import datetime
import warnings
import numpy as np
import pandas as pd
import scanpy as sc
try:
    import moxing as mox
except ImportError:
    mox = None
from copy import deepcopy
from dataclasses import dataclass
from contextlib import nullcontext
from pathlib import Path

import torch
import torch._dynamo
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

import apex
import torch_npu
from torch_npu.contrib import transfer_to_npu
from torch_npu.npu.amp import GradScaler, autocast

from nanoBERT.utils import GeneVocab, CustomCollate_3GeneEmb
from nanoBERT.utils import ParquetDataset, LazyParquetDataset, PreindexedParquetDataset
from nanoBERT.utils import masked_mse_loss, masked_relative_error, criterion_neg_log_bernoulli
from nanoBERT.model.nanoBERTmodel_cellmeta2_plusEncode_adbc import BERTConfig, BERTForPreTraining

try:
    from scripts.pretrain_config import load_model_config
except ModuleNotFoundError:
    from pretrain_config import load_model_config

import dask
import dask.dataframe as dd
from dask import delayed
dask.config.set({"dataframe.convert-string": False})
torch._dynamo.config.suppress_errors = True

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = "training_output"


METRIC_FIELDNAMES = [
    "time",
    "node_rank",
    "rank",
    "local_rank",
    "epoch",
    "batch_index",
    "num_batches",
    "update_step",
    "should_step",
    "lr",
    "loss_total",
    "loss_gep",
    "loss_zero_prob",
    "loss_gepc",
    "loss_gepc_zero_prob",
    "step_ms",
    "data_load_s",
    "to_device_s",
    "forward_s",
    "loss_s",
    "backward_s",
    "optimizer_s",
    "grad_sync_s",
    "samples_per_s",
    "tokens_per_s",
    "mfu",
    "grad_norm_raw",
    "grad_norm_ema",
    "grad_clip_threshold",
    "grad_skip_threshold",
    "grad_skip_max",
    "grad_hard_raw_norm_limit",
    "grad_ema_contribution_cap",
    "grad_action",
    "skipped",
    "skipped_count_cum",
    "clipped_count_cum",
    "consecutive_skips",
    "consecutive_clips",
    "cluster_loss_total_mean",
    "cluster_loss_total_min",
    "cluster_loss_total_max",
    "cluster_step_ms_mean",
    "cluster_step_ms_max",
    "cluster_data_load_s_max",
    "cluster_grad_sync_s_max",
    "cluster_samples_per_s_sum",
    "cluster_tokens_per_s_sum",
]


class RankContextFilter(logging.Filter):
    def __init__(self, node_rank, rank, local_rank):
        super().__init__()
        self.node_rank = node_rank
        self.rank = rank
        self.local_rank = local_rank

    def filter(self, record):
        record.node_rank = self.node_rank
        record.rank = self.rank
        record.local_rank = self.local_rank
        return True


def setup_rank_logger(out_dir, node_rank, rank, local_rank, master_process, log_level="INFO",
                      log_all_ranks=False, append=False):
    level = getattr(logging, str(log_level).upper(), logging.INFO)
    logger = logging.getLogger(f"speciesllm.train.node{node_rank}.rank{rank}")
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()
    logger.propagate = False

    context_filter = RankContextFilter(node_rank=node_rank, rank=rank, local_rank=local_rank)
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s node=%(node_rank)s rank=%(rank)s local_rank=%(local_rank)s pid=%(process)d %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    log_path = os.path.join(out_dir, f"log.{node_rank}-{rank}.txt")
    file_handler = logging.FileHandler(log_path, mode=("a" if append else "w"), encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    file_handler.addFilter(context_filter)
    logger.addHandler(file_handler)

    if log_all_ranks or master_process:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(level)
        stream_handler.setFormatter(formatter)
        stream_handler.addFilter(context_filter)
        logger.addHandler(stream_handler)

    return logger


def log_or_print(logger, level, message, exc_info=False):
    if logger is None:
        print(message, flush=True)
        return
    log_method = getattr(logger, level)
    log_method(message, exc_info=exc_info)


def json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    return str(value)


class AdaptiveGradClipState:
    """EMA-based adaptive global-norm clipping with hard skip on spikes.

    The grad norm fed to clip_grad_norm_ already reflects the DDP-synchronized
    gradients, so it is identical on every rank. The EMA / decision logic here
    therefore stays bit-identical across the cluster without any extra
    all_reduce — every rank takes the same branch.

    Three-tier rule (after warmup):
        norm <= clip_threshold     -> pass through
        clip_threshold < norm <= skip_threshold -> clip (rescale)
        norm > skip_threshold or non-finite     -> skip optimizer.step

    During warmup the static --grad_clip is used as the clip threshold; only
    non-finite grads and the optional hard raw-norm fuse are skipped. The EMA
    is still updated so it is ready when warmup ends.

    EMA contribution is bounded by ``min(raw_norm, ema*clip_ratio)`` plus an
    optional cap derived from ``skip_max / skip_ratio``:
      * ``ema*clip_ratio`` stops a single spike from hijacking the EMA.
      * ``skip_max / skip_ratio`` (when ``skip_max`` is set) stops a slow
        runaway from ratcheting the skip threshold above the configured fuse.

    ``max_clip`` is deliberately NOT used to cap the EMA. The clip ceiling is
    the size of the optimizer update after rescaling; the raw grad norm can be
    many orders of magnitude larger for this model. Tying the EMA to
    ``max_clip`` makes a 1e8-scale natural raw norm look catastrophic after
    warmup and causes false skip cascades.

    Safety nets (all opt-in, default to off via 0 / None):
      * ``max_consecutive_skips``: abort after N consecutive skipped optimizer
        steps. Catches the "raw_norm settled into a regime above skip_threshold
        and never recovers" case, e.g. when params have drifted into a
        catastrophic basin.
      * ``ema_runaway_factor``: abort when ``ema > ema_runaway_factor *
        ema_contribution_cap``. Active only when ``skip_max`` is configured,
        because the EMA cap is derived from the skip fuse, not from max_clip.
      * ``hard_raw_norm_limit``: abort whenever a single raw_norm reading
        exceeds this absolute cap. Useful as a "physical fuse" — pick a value
        many orders of magnitude above the normal regime.
    """

    def __init__(self, beta, clip_ratio, skip_ratio, min_clip, max_clip, warmup_steps, static_clip,
                 skip_max=0.0, max_consecutive_skips=0, ema_runaway_factor=0.0,
                 hard_raw_norm_limit=0.0):
        self.beta = float(beta)
        self.clip_ratio = float(clip_ratio)
        self.skip_ratio = float(skip_ratio)
        self.min_clip = float(min_clip)
        self.max_clip = float(max_clip)
        self.warmup_steps = max(0, int(warmup_steps))
        self.static_clip = float(static_clip) if static_clip and static_clip > 0 else 1.0
        self.skip_max = max(0.0, float(skip_max))
        self.max_consecutive_skips = max(0, int(max_consecutive_skips))
        self.ema_runaway_factor = max(0.0, float(ema_runaway_factor))
        self.hard_raw_norm_limit = max(0.0, float(hard_raw_norm_limit))
        if self.skip_max > 0.0 and self.skip_ratio > 0.0 and math.isfinite(self.skip_ratio):
            self.ema_contribution_cap = self.skip_max / self.skip_ratio
        else:
            self.ema_contribution_cap = 0.0
        self.last_ema_contribution = None
        self.ema = None
        self.observed = 0
        self.skipped = 0
        self.clipped = 0
        self.consecutive_skips = 0
        self.consecutive_clips = 0
        self.max_consecutive_skips_seen = 0
        self.max_consecutive_clips_seen = 0

    def warmup_done(self):
        return self.ema is not None and self.observed >= self.warmup_steps

    def thresholds(self):
        if not self.warmup_done():
            clip = self.static_clip
            skip = float("inf")
        else:
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

    def update(self, raw_norm):
        if not math.isfinite(raw_norm):
            return
        if self.ema is None:
            # Bootstrap: don't let a freak step-1 grad (e.g. fresh-init MSE
            # heads producing raw_norm ~1e8) pin the EMA at sky-high values.
            contribution = raw_norm
        else:
            # Spike protection: a single big grad can lift the contribution at
            # most to ``ema*clip_ratio``.
            contribution = min(raw_norm, self.ema * self.clip_ratio)
        if self.ema_contribution_cap > 0.0:
            contribution = min(contribution, self.ema_contribution_cap)
        self.last_ema_contribution = contribution
        if self.ema is None:
            self.ema = contribution
        else:
            self.ema = self.beta * self.ema + (1.0 - self.beta) * contribution
        self.observed += 1

    def note_action(self, action):
        """Update consecutive-event counters. Call once per optimizer decision.

        ``action`` should be one of: 'pass', 'clip', 'skip_norm', 'skip_nan',
        'no_step'. 'no_step' (gradient-accumulation micro-step) is ignored so
        it doesn't reset the streak counters.
        """
        if action == "no_step":
            return
        if action in ("skip_nan", "skip_norm"):
            self.skipped += 1
            self.consecutive_skips += 1
            self.consecutive_clips = 0
            if self.consecutive_skips > self.max_consecutive_skips_seen:
                self.max_consecutive_skips_seen = self.consecutive_skips
        elif action == "clip":
            self.clipped += 1
            self.consecutive_clips += 1
            self.consecutive_skips = 0
            if self.consecutive_clips > self.max_consecutive_clips_seen:
                self.max_consecutive_clips_seen = self.consecutive_clips
        else:  # 'pass'
            self.consecutive_skips = 0
            self.consecutive_clips = 0

    def safety_check(self, raw_norm):
        """Return a non-empty reason string if a safety tripwire fired."""
        if self.hard_raw_norm_limit > 0.0 and math.isfinite(raw_norm) and raw_norm > self.hard_raw_norm_limit:
            return ("raw_norm=%.4e exceeds hard_raw_norm_limit=%.4e"
                    % (raw_norm, self.hard_raw_norm_limit))
        if self.max_consecutive_skips > 0 and self.consecutive_skips >= self.max_consecutive_skips:
            return ("consecutive_skips=%d >= max_consecutive_skips=%d (gradient regime decoupled from EMA)"
                    % (self.consecutive_skips, self.max_consecutive_skips))
        if (self.ema_runaway_factor > 0.0
                and self.ema is not None
                and self.ema_contribution_cap > 0.0
                and self.ema > self.ema_contribution_cap * self.ema_runaway_factor):
            return ("ema=%.4e > ema_runaway_factor=%.2f * ema_contribution_cap=%.4e "
                    "(adaptive controller has decoupled)"
                    % (self.ema, self.ema_runaway_factor, self.ema_contribution_cap))
        return None


class StreamingMetricsWriter:
    def __init__(self, out_dir, node_rank, rank, flush_interval=100, append=False):
        self.flush_interval = max(1, int(flush_interval))
        self.rows_since_flush = 0
        self.jsonl_path = os.path.join(out_dir, f"metrics.{node_rank}-{rank}.jsonl")
        self.csv_path = os.path.join(out_dir, f"loss_to_log.{node_rank}-{rank}.txt")
        csv_has_rows = append and os.path.exists(self.csv_path) and os.path.getsize(self.csv_path) > 0
        self.jsonl_file = open(self.jsonl_path, "a" if append else "w", encoding="utf-8")
        self.csv_file = open(self.csv_path, "a" if append else "w", newline="", encoding="utf-8")
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=METRIC_FIELDNAMES)
        if not csv_has_rows:
            self.csv_writer.writeheader()

    def write(self, row):
        clean_row = {field: row.get(field, None) for field in METRIC_FIELDNAMES}
        self.jsonl_file.write(json.dumps(clean_row, ensure_ascii=False, default=json_default) + "\n")
        self.csv_writer.writerow(clean_row)
        self.rows_since_flush += 1
        if self.rows_since_flush >= self.flush_interval:
            self.flush()

    def flush(self):
        self.jsonl_file.flush()
        self.csv_file.flush()
        self.rows_since_flush = 0

    def close(self):
        self.flush()
        self.jsonl_file.close()
        self.csv_file.close()


def distributed_step_summary(local_values, device, world_size):
    if world_size <= 1 or not dist.is_available() or not dist.is_initialized():
        return {}

    mean_keys = ["loss_total", "step_ms"]
    max_keys = ["loss_total", "step_ms", "data_load_s", "grad_sync_s"]
    min_keys = ["loss_total"]
    sum_keys = ["samples_per_s", "tokens_per_s"]

    def reduce_values(keys, op):
        tensor = torch.tensor(
            [float(local_values.get(key, 0.0) or 0.0) for key in keys],
            dtype=torch.float32,
            device=device,
        )
        dist.all_reduce(tensor, op=op)
        return {key: float(value) for key, value in zip(keys, tensor.detach().cpu().tolist())}

    mean_values = reduce_values(mean_keys, dist.ReduceOp.SUM)
    max_values = reduce_values(max_keys, dist.ReduceOp.MAX)
    min_values = reduce_values(min_keys, dist.ReduceOp.MIN)
    sum_values = reduce_values(sum_keys, dist.ReduceOp.SUM)

    return {
        "cluster_loss_total_mean": mean_values["loss_total"] / world_size,
        "cluster_loss_total_min": min_values["loss_total"],
        "cluster_loss_total_max": max_values["loss_total"],
        "cluster_step_ms_mean": mean_values["step_ms"] / world_size,
        "cluster_step_ms_max": max_values["step_ms"],
        "cluster_data_load_s_max": max_values["data_load_s"],
        "cluster_grad_sync_s_max": max_values["grad_sync_s"],
        "cluster_samples_per_s_sum": sum_values["samples_per_s"],
        "cluster_tokens_per_s_sum": sum_values["tokens_per_s"],
    }


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


CONFIG_ARG_FIELD_MAP = {
    "hidden_size": "hidden_size",
    "num_hidden_layers": "num_hidden_layers",
    "num_attention_heads": "num_attention_heads",
    "intermediate_size": "intermediate_size",
    "hidden_act": "hidden_act",
    "hidden_dropout_prob": "hidden_dropout_prob",
    "cell_hidden_size": "cell_hidden_size",
    "attention_probs_dropout_prob": "attention_probs_dropout_prob",
    "type_vocab_size": "type_vocab_size",
    "initializer_range": "initializer_range",
    "layer_norm_eps": "layer_norm_eps",
    "_attn_implementation": "_attn_implementation",
    "use_batch_labels": "use_batch_labels",
    "num_batch_labels": "num_batch_labels",
    "use_species_labels": "use_species_labels",
    "num_species_labels": "num_species_labels",
    "use_tissue_labels": "use_tissue_labels",
    "num_tissue_labels": "num_tissue_labels",
    "use_seqmethod_labels": "use_seqmethod_labels",
    "num_seqmethod_labels": "num_seqmethod_labels",
    "use_disease_labels": "use_disease_labels",
    "num_disease_labels": "num_disease_labels",
    "use_age_labels": "use_age_labels",
    "num_age_labels": "num_age_labels",
    "use_sex_labels": "use_sex_labels",
    "num_sex_labels": "num_sex_labels",
    "cell_emb_style": "cell_emb_style",
    "chunk_size_feed_forward": "chunk_size_feed_forward",
    "explicit_zero_prob": "explicit_zero_prob",
}


def apply_config_json(args):
    if not args.config_json:
        if args.seq_len is None:
            raise ValueError("--seq_len is required unless --config_json is provided")
        return

    config = load_model_config(Path(args.config_json))
    config_seq_len = int(config["seq_len"])
    if args.seq_len is not None and int(args.seq_len) != config_seq_len:
        raise ValueError(
            f"--seq_len ({args.seq_len}) does not match {args.config_json} seq_len ({config_seq_len})"
        )
    args.seq_len = config_seq_len

    for arg_name, json_key in CONFIG_ARG_FIELD_MAP.items():
        setattr(args, arg_name, config[json_key])


def value_or_default(value, default):
    return value if value is not None else default


def resolve_torch_dtype(name, amp_dtype=None):
    if name == "amp":
        if amp_dtype is None:
            raise ValueError("amp_dtype is required when dtype is 'amp'")
        return amp_dtype
    try:
        return {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype: {name}") from exc


def apply_runtime_overrides(config, args, logger=None):
    overrides = {}

    if args.runtime_attn_implementation is not None:
        config._attn_implementation = args.runtime_attn_implementation
        overrides["_attn_implementation"] = config._attn_implementation
    if args.runtime_explicit_zero_prob is not None:
        config.explicit_zero_prob = str2bool(args.runtime_explicit_zero_prob)
        overrides["explicit_zero_prob"] = config.explicit_zero_prob
    if args.runtime_do_mvc is not None:
        config.do_mvc = str2bool(args.runtime_do_mvc)
        overrides["do_mvc"] = config.do_mvc
    if args.runtime_chunk_size_feed_forward is not None:
        config.chunk_size_feed_forward = int(args.runtime_chunk_size_feed_forward)
        overrides["chunk_size_feed_forward"] = config.chunk_size_feed_forward

    if logger is not None and overrides:
        logger.info("runtime_config_overrides=%s", json.dumps(overrides, sort_keys=True))
    return overrides


def resolve_amp_dtype(args):
    if args.amp_dtype == "auto":
        dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    else:
        dtype = args.amp_dtype
    return dtype, resolve_torch_dtype(dtype)


def npu_memory_stats_gib():
    stats = {}
    for name, func_name in (
            ("allocated_gib", "memory_allocated"),
            ("reserved_gib", "memory_reserved"),
            ("max_allocated_gib", "max_memory_allocated"),
            ("max_reserved_gib", "max_memory_reserved"),
    ):
        func = getattr(torch_npu.npu, func_name, None)
        if func is None:
            stats[name] = None
            continue
        try:
            stats[name] = float(func()) / (1024 ** 3)
        except Exception:
            stats[name] = None
    return stats


def reset_npu_peak_memory_stats(logger=None, tag=None):
    func = getattr(torch_npu.npu, "reset_peak_memory_stats", None)
    if func is None:
        return
    try:
        func()
    except Exception as exc:
        if logger is not None:
            logger.warning("memory_reset_failed tag=%s error=%s", tag, exc)


def log_npu_memory(logger, tag, rank=None, extra=None):
    if logger is None:
        return
    stats = npu_memory_stats_gib()
    parts = []
    if rank is not None:
        parts.append(f"rank={rank}")
    parts.append(f"tag={tag}")
    for key in ("allocated_gib", "reserved_gib", "max_allocated_gib", "max_reserved_gib"):
        value = stats.get(key)
        value_text = "NA" if value is None else f"{value:.4f}"
        parts.append(f"{key}={value_text}")
    if extra:
        for key, value in extra.items():
            parts.append(f"{key}={value}")
    logger.info("MEMORY %s", " ".join(parts))


def should_trace_step(interval, batch_index, should_step=False, final_step=False):
    if interval is None or int(interval) <= 0:
        return False
    interval = int(interval)
    step_number = batch_index + 1
    return step_number == 1 or step_number % interval == 0 or bool(should_step) or bool(final_step)


def tensor_shape_summary(mapping):
    result = {}
    for key, value in mapping.items():
        if torch.is_tensor(value):
            result[key] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "device": str(value.device),
            }
    return result


def resolve_output_dir(out_path):
    path = Path(out_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path.mkdir(parents=True, exist_ok=True)
    return str(path.resolve())


def is_remote_output_path(path):
    return path.startswith("s3://") or path.startswith("obs://")


def remote_output_name(out_path):
    path = Path(out_path.rstrip("/"))
    if path.is_absolute():
        return path.name
    return out_path.rstrip("/")


def get_node_rank():
    return os.getenv("NODE_RANK") or os.getenv("node_rank") or "0"


class DistributedFileSampler:
    def __init__(self, file_paths, num_replicas=None, rank=None, shuffle=True, seed=0, drop_last=False):
        self.file_paths = file_paths
        self.num_samples = len(file_paths)
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last

        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()

        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0

        if self.drop_last and self.num_samples % self.num_replicas != 0:
            self.num_samples_per_rank = int(math.ceil((self.num_samples - self.num_replicas) / self.num_replicas))
        else:
            self.num_samples_per_rank = int(math.ceil(self.num_samples / self.num_replicas))
        self.total_size = self.num_samples_per_rank * self.num_replicas

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        # 1. Shuffle all indices with epoch-based seed
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = list(torch.randperm(self.num_samples, generator=g).tolist())
        else:
            indices = list(range(self.num_samples))

        if not self.drop_last:
            # 2. Pad to total_size if needed
            padding_size = self.total_size - len(indices)
            if padding_size < len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]

        else:
            indices = indices[:self.total_size]

        assert len(indices) == self.total_size

        # 3. Subsample for this rank
        # This is chatgpt's suggestion
        # offset = self.num_samples_per_rank * self.rank
        # indices = indices[offset:offset + self.num_samples_per_rank]

        # This is DistributedFileSampler's solution
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples_per_rank

        return iter(indices)

    def __len__(self):
        return self.num_samples_per_rank

def get_lr(itertion_number, min_learning_rate, warmup_iters, learning_rate, lr_decay_iters):
    assert min_learning_rate <= learning_rate
    # (1) linear warmup for warmup_epoches
    if itertion_number < warmup_iters:
        return learning_rate * (itertion_number + 1) / (warmup_iters + 1)
    if itertion_number > lr_decay_iters:
        return min_learning_rate
    decay_ratio = (itertion_number - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_learning_rate + coeff * (learning_rate - min_learning_rate)


def unwrap_checkpoint_state(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            state = checkpoint.get(key)
            if isinstance(state, dict):
                return state
    return checkpoint


def strip_module_prefix(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    if not state_dict:
        return state_dict
    if all(isinstance(key, str) and key.startswith("module.") for key in state_dict):
        return {key[len("module."):]: value for key, value in state_dict.items()}
    return state_dict


def broadcast_model_state(model, src_rank=0):
    with torch.no_grad():
        for tensor in model.state_dict().values():
            if torch.is_tensor(tensor):
                dist.broadcast(tensor, src=src_rank)


def load_model_checkpoint(model, checkpoint_path, device, logger, ddp=False, rank=0):
    if not checkpoint_path:
        return
    checkpoint_path = os.path.expanduser(checkpoint_path)
    if rank == 0 and not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"init_model_path not found: {checkpoint_path}")
    if rank == 0 or os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = strip_module_prefix(unwrap_checkpoint_state(checkpoint))
        load_result = model.load_state_dict(state_dict, strict=True)
        logger.info("init_model_loaded path=%s missing_keys=%s unexpected_keys=%s",
                    checkpoint_path, load_result.missing_keys, load_result.unexpected_keys)
    elif not ddp:
        raise FileNotFoundError(f"init_model_path not found: {checkpoint_path}")
    else:
        logger.info("init_model_path not local on rank=%s; waiting for rank0 broadcast", rank)
    if ddp:
        broadcast_model_state(model, src_rank=0)
        logger.info("init_model_broadcast src_rank=0")


def load_optimizer_checkpoint(optimizer, checkpoint_path, device, logger):
    if not checkpoint_path:
        return
    checkpoint_path = os.path.expanduser(checkpoint_path)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"init_optimizer_path not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("optimizer_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    optimizer.load_state_dict(state_dict)
    logger.info("init_optimizer_loaded path=%s", checkpoint_path)


def read_parquet(fp):
    return dask.delayed(pd.read_parquet)(fp)


def read_parquet_delayed(fp):
    return dask.dataframe.read_parquet(fp, engine="pyarrow", memory_map=True)


def load_data_for_rank(folder_path, rank, world_size, logger=None):
    file_paths = sorted(glob.glob(f"{folder_path}/*.parquet"))
    if not file_paths:
        raise FileNotFoundError(f"No parquet files found in {folder_path}")

    sub_file_paths = file_paths[rank::world_size]
    log_or_print(logger, "info", f"[Rank {rank}] will read files: {sub_file_paths}")

    delayed_reads = [read_parquet(fp) for fp in sub_file_paths]
    ddf = dask.dataframe.from_delayed(delayed_reads)
    pdf = ddf.compute()
    log_or_print(logger, "info", f"[Rank {rank}] total rows read: {len(pdf)}")

    return pdf


def load_data_for_total(folder_path):
    file_paths = sorted(glob.glob(f"{folder_path}/*.parquet"))
    if not file_paths:
        raise FileNotFoundError(f"No parquet files found in {folder_path}")

    delayed_reads = [read_parquet(fp) for fp in file_paths]
    ddf = dask.dataframe.from_delayed(delayed_reads)
    pdf = ddf.compute()

    return pdf


def load_data(folder_path):
    file_paths = sorted(glob.glob(f"{folder_path}/*.parquet"))
    if not file_paths:
        raise FileNotFoundError(f"No parquet files found in {folder_path}")

    delayed_reads = read_parquet_delayed(file_paths)

    return delayed_reads


def load_data_for_rank_with_DistributedFileSampler(train_data_filelist, file_indices, rank, NODE_RANK, logger=None):
    file_paths = deepcopy(train_data_filelist)
    if not train_data_filelist:
        raise FileNotFoundError(f"No parquet files found in {train_data_filelist}")

    sub_file_paths = [file_paths[i] for i in file_indices]
    log_or_print(
        logger,
        "info",
        f"[Node {NODE_RANK} Rank {rank}] will read files: {[i.split('/')[-1] for i in sub_file_paths]}",
    )

    delayed_reads = [read_parquet(fp) for fp in sub_file_paths]
    ddf = dd.from_delayed(delayed_reads)
    pdf = ddf.compute()
    log_or_print(logger, "info", f"[Node {NODE_RANK} Rank {rank}] total rows read: {len(pdf)}")

    return pdf, sub_file_paths


def get_files(folder_path, number=0):
    if number == 0:
        file_paths = sorted(glob.glob(f"{folder_path}/*.parquet"))
    else:
        file_paths = sorted(glob.glob(f"{folder_path}/*.parquet"))[:number]
    if not file_paths:
        raise FileNotFoundError(f"No parquet files found in {folder_path}")
    return file_paths


def build_bertconfig(vocab, seq_len, args):
    # load config
    config = BERTConfig(vocab_size=seq_len + 1,
                        vocab=vocab,
                        hidden_size=value_or_default(args.hidden_size, 1280),
                        num_hidden_layers=value_or_default(args.num_hidden_layers, 24),
                        num_attention_heads=value_or_default(args.num_attention_heads, 20),
                        intermediate_size=value_or_default(args.intermediate_size, 5120),
                        hidden_act=value_or_default(args.hidden_act, "gelu"),
                        hidden_dropout_prob=value_or_default(args.hidden_dropout_prob, 0.1),
                        cell_hidden_size=value_or_default(args.cell_hidden_size, 128),
                        attention_probs_dropout_prob=value_or_default(args.attention_probs_dropout_prob, 0.1),
                        max_position_embeddings=args.seq_len + 1,
                        type_vocab_size=value_or_default(args.type_vocab_size, 2),
                        initializer_range=value_or_default(args.initializer_range, 0.02),
                        layer_norm_eps=value_or_default(args.layer_norm_eps, 1e-12),
                        _attn_implementation=value_or_default(args._attn_implementation, "sdpa"),
                        use_batch_labels=str2bool(args.use_batch_labels) if args.use_batch_labels is not None else False,
                        num_batch_labels=value_or_default(args.num_batch_labels, 12028),
                        use_species_labels=str2bool(args.use_species_labels) if args.use_species_labels is not None else True,
                        num_species_labels=value_or_default(args.num_species_labels, 11),
                        use_tissue_labels=str2bool(args.use_tissue_labels) if args.use_tissue_labels is not None else True,
                        num_tissue_labels=value_or_default(args.num_tissue_labels, 154),
                        use_seqmethod_labels=str2bool(args.use_seqmethod_labels) if args.use_seqmethod_labels is not None else True,
                        num_seqmethod_labels=value_or_default(args.num_seqmethod_labels, 28),
                        use_disease_labels=str2bool(args.use_disease_labels) if args.use_disease_labels is not None else True,
                        num_disease_labels=value_or_default(args.num_disease_labels, 143),
                        use_age_labels=str2bool(args.use_age_labels) if args.use_age_labels is not None else True,
                        num_age_labels=value_or_default(args.num_age_labels, 5),
                        use_sex_labels=str2bool(args.use_sex_labels) if args.use_sex_labels is not None else True,
                        num_sex_labels=value_or_default(args.num_sex_labels, 3),
                        cell_emb_style=value_or_default(args.cell_emb_style, "cls"),
                        chunk_size_feed_forward=value_or_default(args.chunk_size_feed_forward, 0),
                        explicit_zero_prob=str2bool(args.explicit_zero_prob) if args.explicit_zero_prob is not None else True,
                        )

    return config


def setup_ddp(backend='hccl', device_type='npu'):
    ddp = int(os.environ.get('RANK', -1)) != -1
    if True:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        torch_npu.npu.set_device(local_rank)
        torch.distributed.init_process_group(backend=backend)
        master_process = (rank == 0 and local_rank == 0)
        world_size = int(os.environ['WORLD_SIZE']) if int(os.environ['WORLD_SIZE']) else int(os.environ['NNODES']) * int(os.environ['NPROC_PER_NODE'])
    else:
        local_rank = 0
        rank = 0
        master_process = True
        world_size = 1

    return ddp, rank, local_rank, master_process, world_size


def train_loop(args, model, ddp, rank, local_rank, optimizer, train_data_filelist, train_sampler, device, config, ctx, scaler, grad_clip, out_dir, collate_fn, world_size, logger):
    raw_model = model.module if ddp else model
    iter_num = 0
    update_step = max(0, int(args.resume_update_step))
    resume_start_epoch = max(0, int(args.resume_start_epoch))
    resume_skip_batches = max(0, int(args.resume_skip_batches))
    if resume_start_epoch >= args.epoch:
        raise ValueError(f"resume_start_epoch ({resume_start_epoch}) must be < epoch ({args.epoch})")
    if resume_skip_batches and resume_skip_batches % args.gradient_accumulation_steps != 0:
        raise ValueError(
            f"resume_skip_batches ({resume_skip_batches}) must be divisible by "
            f"gradient_accumulation_steps ({args.gradient_accumulation_steps})"
        )
    runing_mfu = -1.0

    save_step_interval = max(1, int(args.save_data_interval / world_size / args.batch_size))
    NODE_RANK = get_node_rank()
    metrics_writer = StreamingMetricsWriter(
        out_dir=out_dir,
        node_rank=NODE_RANK,
        rank=rank,
        flush_interval=args.metrics_flush_interval,
        append=args.append_output_logs,
    )
    logger.info(
        "save_step_interval=%s save_data_interval=%s world_size=%s batch_size=%s append_output_logs=%s",
        save_step_interval,
        args.save_data_interval,
        world_size,
        args.batch_size,
        args.append_output_logs,
    )
    if update_step or resume_start_epoch or resume_skip_batches:
        logger.info(
            "resume_position resume_update_step=%s resume_start_epoch=%s resume_skip_batches=%s",
            update_step,
            resume_start_epoch,
            resume_skip_batches,
        )
    lr = optimizer.param_groups[0]["lr"]

    adaptive_state = None
    if args.adaptive_grad_clip:
        adaptive_state = AdaptiveGradClipState(
            beta=args.grad_clip_ema_beta,
            clip_ratio=args.grad_clip_ratio,
            skip_ratio=args.grad_skip_ratio,
            min_clip=args.grad_clip_min,
            max_clip=args.grad_clip_max,
            warmup_steps=args.grad_clip_warmup_steps,
            static_clip=grad_clip if grad_clip != 0.0 else 1.0,
            skip_max=args.grad_skip_max,
            max_consecutive_skips=args.grad_clip_max_consecutive_skips,
            ema_runaway_factor=args.grad_clip_ema_runaway_factor,
            hard_raw_norm_limit=args.grad_clip_hard_raw_norm_limit,
        )
        logger.info(
            "adaptive_grad_clip=on beta=%.4f clip_ratio=%.2f skip_ratio=%.2f min_clip=%.4f "
            "max_clip=%.4f skip_max=%.4e warmup_steps=%s static_fallback=%.4f "
            "max_consecutive_skips=%s ema_runaway_factor=%.2f hard_raw_norm_limit=%.4e "
            "ema_contribution_cap=%.4e",
            adaptive_state.beta,
            adaptive_state.clip_ratio,
            adaptive_state.skip_ratio,
            adaptive_state.min_clip,
            adaptive_state.max_clip,
            adaptive_state.skip_max,
            adaptive_state.warmup_steps,
            adaptive_state.static_clip,
            adaptive_state.max_consecutive_skips,
            adaptive_state.ema_runaway_factor,
            adaptive_state.hard_raw_norm_limit,
            adaptive_state.ema_contribution_cap,
        )
    else:
        logger.info("adaptive_grad_clip=off static_grad_clip=%.4f", grad_clip)

    t_temp_1_sum_lr = 0
    t_temp_2_sum_data = 0
    t_temp_3_sum_batchdata = 0
    t_temp_4_sum_sync = 0
    t_temp_5_sum_forward = 0
    t_temp_6_sum_loss_gep = 0
    t_temp_7_sum_loss_gep_zero_prob = 0
    t_temp_8_sum_backward = 0
    t_temp_9_sum_step_update_grad = 0
    t_temp_10_sum_sync = 0

    t_temp_0 = time.time()
    last_completed_epoch = 0
    last_completed_step = 0
    last_completed_loss = 0.0
    last_saved_step = -1
    for epoch in range(resume_start_epoch, args.epoch):

        # Uncomment this if use DistributedSampler
        t_temp_2_data = time.time()
        if ddp:
            train_sampler.set_epoch(epoch)
            data_pt, file_paths = load_data_for_rank_with_DistributedFileSampler(
                train_data_filelist,
                train_sampler,
                rank,
                NODE_RANK,
                logger=logger)
            logger.info(f"Node: {NODE_RANK}, Rank: {rank}, Epoch: {epoch}, Data: {[x.split('/')[-1] for x in file_paths]}")
            full_num_batches = math.ceil(len(data_pt) / args.batch_size)
            step_tensor = torch.tensor([full_num_batches], dtype=torch.long, device=device)
            dist.all_reduce(step_tensor, op=dist.ReduceOp.MIN)
            max_batch_index = int(step_tensor.item())
            resume_batch_offset = resume_skip_batches if epoch == resume_start_epoch else 0
            if resume_batch_offset >= max_batch_index:
                logger.info(
                    "resume_skip_epoch_completed epoch=%s resume_skip_batches=%s max_batch_index=%s",
                    epoch + 1,
                    resume_batch_offset,
                    max_batch_index,
                )
                continue
            if resume_batch_offset:
                rows_before_skip = len(data_pt)
                skip_rows = min(resume_batch_offset * args.batch_size, rows_before_skip)
                data_pt = data_pt.iloc[skip_rows:].reset_index(drop=True)
                logger.info(
                    "resume_skip_batches epoch=%s skipped_batches=%s skipped_rows=%s rows_before=%s rows_after=%s",
                    epoch + 1,
                    resume_batch_offset,
                    skip_rows,
                    rows_before_skip,
                    len(data_pt),
                )
            nw = max(0, int(args.num_workers))
            dl_kw = dict(
                dataset=ParquetDataset(data_pt),
                batch_size=args.batch_size,
                collate_fn=collate_fn,
                shuffle=False,
                drop_last=False,
                num_workers=nw,
                pin_memory=args.pin_memory,
            )
            if nw > 0:
                dl_kw["prefetch_factor"] = max(1, int(args.prefetch_factor))
                dl_kw["persistent_workers"] = bool(args.persistent_workers)
            logger.info(
                "DataLoader rank=%s num_workers=%s prefetch_factor=%s persistent_workers=%s pin_memory=%s",
                rank,
                nw,
                dl_kw.get("prefetch_factor", "n/a"),
                dl_kw.get("persistent_workers", False),
                dl_kw["pin_memory"],
            )
            data_loader = DataLoader(**dl_kw)
        t_temp_2_data = time.time() - t_temp_2_data
        t_temp_2_sum_data += t_temp_2_data

        model.train()
        total_loss = 0.0
        total_loss_tensor = torch.zeros((), dtype=torch.float32, device=device)
        processed_batches = 0
        num_batches = len(data_loader) # 已经处理过 // args.batch_size
        logger.info(
            "Node: %s Rank: %s num_batches_remaining=%s max_batch_index=%s resume_batch_offset=%s",
            NODE_RANK,
            rank,
            num_batches,
            max_batch_index,
            resume_batch_offset,
        )

        lr_decay_epochs = float(args.lr_decay_epochs) if args.lr_decay_epochs else float(args.epoch)
        lr_decay_iters = int(math.ceil(
            math.ceil(max_batch_index / args.gradient_accumulation_steps) * lr_decay_epochs
        ))
        warmup_iters = int(lr_decay_iters * args.warmup_ratio) if args.warmup_ratio else args.warmup_iters
        if args.decay_lr:
            assert lr_decay_iters > warmup_iters, \
                f"lr_decay_iters ({lr_decay_iters}) must be > warmup_iters ({warmup_iters})"
        logger.info(
            "lr_schedule decay_lr=%s learning_rate=%s min_lr=%s lr_decay_epochs=%s "
            "lr_decay_iters=%s warmup_iters=%s",
            args.decay_lr,
            args.learning_rate,
            args.min_lr,
            lr_decay_epochs,
            lr_decay_iters,
            warmup_iters,
        )

        for batch_index, batch_data in enumerate(data_loader):
            absolute_batch_index = resume_batch_offset + batch_index
            absolute_batch_number = absolute_batch_index + 1
            t_temp_1_lr = 0.0
            t_temp_7_loss_other = 0.0
            loss_gep = None
            loss_zero_prob = None
            loss_gepc = None
            loss_gepc_zero_prob = None
            lossf = None
            loss_gep_value = None
            loss_zero_prob_value = None
            loss_gepc_value = None
            loss_gepc_zero_prob_value = None

            final_step_in_epoch = absolute_batch_number >= max_batch_index
            should_step = (
                absolute_batch_number % args.gradient_accumulation_steps == 0
                or final_step_in_epoch
            )
            should_log = (iter_num % max(1, args.log_interval) == 0) or final_step_in_epoch
            should_profile = (
                args.profile_interval > 0
                and ((iter_num % args.profile_interval == 0) or final_step_in_epoch)
            )
            should_collect_scalars = should_log or should_profile
            should_nan_check = (
                args.nan_check_interval > 0
                and ((iter_num % args.nan_check_interval == 0) or final_step_in_epoch)
            )
            should_log_memory = should_trace_step(
                args.memory_log_interval,
                absolute_batch_index,
                should_step=should_step,
                final_step=final_step_in_epoch,
            )
            should_log_shapes = should_trace_step(
                args.tensor_shape_log_interval,
                absolute_batch_index,
                should_step=should_step,
                final_step=final_step_in_epoch,
            )
            if should_log_memory:
                log_npu_memory(
                    logger,
                    "before_to_device",
                    rank=rank,
                    extra={"epoch": epoch + 1, "batch": absolute_batch_number},
                )

            t_temp_3_batchdata = time.time()
            input_gene_values = batch_data["values"].to(device, non_blocking=True)
            target_values = batch_data["target_values"].to(device, non_blocking=True)
            # print(f"rank: {rank} batch done")
            if config.use_batch_labels:
                input_batch_labels = batch_data["batch_labels"].to(device, non_blocking=True)
            if config.use_species_labels:
                input_species_labels = batch_data["species_labels"].to(device, non_blocking=True)
            if config.use_tissue_labels:
                input_tissue_labels = batch_data["tissue_labels"].to(device, non_blocking=True)
            if config.use_seqmethod_labels:
                input_seqmethod_labels = batch_data["seqmethod_labels"].to(device, non_blocking=True)
            if config.use_disease_labels:
                input_disease_labels = batch_data["disease_labels"].to(device, non_blocking=True)
            if config.use_sex_labels:
                input_sex_labels = batch_data["sex_labels"].to(device, non_blocking=True)
            if config.use_age_labels:
                input_age_labels = batch_data["age_labels"].to(device, non_blocking=True)
            # print(f"rank: {rank} batch to device done")
            t_temp_3_batchdata = time.time() - t_temp_3_batchdata
            t_temp_3_sum_batchdata += t_temp_3_batchdata
            if should_log_memory:
                log_npu_memory(
                    logger,
                    "after_to_device",
                    rank=rank,
                    extra={"epoch": epoch + 1, "batch": absolute_batch_number},
                )
            if ddp:
                model.require_backward_grad_sync = should_step
            t_temp_5_forward = time.time()
            if should_log_memory:
                log_npu_memory(
                    logger,
                    "before_forward",
                    rank=rank,
                    extra={"epoch": epoch + 1, "batch": absolute_batch_number},
                )
            with ctx:
                run_mvc = bool(args.train_mvc and config.do_mvc)
                outputs = model(
                    values=input_gene_values,
                    batch_labels=input_batch_labels if config.use_batch_labels else None,
                    species_labels=input_species_labels if config.use_species_labels else None,
                    tissue_labels=input_tissue_labels if config.use_tissue_labels else None,
                    seqmethod_labels=input_seqmethod_labels if config.use_seqmethod_labels else None,
                    disease_labels=input_disease_labels if config.use_disease_labels else None,
                    sex_labels=input_sex_labels if config.use_sex_labels else None,
                    age_labels=input_age_labels if config.use_age_labels else None,
                    CLS=False,
                    MVC=run_mvc,
                    output_hidden_states=False,
                    output_attentions=False, # can not both turned on with _attn_implementation of sdpa
                    )
                # print(f"rank: {rank} outputs done")
                if should_log_shapes:
                    logger.info(
                        "TENSOR_SHAPES epoch=%s batch=%s %s",
                        epoch + 1,
                        absolute_batch_number,
                        json.dumps(tensor_shape_summary(outputs), sort_keys=True),
                    )
                if should_log_memory:
                    log_npu_memory(
                        logger,
                        "after_forward",
                        rank=rank,
                        extra={"epoch": epoch + 1, "batch": absolute_batch_number},
                    )
                mask_positions = input_gene_values.eq(-1)
                t_temp_5_forward = time.time() - t_temp_5_forward
                t_temp_5_sum_forward += t_temp_5_forward

                t_temp_6_loss_gep = time.time()
                # compute loss
                loss = 0.0
                loss_gep = masked_mse_loss(outputs["model_output"],
                                           target_values,
                                           mask_positions)
                loss += loss_gep
                # print(f"rank: {rank} loss_gep done")
                t_temp_6_loss_gep = time.time() - t_temp_6_loss_gep
                t_temp_6_sum_loss_gep += t_temp_6_loss_gep

                t_temp_7_loss_other = time.time()
                if config.explicit_zero_prob:
                    if should_nan_check and torch.isnan(outputs["model_zero_prob"]).any():
                        raise ValueError("There are nan values in zero prob output!")
                    loss_zero_prob = criterion_neg_log_bernoulli(outputs["model_zero_prob"],
                                                                 target_values,
                                                                 mask_positions)
                    loss += loss_zero_prob

                    # print(f"rank: {rank} loss_zero_prob done")
                if "mvc_output" in outputs:
                    loss_gepc = masked_mse_loss(outputs["mvc_output"],
                                                target_values,
                                                mask_positions)
                    loss += loss_gepc

                    # print(f"rank: {rank} mvc_output done")
                if "mvc_output" in outputs and config.explicit_zero_prob:
                    loss_gepc_zero_prob = criterion_neg_log_bernoulli(outputs["mvc_zero_probs"],
                                                                      target_values,
                                                                      mask_positions)
                    loss += loss_gepc_zero_prob

                    # print(f"rank: {rank} loss_gepc_zero_prob done")
                loss = loss / args.gradient_accumulation_steps
                t_temp_7_loss_other = time.time() - t_temp_7_loss_other
                t_temp_7_sum_loss_gep_zero_prob += t_temp_7_loss_other
                if should_log_memory:
                    log_npu_memory(
                        logger,
                        "after_loss",
                        rank=rank,
                        extra={"epoch": epoch + 1, "batch": absolute_batch_number},
                    )

            # backward and optimization
            t_temp_8_backward = time.time()
            scaler.scale(loss).backward()
            # print(f"rank: {rank} scaler backward done")
            t_temp_8_backward = time.time() - t_temp_8_backward
            t_temp_8_sum_backward += t_temp_8_backward
            if should_log_memory:
                log_npu_memory(
                    logger,
                    "after_backward",
                    rank=rank,
                    extra={"epoch": epoch + 1, "batch": absolute_batch_number},
                )

            # for name, param in model.named_parameters():
            #     if param.grad is None:
            #         print(f"WARNING: {name} grad is None! ^^^^^^^^^^^^^^^^^^^^^^")
            t_temp_9_step_update_grad = time.time()
            t_temp_10_sync = 0
            grad_norm_raw_value = None
            grad_norm_ema_value = None
            grad_clip_threshold_value = None
            grad_skip_threshold_value = None
            grad_skip_max_value = None
            grad_hard_raw_norm_limit_value = None
            grad_ema_contribution_cap_value = None
            grad_action = "no_step"
            skipped_this_step = False
            if should_step:
                t_temp_1_lr = time.time()
                lr = get_lr(update_step,
                            args.min_lr,
                            warmup_iters,
                            args.learning_rate,
                            lr_decay_iters) if args.decay_lr else args.learning_rate
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                t_temp_1_lr = time.time() - t_temp_1_lr
                t_temp_1_sum_lr += t_temp_1_lr

                if adaptive_state is not None:
                    clip_threshold, skip_threshold = adaptive_state.thresholds()
                    grad_clip_threshold_value = clip_threshold
                    grad_skip_threshold_value = (
                        skip_threshold if math.isfinite(skip_threshold) else None
                    )
                    grad_skip_max_value = (
                        adaptive_state.skip_max if adaptive_state.skip_max > 0.0 else None
                    )
                    grad_hard_raw_norm_limit_value = (
                        adaptive_state.hard_raw_norm_limit
                        if adaptive_state.hard_raw_norm_limit > 0.0
                        else None
                    )
                    grad_ema_contribution_cap_value = (
                        adaptive_state.ema_contribution_cap
                        if adaptive_state.ema_contribution_cap > 0.0
                        else None
                    )
                    scaler.unscale_(optimizer)
                    raw_norm_tensor = torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=clip_threshold,
                        error_if_nonfinite=False,
                    )
                    raw_norm = float(raw_norm_tensor.detach())
                    grad_norm_raw_value = raw_norm
                    if not math.isfinite(raw_norm):
                        grad_action = "skip_nan"
                        skipped_this_step = True
                    elif math.isfinite(skip_threshold) and raw_norm > skip_threshold:
                        grad_action = "skip_norm"
                        skipped_this_step = True
                    elif raw_norm > clip_threshold:
                        grad_action = "clip"
                    else:
                        grad_action = "pass"
                    if skipped_this_step:
                        logger.warning(
                            "grad_skip update_step=%s raw_norm=%.4f clip_threshold=%.4f "
                            "skip_threshold=%s ema=%s reason=%s",
                            update_step,
                            raw_norm,
                            clip_threshold,
                            (f"{skip_threshold:.4f}" if math.isfinite(skip_threshold) else "inf"),
                            (f"{adaptive_state.ema:.4f}" if adaptive_state.ema is not None else "n/a"),
                            grad_action,
                        )
                    else:
                        scaler.step(optimizer)
                        adaptive_state.update(raw_norm)
                    scaler.update()
                    optimizer.zero_grad()
                    adaptive_state.note_action(grad_action)
                    # Surface clip streaks at the start, then every 50 in a row,
                    # so a slow EMA runaway (the failure mode we hit in prod) is
                    # visible BEFORE it cascades into 100% skips.
                    if grad_action == "clip" and (
                        adaptive_state.consecutive_clips == 1
                        or adaptive_state.consecutive_clips % 50 == 0
                    ):
                        logger.warning(
                            "grad_clipped update_step=%s raw_norm=%.4f clip_threshold=%.4f "
                            "ema=%s consecutive_clips=%d",
                            update_step,
                            raw_norm,
                            clip_threshold,
                            (f"{adaptive_state.ema:.4f}" if adaptive_state.ema is not None else "n/a"),
                            adaptive_state.consecutive_clips,
                        )
                    abort_reason = adaptive_state.safety_check(raw_norm)
                    if abort_reason:
                        logger.error(
                            "adaptive_grad_clip_abort update_step=%s raw_norm=%.4f ema=%s "
                            "clip_threshold=%.4f skip_threshold=%s consecutive_skips=%d "
                            "consecutive_clips=%d reason=%s",
                            update_step,
                            raw_norm,
                            (f"{adaptive_state.ema:.4f}" if adaptive_state.ema is not None else "n/a"),
                            clip_threshold,
                            (f"{skip_threshold:.4f}" if math.isfinite(skip_threshold) else "inf"),
                            adaptive_state.consecutive_skips,
                            adaptive_state.consecutive_clips,
                            abort_reason,
                        )
                        raise RuntimeError(
                            f"adaptive_grad_clip safety abort at update_step={update_step}: {abort_reason}"
                        )
                    grad_norm_ema_value = adaptive_state.ema
                elif grad_clip != 0.0:
                    scaler.unscale_(optimizer)
                    raw_norm_tensor = torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=grad_clip,
                        error_if_nonfinite=False,
                    )
                    raw_norm = float(raw_norm_tensor.detach())
                    grad_norm_raw_value = raw_norm
                    grad_clip_threshold_value = grad_clip
                    if not math.isfinite(raw_norm):
                        grad_action = "skip_nan"
                    elif raw_norm > grad_clip:
                        grad_action = "clip"
                    else:
                        grad_action = "pass"
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                else:
                    grad_action = "pass"
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                update_step += 1
                # print(f"rank: {rank} step update zero_grad done")
                if should_log_memory:
                    log_npu_memory(
                        logger,
                        "after_optimizer_step",
                        rank=rank,
                        extra={"epoch": epoch + 1, "batch": absolute_batch_number},
                    )

                if absolute_batch_number % save_step_interval == 0:
                    lossf = loss.item() * args.gradient_accumulation_steps
                    if rank == 0:
                        if ddp:
                            save_model(model.module, optimizer, epoch, absolute_batch_number, lossf, out_dir, rank, NODE_RANK, s3_remote_dir_path=args.s3_remote_dir_path, logger=logger)
                        else:
                            save_model(model, optimizer, epoch, absolute_batch_number, lossf, out_dir, rank, NODE_RANK, s3_remote_dir_path=args.s3_remote_dir_path, logger=logger)
                    metrics_writer.flush()
                    save_log_to_s3(args, out_dir, NODE_RANK, rank, logger=logger)
                    last_saved_step = absolute_batch_number
            else:
                pass
                # print(f"rank: {rank} not in step update zero_grad")
            t_temp_9_step_update_grad = time.time() - t_temp_9_step_update_grad
            t_temp_9_sum_step_update_grad += t_temp_9_step_update_grad


            t_temp_1 = time.time()
            dt = t_temp_1 - t_temp_0
            t_temp_0 = t_temp_1

            if iter_num % 100 == 0 and iter_num != 0:
                mfu = raw_model.estimate_mfu(args.batch_size * args.gradient_accumulation_steps, dt)
                runing_mfu = mfu if runing_mfu == -1.0 else 0.9 * runing_mfu + 0.1 * mfu

            local_batch_size = int(target_values.shape[0])
            step_ms = dt * 1000.0
            samples_per_s = local_batch_size / dt if dt > 0 else 0.0
            tokens_per_s = local_batch_size * args.seq_len / dt if dt > 0 else 0.0
            mfu_value = None if runing_mfu < 0 else runing_mfu
            total_loss_tensor += loss.detach() * args.gradient_accumulation_steps
            processed_batches += 1

            if should_collect_scalars:
                lossf = loss.item() * args.gradient_accumulation_steps
                loss_gep_value = loss_gep.item() if loss_gep is not None else None
                loss_zero_prob_value = loss_zero_prob.item() if loss_zero_prob is not None else None
                loss_gepc_value = loss_gepc.item() if loss_gepc is not None else None
                loss_gepc_zero_prob_value = (
                    loss_gepc_zero_prob.item() if loss_gepc_zero_prob is not None else None
                )

            local_metric_values = {
                "loss_total": lossf if lossf is not None else 0.0,
                "step_ms": step_ms,
                "data_load_s": t_temp_2_data,
                "grad_sync_s": t_temp_10_sync,
                "samples_per_s": samples_per_s,
                "tokens_per_s": tokens_per_s,
            }

            cluster_summary = distributed_step_summary(local_metric_values, device, world_size) if should_log else {}

            metric_row = {
                "time": datetime.datetime.now().isoformat(timespec="seconds"),
                "node_rank": NODE_RANK,
                "rank": rank,
                "local_rank": local_rank,
                "epoch": epoch + 1,
                "batch_index": absolute_batch_number,
                "num_batches": max_batch_index,
                "update_step": update_step,
                "should_step": should_step,
                "lr": lr,
                "loss_total": lossf,
                "loss_gep": loss_gep_value,
                "loss_zero_prob": loss_zero_prob_value,
                "loss_gepc": loss_gepc_value,
                "loss_gepc_zero_prob": loss_gepc_zero_prob_value,
                "step_ms": step_ms,
                "data_load_s": t_temp_2_data,
                "to_device_s": t_temp_3_batchdata,
                "forward_s": t_temp_5_forward,
                "loss_s": t_temp_6_loss_gep + t_temp_7_loss_other,
                "backward_s": t_temp_8_backward,
                "optimizer_s": t_temp_9_step_update_grad,
                "grad_sync_s": t_temp_10_sync,
                "samples_per_s": samples_per_s,
                "tokens_per_s": tokens_per_s,
                "mfu": mfu_value,
                "grad_norm_raw": grad_norm_raw_value,
                "grad_norm_ema": grad_norm_ema_value,
                "grad_clip_threshold": grad_clip_threshold_value,
                "grad_skip_threshold": grad_skip_threshold_value,
                "grad_skip_max": grad_skip_max_value,
                "grad_hard_raw_norm_limit": grad_hard_raw_norm_limit_value,
                "grad_ema_contribution_cap": grad_ema_contribution_cap_value,
                "grad_action": grad_action,
                "skipped": skipped_this_step,
                "skipped_count_cum": (adaptive_state.skipped if adaptive_state is not None else None),
                "clipped_count_cum": (adaptive_state.clipped if adaptive_state is not None else None),
                "consecutive_skips": (adaptive_state.consecutive_skips if adaptive_state is not None else None),
                "consecutive_clips": (adaptive_state.consecutive_clips if adaptive_state is not None else None),
                **cluster_summary,
            }
            metrics_writer.write(metric_row)

            if should_log:
                mfu_text = "NA" if mfu_value is None else f"{mfu_value * 100:.2f}%"
                logger.info(
                    "Node: %s, rank: %s, [e: %s, %s/%s], loss: %.4f, time %.2fms, mfu: %s, "
                    "cluster_loss_mean=%s, cluster_step_max_ms=%s",
                    NODE_RANK,
                    rank,
                    epoch + 1,
                    absolute_batch_number,
                    max_batch_index,
                    lossf,
                    step_ms,
                    mfu_text,
                    f"{cluster_summary['cluster_loss_total_mean']:.4f}" if cluster_summary else "NA",
                    f"{cluster_summary['cluster_step_ms_max']:.2f}" if cluster_summary else "NA",
                )
            if should_profile:
                logger.info(
                    "profile epoch=%s batch=%s lr_s=%.4f data_load_s=%.4f to_device_s=%.4f forward_s=%.4f "
                    "loss_s=%.4f backward_s=%.4f optimizer_s=%.4f grad_sync_s=%.4f",
                    epoch + 1,
                    absolute_batch_number,
                    t_temp_1_lr,
                    t_temp_2_data,
                    t_temp_3_batchdata,
                    t_temp_5_forward,
                    t_temp_6_loss_gep + t_temp_7_loss_other,
                    t_temp_8_backward,
                    t_temp_9_step_update_grad,
                    t_temp_10_sync,
                )

            iter_num += 1

            if args.max_train_steps > 0 and iter_num >= args.max_train_steps:
                logger.info("max_train_steps_reached=%s", args.max_train_steps)
                break

            if absolute_batch_number >= max_batch_index:
                break

        total_loss = (total_loss_tensor / max(1, processed_batches)).item()
        if processed_batches > 0:
            last_completed_epoch = epoch
            last_completed_step = absolute_batch_number
            last_completed_loss = total_loss
            if absolute_batch_number % save_step_interval != 0:
                logger.info("end_of_epoch_save epoch=%s step=%s", epoch + 1, absolute_batch_number)
                if rank == 0:
                    if ddp:
                        save_model(model.module, optimizer, epoch, absolute_batch_number, total_loss, out_dir, rank, NODE_RANK, s3_remote_dir_path=args.s3_remote_dir_path, logger=logger)
                    else:
                        save_model(model, optimizer, epoch, absolute_batch_number, total_loss, out_dir, rank, NODE_RANK, s3_remote_dir_path=args.s3_remote_dir_path, logger=logger)
                metrics_writer.flush()
                save_log_to_s3(args, out_dir, NODE_RANK, rank, logger=logger)
                last_saved_step = absolute_batch_number
        loss_info = f"Node: {NODE_RANK}, Rank: {rank}, Epoch [{epoch + 1}/{args.epoch}], average loss is: {total_loss:.4f} | Learning rate is: {lr}"
        logger.info(loss_info)
        if adaptive_state is not None:
            logger.info(
                "adaptive_grad_clip_summary epoch=%s ema=%s observed=%s skipped=%s clipped=%s "
                "max_consecutive_skips_seen=%s max_consecutive_clips_seen=%s",
                epoch + 1,
                (f"{adaptive_state.ema:.4f}" if adaptive_state.ema is not None else "n/a"),
                adaptive_state.observed,
                adaptive_state.skipped,
                adaptive_state.clipped,
                adaptive_state.max_consecutive_skips_seen,
                adaptive_state.max_consecutive_clips_seen,
            )
        time_checker = (
                        f"Node: {NODE_RANK} "
                        f"Rank: {rank} "
                        f"1_lr {t_temp_1_sum_lr:.4f} "
                        f"2_epoch_dataloder {t_temp_2_sum_data:.4f} "
                        f"3_step_batched_data {t_temp_3_sum_batchdata:.4f} "
                        # f"4_sync {t_temp_4_sum_sync:.4f} "
                        f"5_forward {t_temp_5_sum_forward:.4f} "
                        f"6_loss_gep {t_temp_6_sum_loss_gep:.4f} "
                        f"7_loss_gep_zero_prob {t_temp_7_sum_loss_gep_zero_prob:.4f} "
                        f"8_backward {t_temp_8_sum_backward:.4f} "
                        f"9_step_update_grad {t_temp_9_sum_step_update_grad:.4f} "
                        f"10_sync {t_temp_10_sum_sync:.4f}")
        logger.info(time_checker)
        total_loss = 0
        # 释放本 epoch 的 in-RAM 分片与 DataLoader：否则下个 epoch 的 ddf.compute()
        # 会在旧 DataFrame 仍存活时构建新分片，主机内存瞬时翻倍 → 边界 OOM
        del data_loader
        del data_pt
        gc.collect()
        if args.max_train_steps > 0 and iter_num >= args.max_train_steps:
            break

    metrics_writer.close()
    save_log_to_s3(args, out_dir, NODE_RANK, rank, logger=logger)

    NODE_RANK = get_node_rank()
    if args.skip_final_save:
        logger.info("skip_final_save=true")
    elif last_saved_step == last_completed_step:
        logger.info("final_save_skipped step=%s already_saved_at_epoch_end", last_completed_step)
    elif rank == 0:
        if ddp:
            save_model(model.module, optimizer, last_completed_epoch, last_completed_step, last_completed_loss, out_dir, rank, NODE_RANK, s3_remote_dir_path=args.s3_remote_dir_path, logger=logger)
        else:
            save_model(model, optimizer, last_completed_epoch, last_completed_step, last_completed_loss, out_dir, rank, NODE_RANK, s3_remote_dir_path=args.s3_remote_dir_path, logger=logger)
    save_log_to_s3(args, out_dir, NODE_RANK, rank, logger=logger)

def save_log_to_s3(args, out_dir, NODE_RANK, rank, logger=None):
    if not args.s3_remote_dir_path or not is_remote_output_path(args.s3_remote_dir_path):
        return
    if mox is None:
        log_or_print(logger, "warning", "moxing is not available; skip log upload to remote output path")
        return
    try:
        mox.file.mk_dir(args.s3_remote_dir_path)
        uploaded = []
        for filename in (
            f"loss_to_log.{NODE_RANK}-{rank}.txt",
            f"metrics.{NODE_RANK}-{rank}.jsonl",
            f"log.{NODE_RANK}-{rank}.txt",
        ):
            local_path = os.path.join(out_dir, filename)
            if not os.path.exists(local_path):
                continue
            remote_path = args.s3_remote_dir_path.strip("/") + "/" + filename
            mox.file.copy(local_path, remote_path)
            uploaded.append(remote_path)
        if uploaded:
            log_or_print(logger, "info", f"log_sync remote_paths={uploaded}")
    except Exception as e:
        log_or_print(logger, "exception", f"log_sync_failed error={e}", exc_info=True)

def save_model(model, optimizer, epoch, step, loss, out_dir, local_rank=None, NODE_RANK=None, savepath=None, s3_remote_dir_path=None, logger=None):
    if NODE_RANK is None:
        NODE_RANK = get_node_rank()
    if not savepath:
        savepath = "SC-node-{node:02d}-rank-{rank:02d}-epoch-{epoch:02d}-step-{step}-loss-{loss:.6f}.pt"
    save_path = savepath.format(
        node=int(NODE_RANK),
        rank=int(local_rank),
        epoch=int(epoch),
        step=str(step),
        loss=loss)
    save_other_path = save_path.replace(".pt", ".optimizer.pt")
    with torch.no_grad():
        torch.save(model.state_dict(), os.path.join(out_dir, save_path))
        torch.save(optimizer.state_dict(), os.path.join(out_dir, save_other_path))
    if s3_remote_dir_path and (s3_remote_dir_path.startswith("s3://") or s3_remote_dir_path.startswith("obs://")):
        if mox is not None:
            mox.file.mk_dir(s3_remote_dir_path)
            mox.file.copy(os.path.join(out_dir, save_path), s3_remote_dir_path.strip("/") + "/" + save_path)
            mox.file.copy(os.path.join(out_dir, save_other_path), s3_remote_dir_path.strip("/") + "/" + save_other_path)
            log_or_print(
                logger,
                "info",
                f'checkpoint_saved remote_model={s3_remote_dir_path.strip("/") + "/" + save_path} '
                f'remote_optimizer={s3_remote_dir_path.strip("/") + "/" + save_other_path}',
            )
            os.remove(os.path.join(out_dir, save_path))
            os.remove(os.path.join(out_dir, save_other_path))
        else:
            log_or_print(
                logger,
                "warning",
                f"moxing not available; checkpoints kept locally only: "
                f"{os.path.join(out_dir, save_path)} {os.path.join(out_dir, save_other_path)}"
            )
    else:
        log_or_print(logger, "info", f"checkpoint_saved model={save_path} optimizer={save_other_path}")


def main(args):
    data_path = args.data_path
    emb_path = args.emb_path
    out_dir = args.out_path
    seq_len = args.seq_len
    batch_size = args.batch_size  # if gradient_accumulation_steps > 1, this is the mini-batch size
    num_epochs = args.epoch

    # Learning rate setting
    learning_rate = args.learning_rate
    # lr_decay_iters = args.lr_decay_iters
    # max_iters = args.max_iters
    min_lr = args.min_lr
    decay_lr = args.decay_lr
    warmup_iters = args.warmup_iters
    warmup_ratio = args.warmup_ratio
    weight_decay = args.weight_decay
    beta1 = args.beta1
    beta2 = args.beta2

    grad_clip = args.grad_clip  # clip gradients at this value, or disable if == 0.0
    compile = args.compile  # complie model if pytorch > 2.0
    gradient_accumulation_steps = args.gradient_accumulation_steps  # used to simulate larger batch sizes
    backend = args.backend
    device = args.device
    device_type = args.device_type

    dtype, ptdtype = resolve_amp_dtype(args)  # note: float16 data type will automatically use a GradScaler
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # time of start
    t0 = time.time()
    src = np.arange(1, seq_len + 1)
    #######################################
    # original reading section
    esm_embeddings = np.load(emb_path + "/2nd_run_macrogene_features_sum_esm2.npy")
    desc_embeddings = np.load(emb_path + "/2nd_run_macrogene_features_sum_gene_desc.npy")
    dna_embeddings = np.load(emb_path + "/2nd_run_macrogene_features_sum_dnaseq.npy")
    #######################################

    # Build vocabulary
    gene_ids = ["gene_" + str(id) for id in range(1, seq_len + 1)]
    cls_token = ["<cls>"]
    special_tokens = ["<cls>", "<sep>", "<pad>", "<mask>"]
    combined_vocab = cls_token + gene_ids + special_tokens
    vocab = {token: idx for idx, token in enumerate(combined_vocab)}
    config = build_bertconfig(vocab, seq_len, args)
    runtime_overrides = apply_runtime_overrides(config, args)
    collate_fn = CustomCollate_3GeneEmb(config=config,
                                        genes=src,
                                        esm_embeddings=esm_embeddings,
                                        desc_embeddings=desc_embeddings,
                                        dna_embeddings=dna_embeddings,
                                        return_static_inputs=False)
    
    out_dir = out_dir.format(hidden_size=config.hidden_size, num_hidden_layers=config.num_hidden_layers,
                             num_attention_heads=config.num_attention_heads, hidden_dropout_prob=config.hidden_dropout_prob,
                             learning_rate=learning_rate, min_lr=min_lr, weight_decay=weight_decay, warmup_ratio=warmup_ratio)
    if args.s3_remote_dir_path and is_remote_output_path(args.s3_remote_dir_path):
        args.s3_remote_dir_path = os.path.join(
            "/".join(args.s3_remote_dir_path.split("/")[:-1]),
            remote_output_name(out_dir) + "_" + args.s3_remote_dir_path.split("/")[-1],
        )
    out_dir = resolve_output_dir(out_dir)
    if not os.path.exists(out_dir):
        raise FileNotFoundError(f"Output directory {out_dir} does not exist!")
        sys.exit(1)

    # set up for multiple GPUs run
    ddp, rank, local_rank, master_process, world_size = setup_ddp(backend=backend, device_type=device_type)
    device = f"npu:{local_rank}"
    tokens_per_epoch = gradient_accumulation_steps * world_size * batch_size * seq_len
    NODE_RANK = get_node_rank()
    logger = setup_rank_logger(
        out_dir=out_dir,
        node_rank=NODE_RANK,
        rank=rank,
        local_rank=local_rank,
        master_process=master_process,
        log_level=args.log_level,
        log_all_ranks=args.log_all_ranks,
        append=args.append_output_logs,
    )
    logger.info("sys.argv=%s", sys.argv)
    logger.info("args=%s", json.dumps(vars(args), ensure_ascii=False, indent=2, default=json_default))
    logger.info("runtime_config_overrides=%s", json.dumps(runtime_overrides, sort_keys=True))
    logger.info("model_config=%s", config)
    logger.info(
        "experiment_controls amp_dtype=%s static_gene_dtype=%s train_mvc=%s gradient_checkpointing=%s "
        "memory_log_interval=%s tensor_shape_log_interval=%s max_train_steps=%s skip_final_save=%s "
        "ddp_find_unused_parameters=%s append_output_logs=%s",
        dtype,
        args.static_gene_dtype,
        args.train_mvc,
        args.gradient_checkpointing,
        args.memory_log_interval,
        args.tensor_shape_log_interval,
        args.max_train_steps,
        args.skip_final_save,
        args.ddp_find_unused_parameters,
        args.append_output_logs,
    )
    logger.info("output_dir=%s remote_output_dir=%s", os.path.abspath(out_dir), args.s3_remote_dir_path)
    logger.info(
        "ddp=%s rank=%s local_rank=%s world_size=%s master_process=%s device=%s tokens_per_epoch=%s",
        ddp,
        rank,
        local_rank,
        world_size,
        master_process,
        device,
        tokens_per_epoch,
    )

    # Since we do not use DistributedSampler to distribute data, so commenting it
    train_data_filelist = get_files(data_path, args.num_of_used_data)  # get all parquet files, use for PreindexedParquetDataset
    # train_data_filelist = load_data_for_total(data_path) # get all parquet files, use for ParquetDataset
    train_sampler = DistributedFileSampler(train_data_filelist,
                                     num_replicas=world_size,
                                     rank=rank,
                                     drop_last=True)

    model = BERTForPreTraining(config).to(device)
    if args.gradient_checkpointing:
        model.gradient_checkpointing = True
        model.bert.gradient_checkpointing = True
    if args.memory_log_interval > 0:
        reset_npu_peak_memory_stats(logger, "after_model_to")
        log_npu_memory(logger, "after_model_to", rank=rank)
    model.set_static_gene_inputs(
        src,
        esm_embeddings,
        desc_embeddings,
        dna_embeddings,
        cls_id=vocab["<cls>"],
        append_cls=True,
        dtype=resolve_torch_dtype(args.static_gene_dtype, ptdtype),
    )
    if args.memory_log_interval > 0:
        log_npu_memory(logger, "after_static_gene_inputs", rank=rank)
    load_model_checkpoint(model, args.init_model_path, device, logger, ddp=ddp, rank=rank)
    # optimizer and initialize a GradSclaer.
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2))
    load_optimizer_checkpoint(optimizer, args.init_optimizer_path, device, logger)
    if args.memory_log_interval > 0:
        log_npu_memory(logger, "after_optimizer_init", rank=rank)
    # NanoGPT's way to initialize optimizer. But using this with turn use_fused = False in model
    # optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
    scaler = GradScaler(enabled=(dtype == 'float16'))
    # compile the model
    if compile:
        if master_process:
            logger.info("%s compiling the model...(take a ~minute)", compile)
        unoptimized_model = model
        model = torch.compile(model)  # requires P
    if ddp:
        model = DDP(
            model,
            device_ids=[local_rank],
            find_unused_parameters=args.ddp_find_unused_parameters,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
        )
    if args.memory_log_interval > 0:
        log_npu_memory(logger, "after_ddp_wrap", rank=rank)

    ctx = nullcontext() if device_type == 'cpu' or dtype == 'float32' else torch.autocast(device_type=device_type, dtype=ptdtype)
    train_loop(args, model, ddp, rank, local_rank, optimizer, train_data_filelist, train_sampler, device, config, ctx, scaler, grad_clip, out_dir, collate_fn, world_size, logger)

    t1 = time.time()
    logger.info("Complete pretraining! Running time is %.2fs.", t1 - t0)


def argumentparser():
    parser = argparse.ArgumentParser(description="Distributed Multi-Node Multi-GPU parquet training")
    parser.add_argument("--data_path",
                        type=str,
                        required=True)
    parser.add_argument("--num_of_used_data",
                        type=int,
                        required=0)
    parser.add_argument("--emb_path",
                        type=str,
                        required=True)
    parser.add_argument("--config_json",
                        type=str,
                        default=None,
                        help="Strict model config JSON. Overrides seq_len, model structure, and label settings.")
    parser.add_argument("--seq_len",
                        type=int,
                        default=None)
    parser.add_argument("--out_path",
                        type=str,
                        default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch_size",
                        type=int,
                        default=64)
    parser.add_argument("--epoch",
                        type=int,
                        default=10)
    parser.add_argument("--gradient_accumulation_steps",
                        type=int,
                        default=4)
    parser.add_argument("--learning_rate",
                        type=float,
                        default=1e-4)
    # parser.add_argument('--lr_decay_iters',
    #                     type=int,
    #                     default=600000)
    # parser.add_argument('--max_iters',
    #                     type=int,
    #                     default=999999999)
    parser.add_argument('--min_lr',
                        type=float,
                        default=6e-5)
    parser.add_argument('--decay_lr',
                        type=str2bool,
                        default=True)
    parser.add_argument('--warmup_iters',
                        type=int,
                        default=2000)
    parser.add_argument('--warmup_ratio',
                        type=float,
                        default=0.01,
                        help='if warmup_ratio, then discard warmup_iters')
    parser.add_argument('--lr_decay_epochs',
                        type=float,
                        default=0.0,
                        help="Epoch-equivalent horizon for the LR decay schedule. "
                             "0 means use --epoch. Set 1 with --epoch=5 to finish "
                             "the 1-epoch decay schedule by the end of epoch 1.")
    parser.add_argument('--weight_decay',
                        type=float,
                        default=1e-1)
    parser.add_argument('--save_data_interval',
                        type=int,
                        default=5120000)
    parser.add_argument('--beta1',
                        type=float,
                        default=0.9)
    parser.add_argument('--beta2',
                        type=float,
                        default=0.95)
    parser.add_argument('--grad_clip',
                        type=float,
                        default=1.0,
                        help="Static global L2 grad-norm clip. Used as fallback during adaptive warmup or when adaptive is off. 0 disables.")
    parser.add_argument('--adaptive_grad_clip',
                        type=str2bool,
                        default=True,
                        help="Enable EMA-based adaptive grad-norm clipping + skip-on-spike. Pass --adaptive_grad_clip=false to disable.")
    parser.add_argument('--grad_clip_ema_beta',
                        type=float,
                        default=0.98,
                        help="EMA smoothing factor for tracking grad-norm baseline. Higher = smoother / slower to react.")
    parser.add_argument('--grad_clip_ratio',
                        type=float,
                        default=3.0,
                        help="Adaptive clip threshold = ema * grad_clip_ratio. Grads with norm above this get rescaled.")
    parser.add_argument('--grad_skip_ratio',
                        type=float,
                        default=10.0,
                        help="Skip threshold = ema * grad_skip_ratio. Updates with norm above this are dropped entirely (no optimizer.step).")
    parser.add_argument('--grad_skip_max',
                        type=float,
                        default=0.0,
                        help="Optional absolute upper bound on the adaptive skip threshold. "
                             "0 disables. This is independent from --grad_clip_max; use it as the raw-norm fuse scale.")
    parser.add_argument('--grad_clip_min',
                        type=float,
                        default=0.5,
                        help="Lower bound on the adaptive clip threshold (prevents over-tight clipping when EMA is small).")
    parser.add_argument('--grad_clip_max',
                        type=float,
                        default=1000.0,
                        help="Upper bound on the adaptive clip threshold (prevents runaway thresholds during transient spikes).")
    parser.add_argument('--grad_clip_warmup_steps',
                        type=int,
                        default=200,
                        help="Number of optimizer steps to observe before adaptive thresholds take over. Static --grad_clip is used during warmup.")
    parser.add_argument('--grad_clip_max_consecutive_skips',
                        type=int,
                        default=50,
                        help="Safety net: abort training when this many optimizer steps are skipped back-to-back. "
                             "0 disables. Default 50 ~= 5 min of wasted compute at the 500M cluster step time.")
    parser.add_argument('--grad_clip_ema_runaway_factor',
                        type=float,
                        default=2.0,
                        help="Safety net: abort when ema > ema_runaway_factor * (grad_skip_max / grad_skip_ratio). "
                             "0 disables. Active only when --grad_skip_max is set.")
    parser.add_argument('--grad_clip_hard_raw_norm_limit',
                        type=float,
                        default=0.0,
                        help="Safety net: abort when any single raw grad norm exceeds this absolute cap. "
                             "0 disables. Use as a physical fuse far above the normal regime (e.g. 1e11 for 500M).")
    parser.add_argument('--compile',
                        type=str2bool,
                        default=False)
    parser.add_argument('--backend',
                        type=str,
                        default='hccl')
    parser.add_argument('--device',
                        type=str,
                        default='npu')
    parser.add_argument('--device_type',
                        type=str,
                        default='npu')
    parser.add_argument('--s3_remote_dir_path',
                        type=str,
                        default="")
    parser.add_argument('--log_interval',
                        type=int,
                        default=10,
                        help="Write human-readable step summaries every N local batches.")
    parser.add_argument('--profile_interval',
                        type=int,
                        default=100,
                        help="Write detailed timing profiles every N local batches. Use 0 to disable.")
    parser.add_argument('--nan_check_interval',
                        type=int,
                        default=0,
                        help="Check output tensors for NaN every N local batches. Use 0 to disable.")
    parser.add_argument('--metrics_flush_interval',
                        type=int,
                        default=100,
                        help="Flush JSONL/CSV metric files every N rows.")
    parser.add_argument(
        '--num_workers',
        type=int,
        default=8,
        help="DataLoader workers per rank. Lower values reduce host RAM from worker processes and prefetch buffers.",
    )
    parser.add_argument(
        '--prefetch_factor',
        type=int,
        default=1,
        help="Batches prefetched per worker when num_workers>0. Default 1 lowers RAM vs PyTorch default 2.",
    )
    parser.add_argument(
        '--persistent_workers',
        type=str2bool,
        default=True,
        help="When num_workers>0, keep worker processes across epochs (recommended). Ignored if num_workers is 0.",
    )
    parser.add_argument(
        '--pin_memory',
        type=str2bool,
        default=True,
        help="DataLoader pin_memory (pinned CPU pages for faster H2D-style copies). Default True.",
    )
    parser.add_argument('--log_level',
                        type=str,
                        default="INFO")
    parser.add_argument('--log_all_ranks',
                        type=str2bool,
                        default=False,
                        help="If true, all ranks also write summaries to stdout; otherwise only global rank 0 does.")
    parser.add_argument('--amp_dtype',
                        type=str,
                        choices=["auto", "float16", "bfloat16", "float32"],
                        default="auto",
                        help="Autocast dtype. Default auto preserves the current cuda-based float16/bfloat16 selection.")
    parser.add_argument('--static_gene_dtype',
                        type=str,
                        choices=["float32", "float16", "bfloat16", "amp"],
                        default="float32",
                        help="dtype for static gene embedding buffers. Default float32 preserves current behavior.")
    parser.add_argument('--runtime_attn_implementation',
                        type=str,
                        choices=["eager", "sdpa"],
                        default="sdpa",
                        help="Optional runtime override for config._attn_implementation.")
    parser.add_argument('--runtime_explicit_zero_prob',
                        type=str,
                        default=None,
                        help="Optional runtime override for config.explicit_zero_prob.")
    parser.add_argument('--runtime_do_mvc',
                        type=str,
                        default=None,
                        help="Optional runtime override for config.do_mvc before model construction.")
    parser.add_argument('--runtime_chunk_size_feed_forward',
                        type=int,
                        default=None,
                        help="Optional runtime override for config.chunk_size_feed_forward.")
    parser.add_argument('--train_mvc',
                        type=str2bool,
                        default=True,
                        help="Whether to execute the MVC forward/loss branch. Default true preserves current behavior.")
    parser.add_argument('--gradient_checkpointing',
                        type=str2bool,
                        default=True,
                        help="Enable transformer layer activation checkpointing. Default true reduces activation memory.")
    parser.add_argument('--memory_log_interval',
                        type=int,
                        default=0,
                        help="Log torch-npu allocated/reserved memory every N local batches. 0 disables.")
    parser.add_argument('--tensor_shape_log_interval',
                        type=int,
                        default=0,
                        help="Log output tensor shapes every N local batches. 0 disables.")
    parser.add_argument('--max_train_steps',
                        type=int,
                        default=0,
                        help="Stop after N local batches per rank. 0 runs the full epoch.")
    parser.add_argument('--skip_final_save',
                        type=str2bool,
                        default=False,
                        help="Skip final checkpoint save. Default false preserves current behavior.")
    parser.add_argument('--ddp_find_unused_parameters',
                        type=str2bool,
                        default=False,
                        help="DDP find_unused_parameters. Default false preserves current behavior.")
    parser.add_argument('--init_model_path',
                        type=str,
                        default="",
                        help="Optional model state_dict checkpoint to load before training. "
                             "Rank 0 reads it and broadcasts the loaded weights under DDP.")
    parser.add_argument('--init_optimizer_path',
                        type=str,
                        default="",
                        help="Optional optimizer state_dict checkpoint to load after optimizer creation. "
                             "Use only for exact resume-style runs.")
    parser.add_argument('--resume_update_step',
                        type=int,
                        default=0,
                        help="Initial update_step used for LR schedule and metrics. "
                             "0 starts a fresh schedule from the loaded weights.")
    parser.add_argument('--resume_start_epoch',
                        type=int,
                        default=0,
                        help="Zero-based epoch index to start from when resuming within an output stream.")
    parser.add_argument('--resume_skip_batches',
                        type=int,
                        default=0,
                        help="Batches already completed in resume_start_epoch. "
                             "The first trained batch will be resume_skip_batches + 1.")
    parser.add_argument('--append_output_logs',
                        type=str2bool,
                        default=False,
                        help="Append log/metric files instead of truncating them. "
                             "Use with a seeded resume output directory.")
    parser.add_argument('--hidden_size',
        type=int,
        default=None)
    parser.add_argument('--num_hidden_layers',
        type=int,
        default=None)
    parser.add_argument('--num_attention_heads',
        type=int,
        default=None)
    parser.add_argument('--intermediate_size',
        type=int,
        default=None)
    parser.add_argument('--hidden_act',
        type=str,
        default=None)
    parser.add_argument('--hidden_dropout_prob',
        type=float,
        default=None)
    parser.add_argument('--cell_hidden_size',
        type=int,
        default=None)
    parser.add_argument('--attention_probs_dropout_prob',
        type=float,
        default=None)
    parser.add_argument('--type_vocab_size',
        type=int,
        default=None)
    parser.add_argument('--initializer_range',
        type=float,
        default=None)
    parser.add_argument('--layer_norm_eps',
        type=float,
        default=None)
    parser.add_argument('--_attn_implementation',
        type=str,
        default=None)
    parser.add_argument('--use_batch_labels',
        type=str,
        default=None)
    parser.add_argument('--num_batch_labels',
        type=int,
        default=None)
    parser.add_argument('--use_species_labels',
        type=str,
        default=None)
    parser.add_argument('--num_species_labels',
        type=int,
        default=None)
    parser.add_argument('--use_tissue_labels',
        type=str,
        default=None)
    parser.add_argument('--num_tissue_labels',
        type=int,
        default=None)
    parser.add_argument('--use_seqmethod_labels',
        type=str,
        default=None)
    parser.add_argument('--num_seqmethod_labels',
        type=int,
        default=None)
    parser.add_argument('--use_disease_labels',
        type=str,
        default=None)
    parser.add_argument('--num_disease_labels',
        type=int,
        default=None)
    parser.add_argument('--use_age_labels',
        type=str,
        default=None)
    parser.add_argument('--num_age_labels',
        type=int,
        default=None)
    parser.add_argument('--use_sex_labels',
        type=str,
        default=None)
    parser.add_argument('--num_sex_labels',
        type=int,
        default=None)
    parser.add_argument('--cell_emb_style',
        type=str,
        default=None)
    parser.add_argument('--chunk_size_feed_forward',
        type=int,
        default=None)
    parser.add_argument('--explicit_zero_prob',
        type=str,
        default=None)


    args = parser.parse_args()
    try:
        apply_config_json(args)
    except ValueError as exc:
        parser.error(str(exc))
    return args


if __name__ == "__main__":
    args = argumentparser()
    main(args)

"""
# para 1.1 - 1.4
--learning_rate=0.0005 --min_lr=0.00006 --weight_decay=0.1 --warmup_ratio=0.05 --hidden_dropout_prob=0.1
--learning_rate=0.0001 --min_lr=0.00006 --weight_decay=0.1 --warmup_ratio=0.05 --hidden_dropout_prob=0.1
--learning_rate=0.00005 --min_lr=0.00006 --weight_decay=0.1 --warmup_ratio=0.05 --hidden_dropout_prob=0.1
--learning_rate=0.00001 --min_lr=0.00006 --weight_decay=0.1 --warmup_ratio=0.05 --hidden_dropout_prob=0.1

# para 2.1 - 2.3
--learning_rate=0.0001 --min_lr=0.000001 --weight_decay=0.1 --warmup_ratio=0.05 --hidden_dropout_prob=0.1
--learning_rate=0.0001 --min_lr=0.00006 --weight_decay=0.1 --warmup_ratio=0.05 --hidden_dropout_prob=0.1
--learning_rate=0.0001 --min_lr=0.00001 --weight_decay=0.1 --warmup_ratio=0.05 --hidden_dropout_prob=0.1

# para 3.1 - 3.3
--learning_rate=0.00001 --min_lr=0.00006 --weight_decay=0.1 --warmup_ratio=0.01 --hidden_dropout_prob=0.1
--learning_rate=0.00001 --min_lr=0.00006 --weight_decay=0.1 --warmup_ratio=0.02 --hidden_dropout_prob=0.1
--learning_rate=0.00001 --min_lr=0.00006 --weight_decay=0.1 --warmup_ratio=0.05 --hidden_dropout_prob=0.1

# para 4.1 - 4.4
--learning_rate=0.00001 --min_lr=0.00006 --weight_decay=0.1 --warmup_ratio=0.01 --hidden_dropout_prob=0
--learning_rate=0.00001 --min_lr=0.00006 --weight_decay=0.1 --warmup_ratio=0.01 --hidden_dropout_prob=0.01
--learning_rate=0.00001 --min_lr=0.00006 --weight_decay=0.1 --warmup_ratio=0.01 --hidden_dropout_prob=0.05
--learning_rate=0.00001 --min_lr=0.00006 --weight_decay=0.1 --warmup_ratio=0.01 --hidden_dropout_prob=0.1



# para 1.1 - 1.4 SpeciesLLM-exp2-para1_4-fixminlr
--learning_rate=0.0005  --min_lr=0.00006  --weight_decay=0.1 --warmup_ratio=0.05 --hidden_dropout_prob=0.1 # 不用重跑
--learning_rate=0.0001  --min_lr=0.00006  --weight_decay=0.1 --warmup_ratio=0.05 --hidden_dropout_prob=0.1 # 不用重跑
--learning_rate=0.00005 --min_lr=0.000005 --weight_decay=0.1 --warmup_ratio=0.05 --hidden_dropout_prob=0.1
--learning_rate=0.00001 --min_lr=0.000001 --weight_decay=0.1 --warmup_ratio=0.05 --hidden_dropout_prob=0.1

# para 2.1 - 2.3
--learning_rate=0.0001  --min_lr=0.000001 --weight_decay=0.1 --warmup_ratio=0.05 --hidden_dropout_prob=0.1 # 不用重跑
--learning_rate=0.0001  --min_lr=0.00006  --weight_decay=0.1 --warmup_ratio=0.05 --hidden_dropout_prob=0.1 # 不用重跑
--learning_rate=0.0001  --min_lr=0.00001  --weight_decay=0.1 --warmup_ratio=0.05 --hidden_dropout_prob=0.1 # 不用重跑

# para 3.1 - 3.3
--learning_rate=0.00001 --min_lr=0.000001 --weight_decay=0.1 --warmup_ratio=0.01 --hidden_dropout_prob=0.1
--learning_rate=0.00001 --min_lr=0.000001 --weight_decay=0.1 --warmup_ratio=0.02 --hidden_dropout_prob=0.1
--learning_rate=0.00001 --min_lr=0.000001 --weight_decay=0.1 --warmup_ratio=0.05 --hidden_dropout_prob=0.1

# para 4.1 - 4.4
--learning_rate=0.00001 --min_lr=0.000001 --weight_decay=0.1 --warmup_ratio=0.01 --hidden_dropout_prob=0
--learning_rate=0.00001 --min_lr=0.000001 --weight_decay=0.1 --warmup_ratio=0.01 --hidden_dropout_prob=0.01
--learning_rate=0.00001 --min_lr=0.000001 --weight_decay=0.1 --warmup_ratio=0.01 --hidden_dropout_prob=0.05
--learning_rate=0.00001 --min_lr=0.000001 --weight_decay=0.1 --warmup_ratio=0.01 --hidden_dropout_prob=0.1
"""
