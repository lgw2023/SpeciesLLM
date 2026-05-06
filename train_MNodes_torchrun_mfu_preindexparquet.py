import os
import sys
import dask
import json
import time
import math
import glob
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


def setup_rank_logger(out_dir, node_rank, rank, local_rank, master_process, log_level="INFO", log_all_ranks=False):
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
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
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


class StreamingMetricsWriter:
    def __init__(self, out_dir, node_rank, rank, flush_interval=100):
        self.flush_interval = max(1, int(flush_interval))
        self.rows_since_flush = 0
        self.jsonl_path = os.path.join(out_dir, f"metrics.{node_rank}-{rank}.jsonl")
        self.csv_path = os.path.join(out_dir, f"loss_to_log.{node_rank}-{rank}.txt")
        self.jsonl_file = open(self.jsonl_path, "w", encoding="utf-8")
        self.csv_file = open(self.csv_path, "w", newline="", encoding="utf-8")
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=METRIC_FIELDNAMES)
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
        torch.distributed.init_process_group(backend=backend)
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        torch_npu.npu.set_device(local_rank)
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
    iter_num, update_step, runing_mfu = 0, 0, -1.0

    save_step_interval = max(1, int(args.save_data_interval / world_size / args.batch_size))
    NODE_RANK = get_node_rank()
    metrics_writer = StreamingMetricsWriter(
        out_dir=out_dir,
        node_rank=NODE_RANK,
        rank=rank,
        flush_interval=args.metrics_flush_interval,
    )
    logger.info(
        "save_step_interval=%s save_data_interval=%s world_size=%s batch_size=%s",
        save_step_interval,
        args.save_data_interval,
        world_size,
        args.batch_size,
    )
    lr = optimizer.param_groups[0]["lr"]

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
    for epoch in range(args.epoch):

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
            data_loader = DataLoader(dataset=ParquetDataset(data_pt),
                batch_size=args.batch_size,
                collate_fn=collate_fn,
                shuffle=False,
                drop_last=False,
                num_workers=min(len(os.sched_getaffinity(0)), args.batch_size // 2),
                pin_memory=True)
        t_temp_2_data = time.time() - t_temp_2_data
        t_temp_2_sum_data += t_temp_2_data

        model.train()
        total_loss = 0.0
        total_loss_tensor = torch.zeros((), dtype=torch.float32, device=device)
        processed_batches = 0
        num_batches = len(data_loader) # 已经处理过 // args.batch_size

        # 确保 local_step 是一个 tensor，并且在 GPU 上或 CPU 上与进程环境一致
        step_tensor = torch.tensor([num_batches], dtype=torch.long, device=device)
        # 使用 all_reduce 进行全局最小值操作
        dist.all_reduce(step_tensor, op=dist.ReduceOp.MIN)
        # 返回全局最小值
        max_batch_index = int(step_tensor.item())
        logger.info("Node: %s Rank: %s num_batches=%s max_batch_index=%s", NODE_RANK, rank, num_batches, max_batch_index)

        total_update_steps = math.ceil(max_batch_index / args.gradient_accumulation_steps) * args.epoch
        lr_decay_iters = total_update_steps
        warmup_iters = int(total_update_steps * args.warmup_ratio) if args.warmup_ratio else args.warmup_iters
        if args.decay_lr:
            assert lr_decay_iters > warmup_iters, \
                f"lr_decay_iters ({lr_decay_iters}) must be > warmup_iters ({warmup_iters})"

        for batch_index, batch_data in enumerate(data_loader):
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

            final_step_in_epoch = (batch_index + 1) >= max_batch_index
            should_step = (
                (batch_index + 1) % args.gradient_accumulation_steps == 0
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
            if ddp:
                model.require_backward_grad_sync = should_step
            t_temp_5_forward = time.time()
            with ctx:
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
                    MVC=True,
                    output_hidden_states=False,
                    output_attentions=False, # can not both turned on with _attn_implementation of sdpa
                    )
                # print(f"rank: {rank} outputs done")
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

            # backward and optimization
            t_temp_8_backward = time.time()
            scaler.scale(loss).backward()
            # print(f"rank: {rank} scaler backward done")
            t_temp_8_backward = time.time() - t_temp_8_backward
            t_temp_8_sum_backward += t_temp_8_backward

            # for name, param in model.named_parameters():
            #     if param.grad is None:
            #         print(f"WARNING: {name} grad is None! ^^^^^^^^^^^^^^^^^^^^^^")
            t_temp_9_step_update_grad = time.time()
            t_temp_10_sync = 0
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

                if grad_clip != 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                update_step += 1
                # print(f"rank: {rank} step update zero_grad done")

                if (batch_index + 1) % save_step_interval == 0:
                    lossf = loss.item() * args.gradient_accumulation_steps
                    if ddp:
                        save_model(model.module, optimizer, epoch, batch_index + 1, lossf, out_dir, rank, NODE_RANK, s3_remote_dir_path=args.s3_remote_dir_path, logger=logger)
                    else:
                        save_model(model, optimizer, epoch, batch_index + 1, lossf, out_dir, rank, NODE_RANK, s3_remote_dir_path=args.s3_remote_dir_path, logger=logger)
                    metrics_writer.flush()
                    save_log_to_s3(args, out_dir, NODE_RANK, rank, logger=logger)
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
                "batch_index": batch_index + 1,
                "num_batches": num_batches,
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
                    batch_index + 1,
                    num_batches,
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
                    batch_index + 1,
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

            if (batch_index + 1) >= max_batch_index:
                break

        total_loss = (total_loss_tensor / max(1, processed_batches)).item()
        loss_info = f"Node: {NODE_RANK}, Rank: {rank}, Epoch [{epoch + 1}/{args.epoch}], average loss is: {total_loss:.4f} | Learning rate is: {lr}"
        logger.info(loss_info)
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

    metrics_writer.close()
    save_log_to_s3(args, out_dir, NODE_RANK, rank, logger=logger)

    NODE_RANK = get_node_rank()
    if ddp:
        save_model(model.module, optimizer, args.epoch+1, 0, 0, out_dir, rank, NODE_RANK, s3_remote_dir_path=args.s3_remote_dir_path, logger=logger)
    else:
        save_model(model, optimizer, args.epoch+1, 0, 0, out_dir, rank, NODE_RANK, s3_remote_dir_path=args.s3_remote_dir_path, logger=logger)
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

    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[
        dtype]  # note: float16 data type will automatically use a GradScaler
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
    )
    logger.info("sys.argv=%s", sys.argv)
    logger.info("args=%s", json.dumps(vars(args), ensure_ascii=False, indent=2, default=json_default))
    logger.info("model_config=%s", config)
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
    model.set_static_gene_inputs(
        src,
        esm_embeddings,
        desc_embeddings,
        dna_embeddings,
        cls_id=vocab["<cls>"],
        append_cls=True,
        dtype=torch.float32,
    )
    # optimizer and initialize a GradSclaer.
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2))
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
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False, broadcast_buffers=False)

    ctx = nullcontext() if device_type == 'cpu' else torch.autocast(device_type=device_type, dtype=ptdtype)
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
                        type=bool,
                        default=True)
    parser.add_argument('--warmup_iters',
                        type=int,
                        default=2000)
    parser.add_argument('--warmup_ratio',
                        type=float,
                        default=0.01,
                        help='if warmup_ratio, then discard warmup_iters')
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
                        default=1.0)
    parser.add_argument('--compile',
                        type=bool,
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
    parser.add_argument('--log_level',
                        type=str,
                        default="INFO")
    parser.add_argument('--log_all_ranks',
                        type=str2bool,
                        default=False,
                        help="If true, all ranks also write summaries to stdout; otherwise only global rank 0 does.")
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
