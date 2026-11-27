"""Joint or single-source pretraining with FP32 masked logits BCE.

Select a JSON model preset and scratch, resume, or warm_start initialization.
Resume preserves optimizer and schedule where compatible. An unfinished epoch
is independently resampled. Each invocation writes to an empty output directory.
"""

import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import math
import random
import argparse
import json
import shutil
from pathlib import Path
from scripts.stable_logits_branch import (
    masked_fp32_logits_bce, branch_budget, restore_healthy_branch, restore_model_weights, validate_branch_output,
)
from contextlib import nullcontext
from typing import Dict, Iterable, List, Sequence, Tuple
import numpy as np
import pyarrow.parquet as pq
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from torch.cuda.amp import GradScaler
from scripts.species_epoch_sampling import (
    SpeciesParquetBatchDataset, prepare_species_data,
    resolve_epoch_cells, build_epoch_plan,
)

from scripts.training_diagnostics import (
    DiagnosticSpeciesParquetBatchDataset, LightweightDiagnostics,
)
from scripts.species_replay_sampling import (
    DEFAULT_BATCH1, DEFAULT_BATCH2, prepare_replay_data, resolve_replay_cells,
    build_replay_plan, source_weights, stage2_lr,
)


def prepare_training_data(args, required_columns, split_seed=20261022):
    """Share exactly the continuation splits, while preserving explicit single input."""
    if args.joint_sampling:
        return prepare_replay_data(
            args.batch1_path, args.batch2_path, required_columns, args.shard_rows,
            args.validation_fraction, args.small_species_validation, batch3=args.batch3_path,
        )
    train, validation, manifest = prepare_species_data(
        args.data_path, required_columns, args.shard_rows,
        args.validation_fraction, split_seed, args.validation_data_path,
    )
    return train, {"single": validation}, manifest


def build_training_epoch_plan(args, train, cells, world_size, epoch):
    if args.joint_sampling:
        return build_replay_plan(
            train, cells, args.batch_size, world_size, args.num_workers,
            epoch, args.sampling_seed, args.sampling_alpha,
            args.allow_species_repeats, args.new_data_fraction, args.batch3_fraction,
        )
    return build_epoch_plan(
        train, cells, args.batch_size, world_size, args.num_workers,
        epoch, args.sampling_seed, args.sampling_alpha, args.allow_species_repeats,
    )


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")

class DataPrefetcher:
    def __init__(self, loader, device):
        self.loader = iter(loader)
        self.device = torch.device(device)
        self.stream = torch.cuda.Stream(device=self.device)
        self.batch = None
        self.preload()

    def preload(self):
        try:
            cpu_batch = next(self.loader)
        except StopIteration:
            self.batch = None
            return
        with torch.cuda.stream(self.stream):
            self.batch = {
                key: value.to(device=self.device, non_blocking=True)
                if torch.is_tensor(value) else value
                for key, value in cpu_batch.items()
            }

    def next(self):
        current_stream = torch.cuda.current_stream(self.device)
        current_stream.wait_stream(self.stream)
        batch = self.batch
        if batch is not None:
            for value in batch.values():
                if torch.is_tensor(value) and value.is_cuda:
                    value.record_stream(current_stream)
        self.preload()
        return batch

LABEL_COLUMNS = {
    "batch_labels": "tech_sample",
    "species_labels": "species",
    "tissue_labels": "tissue",
    "seqmethod_labels": "assay",
    "disease_labels": "disease",
    "sex_labels": "sex",
    "age_labels": "development_stage",
}

class ExpressionBatchCollator:
    """Build only cell-dependent tensors; gene features stay resident on the GPU."""

    def __init__(
        self,
        seq_len: int,
        enabled_label_keys: Sequence[str],
        mlm_probability: float = 0.15,
        deterministic_mask_seed=None,
    ):
        self.seq_len = seq_len
        self.enabled_label_keys = tuple(enabled_label_keys)
        self.mlm_probability = mlm_probability
        self.deterministic_mask_seed = deterministic_mask_seed
        self._mask_generator = None
        self.reset_mask_generator()

    def reset_mask_generator(self):
        # Initialize lazily in the consuming process, not in the parent before spawn.
        self._mask_generator = None

    def __call__(self, rows: List[Dict]):
        values_np = np.stack(
            [np.asarray(row["X"], dtype=np.float32) for row in rows], axis=0
        )
        if values_np.shape[1] != self.seq_len:
            raise ValueError(
                f"Expected {self.seq_len} expression values, got {values_np.shape[1]}"
            )

        values = torch.from_numpy(values_np)
        cls_values = torch.zeros((values.shape[0], 1), dtype=values.dtype)
        target_values = torch.cat((cls_values, values), dim=1)
        if self.deterministic_mask_seed is None:
            mask = torch.rand(target_values.shape) < self.mlm_probability
        else:
            if self._mask_generator is None:
                self._mask_generator = torch.Generator()
                self._mask_generator.manual_seed(self.deterministic_mask_seed)
            mask = (
                torch.rand(target_values.shape, generator=self._mask_generator)
                < self.mlm_probability
            )
        mask[:, 0] = False

        batch = {
            "target_values": target_values,
            "values": target_values.masked_fill(mask, -1),
        }
        for key in self.enabled_label_keys:
            column = LABEL_COLUMNS[key]
            batch[key] = torch.tensor([row[column] for row in rows], dtype=torch.long)
        if '_provenance' in rows[0]:
            batch['_provenance'] = [row['_provenance'] for row in rows]
        return batch

class ParquetRowGroupBatchDataset(IterableDataset):
    """Stream exact, full batches from shuffled Parquet row groups.

    Every DDP rank receives the same number of batches. DataLoader workers split
    a rank's batch range without duplicating or dropping samples.
    """

    def __init__(
        self,
        row_groups: Sequence[Tuple[str, int, int]],
        columns: Sequence[str],
        rank: int,
        world_size: int,
        batch_size: int,
        num_batches_per_rank: int,
        epoch: int,
        seed: int = 1337,
    ):
        super().__init__()
        self.row_groups = tuple(row_groups)
        self.columns = tuple(columns)
        self.rank = rank
        self.world_size = world_size
        self.batch_size = batch_size
        self.num_batches_per_rank = num_batches_per_rank
        self.epoch = epoch
        self.seed = seed

    def __len__(self):
        return self.num_batches_per_rank

    @staticmethod
    def _split_batches(num_batches: int, worker_id: int, num_workers: int):
        base, remainder = divmod(num_batches, num_workers)
        start = worker_id * base + min(worker_id, remainder)
        count = base + (1 if worker_id < remainder else 0)
        return start, start + count

    def _iter_rows(self, ordered_row_groups, start: int, end: int) -> Iterable[Dict]:
        cursor = 0
        parquet_files = {}
        for file_path, row_group_index, num_rows in ordered_row_groups:
            row_group_end = cursor + num_rows
            if row_group_end <= start:
                cursor = row_group_end
                continue
            if cursor >= end:
                break

            local_start = max(0, start - cursor)
            local_end = min(num_rows, end - cursor)
            parquet_file = parquet_files.get(file_path)
            if parquet_file is None:
                parquet_file = pq.ParquetFile(file_path, memory_map=True)
                parquet_files[file_path] = parquet_file
            table = parquet_file.read_row_group(
                row_group_index, columns=self.columns, use_threads=False
            ).slice(local_start, local_end - local_start)
            columns = table.to_pydict()
            for row_index in range(table.num_rows):
                yield {key: values[row_index] for key, values in columns.items()}
            cursor = row_group_end

    def __iter__(self):
        ordered_row_groups = list(self.row_groups)
        random.Random(self.seed + self.epoch).shuffle(ordered_row_groups)

        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        num_workers = 1 if worker is None else worker.num_workers
        worker_batch_start, worker_batch_end = self._split_batches(
            self.num_batches_per_rank, worker_id, num_workers
        )

        rank_sample_start = self.rank * self.num_batches_per_rank * self.batch_size
        sample_start = rank_sample_start + worker_batch_start * self.batch_size
        sample_end = rank_sample_start + worker_batch_end * self.batch_size

        batch = []
        for row in self._iter_rows(ordered_row_groups, sample_start, sample_end):
            batch.append(row)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch:
            raise RuntimeError("Internal error: worker received a partial batch")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_mode", choices=("scratch", "resume", "warm_start"), default="resume",
                        help="scratch: random init; resume: continue existing plan; warm_start: weights only, new plan")
    parser.add_argument("--checkpoint_path")
    parser.add_argument("--model_config", type=Path, default=Path(__file__).resolve().parents[1] /
                        "configuration/350m_batch1_batch2.json")
    parser.add_argument("--optimizer_state", choices=("auto", "reset", "keep"), default="auto",
                        help="auto: reset for old loss, keep for logits-branch checkpoints")
    parser.add_argument("--warmup_steps", type=int, default=300,
                        help="New branch warmup in optimizer updates; preserved on branch resume")
    parser.add_argument("--warmup_start_lr", type=float, default=1e-6)

    parser.add_argument("--seq_len", type=int, required=True)

    parser.add_argument("--data_path", default=None,
                        help="Single-source data root (alias for --batch1_path with the single-source preset)")
    parser.add_argument("--batch1_path", default=None, help="First dataset root; built-in path used in joint mode")
    parser.add_argument("--batch2_path", default=None, help="Second dataset root; built-in path used in joint mode")
    parser.add_argument("--batch3_path", help="Optional third dataset root")
    parser.add_argument("--batch2_fraction", "--new_data_fraction", dest="new_data_fraction",
                        type=float, default=0.75, help="Absolute fraction of draws from batch2")
    parser.add_argument("--batch3_fraction", type=float, default=0.0,
                        help="Absolute fraction from batch3; batch1 receives the remainder")
    parser.add_argument("--embedding_dir", default="/mnt/data/u/wangj0g/projects/SpeciesLLM/macrogene/2ndrun_K1500",
                        help="Default directory for the three macrogene embedding files")
    for kind in ("esm", "desc", "dna"):
        parser.add_argument(f"--{kind}_embedding_dir", help=f"Override --embedding_dir for {kind}")
    parser.add_argument("--small_species_validation", choices=("error", "keep_one_train", "validation_only"),
                        default="keep_one_train")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--min_lr", type=float, default=1e-6)

    parser.add_argument("--out_path", type=str, default="out")

    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument("--epoch", type=int, default=3)

    parser.add_argument("--wanlog", type=str_to_bool, nargs="?", const=True, default=False)

    parser.add_argument("--compile_flag", type=str_to_bool, nargs="?", const=True, default=False)

    parser.add_argument("--num_workers", type=int, default=2,
                        help="DataLoader workers per rank; start with 2 and tune using cells/s")

    parser.add_argument("--prefetch_factor", type=int, default=1,
                        help="batches prefetched by each worker")

    parser.add_argument("--log_interval", type=int, default=50)

    parser.add_argument(
        "--huber_delta", type=float, default=1.0,
        help="Huber threshold in expression units for GEP and GEPC (default: 1.0)",
    )

    parser.add_argument("--sampling_alpha", type=float, default=0.6)

    parser.add_argument("--sampling_seed", type=int, default=20261115)

    parser.add_argument("--epoch_cells", type=int, default=0,
                        help="Global draws/epoch; 0 uses batch2_train_cells/new_data_fraction in joint mode, no-repeat cap in single mode")

    parser.add_argument("--allow_species_repeats", action=argparse.BooleanOptionalAction, default=None,
                        help="Joint mode defaults to repeats; single mode defaults to no repeats")

    parser.add_argument("--shard_rows", type=int, default=16384,
                        help="Required cells per training shard; shorter tail shards are allowed in validation")

    parser.add_argument("--shuffle_buffer_cells", type=int, default=4096,
                        help="Cell shuffle buffer per DataLoader worker")

    parser.add_argument("--read_batch_rows", type=int, default=256,
                        help="Arrow decoding batch size per active species per worker")

    parser.add_argument("--validation_data_path", default=None,
                        help="Optional separate species-directory validation tree")

    parser.add_argument("--validation_fraction", type=float, default=0.005,
                        help="Hold out this fraction from the last numbered shards backwards (at least one when >=2 shards)")

    parser.add_argument("--diagnostics", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--diagnostic_stats_interval", type=int, default=50,
                        help="Sample GEP/input statistics every N optimizer updates")
    parser.add_argument("--diagnostic_capture_batches", type=int, default=50,
                        help="Bounded number of retained micro-batches per rank; tensor references only")
    parser.add_argument("--diagnostic_max_snapshots", type=int, default=3,
                        help="Maximum anomaly batch files per rank for the entire run")
    parser.add_argument("--diagnostic_cooldown_updates", type=int, default=1000)
    parser.add_argument("--diagnostic_burnin_updates", type=int, default=100)
    parser.add_argument("--diagnostic_loss_ratio", type=float, default=5.0)
    parser.add_argument("--diagnostic_loss_floor", type=float, default=1.0)
    parser.add_argument("--diagnostic_grad_ratio", type=float, default=10.0)
    parser.add_argument("--diagnostic_grad_floor", type=float, default=10.0)

    args = parser.parse_args()
    if (min(args.diagnostic_stats_interval, args.diagnostic_capture_batches,
            args.diagnostic_cooldown_updates) < 1
            or min(args.diagnostic_max_snapshots, args.diagnostic_burnin_updates) < 0):
        parser.error("Invalid diagnostic interval or buffer settings")
    if any(not math.isfinite(x) or x <= 0 for x in (
            args.diagnostic_loss_ratio, args.diagnostic_loss_floor,
            args.diagnostic_grad_ratio, args.diagnostic_grad_floor)):
        parser.error("Diagnostic thresholds must be finite and positive")
    try:
        with args.model_config.open(encoding="utf-8") as file:
            preset = json.load(file)
        if preset["default_sources"] not in (1, 2) or not isinstance(preset["model"], dict):
            raise ValueError("Expected default_sources (1 or 2) and a model object")
        if not preset["model"].get("explicit_zero_prob", False) or not preset["model"].get("do_mvc", True):
            raise ValueError("This trainer requires explicit_zero_prob and do_mvc")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(f"Invalid --model_config: {exc}")
    if args.data_path and (args.batch1_path or args.batch2_path or args.batch3_path):
        parser.error("Use either --data_path or the --batchN_path options")
    args.joint_sampling = not args.data_path and preset["default_sources"] == 2
    if args.joint_sampling:
        if args.validation_data_path:
            parser.error("--validation_data_path applies only to single-source mode")
        args.batch1_path = args.batch1_path or DEFAULT_BATCH1
        args.batch2_path = args.batch2_path or DEFAULT_BATCH2
        if bool(args.batch3_path) != (args.batch3_fraction > 0):
            parser.error("--batch3_path and a positive --batch3_fraction must be set together")
        try:
            weights = source_weights(args.new_data_fraction, args.batch3_fraction)
        except ValueError as exc:
            parser.error(str(exc))
    else:
        if args.batch2_path or args.batch3_path or args.batch3_fraction != 0:
            parser.error("Use a joint preset for batch2 or batch3")
        args.data_path = args.data_path or args.batch1_path or DEFAULT_BATCH1
        weights = {"single": 1.0}
    if args.allow_species_repeats is None:
        args.allow_species_repeats = args.joint_sampling
    if not (math.isfinite(args.learning_rate) and math.isfinite(args.min_lr)
            and 0 < args.min_lr <= args.learning_rate):
        parser.error("Require finite 0 < min_lr <= learning_rate")

    if args.batch_size < 1 or args.epoch < 1 or args.num_workers < 0:
        parser.error("batch_size and epoch must be positive; num_workers must be nonnegative")

    if args.epoch_cells < 0 or args.shard_rows < 1:
        parser.error("epoch_cells must be nonnegative and shard_rows must be positive")

    if args.shuffle_buffer_cells < 1 or args.read_batch_rows < 1:
        parser.error("shuffle_buffer_cells and read_batch_rows must be positive")

    if not 0 <= args.sampling_alpha <= 1:
        parser.error("sampling_alpha must be in [0,1]")
    if args.warmup_steps < 0 or not (math.isfinite(args.warmup_start_lr)
                                   and 0 < args.warmup_start_lr <= args.learning_rate):
        parser.error("Require warmup_steps >= 0 and 0 < warmup_start_lr <= learning_rate")
    if args.train_mode == "scratch":
        if args.checkpoint_path:
            parser.error("--train_mode scratch does not use --checkpoint_path")
    elif not args.checkpoint_path:
        parser.error(f"--train_mode {args.train_mode} requires --checkpoint_path")
    if args.train_mode != "resume" and args.optimizer_state == "keep":
        parser.error("--optimizer_state keep requires --train_mode resume")
    validate_branch_output(args.checkpoint_path, args.out_path)
    checkpoint_state = None
    if args.checkpoint_path:
        checkpoint_state = torch.load(args.checkpoint_path, map_location="cpu", mmap=True,
                                      weights_only=False)
    schedule_state = checkpoint_state if args.train_mode == "resume" else {"optimizer_step": 0}

    if not 0 < args.validation_fraction < 1:
        parser.error("validation_fraction must be in (0,1)")

    if not math.isfinite(args.huber_delta) or args.huber_delta <= 0:
        parser.error("--huber_delta must be finite and positive")

    from nanoBERT.utils import GeneVocab
    from nanoBERT.utils import masked_huber_loss
    from nanoBERT.model.nanoBERTmodel_cellmeta2_plusEncode_adbc_logits import BERTConfig, BERTForPreTraining

    reconstruction_loss_config = {"name": "huber", "delta": args.huber_delta,
                                  "nonzero_loss": "masked_fp32_bce_with_logits_v1"}

    wandb_log = args.wanlog

    wandb_project = 'SpeciesLLM'

    wandb_run_name = 'BERT_logits_pretraining'
    wandb_run_name += f"_lr{args.learning_rate:g}_lightdiag"


    out_dir = args.out_path

    num_epochs = args.epoch

    learning_rate = args.learning_rate

    lr_decay_steps = None

    max_optimizer_steps = None

    min_lr = args.min_lr

    decay_lr = True

    warmup_steps = None

    weight_decay = 1e-2

    beta1 = 0.9

    beta2 = 0.999  # actual AdamW default used in the healthy checkpoint

    seq_len = args.seq_len

    grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0

    compile = args.compile_flag  # complie model if pytorch > 2.0

    gradient_accumulation_steps = 10 * 8  # used to simulate larger batch sizes

    batch_size = args.batch_size  # if gradient_accumulation_steps > 1, this is the mini-batch size

    VALIDATION_FRACTION = args.validation_fraction

    VALIDATION_SPLIT_SEED = 20261110

    VALIDATION_MASK_SEED = 20261110

    VALIDATION_INTERVAL_STEPS = 5_000

    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'

    backend = 'nccl'  # 'nccl', 'gloo', etc.

    device = 'cuda'

    def get_lr(optimizer_step):
        return stage2_lr(
            optimizer_step - branch_config["schedule_start_optimizer_step"],
            branch_config["phase_total_optimizer_steps"], warmup_steps, learning_rate,
            min_lr, args.warmup_start_lr if warmup_steps else learning_rate,
        )

    ddp = int(os.environ.get('RANK', -1)) != -1  # Torchrun approach to check if DDP is enabled

    if ddp:
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])

        if ddp_rank == 0:
            print("++++++++++++++++++++++++++++++++++++++++++++++")
            print("ddp_world_size")
            print(ddp_world_size)
            print("++++++++++++++++++++++++++++++++++++++++++++++")

        init_process_group(backend=backend, rank=ddp_rank, world_size=ddp_world_size)
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0  # this process will do logging, checkpointing, etc.
        seed_offset = ddp_rank  # each process gets a different seed
        # world_size number of processes will be training simultaneously, so we can scale down
        # the desired gradient accumulation iterations per process proportionally
        assert gradient_accumulation_steps % ddp_world_size == 0
        gradient_accumulation_steps //= ddp_world_size
    else:
        # if not ddp, we are running on a single GPU, and one process
        print("==============not ddp")
        ddp_rank = 0
        master_process = True
        seed_offset = 0
        ddp_world_size = 1

    tokens_per_optimizer_step = (
        gradient_accumulation_steps * ddp_world_size * batch_size * seq_len
    )

    if master_process:
        print(f"tokens per optimizer step: {tokens_per_optimizer_step:,}")

    if master_process:
        os.makedirs(out_dir, exist_ok=True)

    torch.manual_seed(1337 + seed_offset)

    torch.backends.cuda.matmul.allow_tf32 = True

    torch.backends.cudnn.allow_tf32 = True

    device_type = 'cuda' if 'cuda' in device else 'cpu'

    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]

    ctx = nullcontext() if device_type == 'cpu' else torch.autocast(device_type=device_type, dtype=ptdtype)

    t0 = time.time()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    vocab_file = os.path.join(project_root, "configuration", f"customized_gene{seq_len}_vocab.json")

    if not os.path.exists(vocab_file):
        vocab_file = f"customized_gene{seq_len}_vocab.json"

    vocab = GeneVocab.from_file(vocab_file)

    if "<cls>" not in vocab:
        vocab.append_token("<cls>")

    gene_tokens = [f"gene_{gene_index}" for gene_index in range(1, seq_len + 1)]

    missing_genes = [token for token in gene_tokens if token not in vocab]

    if missing_genes:
        raise ValueError(f"Vocabulary is missing gene tokens, first entries: {missing_genes[:5]}")

    src = np.asarray([vocab[token] for token in gene_tokens], dtype=np.int64)

    cls_id = vocab["<cls>"]

    if len(np.unique(src)) != seq_len or cls_id in set(src.tolist()):
        raise ValueError("Gene IDs must be unique and must not overlap the CLS ID")

    vocab_size = max(int(src.max()), int(cls_id)) + 1

    def load_embedding(kind, filename):
        directory = getattr(args, f"{kind}_embedding_dir") or args.embedding_dir
        return np.load(os.path.join(directory, filename), mmap_mode="r")

    esm_embeddings_np = load_embedding("esm", "3rdrun_K1500_Eukaryota_macrogene_esm2_weighted_average.npy")
    desc_embeddings_np = load_embedding("desc", "3rdrun_K1500_Eukaryota_macrogene_gene_desc_weighted_average.npy")
    dna_embeddings_np = load_embedding("dna", "3rdrun_K1500_Eukaryota_macrogene_DNAseq_weighted_average.npy")

    for name, embeddings, width in (
        ("ESM", esm_embeddings_np, 1280),
        ("description", desc_embeddings_np, 1280),
        ("DNA", dna_embeddings_np, 2560),
    ):
        if embeddings.shape != (seq_len, width):
            raise ValueError(f"{name} embedding shape {embeddings.shape}; expected {(seq_len, width)}")

    config = BERTConfig(
        vocab_size=vocab_size, vocab=vocab, max_position_embeddings=args.seq_len + 1,
        **preset["model"],
    )

    enabled_label_keys = [
        key for key, enabled in (
            ("batch_labels", config.use_batch_labels),
            ("species_labels", config.use_species_labels),
            ("tissue_labels", config.use_tissue_labels),
            ("seqmethod_labels", config.use_seqmethod_labels),
            ("disease_labels", config.use_disease_labels),
            ("sex_labels", config.use_sex_labels),
            ("age_labels", config.use_age_labels),
        ) if enabled
    ]

    required_columns = ["X"] + [LABEL_COLUMNS[key] for key in enabled_label_keys]

    collator = ExpressionBatchCollator(seq_len, enabled_label_keys)

    validation_collator = ExpressionBatchCollator(
        seq_len,
        enabled_label_keys,
        deterministic_mask_seed=VALIDATION_MASK_SEED,
    )

    global_batch_size = ddp_world_size * batch_size

    split_objects = [None]

    if master_process:
        try:
            split_objects[0] = {"data": prepare_training_data(
                args, required_columns, VALIDATION_SPLIT_SEED,
            )}
        except Exception as exc:
            split_objects[0] = {"error": f"{type(exc).__name__}: {exc}"}

    if ddp:
        dist.broadcast_object_list(split_objects, src=0)

    if "error" in split_objects[0]:
        raise ValueError(split_objects[0]["error"])

    train_species_shards, validation_by_source, sampling_manifest = split_objects[0]["data"]
    validation_source_rows = {source: sum(s.rows for s in shards)
                              for source, shards in validation_by_source.items()}
    for source, count in validation_source_rows.items():
        if count < global_batch_size:
            raise ValueError(f"{source} validation is smaller than one global batch")
    train_rows = sum(s.rows for shards in train_species_shards.values() for s in shards)
    validation_rows = sum(validation_source_rows.values())
    all_rows = train_rows + validation_rows
    validation_rows_used = sum(count // global_batch_size * global_batch_size
                               for count in validation_source_rows.values())
    if args.joint_sampling:
        used_train_rows = resolve_replay_cells(
            train_species_shards, args.sampling_alpha, args.epoch_cells,
            global_batch_size * gradient_accumulation_steps,
            args.new_data_fraction, args.allow_species_repeats, args.batch3_fraction,
        )
    else:
        used_train_rows = resolve_epoch_cells(
            train_species_shards, args.sampling_alpha, args.epoch_cells,
            global_batch_size * gradient_accumulation_steps, args.allow_species_repeats,
        )

    optimizer_steps_per_epoch = used_train_rows // (global_batch_size * gradient_accumulation_steps)

    num_batches_per_rank = optimizer_steps_per_epoch * gradient_accumulation_steps

    validation_batches_per_rank = validation_rows_used // global_batch_size

    max_optimizer_steps = num_epochs * optimizer_steps_per_epoch

    lr_decay_steps = max_optimizer_steps

    branch_config = branch_budget(
        schedule_state, num_epochs, used_train_rows,
        global_batch_size * gradient_accumulation_steps, args.warmup_steps,
    )
    warmup_steps = branch_config["warmup_steps"]
    branch_config.update(
        checkpoint_path=os.path.realpath(args.checkpoint_path) if args.checkpoint_path else None,
        train_mode=args.train_mode,
        learning_rate=learning_rate, min_lr=min_lr,
        warmup_start_lr=args.warmup_start_lr, sampling_seed=args.sampling_seed,
        nonzero_loss="masked_fp32_bce_with_logits_v1",
        old_loss_validation_is_not_directly_comparable=True,
    )
    previous_branch = schedule_state.get("logits_branch") or {}
    if previous_branch and previous_branch["warmup_start_lr"] != args.warmup_start_lr:
        raise ValueError("Existing logits branch must retain warmup_start_lr")

    num_workers = args.num_workers

    if args.prefetch_factor < 1 or args.log_interval < 1:
        raise ValueError("prefetch_factor and log_interval must be positive")

    sampling_config = {
        "alpha": args.sampling_alpha, "seed": args.sampling_seed,
        "requested_epoch_cells": args.epoch_cells, "epoch_cells": used_train_rows,
        "allow_species_repeats": args.allow_species_repeats,
        "shard_rows": args.shard_rows, "num_workers": num_workers,
        "shuffle_buffer_cells": args.shuffle_buffer_cells,
        "read_batch_rows": args.read_batch_rows,
        "world_size": ddp_world_size, "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "manifest_fingerprint": sampling_manifest["fingerprint"],
        "max_optimizer_steps": max_optimizer_steps, "warmup_steps": warmup_steps,
        "independent_epochs": True,
        "initialization": args.train_mode,
        "model_config": preset["model"],
        "model_config_path": str(args.model_config.resolve()),
        "embedding_dirs": {kind: os.path.realpath(getattr(args, f"{kind}_embedding_dir")
                                                   or args.embedding_dir)
                           for kind in ("esm", "desc", "dna")},
        "source_weights": weights,
        "batch3_path": args.batch3_path,
        "batch3_fraction": args.batch3_fraction,
        "logits_branch": branch_config,
        "joint_sampling": args.joint_sampling,
        "batch1_path": args.batch1_path, "batch2_path": args.batch2_path,
        "data_path": args.data_path,
        "new_data_fraction": args.new_data_fraction if args.joint_sampling else None,
        "small_species_validation": args.small_species_validation if args.joint_sampling else None,
        "learning_rate": learning_rate, "min_lr": min_lr,
        "warmup_start_lr": args.warmup_start_lr,
        "validation_source_rows": validation_source_rows,
        "metadata_label_sizes": {key: getattr(config, "num_" + key.removesuffix("_labels") + "_labels")
                                 for key in enabled_label_keys},
    }

    diagnostic_config = {
        "enabled": args.diagnostics,
        "stats_interval_updates": args.diagnostic_stats_interval,
        "capture_batches": args.diagnostic_capture_batches,
        "max_snapshots_per_rank": args.diagnostic_max_snapshots,
        "cooldown_updates": args.diagnostic_cooldown_updates,
        "burnin_updates": args.diagnostic_burnin_updates,
        "burnin_until_optimizer_step": branch_config["resume_optimizer_step"] + args.diagnostic_burnin_updates,
        "loss_ratio": args.diagnostic_loss_ratio, "loss_floor": args.diagnostic_loss_floor,
        "grad_ratio": args.diagnostic_grad_ratio, "grad_floor": args.diagnostic_grad_floor,
        "scope": "rank_local; synchronized DDP gradients",
        "detection_interval_micro_batches": args.log_interval,
    }
    sampling_config["diagnostics"] = diagnostic_config
    diagnostics = LightweightDiagnostics(
        out_dir, ddp_rank, device,
        stats_interval=args.diagnostic_stats_interval,
        capture_batches=args.diagnostic_capture_batches,
        max_snapshots=args.diagnostic_max_snapshots,
        cooldown_updates=args.diagnostic_cooldown_updates,
        burnin_updates=branch_config["resume_optimizer_step"] + args.diagnostic_burnin_updates,
        loss_ratio=args.diagnostic_loss_ratio, loss_floor=args.diagnostic_loss_floor,
        grad_ratio=args.diagnostic_grad_ratio, grad_floor=args.diagnostic_grad_floor,
    ) if args.diagnostics else None

    if master_process:
        print(f"Species sampling: {used_train_rows:,} draws/epoch from {train_rows:,} train cells; "
              f"{num_batches_per_rank:,} batches/rank, {optimizer_steps_per_epoch:,} updates/epoch")
        print(f"Fixed validation: {validation_rows_used:,}/{validation_rows:,} cells; "
              f"actual holdout fraction={validation_rows / all_rows:.3%}")
        print(f"Branch: resume update {branch_config['resume_optimizer_step']:,}; "
              f"{branch_config['remaining_optimizer_steps']:,} remaining updates / "
              f"{branch_config['remaining_cell_draws']:,} remaining cell draws")
        print(f"Schedule: original target {max_optimizer_steps:,} total updates, "
              f"configured {warmup_steps:,}-update warmup from {args.warmup_start_lr:g}, "
              f"peak {learning_rate:g}, minimum {min_lr:g}")
        if args.train_mode == "resume":
            print("Unfinished epoch uses independent resampling; no old data-loader cursor is restored.")
        for warning in sampling_manifest["warnings"]:
            print(f"Sampling warning: {warning}")

    if master_process:
        print(f"Using device: {device_type}")
        print(f"GEP/GEPC reconstruction loss: Huber, delta={args.huber_delta}")

    model = BERTForPreTraining(config)

    model.to(device)

    static_gene_ids = torch.tensor(
        np.concatenate(([cls_id], src)), dtype=torch.long, device=device
    )

    def prepare_static_embeddings(embeddings):
        cpu_tensor = torch.from_numpy(
            np.asarray(embeddings, dtype=np.float32).copy()
        )
        gpu_tensor = cpu_tensor.to(device=device, non_blocking=False)
        cls_row = torch.zeros(
            (1, gpu_tensor.shape[1]), dtype=gpu_tensor.dtype, device=device
        )
        return torch.cat((cls_row, gpu_tensor), dim=0)

    static_esm_embeddings = prepare_static_embeddings(esm_embeddings_np)

    static_desc_embeddings = prepare_static_embeddings(desc_embeddings_np)

    static_dna_embeddings = prepare_static_embeddings(dna_embeddings_np)

    del esm_embeddings_np, desc_embeddings_np, dna_embeddings_np

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                  betas=(beta1, beta2), weight_decay=weight_decay)
    scaler = GradScaler(enabled=(dtype == 'float16'))
    if args.train_mode == "resume":
        branch_restore = restore_healthy_branch(
            model, optimizer, scaler, checkpoint_state, torch.from_numpy(src.copy()),
            cls_id, sampling_manifest["fingerprint"], sampling_config, args.optimizer_state,
        )
    else:
        branch_restore = {"effective_optimizer_mode": "reset", "optimizer_parameters_restored": 0}
        if args.train_mode == "warm_start":
            restore_model_weights(model, checkpoint_state, torch.from_numpy(src.copy()), cls_id)
            branch_restore["weight_source_optimizer_step"] = checkpoint_state["optimizer_step"]
    branch_config.update(branch_restore)
    if master_process:
        print("Initialization:", json.dumps(branch_restore))
    del checkpoint_state, schedule_state

    if compile:
        if master_process:
            print("compiling the model...(take a ~minute)")
        unoptimized_model = model
        model = torch.compile(model)  # requires PyTorch 2.0

    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=False)

    if wandb_log and master_process:
        import wandb

        wandb.init(project=wandb_project, name=wandb_run_name, config=config)
        wandb.config.update({"reconstruction_loss": reconstruction_loss_config, "species_sampling": sampling_config})

    raw_model = model.module if ddp else model

    if hasattr(raw_model, "_orig_mod"):
        raw_model = raw_model._orig_mod

    optimizer_step = branch_config["resume_optimizer_step"]
    # Counter uses current accumulation; optimizer_step is comparable across GPU counts.
    iter_num = optimizer_step * gradient_accumulation_steps

    running_mfu = -1.0

    stop_training = False

    best_total_loss = float("inf")

    best_epoch = None

    best_validation_loss = float("inf")

    best_validation_step = None

    last_checkpoint_path = None

    loss_history = {
        "species_sampling": sampling_config,
        "reconstruction_loss": reconstruction_loss_config,
        "metric": "total_loss = GEP + nzlp + GEPC + GEPC_nzlp",
        "log_interval": args.log_interval,
        "validation_interval_steps": VALIDATION_INTERVAL_STEPS,
        "validation_rows": validation_rows,
        "validation_rows_evaluated": validation_rows_used,
        "steps": [],
        "epochs": [],
        "validation": [],
        "best_epoch": None,
        "best_total_loss": None,
        "best_validation_step": None,
        "best_validation_loss": None,
    }

    loss_history_path = os.path.join(out_dir, "total_loss_history.json")

    def atomic_torch_save(obj, path):
        """Write a checkpoint without exposing a partially written target file."""
        temporary_path = f"{path}.tmp"
        try:
            torch.save(obj, temporary_path)
            os.replace(temporary_path, path)
        finally:
            if os.path.exists(temporary_path):
                os.remove(temporary_path)

    def atomic_json_save(obj, path):
        """Persist the buffered loss history once per epoch."""
        temporary_path = f"{path}.tmp"
        try:
            with open(temporary_path, "w", encoding="utf-8") as file_handle:
                json.dump(obj, file_handle, indent=2)
            os.replace(temporary_path, path)
        finally:
            if os.path.exists(temporary_path):
                os.remove(temporary_path)

    def update_checkpoint_alias(source_path, alias_path):
        """Atomically point an alias at a saved checkpoint without rewriting it."""
        temporary_path = f"{alias_path}.tmp"
        try:
            if os.path.exists(temporary_path):
                os.remove(temporary_path)
            try:
                os.link(source_path, temporary_path)
            except OSError:
                shutil.copy2(source_path, temporary_path)
            os.replace(temporary_path, alias_path)
        finally:
            if os.path.exists(temporary_path):
                os.remove(temporary_path)

    def run_validation(source):
        """Evaluate the fixed holdout with equal work on every DDP rank."""
        validation_collator.reset_mask_generator()
        validation_row_groups = [(s.path, index, rows)
                                 for s in validation_by_source[source]
                                 for index, rows in enumerate(s.row_groups)]
        validation_batches_per_rank = validation_source_rows[source] // global_batch_size
        validation_dataset = ParquetRowGroupBatchDataset(
            row_groups=validation_row_groups,
            columns=required_columns,
            rank=ddp_rank,
            world_size=ddp_world_size,
            batch_size=batch_size,
            num_batches_per_rank=validation_batches_per_rank,
            epoch=0,
            seed=VALIDATION_SPLIT_SEED,
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=None,
            collate_fn=validation_collator,
            num_workers=0,
            pin_memory=True,
        )
        if len(validation_loader) != validation_batches_per_rank:
            raise RuntimeError("Validation DataLoader length differs from its batch plan")

        validation_prefetcher = DataPrefetcher(validation_loader, device=device)
        validation_batch = validation_prefetcher.next()
        metric_names = ("GEP", "nzlp", "GEPC", "GEPC_nzlp", "total_loss")
        validation_metric_sum = torch.zeros(
            len(metric_names), dtype=torch.float64, device=device
        )
        validation_batch_count = 0
        model.eval()

        with torch.no_grad():
            while validation_batch is not None:
                input_gene_values = validation_batch["values"]
                target_values = validation_batch["target_values"]
                input_gene_ids = static_gene_ids.unsqueeze(0).expand(
                    input_gene_values.shape[0], -1
                )

                with ctx:
                    outputs = model(
                        src=input_gene_ids,
                        values=input_gene_values,
                        esm_embeddings=static_esm_embeddings,
                        desc_embeddings=static_desc_embeddings,
                        dna_embeddings=static_dna_embeddings,
                        batch_labels=validation_batch.get("batch_labels"),
                        species_labels=validation_batch.get("species_labels"),
                        tissue_labels=validation_batch.get("tissue_labels"),
                        seqmethod_labels=validation_batch.get("seqmethod_labels"),
                        disease_labels=validation_batch.get("disease_labels"),
                        sex_labels=validation_batch.get("sex_labels"),
                        age_labels=validation_batch.get("age_labels"),
                        CLS=False,
                        MVC=True,
                        output_hidden_states=False,
                        output_attentions=False,
                    )

                    mask_positions = input_gene_values.eq(-1)
                    loss_gep = masked_huber_loss(
                        outputs["model_output"], target_values, mask_positions, delta=args.huber_delta
                    )
                    loss_zero_prob = masked_fp32_logits_bce(
                        outputs["model_zero_logits"], target_values, mask_positions
                    )
                    loss_gepc = masked_huber_loss(
                        outputs["mvc_output"], target_values, mask_positions, delta=args.huber_delta
                    )
                    loss_gepc_zero_prob = masked_fp32_logits_bce(
                        outputs["mvc_zero_logits"], target_values, mask_positions
                    )
                    total_loss = (
                        loss_gep + loss_zero_prob + loss_gepc + loss_gepc_zero_prob
                    )

                validation_metric_sum += torch.stack(
                    (
                        loss_gep.detach(),
                        loss_zero_prob.detach(),
                        loss_gepc.detach(),
                        loss_gepc_zero_prob.detach(),
                        total_loss.detach(),
                    )
                ).to(torch.float64)
                validation_batch_count += 1
                validation_batch = validation_prefetcher.next()

        if validation_batch_count != validation_batches_per_rank:
            raise RuntimeError("Validation consumed a different number of batches")
        reduced_validation_metrics = validation_metric_sum.clone()
        if ddp:
            dist.all_reduce(reduced_validation_metrics, op=dist.ReduceOp.SUM)
        validation_metrics = (
            reduced_validation_metrics
            / (validation_batch_count * ddp_world_size)
        ).cpu().tolist()

        del validation_prefetcher, validation_loader, validation_dataset
        model.train()
        return metric_names, validation_metrics

    def run_training_validation():
        results = {source: run_validation(source) for source in weights}
        names = next(iter(results.values()))[0]
        combined = [sum(weights[source] * values[i] for source, (_, values) in results.items())
                    for i in range(len(names))]
        if master_process:
            record = {"optimizer_step": optimizer_step,
                      **{source: dict(zip(names, values)) for source, (_, values) in results.items()},
                      "source_weighted_total_loss": combined[-1]}
            loss_history.setdefault("validation_by_source", []).append(record)
            print("Validation by source:", json.dumps(record))
            if wandb_log:
                wandb.log({f"valid/{source}/{name}": value
                           for source, (_, values) in results.items()
                           for name, value in zip(names, values)}, step=iter_num)
        return names, combined

    if master_process:
        atomic_json_save(sampling_manifest, os.path.join(out_dir, "sampling_manifest.json"))
        atomic_json_save(sampling_config, os.path.join(out_dir, "sampling_config.json"))

    # Rebaseline under the corrected objective. Never compare against old clipped BCE.
    baseline_names, baseline_values = run_training_validation()
    best_validation_loss = baseline_values[-1]
    if not math.isfinite(best_validation_loss):
        raise ValueError("Initial model has non-finite validation under logits loss")
    best_validation_step = optimizer_step
    if master_process:
        loss_history["branch_start_validation"] = dict(zip(baseline_names, baseline_values))
        loss_history["best_validation_step"] = best_validation_step
        loss_history["best_validation_loss"] = best_validation_loss
        initial_checkpoint = {
            "model": raw_model.state_dict(), "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(), "optimizer_step": optimizer_step,
            "iter_num": iter_num, "epoch": branch_config["start_epoch_index"] + 1,
            "epoch_complete": False,
            "validation_total_loss": best_validation_loss,
            "best_validation_loss": best_validation_loss,
            "best_validation_step": best_validation_step,
            "species_sampling": sampling_config, "logits_branch": branch_config,
            "reconstruction_loss": reconstruction_loss_config,
            "gene_token_ids": torch.from_numpy(src.copy()), "cls_token_id": int(cls_id),
            "input_id_scheme": "gene_vocab_ids_with_disjoint_cls_v2",
            "total_loss_history": loss_history,
        }
        initial_path = os.path.join(out_dir, "checkpoint_branch_start.pth")
        atomic_torch_save(initial_checkpoint, initial_path)
        update_checkpoint_alias(initial_path, os.path.join(out_dir, "checkpoint_best_validation_loss.pth"))
        update_checkpoint_alias(initial_path, os.path.join(out_dir, "checkpoint_latest.pth"))
        atomic_json_save(loss_history, loss_history_path)
        print("Branch start validation:", json.dumps(loss_history["branch_start_validation"]))
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(branch_config["start_epoch_index"], num_epochs):
        completed_before_epoch = (branch_config["completed_updates_in_start_epoch"]
                                  if epoch == branch_config["start_epoch_index"] else 0)
        epoch_updates = optimizer_steps_per_epoch - completed_before_epoch
        epoch_cells_this_invocation = epoch_updates * global_batch_size * gradient_accumulation_steps
        num_batches_per_rank = epoch_updates * gradient_accumulation_steps
        if master_process:
            print("++++++++++++++++++++++++++++++++++++++++++++++")
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print("++++++++++++++++++++++++++++++++++++++++++++++")

        epoch_plan = build_training_epoch_plan(
            args, train_species_shards, epoch_cells_this_invocation, ddp_world_size,
            epoch + branch_config["resume_optimizer_step"],
        )
        epoch_plan.summary.update(
            original_epoch_index=epoch, original_epoch_cells=used_train_rows,
            completed_optimizer_updates_before_invocation=completed_before_epoch,
            independent_remaining_budget_resampling=True,
        )
        if master_process:
            atomic_json_save(epoch_plan.summary, os.path.join(out_dir, f"sampling_epoch_{epoch + 1:04d}.json"))
            for species, stats in epoch_plan.summary["species"].items():
                print(f"  {species}: {stats['sampled_cells']:,} draws "
                      f"({stats['probability']:.2%} target), "
                      f"{stats['passes']:.3f} passes, "
                      f"{stats['selected_unique_shards']} unique shards")
        if diagnostics is not None:
            diagnostics.set_epoch(epoch + 1, sampling_manifest["fingerprint"], epoch_plan.summary)
        dataset_class = DiagnosticSpeciesParquetBatchDataset if args.diagnostics and args.diagnostic_max_snapshots else SpeciesParquetBatchDataset
        dataset = dataset_class(
            epoch_plan, ddp_rank, required_columns,
            args.shuffle_buffer_cells, args.read_batch_rows,
        )
        loader_kwargs = {
            "dataset": dataset,
            "batch_size": None,
            "collate_fn": collator,
            "num_workers": num_workers,
            "pin_memory": True,
            "generator": torch.Generator().manual_seed(args.sampling_seed + epoch * ddp_world_size + ddp_rank),
        }
        if num_workers > 0:
            loader_kwargs["prefetch_factor"] = args.prefetch_factor
            loader_kwargs["multiprocessing_context"] = "spawn"
        data_loader = DataLoader(**loader_kwargs)
        if len(data_loader) != num_batches_per_rank:
            raise RuntimeError("DataLoader length differs from the synchronized batch plan")

        prefetcher = DataPrefetcher(data_loader, device=device)
        batch_data = prefetcher.next()
        model.train()

        metric_names = ("GEP", "nzlp", "GEPC", "GEPC_nzlp", "total_loss")
        window_metrics = torch.zeros(len(metric_names), dtype=torch.float64, device=device)
        epoch_loss_sum = torch.zeros((), dtype=torch.float64, device=device)
        window_steps = 0
        batch_id = 0
        torch.cuda.synchronize()
        window_start = time.perf_counter()

        while batch_data is not None:
            should_step = (batch_id + 1) % gradient_accumulation_steps == 0
            if ddp:
                model.require_backward_grad_sync = should_step

            # Keep one LR for every gradient-accumulation window. This makes the
            # schedule invariant to how the same global batch is split across ranks.
            if batch_id % gradient_accumulation_steps == 0:
                lr = get_lr(optimizer_step) if decay_lr else learning_rate
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

            input_gene_values = batch_data["values"]
            target_values = batch_data["target_values"]
            input_gene_ids = static_gene_ids.unsqueeze(0).expand(
                input_gene_values.shape[0], -1
            )

            with ctx:
                outputs = model(
                    src=input_gene_ids,
                    values=input_gene_values,
                    esm_embeddings=static_esm_embeddings,
                    desc_embeddings=static_desc_embeddings,
                    dna_embeddings=static_dna_embeddings,
                    batch_labels=batch_data.get("batch_labels"),
                    species_labels=batch_data.get("species_labels"),
                    tissue_labels=batch_data.get("tissue_labels"),
                    seqmethod_labels=batch_data.get("seqmethod_labels"),
                    disease_labels=batch_data.get("disease_labels"),
                    sex_labels=batch_data.get("sex_labels"),
                    age_labels=batch_data.get("age_labels"),
                    CLS=False,
                    MVC=True,
                    output_hidden_states=False,
                    output_attentions=False,
                )

                mask_positions = input_gene_values.eq(-1)
                loss_gep = masked_huber_loss(
                    outputs["model_output"], target_values, mask_positions, delta=args.huber_delta
                )
                loss_zero_prob = masked_fp32_logits_bce(
                    outputs["model_zero_logits"], target_values, mask_positions
                )
                loss_gepc = masked_huber_loss(
                    outputs["mvc_output"], target_values, mask_positions, delta=args.huber_delta
                )
                loss_gepc_zero_prob = masked_fp32_logits_bce(
                    outputs["mvc_zero_logits"], target_values, mask_positions
                )
                unscaled_loss = (
                    loss_gep + loss_zero_prob + loss_gepc + loss_gepc_zero_prob
                )
                loss = unscaled_loss / gradient_accumulation_steps

            detached_metrics = torch.stack(
                (
                    loss_gep.detach(),
                    loss_zero_prob.detach(),
                    loss_gepc.detach(),
                    loss_gepc_zero_prob.detach(),
                    unscaled_loss.detach(),
                )
            ).to(torch.float64)
            if diagnostics is not None:
                diagnostics.observe_micro(
                    detached_metrics, batch_data, outputs, mask_positions,
                    iter_num + 1, optimizer_step + 1,
                    sample=should_step and diagnostics.sample_due(optimizer_step + 1),
                )

            scaler.scale(loss).backward()
            if should_step:
                if grad_clip != 0.0 or diagnostics is not None:
                    scaler.unscale_(optimizer)
                    grad_norm_before_clip = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), grad_clip if grad_clip != 0.0 else float("inf"),
                    )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1
                if diagnostics is not None:
                    diagnostics.finish_update(grad_norm_before_clip, optimizer_step, iter_num + 1, lr)

            window_metrics += detached_metrics
            epoch_loss_sum += unscaled_loss.detach().to(torch.float64)
            window_steps += 1
            iter_num += 1
            batch_id += 1

            stop_after_step = (
                should_step and optimizer_step >= max_optimizer_steps
            )
            if not stop_after_step:
                # Fetch only after backward/optimizer; the next H2D copy was already
                # launched when the current batch was returned.
                batch_data = prefetcher.next()

            should_log = (
                window_steps >= args.log_interval
                or batch_id == num_batches_per_rank
                or stop_after_step
            )
            if should_log:
                torch.cuda.synchronize()
                reduced_metrics = window_metrics.clone()
                if ddp:
                    dist.all_reduce(reduced_metrics, op=dist.ReduceOp.SUM)
                now = time.perf_counter()
                elapsed = now - window_start
                metrics = (
                    reduced_metrics / (window_steps * ddp_world_size)
                ).cpu().tolist()
                cells_per_second = (
                    window_steps * batch_size * ddp_world_size / elapsed
                )
                step_time = elapsed / window_steps
                mfu = raw_model.estimate_mfu(batch_size, step_time)
                running_mfu = (
                    mfu if running_mfu < 0 else 0.9 * running_mfu + 0.1 * mfu
                )

                diagnostic_metrics = diagnostics.flush() if diagnostics is not None else {}
                if master_process:
                    loss_to_log = {
                        f"train/{name}": value
                        for name, value in zip(metric_names, metrics)
                    }
                    loss_to_log.update(
                        {
                            "train/cells_per_second": cells_per_second,
                            "train/mfu": running_mfu,
                            "train/lr": lr,
                        }
                    )
                    loss_to_log.update(diagnostic_metrics)
                    print(
                        f"[micro_step={iter_num}, optimizer_step={optimizer_step}, "
                        f"epoch_batch={batch_id}/{num_batches_per_rank}] "
                        f"loss={metrics[-1]:.4f}, "
                        f"{cells_per_second:,.1f} cells/s, "
                        f"{step_time * 1000:.1f} ms/batch, "
                        f"MFU={running_mfu * 100:.2f}%"
                    )
                    loss_history["steps"].append(
                        {
                            "epoch": epoch + 1,
                            "batch": batch_id,
                            "iter_num": iter_num,
                            "optimizer_step": optimizer_step,
                            "total_loss": metrics[-1],
                        }
                    )
                    if wandb_log:
                        wandb.log(loss_to_log, step=iter_num)

                window_metrics.zero_()
                window_steps = 0
                window_start = now

            should_validate = should_step and (
                optimizer_step == branch_config["schedule_start_optimizer_step"] + warmup_steps
                or optimizer_step % VALIDATION_INTERVAL_STEPS == 0
                or stop_after_step
                or (batch_id == num_batches_per_rank and epoch == num_epochs - 1)
            )
            if should_validate:
                validation_metric_names, validation_metrics = run_training_validation()
                validation_total_loss = validation_metrics[-1]

                if master_process:
                    is_best_validation = math.isfinite(validation_total_loss) and (
                        validation_total_loss < best_validation_loss
                    )
                    if is_best_validation:
                        best_validation_loss = validation_total_loss
                        best_validation_step = optimizer_step

                    validation_to_log = {
                        f"valid/{name}": value
                        for name, value in zip(
                            validation_metric_names, validation_metrics
                        )
                    }
                    validation_to_log["valid/optimizer_step"] = optimizer_step
                    print(
                        f"Validation [optimizer_step={optimizer_step}] "
                        f"total_loss={validation_total_loss:.4f}"
                    )
                    loss_history["validation"].append(
                        {
                            "epoch": epoch + 1,
                            "batch": batch_id,
                            "iter_num": iter_num,
                            "optimizer_step": optimizer_step,
                            "total_loss": validation_total_loss,
                        }
                    )
                    loss_history["best_validation_step"] = best_validation_step
                    loss_history["best_validation_loss"] = (
                        best_validation_loss
                        if math.isfinite(best_validation_loss)
                        else None
                    )
                    if wandb_log:
                        wandb.log(validation_to_log, step=iter_num)

                    validation_checkpoint = {
                        "model": raw_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scaler": scaler.state_dict(),
                        "epoch": epoch + 1,
                        "iter_num": iter_num,
                        "optimizer_step": optimizer_step,
                        "batches_completed": batch_id + completed_before_epoch * gradient_accumulation_steps,
                        "epoch_complete": False,
                        "average_total_loss": None,
                        "validation_total_loss": validation_total_loss,
                        "best_epoch": best_epoch,
                        "best_total_loss": (
                            best_total_loss
                            if math.isfinite(best_total_loss)
                            else None
                        ),
                        "best_validation_step": best_validation_step,
                        "best_validation_loss": best_validation_loss,
                        "total_loss_history": loss_history,
                        "species_sampling": sampling_config,
                        "logits_branch": branch_config,
                        "sampling_epoch_summary": epoch_plan.summary,
                        "reconstruction_loss": reconstruction_loss_config,
                        "gene_token_ids": torch.from_numpy(src.copy()),
                        "cls_token_id": int(cls_id),
                        "input_id_scheme": "gene_vocab_ids_with_disjoint_cls_v2",
                    }
                    validation_checkpoint_path = os.path.join(out_dir, "checkpoint_latest.pth")
                    atomic_torch_save(validation_checkpoint, validation_checkpoint_path)
                    if is_best_validation:
                        update_checkpoint_alias(validation_checkpoint_path, os.path.join(
                            out_dir, "checkpoint_best_validation_loss.pth"))
                        print(f"New best validation checkpoint: step={best_validation_step}, "
                              f"total loss={best_validation_loss:.6f}")
                    atomic_json_save(loss_history, loss_history_path)

                # Do not charge validation I/O and inference time to the next
                # training-throughput window.
                torch.cuda.synchronize()
                window_start = time.perf_counter()

            if stop_after_step:
                stop_training = True
                break

        if batch_id != num_batches_per_rank:
            raise RuntimeError("Training loader did not consume the planned epoch budget")
        reduced_epoch_loss = epoch_loss_sum.clone()
        if ddp:
            dist.all_reduce(reduced_epoch_loss, op=dist.ReduceOp.SUM)
        average_epoch_loss = (
            reduced_epoch_loss / (batch_id * ddp_world_size)
        ).item()
        if master_process:
            is_best = math.isfinite(average_epoch_loss) and (
                average_epoch_loss < best_total_loss
            )
            if is_best:
                best_total_loss = average_epoch_loss
                best_epoch = epoch + 1

            loss_history["epochs"].append(
                {
                    "epoch": epoch + 1,
                    "iter_num": iter_num,
                    "optimizer_step": optimizer_step,
                    "batches_completed": batch_id,
                    "epoch_complete": batch_id == num_batches_per_rank,
                    "average_total_loss": average_epoch_loss,
                }
            )
            loss_history["best_epoch"] = best_epoch
            loss_history["best_total_loss"] = (
                best_total_loss if math.isfinite(best_total_loss) else None
            )

            print(
                f"Epoch [{epoch + 1}/{num_epochs}], "
                f"average loss={average_epoch_loss:.4f}, lr={lr}"
            )

            checkpoint = {
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch + 1,
                "iter_num": iter_num,
                "optimizer_step": optimizer_step,
                "batches_completed": batch_id + completed_before_epoch * gradient_accumulation_steps,
                "batches_completed_this_invocation": batch_id,
                "epoch_complete": batch_id == num_batches_per_rank,
                "average_total_loss": average_epoch_loss,
                "best_epoch": best_epoch,
                "best_total_loss": (
                    best_total_loss if math.isfinite(best_total_loss) else None
                ),
                "best_validation_step": best_validation_step,
                "best_validation_loss": (
                    best_validation_loss
                    if math.isfinite(best_validation_loss)
                    else None
                ),
                "total_loss_history": loss_history,
                "species_sampling": sampling_config,
                "logits_branch": branch_config,
                "sampling_epoch_summary": epoch_plan.summary,
                "reconstruction_loss": reconstruction_loss_config,
                "gene_token_ids": torch.from_numpy(src.copy()),
                "cls_token_id": int(cls_id),
                "input_id_scheme": "gene_vocab_ids_with_disjoint_cls_v2",
            }
            last_checkpoint_path = os.path.join(
                out_dir, f"checkpoint_epoch_{epoch + 1:04d}.pth"
            )
            atomic_torch_save(checkpoint, last_checkpoint_path)
            update_checkpoint_alias(last_checkpoint_path, os.path.join(out_dir, "checkpoint_latest.pth"))
            atomic_json_save(loss_history, loss_history_path)
            print(f"Saved epoch checkpoint: {last_checkpoint_path}")

        del prefetcher, data_loader, dataset
        if stop_training:
            break

    torch.cuda.synchronize()

    t1 = time.time()

    if master_process:
        if last_checkpoint_path is not None:
            final_checkpoint_path = os.path.join(
                out_dir,
                "nanoBERTmodel_healthy_checkpoint_logits_branch.pth",
            )
            update_checkpoint_alias(last_checkpoint_path, final_checkpoint_path)
        print(f"Complete pretraining! Running time is {t1 - t0:.2f}s.")

    if diagnostics is not None:
        diagnostics.close()
    if ddp:
        destroy_process_group()


if __name__ == "__main__":
    main()
