"""Epoch-independent species sampling for source-mixed, equal-sized Parquet shards.

Layout: ROOT/species_name/*.parquet (nested files inside species dirs are allowed).
Training shards must have exactly shard_rows cells (default 16384). A shorter
final shard is accepted into the fixed validation holdout. Directory names
identify sampling strata; species labels are not required as model inputs.

p_s = N_s**alpha / sum(N_j**alpha), with alpha=0.6 by default. A zero epoch_cells
budget chooses the largest approximately proportional downsampling-only epoch,
rounded down to complete global optimizer updates. Explicit larger budgets need
allow_repeats=True; each species is exhausted before reusing its shards WITHIN
that epoch. There is deliberately no coverage state shared between epochs.

Sampling is cluster sampling, not independent cell sampling. Full shards are
selected in random order; a partial final shard uses a uniformly random circular
row interval. Equal shard sizes give cells equal marginal inclusion probability
within a species. Input shards should already mix donors, tissues and batches.

DDP planning is global but reading/mixing is local. Each rank/worker gets exactly
its batch budget and nearly proportional species quotas. Only shard boundaries
may be shared by readers; no selected occurrence is assigned twice. Workers
interleave species by remaining quotas BEFORE a bounded cell shuffle buffer.

CPU-only inspection (no model or GPU needed):
  python -m scripts.species_epoch_sampling --data_path /data/train \\
    --world_size 8 --batch_size 64 --accumulation_steps 5 --num_workers 2
Use --validation_data_path for a separate fixed validation tree with the same
layout; its shards may have 1..shard_rows cells. Otherwise hold out the last
max(1, round(num_shards * validation_fraction)) shards, capped at num_shards-1,
from species with >=2 shards. Filenames use natural numeric ordering (part-10
comes after part-9), not modification time; nested paths sort component-wise.
The holdout is chosen from the end backwards, without using a random seed.
Species with one full shard remain train-only. A lone partial shard or any
partial shard remaining in training fails explicitly; no cells are discarded.
If there is no validation holdout, fail explicitly.
A non-default --shard_rows is useful for tests or a different uniform shard size.
"""

import argparse
import hashlib
import json
import math
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from torch.utils.data import IterableDataset, get_worker_info


@dataclass(frozen=True)
class Shard:
    path: str
    species: str
    rows: int
    row_groups: Tuple[int, ...]
    size_bytes: int
    mtime_ns: int


@dataclass(frozen=True)
class Span:
    shard: Shard
    start: int
    stop: int

    @property
    def rows(self):
        return self.stop - self.start


@dataclass
class EpochPlan:
    epoch: int
    seed: int
    cells: int
    world_size: int
    workers_per_rank: int
    batch_size: int
    batches_per_rank: int
    assignments: List[Dict[str, List[Span]]]
    consumer_cells: List[int]
    summary: dict


def derived_seed(seed, *parts):
    """Stable across processes and PYTHONHASHSEED values."""
    payload = json.dumps([seed, *parts], ensure_ascii=True).encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


def _natural_path_key(path):
    """Compare relative path components numerically, with deterministic tie breaks."""
    parts = tuple(
        tuple((1, int(token)) if token.isdigit() else (0, token)
              for token in re.split(r"([0-9]+)", part))
        for part in path.parts
    )
    return parts, path.as_posix()


def scan_species_shards(root, required_columns=("X",), shard_rows=16384,
                        allow_partial=False, exclude_suffixes=("_macrogene_lookup.parquet",)):
    """Read footers; optionally allow shorter shards before holdout assignment."""
    root = Path(root).resolve()
    if shard_rows < 1:
        raise ValueError("shard_rows must be positive")
    if not root.is_dir():
        raise ValueError(f"Not a data directory: {root}")
    if list(root.glob("*.parquet")):
        raise ValueError(f"Put Parquet files inside species subdirectories: {root}")
    result = {}
    seen_paths = set()
    for folder in sorted(p for p in root.iterdir() if p.is_dir()
                         and not p.name.startswith(".") and ".buckets-" not in p.name):
        shards = []
        for path in sorted(folder.rglob("*.parquet"),
                           key=lambda p: _natural_path_key(p.relative_to(folder))):
            if any(path.name.endswith(suffix) for suffix in exclude_suffixes):
                continue
            path = path.resolve()
            if path in seen_paths:
                raise ValueError(f"Duplicate/symlinked input file: {path}")
            seen_paths.add(path)
            with pq.ParquetFile(path) as parquet:
                missing = set(required_columns) - set(parquet.schema_arrow.names)
                if missing:
                    raise ValueError(f"{path}: missing columns {sorted(missing)}")
                rows = parquet.metadata.num_rows
                if allow_partial:
                    if not 1 <= rows <= shard_rows:
                        raise ValueError(
                            f"{path}: expected 1..{shard_rows} rows, got {rows}"
                        )
                elif rows != shard_rows:
                    raise ValueError(
                        f"{path}: expected exactly {shard_rows} rows, got {rows}. "
                        "Training shards must be full; partial shards belong in validation."
                    )
                groups = tuple(parquet.metadata.row_group(i).num_rows
                               for i in range(parquet.num_row_groups))
            stat = path.stat()
            shards.append(Shard(str(path), folder.name, rows, groups,
                                stat.st_size, stat.st_mtime_ns))
        if shards:
            result[folder.name] = tuple(shards)
    if not result:
        raise ValueError(f"No species subdirectories containing Parquet in {root}")
    return result


def prepare_species_data(root, required_columns=("X",), shard_rows=16384,
                         validation_fraction=0.005, split_seed=20260907,
                         validation_root=None):
    """Hold out files from the natural-order tail before validating train sizes.

    split_seed is retained for caller compatibility but is no longer used to
    choose the split. Epoch sampling seeds likewise never change this holdout.
    """
    all_shards = scan_species_shards(root, required_columns, shard_rows,
                                     allow_partial=True)
    warnings = []
    if validation_root:
        train = all_shards
        val_map = scan_species_shards(validation_root, required_columns, shard_rows,
                                      allow_partial=True)
        validation = [s for species in sorted(val_map) for s in val_map[species]]
        train_files = {s.path for shards in train.values() for s in shards}
        if train_files.intersection(s.path for s in validation):
            raise ValueError("Training and validation paths overlap")
    else:
        if not 0 < validation_fraction < 1:
            raise ValueError("validation_fraction must be between 0 and 1")
        train, validation = {}, []
        for species, shards in all_shards.items():
            if len(shards) == 1:
                train[species] = shards
                warnings.append(f"{species}: one shard, no validation holdout for this species")
            else:
                count = min(len(shards) - 1,
                            max(1, round(len(shards) * validation_fraction)))
                # Start with the final shard (possibly short), then move backwards.
                validation.extend(reversed(shards[-count:]))
                train[species] = shards[:-count]
    for species, shards in train.items():
        for shard in shards:
            if shard.rows != shard_rows:
                detail = (
                    "This species has only one partial shard; it cannot supply a full "
                    "training shard while reserving validation."
                    if len(all_shards[species]) == 1 else
                    "A partial shard remains in training after validation selection. "
                    "Check the shard numbering/order or repack the training shards."
                )
                raise ValueError(
                    f"{shard.path}: training requires exactly {shard_rows} rows, "
                    f"got {shard.rows}. {detail}"
                )
    if not validation:
        raise ValueError("No validation shards; provide --validation_data_path or more shards")
    fingerprint = hashlib.sha256()
    for split, shards in (("train", [s for v in train.values() for s in v]),
                          ("validation", validation)):
        for shard in sorted(shards, key=lambda s: s.path):
            fingerprint.update(json.dumps([split, asdict(shard)], sort_keys=True).encode())
    manifest = {
        "fingerprint": fingerprint.hexdigest(),
        "split_seed": None,
        "split_strategy": "external_validation" if validation_root else "natural_order_tail",
        "validation_fraction": None if validation_root else validation_fraction,
        "shard_rows": shard_rows,
        "train": {s: [asdict(x) for x in v] for s, v in train.items()},
        "validation": [asdict(x) for x in validation],
        "warnings": warnings,
    }
    return train, tuple(validation), manifest


def species_probabilities(train, alpha=0.6):
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must be finite and in [0, 1]")
    counts = {s: sum(x.rows for x in shards) for s, shards in sorted(train.items())}
    if not counts or any(n <= 0 for n in counts.values()):
        raise ValueError("Every training species must have cells")
    weights = {s: n ** alpha for s, n in counts.items()}
    total = sum(weights.values())
    return counts, {s: weight / total for s, weight in weights.items()}


def cell_quotas(cells, probabilities):
    raw = {s: cells * p for s, p in probabilities.items()}
    quotas = {s: math.floor(v) for s, v in raw.items()}
    order = sorted(raw, key=lambda s: (-(raw[s] - quotas[s]), s))
    for s in order[:cells - sum(quotas.values())]:
        quotas[s] += 1
    return quotas


def resolve_epoch_cells(train, alpha, requested_cells, global_update_cells,
                        allow_repeats=False):
    if requested_cells < 0 or global_update_cells < 1:
        raise ValueError("epoch_cells must be >=0 and global update size must be positive")
    counts, probabilities = species_probabilities(train, alpha)
    cap = math.floor(min(counts[s] / probabilities[s] for s in counts))
    budget = cap if requested_cells == 0 else requested_cells
    budget = budget // global_update_cells * global_update_cells
    # Protect against floating rounding at the strict downsampling limit.
    if requested_cells == 0:
        while budget > 0 and any(k > counts[s] for s, k in cell_quotas(budget, probabilities).items()):
            budget -= global_update_cells
    if budget == 0:
        raise ValueError("Epoch budget is smaller than one global optimizer update; "
                         "reduce batch/accumulation or explicitly allow a larger repeated budget")
    quotas = cell_quotas(budget, probabilities)
    over = {s: (k, counts[s]) for s, k in quotas.items() if k > counts[s]}
    if over and not allow_repeats:
        raise ValueError(f"Species quotas exceed available cells: {over}. "
                         "Reduce --epoch_cells or explicitly set --allow_species_repeats")
    return budget


def _allocate_capacity(total, capacities):
    """Integer proportional allocation, exact sum, never exceeds capacity."""
    capacity = sum(capacities)
    if total < 0 or total > capacity:
        raise ValueError("Allocation exceeds consumer capacity")
    if capacity == 0:
        return [0] * len(capacities)
    result = [total * c // capacity for c in capacities]
    order = sorted(range(len(capacities)),
                   key=lambda i: (-(total * capacities[i] % capacity), i))
    for i in order[:total - sum(result)]:
        result[i] += 1
    return result


def _select_spans(shards, count, rng):
    """Whole shards plus an unbiased circular slice of the last shard."""
    spans = []
    while count:
        order = list(shards)
        rng.shuffle(order)
        for shard in order:
            take = min(shard.rows, count)
            start = rng.randrange(shard.rows) if take < shard.rows else 0
            first = min(take, shard.rows - start)
            spans.append(Span(shard, start, start + first))
            if first < take:
                spans.append(Span(shard, 0, take - first))
            count -= take
            if count == 0:
                break
    return spans


def build_epoch_plan(train, cells, batch_size, world_size=1, num_workers=0,
                     epoch=0, seed=1337, alpha=0.6, allow_repeats=False,
                     probabilities_override=None, quotas_override=None):
    if batch_size < 1 or world_size < 1 or num_workers < 0 or epoch < 0:
        raise ValueError("Invalid batch/world/worker/epoch settings")
    if cells <= 0 or cells % (batch_size * world_size):
        raise ValueError("Epoch cells must form complete equal batches on all ranks")
    counts, probabilities = species_probabilities(train, alpha)
    if probabilities_override is not None:
        probabilities = dict(probabilities_override)
        if (set(probabilities) != set(counts)
                or any(not math.isfinite(v) or v < 0 for v in probabilities.values())
                or not math.isclose(sum(probabilities.values()), 1.0)):
            raise ValueError("Invalid probability override")
    quotas = cell_quotas(cells, probabilities) if quotas_override is None else dict(quotas_override)
    if (set(quotas) != set(counts) or sum(quotas.values()) != cells
            or any(not isinstance(v, int) or v < 0 for v in quotas.values())):
        raise ValueError("Invalid quota override")
    if not allow_repeats and any(quotas[s] > counts[s] for s in counts):
        raise ValueError("Species oversampling requires allow_repeats=True")
    workers = max(1, num_workers)
    batches = cells // (batch_size * world_size)
    base, extra = divmod(batches, workers)
    capacities = [(base + (worker < extra)) * batch_size
                  for rank in range(world_size) for worker in range(workers)]
    remaining = capacities.copy()
    assignments = [{} for _ in capacities]
    species_order = list(counts)
    random.Random(derived_seed(seed, epoch, "allocation")).shuffle(species_order)
    summaries = {}
    for species in species_order:
        quota = quotas[species]
        shares = _allocate_capacity(quota, remaining)
        selected = _select_spans(
            train[species], quota,
            random.Random(derived_seed(seed, epoch, species, "selection")),
        )
        pos, consumed = 0, 0
        # Randomize ownership order without changing worker capacities.
        consumers = list(range(len(capacities)))
        random.Random(derived_seed(seed, epoch, species, "ownership")).shuffle(consumers)
        for consumer in consumers:
            need = shares[consumer]
            remaining[consumer] -= need
            segments = []
            while need:
                span = selected[pos]
                take = min(need, span.rows - consumed)
                segments.append(Span(span.shard, span.start + consumed,
                                     span.start + consumed + take))
                consumed += take
                need -= take
                if consumed == span.rows:
                    consumed = 0
                    pos += 1
            if segments:
                assignments[consumer][species] = segments
        assert pos == len(selected) and consumed == 0
        summaries[species] = {
            "train_cells": counts[species], "probability": probabilities[species],
            "sampled_cells": quota, "unique_cells": min(quota, counts[species]),
            "passes": quota / counts[species],
            "selected_unique_shards": len({x.shard.path for x in selected}),
            "reader_shard_visits": sum(
                len({x.shard.path for x in a.get(species, [])}) for a in assignments),
        }
    assert not any(remaining)
    summary = {
        "epoch": epoch + 1, "seed": seed,
        "epoch_seed": derived_seed(seed, epoch, "selection"),
        "alpha": alpha, "cells": cells, "world_size": world_size,
        "workers_per_rank": workers, "batch_size": batch_size,
        "batches_per_rank": batches, "allow_repeats": allow_repeats,
        "species": dict(sorted(summaries.items())),
    }
    return EpochPlan(epoch, seed, cells, world_size, workers, batch_size, batches,
                     assignments, capacities, summary)


def _record_rows(batch):
    """Keep expression vectors in NumPy, not millions of Python float objects."""
    arrays = {}
    for name, array in zip(batch.schema.names, batch.columns):
        if array.null_count:
            raise ValueError(f"Null values in training column {name}")
        if name == "X":
            if pa.types.is_fixed_size_list(array.type):
                width = array.type.list_size
            elif pa.types.is_list(array.type) or pa.types.is_large_list(array.type):
                lengths = np.diff(array.offsets.to_numpy(zero_copy_only=False))
                if not len(lengths) or not np.all(lengths == lengths[0]):
                    raise ValueError("X must have a fixed vector length across cells")
                width = int(lengths[0])
            else:
                raise ValueError("X must be a list or fixed-size-list expression column")
            flat = array.flatten()
            if flat.null_count or width < 1:
                raise ValueError("X contains nulls or empty vectors")
            arrays[name] = flat.to_numpy(zero_copy_only=False).astype(
                np.float32, copy=False).reshape(len(array), width)
        else:
            arrays[name] = array.to_numpy(zero_copy_only=False)
    for i in range(batch.num_rows):
        yield {name: (values[i].copy() if name == "X" else values[i])
               for name, values in arrays.items()}


def _span_rows(spans, columns, read_batch_rows):
    for span in spans:
        # One open file/stream per active species, closed at every span boundary.
        with pq.ParquetFile(span.shard.path, memory_map=True) as parquet:
            group_start = 0
            for group, size in enumerate(span.shard.row_groups):
                group_stop = group_start + size
                if group_start >= span.stop:
                    break
                if group_stop > span.start:
                    position = group_start
                    for batch in parquet.iter_batches(
                            batch_size=read_batch_rows, row_groups=[group],
                            columns=list(columns), use_threads=False):
                        end = position + batch.num_rows
                        lo = max(span.start, position)
                        hi = min(span.stop, end)
                        if lo < hi:
                            yield from _record_rows(batch.slice(lo - position, hi - lo))
                        position = end
                        if position >= span.stop:
                            break
                group_start = group_stop


class SpeciesParquetBatchDataset(IterableDataset):
    def __init__(self, plan, rank, columns=("X",), shuffle_buffer_cells=4096,
                 read_batch_rows=256):
        super().__init__()
        if not 0 <= rank < plan.world_size:
            raise ValueError("Rank outside epoch plan")
        if shuffle_buffer_cells < 1 or read_batch_rows < 1:
            raise ValueError("Buffer and read batch sizes must be positive")
        self.plan = plan
        self.rank = rank
        self.columns = tuple(columns)
        self.shuffle_buffer_cells = shuffle_buffer_cells
        self.read_batch_rows = read_batch_rows

    def __len__(self):
        return self.plan.batches_per_rank

    def __iter__(self):
        worker = get_worker_info()
        worker_id = worker.id if worker else 0
        workers = worker.num_workers if worker else 1
        if workers != self.plan.workers_per_rank:
            raise ValueError("DataLoader worker count differs from epoch plan")
        consumer = self.rank * workers + worker_id
        assigned = self.plan.assignments[consumer]
        species = sorted(assigned)
        left = [sum(x.rows for x in assigned[s]) for s in species]
        streams = [iter(_span_rows(assigned[s], self.columns, self.read_batch_rows))
                   for s in species]
        total = sum(left)
        assert total == self.plan.consumer_cells[consumer]
        rng = random.Random(derived_seed(self.plan.seed, self.plan.epoch,
                                        self.rank, worker_id, "mix"))

        def mixed_rows():
            remaining = total
            while remaining:
                draw = rng.randrange(remaining)
                for i, count in enumerate(left):
                    if draw < count:
                        try:
                            row = next(streams[i])
                        except StopIteration as exc:
                            raise RuntimeError("Parquet stream shorter than planned") from exc
                        left[i] -= 1
                        remaining -= 1
                        yield row
                        break
                    draw -= count

        buffer, batch = [], []
        emitted = 0
        try:
            for row in mixed_rows():
                if len(buffer) < self.shuffle_buffer_cells:
                    buffer.append(row)
                    continue
                slot = rng.randrange(len(buffer))
                batch.append(buffer[slot])
                buffer[slot] = row
                if len(batch) == self.plan.batch_size:
                    yield batch
                    emitted += len(batch)
                    batch = []
            rng.shuffle(buffer)
            for row in buffer:
                batch.append(row)
                if len(batch) == self.plan.batch_size:
                    yield batch
                    emitted += len(batch)
                    batch = []
            if batch or emitted != total:
                raise RuntimeError("Worker did not emit its exact full-batch allocation")
        finally:
            for stream in streams:
                stream.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--validation_data_path")
    parser.add_argument("--shard_rows", type=int, default=16384)
    parser.add_argument("--validation_fraction", type=float, default=0.005)
    parser.add_argument("--sampling_alpha", type=float, default=0.6)
    parser.add_argument("--sampling_seed", type=int, default=1337)
    parser.add_argument("--epoch_cells", type=int, default=0)
    parser.add_argument("--allow_species_repeats", action="store_true")
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--accumulation_steps", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--epoch_index", type=int, default=0)
    args = parser.parse_args()
    train, validation, manifest = prepare_species_data(
        args.data_path, shard_rows=args.shard_rows,
        validation_fraction=args.validation_fraction,
        validation_root=args.validation_data_path)
    cells = resolve_epoch_cells(
        train, args.sampling_alpha, args.epoch_cells,
        args.world_size * args.batch_size * args.accumulation_steps,
        args.allow_species_repeats)
    plan = build_epoch_plan(
        train, cells, args.batch_size, args.world_size, args.num_workers,
        args.epoch_index, args.sampling_seed, args.sampling_alpha,
        args.allow_species_repeats)
    print(json.dumps({"manifest_fingerprint": manifest["fingerprint"],
                      "warnings": manifest["warnings"],
                      "validation_cells": sum(s.rows for s in validation),
                      "plan": plan.summary}, indent=2))


if __name__ == "__main__":
    main()
