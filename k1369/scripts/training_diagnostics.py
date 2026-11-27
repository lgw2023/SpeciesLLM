"""Opt-in, rank-local training diagnostics outside the compiled model.

No extra collectives, autograd hooks, or random draws. Reuse the pre-clip norm
already computed by training. Accumulate detached scalar statistics on device;
transfer them only at the training loop's existing synchronized log boundaries.
Full output statistics run periodically. A bounded ring holds input tensor
references (not activations/graphs) until a log boundary; only anomalous windows
copy a batch to CPU and write a capped, cooldown-limited snapshot.

JSONL records are rank-local; synchronized DDP pre-clip gradient norms describe
all-rank gradients, but losses/maxima/output statistics are NOT global maxima.
Detection is delayed until the existing log boundary (50 micro-batches by
 default). A snapshot is a single batch, not a full optimizer/RNG checkpoint.
"""
import json
import math
import os
import time
from collections import deque

import torch
import pyarrow.parquet as pq
from torch.utils.data import get_worker_info

from scripts.species_epoch_sampling import (
    SpeciesParquetBatchDataset, _record_rows, derived_seed,
)
import random


def provenance_rows(spans, columns, read_batch_rows, stratum):
    """Same sequential reads as the original loader, with compact CPU provenance."""
    source, _, species = stratum.partition('::')
    if not species:
        species, source = source, 'single'
    for span in spans:
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
                        lo, hi = max(span.start, position), min(span.stop, end)
                        if lo < hi:
                            rows = _record_rows(batch.slice(lo-position, hi-lo))
                            for offset, row in enumerate(rows, lo):
                                row['_provenance'] = (
                                    source, species, span.shard.path, group,
                                    offset, offset-group_start,
                                )
                                yield row
                        position = end
                        if position >= span.stop:
                            break
                group_start = group_stop


class DiagnosticSpeciesParquetBatchDataset(SpeciesParquetBatchDataset):
    """Preserve exact original sampling/mixing RNG while annotating each cell."""
    def __iter__(self):
        worker = get_worker_info()
        worker_id = worker.id if worker else 0
        workers = worker.num_workers if worker else 1
        if workers != self.plan.workers_per_rank:
            raise ValueError('DataLoader worker count differs from epoch plan')
        consumer = self.rank * workers + worker_id
        assigned = self.plan.assignments[consumer]
        species = sorted(assigned)
        left = [sum(x.rows for x in assigned[s]) for s in species]
        streams = [iter(provenance_rows(assigned[s], self.columns,
                                       self.read_batch_rows, s)) for s in species]
        total = sum(left)
        assert total == self.plan.consumer_cells[consumer]
        rng = random.Random(derived_seed(self.plan.seed, self.plan.epoch,
                                        self.rank, worker_id, 'mix'))

        def mixed_rows():
            remaining = total
            while remaining:
                draw = rng.randrange(remaining)
                for i, count in enumerate(left):
                    if draw < count:
                        try:
                            row = next(streams[i])
                        except StopIteration as exc:
                            raise RuntimeError('Parquet stream shorter than planned') from exc
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
                raise RuntimeError('Worker did not emit its exact full-batch allocation')
        finally:
            for stream in streams:
                stream.close()


METRICS = ('GEP', 'nzlp', 'GEPC', 'GEPC_nzlp', 'total_loss')
STATS = (
    'gep_pred_mean', 'gep_pred_std', 'gep_pred_min', 'gep_pred_max',
    'masked_target_mean', 'masked_target_std', 'masked_target_min', 'masked_target_max',
    'nonzero_prob_mean', 'nonzero_prob_std', 'nonzero_prob_min', 'nonzero_prob_max',
    'input_min', 'input_max', 'input_nonzero_fraction', 'input_finite_fraction',
)


def masked_stats(values, mask):
    """Fixed-shape reductions; no CUDA nonzero()/masked indexing size sync."""
    values = values.detach().float()
    count = mask.sum().clamp_min(1)
    selected = values.masked_fill(~mask, 0)
    mean = selected.sum() / count
    variance = (values-mean).square().masked_fill(~mask, 0).sum() / count
    return torch.stack((mean, variance.clamp_min(0).sqrt(),
                        values.masked_fill(~mask, float('inf')).amin(),
                        values.masked_fill(~mask, float('-inf')).amax()))


@torch.no_grad()
def prediction_stats(outputs, batch, mask):
    expression = batch['target_values'][:, 1:].detach().float()
    return torch.cat((
        masked_stats(outputs['model_output'], mask),
        masked_stats(batch['target_values'], mask),
        masked_stats(outputs['model_zero_prob'], mask),
        torch.stack((expression.amin(), expression.amax(),
                     (expression > 0).float().mean(),
                     torch.isfinite(expression).float().mean())),
    ))


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return 'nan' if math.isnan(value) else ('inf' if value > 0 else '-inf')
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


class LightweightDiagnostics:
    def __init__(self, out_dir, rank, device, stats_interval=50,
                 capture_batches=50, max_snapshots=3, cooldown_updates=1000,
                 burnin_updates=100, loss_ratio=5., loss_floor=1.,
                 grad_ratio=10., grad_floor=10.):
        if min(stats_interval, capture_batches, cooldown_updates) < 1 or max_snapshots < 0:
            raise ValueError('Invalid diagnostic intervals or buffer limits')
        self.rank, self.device = rank, device
        self.stats_interval = stats_interval
        self.max_snapshots, self.cooldown = max_snapshots, cooldown_updates
        self.burnin = burnin_updates
        self.loss_ratio, self.loss_floor = loss_ratio, loss_floor
        self.grad_ratio, self.grad_floor = grad_ratio, grad_floor
        self.directory = os.path.join(out_dir, 'diagnostics', f'rank{rank}')
        os.makedirs(self.directory, exist_ok=True)
        self.log = open(os.path.join(self.directory, 'metrics.jsonl'), 'a', encoding='utf-8')
        self.ring = deque(maxlen=capture_batches)
        self.pending, self.details = [], []
        self.loss_sum = torch.zeros(5, dtype=torch.float64, device=device)
        self.loss_max = torch.full((5,), float('-inf'), dtype=torch.float64, device=device)
        self.micro_count = 0
        self.loss_ema = self.grad_ema = None
        self.snapshots, self.last_snapshot = 0, -cooldown_updates
        self.context = {}

    def set_epoch(self, epoch, manifest_fingerprint, sampling_summary):
        if self.pending or self.micro_count:
            raise RuntimeError('Flush diagnostics before changing epochs')
        self.context = {'epoch': epoch, 'manifest_fingerprint': manifest_fingerprint,
                        'sampling_summary': sampling_summary}
        self.ring.clear()

    def sample_due(self, completed_update):
        return completed_update % self.stats_interval == 0

    def observe_micro(self, metrics, batch, outputs, mask, micro_step,
                      completed_update, sample=False):
        values = metrics.detach().to(dtype=torch.float64)
        self.loss_sum.add_(values)
        self.loss_max = torch.maximum(self.loss_max, values)
        self.micro_count += 1
        # Input/label tensors have no autograd graph; references add no GPU copies.
        if (self.snapshots < self.max_snapshots
                and completed_update-self.last_snapshot >= self.cooldown):
            self.ring.append((micro_step, completed_update, values[-1], {
                key: value.detach() if torch.is_tensor(value) else value
                for key, value in batch.items()
            }))
        if sample:
            self.details.append((completed_update, micro_step,
                                 prediction_stats(outputs, batch, mask)))

    def finish_update(self, grad_norm, completed_update, micro_step, lr):
        if not self.micro_count:
            raise RuntimeError('No micro-batches recorded for optimizer update')
        row = torch.cat((self.loss_sum / self.micro_count, self.loss_max,
                         grad_norm.detach().reshape(1).to(dtype=torch.float64)))
        self.pending.append((completed_update, micro_step, lr, row))
        self.loss_sum.zero_()
        self.loss_max.fill_(float('-inf'))
        self.micro_count = 0

    def _save_snapshot(self, reasons, update, window_records):
        if (not self.ring or self.snapshots >= self.max_snapshots
                or update-self.last_snapshot < self.cooldown):
            return None
        losses = torch.stack([x[2] for x in self.ring]).cpu().tolist()
        selected = next((i for i, x in enumerate(losses) if not math.isfinite(x)),
                        max(range(len(losses)), key=lambda i: losses[i]))
        micro_step, batch_update, _, batch = self.ring[selected]
        cpu_batch = {key: value.detach().cpu().clone() if torch.is_tensor(value) else value
                     for key, value in batch.items()}
        target = cpu_batch['target_values'][:, 1:].float()
        payload = {
            'rank': self.rank, **self.context, 'optimizer_step_at_detection': update,
            'snapshot_micro_step': micro_step, 'snapshot_optimizer_step': batch_update,
            'snapshot_total_loss': losses[selected], 'reasons': reasons,
            'batch': cpu_batch, 'window_metrics': window_records,
            'input_stats': {'min': float(target.min()), 'max': float(target.max()),
                            'nonzero_fraction': float((target > 0).float().mean()),
                            'finite_fraction': float(torch.isfinite(target).float().mean())},
            'provenance_fields': ['source', 'species', 'file', 'row_group',
                                  'file_row_offset', 'row_group_row_offset'],
            'note': 'Single retained batch; not a model/optimizer/RNG checkpoint. '
                    'The triggering batch may be earlier than the bounded capture ring.',
        }
        path = os.path.join(self.directory, f'anomaly_update_{update:09d}_micro_{micro_step:09d}.pt')
        temporary = path + '.tmp'
        try:
            torch.save(payload, temporary)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.remove(temporary)
        self.snapshots += 1
        self.last_snapshot = update
        return path

    def flush(self):
        """Call on every rank at an EXISTING synchronized training log boundary."""
        if not self.pending:
            return {}
        start = time.perf_counter()
        # Single tiny normal-path device->host copy for all update scalar records.
        rows = torch.stack([x[3] for x in self.pending]).cpu().tolist()
        detail_map = {}
        if self.details:
            data = torch.stack([x[2] for x in self.details]).cpu().tolist()
            detail_map = {update: (micro, dict(zip(STATS, values)))
                          for (update, micro, _), values in zip(self.details, data)}
        records, reasons = [], []
        for (update, micro, lr, _), values in zip(self.pending, rows):
            record = {'rank': self.rank, 'scope': 'rank_local',
                      'epoch': self.context.get('epoch'),
                      'optimizer_step': update, 'micro_step': micro, 'lr': lr,
                      'loss_mean': dict(zip(METRICS, values[:5])),
                      'loss_max': dict(zip(METRICS, values[5:10])),
                      'grad_norm_before_clip': values[10]}
            if update in detail_map:
                record['output_stats_micro_step'], record['output_stats'] = detail_map[update]
            finite = all(math.isfinite(v) for v in values)
            abnormal = []
            if not finite:
                abnormal.append('nonfinite_loss_or_gradient')
            if update >= self.burnin:
                if values[9] > max(self.loss_floor, (self.loss_ema or 0)*self.loss_ratio):
                    abnormal.append('loss_spike')
                if values[10] > max(self.grad_floor, (self.grad_ema or 0)*self.grad_ratio):
                    abnormal.append('gradient_spike')
            if abnormal:
                record['anomaly'] = abnormal
                reasons.extend(abnormal)
            elif finite:
                self.loss_ema = values[4] if self.loss_ema is None else .95*self.loss_ema+.05*values[4]
                self.grad_ema = values[10] if self.grad_ema is None else .95*self.grad_ema+.05*values[10]
            records.append(record)
        snapshot = self._save_snapshot(sorted(set(reasons)), self.pending[-1][0], records) if reasons else None
        if snapshot:
            records[-1]['snapshot_path'] = snapshot
            print(f'[diagnostics rank={self.rank}] saved anomaly batch: {snapshot}', flush=True)
        self.log.write(''.join(json.dumps(json_safe(x), allow_nan=False)+'\n' for x in records))
        self.log.flush()
        latest = records[-1]
        result = {'diagnostic/optimizer_step': latest['optimizer_step'],
                  'diagnostic/grad_norm_before_clip': latest['grad_norm_before_clip'],
                  'diagnostic/grad_norm_window_max': max(x['grad_norm_before_clip'] for x in records),
                  'diagnostic/anomaly_updates': sum('anomaly' in x for x in records),
                  'diagnostic/snapshots_saved': self.snapshots}
        for i, metric in enumerate(METRICS):
            result[f'diagnostic/{metric}_max'] = max(v[5+i] for v in rows)
        if detail_map:
            update = max(detail_map)
            result['diagnostic/output_stats_optimizer_step'] = update
            result.update({f'diagnostic/{k}': v for k, v in detail_map[update][1].items()})
        result['diagnostic/flush_seconds'] = time.perf_counter()-start
        self.pending.clear()
        self.details.clear()
        self.ring.clear()
        return result

    def close(self):
        self.log.close()
