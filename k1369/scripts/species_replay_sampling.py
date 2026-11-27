"""Multi-source stage-2 sampling: exact source quotas, species^alpha within source.

Source batch1 is replay (default 20%), batch2 new data (80%). Same species
across sources has one biological identity but separate source quotas. This is
an intentional source-prioritized objective, not pooled-count power sampling.
No physical data moves. First-source split matches prepare_species_data's tail
0.5% rule; batch2 holds out the last three shards per species with >=4 shards.
For smaller species, the default keep_one_train policy retains the first shard
for training and holds out the rest; a lone shard fails explicitly. Thus the
current two-shard Equus_caballus and Mesocricetus_auratus each use their last
shard for validation. Exceptions are recorded in the manifest warnings.
Explicit error / validation_only policies are also available.
Paths are checked for overlap, not cell-level biological duplicates.
"""
import argparse
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path

from scripts.species_epoch_sampling import (
    prepare_species_data, scan_species_shards, species_probabilities,
    cell_quotas, build_epoch_plan,
)

BASE = '/mnt/data/u/wangj0g/projects/SpeciesLLM/datasets/pretrain_datasets/'
DEFAULT_BATCH1 = BASE + '1st_pretrain_data_preprocessed_step4_K1500_map_then_log1p_shuffle_within_species'
DEFAULT_BATCH2 = BASE + '2nd_pretrain_data_preprocessed_step4_K1500_map_then_log1p_shuffle_within_species'


def split_new_source(source, root, required_columns, shard_rows, small_species_validation, warnings):
    new_all = scan_species_shards(root, required_columns, shard_rows, allow_partial=True,
                                  exclude_suffixes=('_macrogene_lookup.parquet',))
    if small_species_validation not in ('error','keep_one_train','validation_only'):
        raise ValueError('Unknown small_species_validation policy')
    new, new_val = {}, []
    for species, shards in new_all.items():
        count = 3
        if len(shards) < 4:
            if small_species_validation == 'error':
                raise ValueError(f'{source}/{species}: need >=4 shards to hold out exactly the last 3 and retain training; '
                                 'choose --small_species_validation keep_one_train or validation_only explicitly')
            if small_species_validation == 'keep_one_train':
                if len(shards) < 2:
                    raise ValueError(f'{source}/{species}: need >=2 shards for keep_one_train')
                count = len(shards)-1
            else:
                count = len(shards)
            warnings.append(f'{source}/{species}: {len(shards)} shards; policy={small_species_validation}, '
                            f'{count} validation and {len(shards)-count} training shards')
        if count < len(shards):
            new[species] = shards[:-count]
        new_val.extend(reversed(shards[-count:]))
        for shard in new.get(species, ()):
            if shard.rows != shard_rows:
                raise ValueError(f'{shard.path}: training requires exactly {shard_rows} rows')
    return new, tuple(new_val), {shard.path for shards in new_all.values() for shard in shards}


def prepare_replay_data(batch1, batch2, required_columns=('X',), shard_rows=16384,
                        validation_fraction=.005, small_species_validation='keep_one_train', *, batch3=None):
    old, old_val, old_manifest = prepare_species_data(
        batch1, required_columns, shard_rows, validation_fraction)
    warnings = list(old_manifest['warnings'])
    new, new_val, new_paths = split_new_source(
        'batch2', batch2, required_columns, shard_rows, small_species_validation, warnings)
    old_paths = {shard.path for shards in old.values() for shard in shards}
    old_paths.update(shard.path for shard in old_val)
    if old_paths & new_paths:
        raise ValueError('Sources overlap: the same resolved file appears in both batches')
    train = {f'{source}::{s}': shards
             for source, data in [('batch1', old), ('batch2', new)]
             for s, shards in data.items()}
    validation = {'batch1': old_val, 'batch2': tuple(new_val)}
    manifest = {
        'strategy': 'source_then_species_power',
        'batch1_manifest': old_manifest,
        'batch2_validation_shards_per_species': 3,
        'excluded_batch2_metadata_files': sorted(str(p.resolve()) for p in
            Path(batch2).rglob('*_macrogene_lookup.parquet')),
        'train': {s: [asdict(x) for x in v] for s, v in train.items()},
        'validation': {s: [asdict(x) for x in v] for s, v in validation.items()},
        'warnings': warnings,
        'small_species_validation': small_species_validation,
    }
    if batch3 is not None:
        third, third_val, third_paths = split_new_source(
            'batch3', batch3, required_columns, shard_rows, small_species_validation, warnings)
        if third_paths & (old_paths | new_paths):
            raise ValueError('Sources overlap: batch3 shares a resolved file with batch1 or batch2')
        third_train = {f'batch3::{species}': shards for species, shards in third.items()}
        train.update(third_train)
        validation['batch3'] = third_val
        manifest['train'].update({s: [asdict(x) for x in v] for s, v in third_train.items()})
        manifest['validation']['batch3'] = [asdict(x) for x in third_val]
        manifest['batch3_validation_shards_per_species'] = 3
        manifest['excluded_batch3_metadata_files'] = sorted(str(p.resolve()) for p in
            Path(batch3).rglob('*_macrogene_lookup.parquet'))
    manifest['fingerprint'] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    return train, validation, manifest


def source_weights(new_fraction, batch3_fraction=0.0):
    if not math.isfinite(new_fraction) or not math.isfinite(batch3_fraction):
        raise ValueError('Source fractions must be finite')
    if new_fraction <= 0 or batch3_fraction < 0 or new_fraction + batch3_fraction >= 1:
        raise ValueError('Require batch2_fraction > 0, batch3_fraction >= 0, and their sum < 1')
    weights = {'batch1': 1-new_fraction-batch3_fraction, 'batch2': new_fraction}
    if batch3_fraction > 0:
        weights['batch3'] = batch3_fraction
    return weights


def replay_probabilities(train, alpha=.6, new_fraction=.8, batch3_fraction=0.0):
    result = {}
    for source, weight in source_weights(new_fraction, batch3_fraction).items():
        subset = {s: v for s, v in train.items() if s.startswith(source+'::')}
        _, probabilities = species_probabilities(subset, alpha)
        result.update({s: weight*p for s, p in probabilities.items()})
    return result


def replay_quotas(cells, probabilities, new_fraction, batch3_fraction=0.0):
    budgets = cell_quotas(cells, source_weights(new_fraction, batch3_fraction))
    result = {}
    for source, budget in budgets.items():
        subset = {s: p for s,p in probabilities.items() if s.startswith(source+'::')}
        total = sum(subset.values())
        result.update(cell_quotas(budget, {s:p/total for s,p in subset.items()}))
    return result


def resolve_replay_cells(train, alpha, requested, update_cells, new_fraction=.8,
                         allow_repeats=True, batch3_fraction=0.0):
    if requested < 0 or update_cells < 1:
        raise ValueError('Invalid epoch/update cell budget')
    probabilities = replay_probabilities(train, alpha, new_fraction, batch3_fraction)
    # Default: enough total draws for the new-source draw budget to approximately
    # equal the number of NEW training cells. This is not unique-cell full coverage.
    if requested == 0:
        requested = math.floor(sum(x.rows for s,v in train.items()
                                   if s.startswith('batch2::') for x in v) / new_fraction)
    cells = requested // update_cells * update_cells
    if cells == 0:
        raise ValueError('Budget smaller than one complete global optimizer update')
    quotas = replay_quotas(cells, probabilities, new_fraction, batch3_fraction)
    if not allow_repeats:
        for s, k in quotas.items():
            if k > sum(x.rows for x in train[s]):
                raise ValueError(f'{s}: quota requires repeats; reduce epoch_cells or enable repeats')
    return cells


def build_replay_plan(train, cells, batch_size, world_size=1, num_workers=0,
                      epoch=0, seed=1337, alpha=.6, allow_repeats=True,
                      new_fraction=.8, batch3_fraction=0.0):
    probs = replay_probabilities(train, alpha, new_fraction, batch3_fraction)
    quotas = replay_quotas(cells, probs, new_fraction, batch3_fraction)
    plan = build_epoch_plan(train, cells, batch_size, world_size, num_workers,
                            epoch, seed, alpha, allow_repeats, probs, quotas)
    sources = {}
    biological = {}
    for key, stats in plan.summary['species'].items():
        source, species = key.split('::',1)
        sources[source] = sources.get(source, 0) + stats['sampled_cells']
        merged = biological.setdefault(species, {'train_cells':0, 'sampled_cells':0})
        for field in merged:
            merged[field] += stats[field]
    plan.summary.update(new_fraction=new_fraction, source_draws=sources,
                        biological_species=biological)
    if batch3_fraction:
        plan.summary["source_weights"] = source_weights(new_fraction, batch3_fraction)
    return plan


def stage2_lr(step, total_steps, warmup_steps, peak, minimum, start):
    if total_steps < 1 or not 0 <= step < total_steps:
        raise ValueError('LR step outside second-stage budget')
    if not 0 <= warmup_steps < total_steps:
        raise ValueError('Invalid warmup duration')
    if not all(math.isfinite(x) and x>0 for x in (peak, minimum, start)) or max(minimum,start)>peak:
        raise ValueError('Require positive finite LR with start/minimum <= peak')
    if total_steps == 1:
        return minimum
    # For short runs preserve endpoints even if warmup leaves no decay interval.
    if warmup_steps and step < warmup_steps:
        return start + (peak-start) * step / warmup_steps
    decay_steps = total_steps-1-warmup_steps
    if decay_steps <= 0:
        return minimum
    progress = (step-warmup_steps)/decay_steps
    return minimum + .5*(peak-minimum)*(1+math.cos(math.pi*progress))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch1_path', default=DEFAULT_BATCH1)
    parser.add_argument('--batch2_path', default=DEFAULT_BATCH2)
    parser.add_argument('--batch3_path')
    parser.add_argument('--new_data_fraction', '--batch2_fraction', type=float, default=.8)
    parser.add_argument('--batch3_fraction', type=float, default=0.0)
    parser.add_argument('--small_species_validation', choices=('error','keep_one_train','validation_only'), default='keep_one_train')
    parser.add_argument('--sampling_alpha',type=float, default=.6)
    parser.add_argument('--epoch_cells',type=int, default=0)
    parser.add_argument('--shard_rows',type=int, default=16384)
    parser.add_argument('--world_size',type=int, default=8)
    parser.add_argument('--batch_size',type=int, default=64)
    parser.add_argument('--accumulation_steps',type=int, default=5)
    parser.add_argument('--num_workers',type=int, default=2)
    parser.add_argument('--epoch_index',type=int, default=0)
    parser.add_argument('--sampling_seed',type=int, default=1337)
    parser.add_argument('--allow_species_repeats',action=argparse.BooleanOptionalAction,default=True)
    args=parser.parse_args()
    source_weights(args.new_data_fraction, args.batch3_fraction)
    if bool(args.batch3_path) != (args.batch3_fraction > 0):
        parser.error("--batch3_path and a positive --batch3_fraction must be set together")
    train, val, manifest=prepare_replay_data(args.batch1_path,args.batch2_path,shard_rows=args.shard_rows,
                                             small_species_validation=args.small_species_validation, batch3=args.batch3_path)
    cells=resolve_replay_cells(train,args.sampling_alpha,args.epoch_cells,
                              args.world_size*args.batch_size*args.accumulation_steps,
                              args.new_data_fraction,args.allow_species_repeats,args.batch3_fraction)
    plan=build_replay_plan(train,cells,args.batch_size,args.world_size,args.num_workers,
                           args.epoch_index,args.sampling_seed,args.sampling_alpha,
                           args.allow_species_repeats,args.new_data_fraction,args.batch3_fraction)
    print(json.dumps({'manifest_fingerprint':manifest['fingerprint'],
                      'warnings':manifest['warnings'],
                      'validation_cells':{s:sum(x.rows for x in v) for s,v in val.items()},
                      'plan':plan.summary},indent=2))

if __name__=='__main__':
    main()
