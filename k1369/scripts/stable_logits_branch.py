"""Masked FP32 logits BCE and strict, budget-aware healthy-checkpoint branching."""
from pathlib import Path

import torch
import torch.nn.functional as F


def masked_fp32_logits_bce(logits, target, mask):
    """Mean over the original MLM mask; label 1 means expression > 0.

    Cast RAW logits, never sigmoid probabilities. Fixed-shape masked_fill avoids
    CUDA nonzero/boolean-index size synchronization. Empty masks return zero.
    Unselected values cannot contribute, even when unselected logits are NaN.
    """
    if logits.shape != target.shape or logits.shape != mask.shape:
        raise ValueError('Logits, expression targets and mask must have identical shapes')
    with torch.autocast(device_type=logits.device.type, enabled=False):
        selected = mask.bool()
        values = logits.float().masked_fill(~selected, 0.)
        labels = (target > 0).float().masked_fill(~selected, 0.)
        losses = F.binary_cross_entropy_with_logits(values, labels, reduction='none')
        return losses.masked_fill(~selected, 0.).sum() / selected.sum().clamp_min(1)


def branch_budget(state, epochs, epoch_cells, update_cells, requested_warmup=300):
    """Keep the original total draw budget and use a new LR phase for the branch.

    Each invocation independently samples its unfinished epoch budget. No saved
    shuffle/worker/RNG cursor exists, so this is not exact data-stream replay.
    A checkpoint written by this variant retains the initial branch LR origin.
    """
    if min(epochs, epoch_cells, update_cells) < 1 or epoch_cells % update_cells:
        raise ValueError('Epoch cells must be a positive multiple of global update cells')
    if requested_warmup < 0:
        raise ValueError('Warmup updates must be nonnegative')
    completed = int(state['optimizer_step'])
    per_epoch = epoch_cells // update_cells
    total = epochs * per_epoch
    previous = state.get('logits_branch') or {}
    origin = int(previous.get('schedule_start_optimizer_step', completed))
    if not 0 <= origin <= completed < total:
        raise ValueError('Checkpoint update is outside the requested total training budget')
    if previous and int(previous['max_optimizer_steps']) != total:
        raise ValueError('Existing logits branch must retain its total schedule budget')
    phase_total = total-origin
    warmup = min(requested_warmup, max(0, phase_total-2))
    if previous and int(previous['warmup_steps']) != warmup:
        raise ValueError('Existing logits branch must retain its original warmup duration')
    start_epoch, completed_in_epoch = divmod(completed, per_epoch)
    return {
        'schedule_start_optimizer_step': origin,
        'resume_optimizer_step': completed,
        'max_optimizer_steps': total,
        'optimizer_steps_per_epoch': per_epoch,
        'start_epoch_index': start_epoch,
        'completed_updates_in_start_epoch': completed_in_epoch,
        'phase_total_optimizer_steps': phase_total,
        'remaining_optimizer_steps': total-completed,
        'remaining_cell_draws': (total-completed)*update_cells,
        'first_epoch_remaining_cells': (per_epoch-completed_in_epoch)*update_cells,
        'warmup_steps': warmup,
        'data_continuation': 'independent_resampling_of_remaining_budget',
    }


def restore_model_weights(model, state, gene_ids, cls_id):
    """Load only matching model weights and verify gene-token compatibility."""
    if not torch.equal(state['gene_token_ids'].cpu(), gene_ids.cpu()):
        raise ValueError('Checkpoint gene IDs/order differ')
    if int(state['cls_token_id']) != int(cls_id):
        raise ValueError('Checkpoint CLS ID differs')
    if state.get('input_id_scheme') != 'gene_vocab_ids_with_disjoint_cls_v2':
        raise ValueError('Checkpoint input ID scheme differs')
    for name, value in state['model'].items():
        if torch.is_tensor(value) and not torch.isfinite(value).all():
            raise ValueError(f'Non-finite checkpoint model parameter: {name}')
    model.load_state_dict(state['model'], strict=True)


def restore_healthy_branch(model, optimizer, state, gene_ids, cls_id,
                           manifest_fingerprint, current_sampling, mode='auto'):
    """Strict model restore, no category expansion; optimizer reset for old loss.

    Auto resets old Bernoulli-loss moments; a checkpoint from this logits branch
    instead restores its optimizer/scaler. Explicit keep/reset is also supported.
    """
    if mode not in ('auto', 'keep', 'reset'):
        raise ValueError('Optimizer state must be auto, keep or reset')
    saved = state.get('species_sampling') or {}
    if saved.get('manifest_fingerprint') != manifest_fingerprint:
        raise ValueError('Checkpoint data/validation manifest fingerprint differs')
    for key in ('new_data_fraction', 'alpha', 'epoch_cells', 'joint_sampling',
                'allow_species_repeats', 'shard_rows'):
        if saved.get(key) != current_sampling.get(key):
            raise ValueError(f'Checkpoint sampling setting differs: {key}')
    saved_global_batch = (saved['world_size']*saved['batch_size']
                          *saved['gradient_accumulation_steps'])
    current_global_batch = (current_sampling['world_size']*current_sampling['batch_size']
                            *current_sampling['gradient_accumulation_steps'])
    if saved_global_batch != current_global_batch:
        raise ValueError('Checkpoint global update batch differs')
    previous = state.get('logits_branch') or {}
    if previous:
        for key in ('learning_rate', 'min_lr', 'sampling_seed'):
            wanted = current_sampling['seed'] if key == 'sampling_seed' else current_sampling[key]
            if previous[key] != wanted:
                raise ValueError(f'Existing logits branch setting differs: {key}')
    if saved.get('batch3_fraction', 0.0) != current_sampling.get('batch3_fraction', 0.0):
        raise ValueError('Checkpoint sampling setting differs: batch3_fraction')
    for key in ('model_config', 'embedding_dirs'):
        if key in saved and key in current_sampling and saved[key] != current_sampling[key]:
            raise ValueError(f'Checkpoint setting differs: {key}')
    restore_model_weights(model, state, gene_ids, cls_id)
    initial_empty_optimizer = (previous and not (state.get('optimizer') or {}).get('state')
                               and int(state['optimizer_step']) ==
                               int(previous['schedule_start_optimizer_step']))
    effective_mode = ('keep' if previous and not initial_empty_optimizer else 'reset') if mode == 'auto' else mode
    if effective_mode == 'reset':
        optimizer.state.clear()
    else:
        saved_optimizer = state.get('optimizer') or {}
        if not saved_optimizer.get('state'):
            raise ValueError('Keeping optimizer requires populated checkpoint state')
        # Validate on CPU before transferring moments to the training device.
        for values in saved_optimizer['state'].values():
            for name, value in values.items():
                if torch.is_tensor(value) and not torch.isfinite(value).all():
                    raise ValueError(f'Non-finite checkpoint optimizer tensor: {name}')
        optimizer.load_state_dict(saved_optimizer)
        for parameter, values in optimizer.state.items():
            for name in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
                if name in values and values[name].shape != parameter.shape:
                    raise ValueError(f'Optimizer moment shape differs: {name}')
    return {'requested_optimizer_mode': mode, 'effective_optimizer_mode': effective_mode,
            'optimizer_parameters_restored': len(optimizer.state),
            'checkpoint_validation_loss_under_old_or_saved_objective': state.get('validation_total_loss'),
            'strict_model_restore': True}


def validate_branch_output(checkpoint_path, out_path):
    checkpoint = Path(checkpoint_path).resolve() if checkpoint_path else None
    output = Path(out_path).resolve()
    if checkpoint is not None and not checkpoint.is_file():
        raise ValueError(f'Checkpoint does not exist: {checkpoint}')
    if checkpoint is not None and output == checkpoint.parent:
        raise ValueError('Branch output must differ from the checkpoint directory')
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError('Use a new empty output directory for every branch invocation')
