def combine_pretraining_losses(
    *,
    loss_gep,
    loss_zero_prob,
    loss_gepc,
    loss_gepc_zero_prob,
    gep_weight: float,
    zero_prob_weight: float,
    gepc_weight: float,
    gepc_zero_prob_weight: float,
):
    """Combine enabled pretraining objectives without changing their raw metrics."""
    total = gep_weight * loss_gep
    optional_losses = (
        (zero_prob_weight, loss_zero_prob),
        (gepc_weight, loss_gepc),
        (gepc_zero_prob_weight, loss_gepc_zero_prob),
    )
    for weight, component in optional_losses:
        if component is not None:
            total = total + weight * component
    return total
