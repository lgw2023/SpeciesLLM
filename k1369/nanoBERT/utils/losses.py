import numpy as np
import torch
import torch.nn.functional as F

def masked_mse_loss(
    input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute the masked MSE loss between input and target.
    """
    mask = mask.float()
    loss = F.mse_loss(input * mask, target * mask, reduction="sum")
    return loss / mask.sum()

def masked_huber_loss(
    input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
    delta: float = 1.0,
) -> torch.Tensor:
    """Mean Huber loss over a boolean mask, with FP32 arithmetic under AMP.

    ``delta`` is in target-expression units. An empty mask returns zero loss
    and zero gradients. No per-macrogene variance weighting is applied.
    """
    if not np.isfinite(delta) or delta <= 0:
        raise ValueError("delta must be finite and positive")
    mask = mask.bool()
    masked_input = input.float().masked_fill(~mask, 0.0)
    masked_target = target.float().masked_fill(~mask, 0.0)
    loss = F.huber_loss(masked_input, masked_target, reduction="sum", delta=delta)
    return loss / mask.sum().clamp_min(1)


def criterion_neg_log_bernoulli(
    input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute the negative log-likelihood of Bernoulli distribution
    """
    mask = mask.float()
    bernoulli = torch.distributions.Bernoulli(probs=input, validate_args=False)
    masked_log_probs = bernoulli.log_prob((target > 0).float()) * mask
    return -masked_log_probs.sum() / mask.sum()

def masked_relative_error(
    input: torch.Tensor, target: torch.Tensor, mask: torch.LongTensor
) -> torch.Tensor:
    """
    Compute the masked relative error between input and target
    """
    assert mask.any()
    loss = torch.abs(input[mask] - target[mask]) / (target[mask] + 1e-6)
    return loss.mean()