"""ELBO loss: reconstruction error plus the scaled KL divergence."""
from __future__ import annotations

import torch
from torch.distributions import Normal, kl_divergence

BETA_KL = 0.1


def elbo_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    posterior: Normal,
    prior: Normal,
    beta: float = BETA_KL,
    error_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (total, reconstruction, kl). total = MSE + beta * KL(posterior || prior).

    `error_scale` has the shape of the prediction. It is the usual size of a quarterly
    change of each company and account. The error is divided by it before the square, so
    the reconstruction term is near 1 for a naive forecast. Without it the term is near
    0.01, the KL term is much larger, and the latent stops reading its input.
    """
    error = prediction - target
    if error_scale is not None:
        error = error / error_scale
    reconstruction = (error**2).mean()
    kl = kl_divergence(posterior, prior).mean()
    return reconstruction + beta * kl, reconstruction, kl
