"""Model parts: the accounting projection and the variational graph recurrent cell."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch_geometric.nn import GATv2Conv


class AccountingProjectionLayer(nn.Module):
    """Project a vector onto the set where Assets - Liabilities - Equity = 0.

    Columns: [Assets, Current Liabilities, Non-Current Liabilities, Equity].
    C = [1, -1, -1, -1] and P = I - C^T (C C^T)^-1 C. The layer has no trainable parameter.

    The engine projects the predicted change, not the level. A projected change added to
    a balanced sheet gives a balanced sheet.
    """

    def __init__(self, n_accounts: int = 4):
        super().__init__()
        constraint = torch.tensor([[1.0] + [-1.0] * (n_accounts - 1)])
        projection = torch.eye(n_accounts) - torch.linalg.pinv(constraint) @ constraint
        self.register_buffer("C", constraint)
        self.register_buffer("P", projection)

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        return raw @ self.P.T


class VariationalGraphRSSMCell(nn.Module):
    """One step of the latent state of each company.

    1. GATv2 attention reads the neighbours of each company in the graph.
    2. A GRU cell updates the deterministic state h.
    3. The prior guesses the latent z from the state before the observation.
    4. The posterior corrects z with the observation.

    In training mode z is a sample of the posterior. In evaluation mode z is the
    posterior mean, so a forecast is the same on each run.
    """

    def __init__(self, feature_dim: int, hidden_dim: int, latent_dim: int, heads: int = 4, edge_dim: int | None = None):
        super().__init__()
        self.latent_dim = latent_dim
        self.gat_conv = GATv2Conv(feature_dim, hidden_dim, heads=heads, concat=False, edge_dim=edge_dim)
        self.gru_cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.prior_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(), nn.Linear(hidden_dim, 2 * latent_dim)
        )
        self.posterior_net = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.ELU(), nn.Linear(hidden_dim, 2 * latent_dim)
        )

    def _distribution(self, params: torch.Tensor) -> Normal:
        mean, logvar = torch.split(params, self.latent_dim, dim=-1)
        return Normal(mean, torch.exp(0.5 * logvar.clamp(-10.0, 5.0)) + 1e-6)

    def forward(self, x_t, edge_index, h_prev, edge_attr=None):
        """Return z, h, prior, posterior and the attention (edge_index, weight for each head)."""
        spatial, attention = self.gat_conv(x_t, edge_index, edge_attr=edge_attr, return_attention_weights=True)
        # ELU, not ReLU: with ReLU all units of this layer went to zero in training, and then
        # no input reached the recurrent state.
        spatial = F.elu(spatial)
        h_t = self.gru_cell(spatial, h_prev)
        prior = self._distribution(self.prior_net(h_prev))
        posterior = self._distribution(self.posterior_net(torch.cat([h_t, spatial], dim=-1)))
        z_t = posterior.rsample() if self.training else posterior.mean
        return z_t, h_t, prior, posterior, attention
