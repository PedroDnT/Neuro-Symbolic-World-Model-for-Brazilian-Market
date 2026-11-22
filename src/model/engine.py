"""The contagion engine: graph recurrent cell, decoder and accounting projection."""
from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn
from torch.distributions import Normal

from src.model.components import AccountingProjectionLayer, VariationalGraphRSSMCell


class Step(NamedTuple):
    prediction: torch.Tensor  # balance sheet of the next quarter, shape (N, 4), scaled units
    h: torch.Tensor  # deterministic state, shape (N, hidden_dim)
    prior: Normal
    posterior: Normal
    attention: tuple  # (edge_index with self-loops, weight of each edge for each head)


class EnterpriseContagionEngine(nn.Module):
    """Predict the balance sheet of the next quarter for each company.

    prediction = this quarter + trend of the company + learned correction

    The trend is the mean quarterly change of the company in the train split (the
    "mean change" baseline). The decoder gives a raw correction, and the projection makes
    it balanced. The trend is projected too, so the prediction balances when the input
    balances.

    The decoder reads the deterministic state h and the latent z together. With z alone
    (the starter design) the KL term makes z ignore its input, and then the prediction
    cannot react to the company or to the macro data.
    """

    def __init__(
        self,
        input_features: int = 7,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        output_features: int = 4,
        gat_heads: int = 4,
        edge_dim: int | None = None,
        n_nodes: int = 1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_features = output_features
        # Mean quarterly change of each company in the train split. It is a constant, not a
        # trained weight. Zero by default. `set_trend` fills it.
        self.register_buffer("trend", torch.zeros(n_nodes, output_features))
        self.transition_model = VariationalGraphRSSMCell(input_features, hidden_dim, latent_dim, gat_heads, edge_dim)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, 64), nn.ELU(), nn.Linear(64, output_features)
        )
        self.accounting_projection = AccountingProjectionLayer(output_features)

    def initial_state(self, n_nodes: int) -> torch.Tensor:
        return torch.zeros(n_nodes, self.hidden_dim)

    def set_trend(self, trend: torch.Tensor) -> None:
        """Set the base forecast: prediction = this quarter + trend + learned correction."""
        self.trend.copy_(self.accounting_projection(trend))

    def forward(self, x_t, edge_index, h_prev, edge_attr=None) -> Step:
        z_t, h_t, prior, posterior, attention = self.transition_model(x_t, edge_index, h_prev, edge_attr)
        correction = self.accounting_projection(self.decoder(torch.cat([h_t, z_t], dim=-1)))
        prediction = x_t[:, : self.output_features] + self.trend + correction
        return Step(prediction, h_t, prior, posterior, attention)
