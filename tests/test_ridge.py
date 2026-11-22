"""The seasonal ridge: repeatable, balanced, and exactly mean change when the weights are 0."""
import torch

from src.model.ridge import SeasonalRidge


def levels(quarters: int = 30, companies: int = 6, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    liabilities = torch.rand(quarters, companies, 3, generator=generator, dtype=torch.float64).cumsum(dim=0) * 0.1 + 1.0
    assets = liabilities.sum(dim=-1, keepdim=True)
    return torch.cat([assets, liabilities], dim=-1)


def test_fit_is_repeatable():
    data = levels()
    first = SeasonalRidge().fit(data, list(range(25)))
    second = SeasonalRidge().fit(data, list(range(25)))
    assert torch.equal(first.weights, second.weights)
    assert first.alphas == second.alphas


def test_prediction_and_rollout_balance():
    data = levels()
    ridge = SeasonalRidge().fit(data, list(range(25)))
    path = ridge.rollout(data[:26], horizon=8)
    assert float((path[..., 0] - path[..., 1:].sum(dim=-1)).abs().max()) <= 1e-9


def test_zero_weights_give_mean_change():
    data = levels()
    ridge = SeasonalRidge().fit(data, list(range(25)))
    ridge.weights = torch.zeros_like(ridge.weights)
    assert torch.allclose(ridge.predict(data, 26), data[26].double() + ridge.trend @ ridge.P.T)


def test_features_read_only_the_past():
    data = levels()
    ridge = SeasonalRidge().fit(data, list(range(25)))
    changed = data.clone()
    changed[27:] += 5.0  # the future changes
    assert torch.equal(ridge.predict(data, 26), ridge.predict(changed, 26))
