"""The accounting guarantee: residual at or below 1e-5, with a threshold and never equality."""
import pytest
import torch

from src.model.components import AccountingProjectionLayer
from src.model.engine import EnterpriseContagionEngine
from src.protocol import BALANCE_THRESHOLD, check_balance


def residual(balance: torch.Tensor) -> float:
    return float((balance[:, 0] - balance[:, 1:].sum(dim=1)).abs().max())


def balanced_input(n_nodes: int, n_features: int = 7) -> torch.Tensor:
    x = torch.rand(n_nodes, n_features)
    x[:, 0] = x[:, 1:4].sum(dim=1)
    return x


@pytest.mark.parametrize("sampler", [torch.randn, torch.rand])
def test_projection_residual_on_random_tensors(sampler):
    torch.manual_seed(0)
    projected = AccountingProjectionLayer()(sampler(1000, 4))
    assert residual(projected) <= BALANCE_THRESHOLD


def test_projection_matrix_properties():
    layer = AccountingProjectionLayer()
    assert torch.allclose(layer.P @ layer.P, layer.P, atol=1e-6)
    assert float((layer.C @ layer.P).abs().max()) <= 1e-6


def test_projection_keeps_a_balanced_vector():
    torch.manual_seed(0)
    balanced = balanced_input(50)[:, :4]
    assert torch.allclose(AccountingProjectionLayer()(balanced), balanced, atol=1e-6)


def test_four_step_rollout_stays_balanced():
    torch.manual_seed(0)
    n_nodes = 30
    model = EnterpriseContagionEngine().eval()
    x = balanced_input(n_nodes)
    edge_index = torch.randint(0, n_nodes, (2, 90))
    h = model.initial_state(n_nodes)
    with torch.no_grad():
        for _ in range(4):
            step = model(x, edge_index, h)
            check_balance(step.prediction)
            x = torch.cat([step.prediction, x[:, 4:]], dim=1)
            h = step.h


def test_evaluation_mode_is_deterministic():
    torch.manual_seed(0)
    model = EnterpriseContagionEngine().eval()
    x, edge_index = balanced_input(20), torch.randint(0, 20, (2, 60))
    with torch.no_grad():
        first = model(x, edge_index, model.initial_state(20)).prediction
        second = model(x, edge_index, model.initial_state(20)).prediction
    assert torch.equal(first, second)


def test_typed_edges_run():
    torch.manual_seed(0)
    model = EnterpriseContagionEngine(edge_dim=4).eval()
    x, edge_index = balanced_input(20), torch.randint(0, 20, (2, 60))
    edge_attr = torch.nn.functional.one_hot(torch.randint(0, 4, (60,)), num_classes=4).float()
    with torch.no_grad():
        check_balance(model(x, edge_index, model.initial_state(20), edge_attr).prediction)


def test_check_balance_raises_on_a_violation():
    broken = torch.tensor([[1.0, 0.2, 0.2, 0.2]])
    with pytest.raises(ValueError, match="ERR_ACCOUNTING_BALANCE"):
        check_balance(broken)


def test_trend_keeps_the_balance():
    torch.manual_seed(0)
    model = EnterpriseContagionEngine(n_nodes=20).eval()
    model.set_trend(torch.randn(20, 4))  # an unbalanced trend is projected first
    x, edge_index = balanced_input(20), torch.randint(0, 20, (2, 60))
    with torch.no_grad():
        check_balance(model(x, edge_index, model.initial_state(20)).prediction)
