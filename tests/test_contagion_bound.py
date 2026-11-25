"""The step 6 contagion bound on a small built network. No download."""
import pandas as pd
import pytest

from src.evaluation.contagion_bound import links, propagate


def network():
    index = pd.Index([1, 2, 3], name="cd_cvm")
    direct = pd.Series([50.0, 0.0, 0.0], index=index)
    equity = pd.Series([100.0, 1000.0, 1000.0], index=index)
    receivables = pd.Series([0.0, 80.0, 0.0], index=index)
    return direct, equity, receivables


def test_a_supplier_loses_its_receivables_times_the_customer_distress():
    direct, equity, receivables = network()
    edges = links(pd.DataFrame(columns=["relation", "src_cd_cvm", "dst_cd_cvm", "weight"]),
                  pd.DataFrame({"src_cd_cvm": [2], "dst_cd_cvm": [1]}))  # 2 supplies 1
    loss, added = propagate(direct, equity, receivables, edges)
    assert loss[2] == pytest.approx(80.0 * 0.5)  # customer 1 lost half its equity
    assert added["supplier"] == pytest.approx(40.0)


def test_an_owner_loses_its_stake_of_the_owned_company_loss():
    direct, equity, receivables = network()
    fre = pd.DataFrame({"relation": ["ownership"], "src_cd_cvm": [3], "dst_cd_cvm": [1], "weight": [30.0]})
    edges = links(fre, pd.DataFrame(columns=["src_cd_cvm", "dst_cd_cvm"]))
    loss, _ = propagate(direct, equity, receivables, edges)
    assert loss[3] == pytest.approx(15.0)


def test_losses_are_capped_at_equity_and_a_cycle_converges():
    index = pd.Index([1, 2], name="cd_cvm")
    direct = pd.Series([150.0, 10.0], index=index)
    equity = pd.Series([100.0, 50.0], index=index)
    receivables = pd.Series([500.0, 500.0], index=index)
    fre = pd.DataFrame({"relation": ["related_party"], "src_cd_cvm": [1], "dst_cd_cvm": [2], "weight": [1.0]})
    edges = links(fre, pd.DataFrame(columns=["src_cd_cvm", "dst_cd_cvm"]))
    loss, _ = propagate(direct, equity, receivables, edges)
    assert loss[1] == pytest.approx(100.0)
    assert loss[2] == pytest.approx(50.0)  # full distress of 1, capped at its own equity


def test_no_links_give_no_second_round():
    direct, equity, receivables = network()
    edges = links(pd.DataFrame(columns=["relation", "src_cd_cvm", "dst_cd_cvm", "weight"]), pd.DataFrame(columns=["src_cd_cvm", "dst_cd_cvm"]))
    loss, added = propagate(direct, equity, receivables, edges)
    assert loss.sum() == pytest.approx(direct.sum())
    assert sum(added.values()) == 0.0
