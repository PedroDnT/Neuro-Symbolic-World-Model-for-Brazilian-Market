"""The debt profile: unit of the table, total check and amounts in reais. No model call."""
import pytest

from src.extraction.aggregate import debt_profile, unit_factor


def line(cd_cvm: int, indexer: str, amount: float, text: str, hedged: bool = False) -> dict:
    return {"task": "debt", "cd_cvm": cd_cvm, "indexer": indexer, "amount": amount, "hedged_to_cdi": hedged, "quote": text}


def run(cd_cvm: int, unit: str, total: float | None) -> dict:
    return {"cd_cvm": cd_cvm, "ticker": f"T{cd_cvm}", "task": "debt", "status": "ok", "summary": {"unit": unit, "total_debt": total}}


def test_a_stated_thousand_or_million_is_kept():
    assert unit_factor("BRL_thousand", 3.0, 1.0) == (1e3, "stated")
    assert unit_factor("BRL_million", 3.0, 1.0) == (1e6, "stated")


def test_a_wrong_brl_label_is_inferred_from_the_filed_debt():
    # The table is in thousands: 2,725,880 x 1,000 is the filed debt.
    assert unit_factor("BRL", 2_725_880.0, 2_725_880_000.0) == (1e3, "inferred")


def test_an_inferred_unit_outside_the_band_is_unclear():
    # Gol-like: 211,428 is 9.3 times the filed debt at x1,000,000 and 0.009 times at x1,000.
    assert unit_factor("BRL", 211_428.0, 22_600_000_000.0) == (None, "unclear")
    assert unit_factor("not_stated", 100.0, float("nan")) == (None, "unclear")


def test_amounts_in_reais_and_shares_by_channel():
    facts = [
        line(1, "CDI", 600.0, "Debêntures CDI + 1,2% 600"),
        line(1, "IPCA", 100.0, "Debêntures IPCA + 5% 100"),
        line(1, "FIXED", 300.0, "Bonds USD Fixo 5,0% 300"),
    ]
    profile = debt_profile(facts, [run(1, "BRL_thousand", 1000.0)], {1: 1_000_000.0}).iloc[0]
    assert profile["status"] == "ok"
    assert profile["floating_debt_brl"] == pytest.approx(600_000.0)
    assert profile["ipca_debt_brl"] == pytest.approx(100_000.0)
    assert profile["fx_debt_brl"] == pytest.approx(300_000.0)
    assert profile["fx_share"] == pytest.approx(0.3)


def test_a_wrong_stated_total_passes_through_the_filed_debt():
    # Taurus-like: the model read the total of the year before (528,977); the lines match the filed debt.
    facts = [line(2, "USD", 747_538.0, "Adiantamentos de câmbio USD 747.538")]
    profile = debt_profile(facts, [run(2, "not_stated", 528_977.0)], {2: 747_538_000.0}).iloc[0]
    assert profile["status"] == "ok_filed_total"
    assert profile["unit_basis"] == "inferred" and profile["unit_factor"] == 1e3


def test_lines_that_miss_both_totals_are_not_usable():
    facts = [line(3, "CDI", 400.0, "NCE CDI 400")]
    profile = debt_profile(facts, [run(3, "BRL_thousand", 1000.0)], {3: 1_000_000.0}).iloc[0]
    assert profile["status"] == "sum_differs_from_total"
    assert profile["floating_debt_brl"] is None or profile["floating_debt_brl"] != profile["floating_debt_brl"]


def test_fx_check_statuses():
    from src.extraction.aggregate import fx_check

    assert fx_check(95.0, 100.0) == "ok"
    assert fx_check(42.0, 100.0) == "fx_lines_missing"
    assert fx_check(1337.0, 100.0) == "fx_above_filed"
    assert fx_check(500.0, 0.0) == "no_filed_split"  # a filed 0 is not proof of no FX debt
    assert fx_check(500.0, float("nan")) == "no_filed_split"


def test_the_fx_check_counts_swapped_lines_too():
    facts = [
        line(4, "USD", 600.0, "Bonds USD 600", hedged=True),
        line(4, "CDI", 400.0, "NCE CDI 400"),
    ]
    profile = debt_profile(facts, [run(4, "BRL_thousand", 1000.0)], {4: 1_000_000.0}, {4: 600_000.0}).iloc[0]
    assert profile["fx_debt_brl"] == 0  # swapped: no PTAX shock
    assert profile["fx_lines_brl"] == pytest.approx(600_000.0)
    assert profile["fx_check"] == "ok"


def test_the_fx_check_counts_a_dollar_line_indexed_to_cdi():
    from src.extraction.aggregate import is_fx, names_currency

    swapped = line(5, "CDI", 99.0, "PPE com swap (b) US$ nov-29 97,00% CDI 99", hedged=True)
    assert names_currency(swapped) and not is_fx(swapped)  # in the check, but no PTAX shock
