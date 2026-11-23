"""Arithmetic of the accounting transmission, against values computed by hand."""
import numpy as np
import pytest

from src.simulator.transmission import TAX, Exposure, cumulative_delta, quarter_effects

EXPOSURE = Exposure(
    gross_debt=1000.0, short_share=0.25, floating_share=0.5, floating_basis="extracted",
    ipca_share=0.2, fx_share=0.1, profile_basis="extracted",
    floating_debt=500.0, ipca_debt=200.0, fx_debt=100.0, amount_source="test",
)


def residual(delta: np.ndarray) -> float:
    return float(np.abs(delta[..., 0] - delta[..., 1:].sum(axis=-1)).max())


def test_selic_shock_by_hand():
    # Extra interest each quarter: 1000 * 0.5 * 3 / 100 / 4 = 3.75, paid in cash.
    effects = quarter_effects(EXPOSURE, 3.0, 0.0, 0.0, horizon=4)
    assert effects[0].interest == pytest.approx(3.75)
    assert effects[0].delta == pytest.approx([-3.75, -TAX * 3.75, 0.0, -(1 - TAX) * 3.75])
    assert cumulative_delta(effects)[-1, 3] == pytest.approx(-(1 - TAX) * 15.0)


def test_ptax_shock_hits_quarter_1_only():
    # Revaluation: 1000 * 0.1 * 15 / 100 = 15, in quarter 1 only.
    effects = quarter_effects(EXPOSURE, 0.0, 0.0, 15.0, horizon=3)
    assert [effect.fx_revaluation for effect in effects] == pytest.approx([15.0, 0.0, 0.0])
    assert effects[0].delta == pytest.approx([TAX * 15.0, 0.25 * 15.0, 0.75 * 15.0, -(1 - TAX) * 15.0])


def test_ipca_shock_accrues_each_quarter():
    # Accrual each quarter: 1000 * 0.2 * 1 / 100 = 2.
    effects = quarter_effects(EXPOSURE, 0.0, 1.0, 0.0, horizon=2)
    assert [effect.ipca_accrual for effect in effects] == pytest.approx([2.0, 2.0])


def test_each_effect_balances():
    effects = quarter_effects(EXPOSURE, 3.0, 1.0, 15.0, horizon=4)
    assert residual(np.stack([effect.delta for effect in effects])) <= 1e-9
    assert residual(cumulative_delta(effects)) <= 1e-9


def test_zero_shock_gives_zero():
    assert not cumulative_delta(quarter_effects(EXPOSURE, 0.0, 0.0, 0.0, horizon=4)).any()


def test_unknown_floating_share_gives_an_empty_effect_not_zero():
    unknown = Exposure(1000.0, 0.25, float("nan"), "none", 0.0, 0.0, "none", float("nan"), 0.0, 0.0, "test")
    delta = cumulative_delta(quarter_effects(unknown, 3.0, 0.0, 0.0, horizon=4))
    assert np.isnan(delta[:, 3]).all()  # equity
    assert np.isnan(delta[:, 0]).all()  # assets: the interest is paid in cash
    # A zero shock gives 0, also with an unknown share.
    assert not cumulative_delta(quarter_effects(unknown, 0.0, 0.0, 0.0, horizon=4)).any()


def test_exposure_without_profile_or_estimate():
    import pandas as pd

    from src.simulator.transmission import exposures

    statements = pd.DataFrame({"debt_short": [100.0, 0.0], "debt_long": [300.0, 0.0]}, index=pd.Index([1, 2], name="cd_cvm"))
    estimate = pd.DataFrame({"cd_cvm": [], "floating_share": [], "basis": []})
    result = exposures(statements, pd.DataFrame(columns=["cd_cvm", "floating_share"]), estimate)
    assert np.isnan(result[1].floating_share) and np.isnan(result[1].floating_debt)  # debt, share unknown
    assert result[2].floating_share == 0.0 and result[2].floating_debt == 0.0  # no debt: the share does not matter


def test_the_shock_uses_the_note_amounts_not_a_share_of_the_filed_debt():
    import pandas as pd

    from src.simulator.transmission import exposures

    # Minerva-like: the note lines (38) come before swap lines, the filed debt is 30.
    statements = pd.DataFrame({"debt_short": [5.0], "debt_long": [25.0]}, index=pd.Index([7], name="cd_cvm"))
    profile = pd.DataFrame([{
        "cd_cvm": 7, "floating_share": 0.6, "ipca_share": 0.0, "fx_share": 0.4, "amount_sum": 38.0,
        "floating_debt_brl": 22.8, "ipca_debt_brl": 0.0, "fx_debt_brl": 15.2, "unit_factor": 1000.0, "unit_basis": "stated",
    }])
    exposure = exposures(statements, profile, pd.DataFrame({"cd_cvm": [], "floating_share": [], "basis": []}))[7]
    effects = quarter_effects(exposure, 0.0, 0.0, 10.0, horizon=1)
    assert effects[0].fx_revaluation == pytest.approx(1.52)  # 15.2 * 10 %, not 30 * 0.4 * 10 % = 1.2
    assert "x1,000 (stated)" in exposure.amount_source


def test_an_old_profile_file_is_refused():
    import pandas as pd

    from src.simulator.transmission import exposures

    statements = pd.DataFrame({"debt_short": [1.0], "debt_long": [1.0]}, index=pd.Index([7], name="cd_cvm"))
    old = pd.DataFrame([{"cd_cvm": 7, "floating_share": 0.5}])
    with pytest.raises(ValueError, match="older than the code"):
        exposures(statements, old, pd.DataFrame({"cd_cvm": [], "floating_share": [], "basis": []}))
