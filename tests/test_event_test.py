"""The step 3 event test on built data. No download."""
import numpy as np
import pandas as pd
import pytest

from src.evaluation import event_test as et

QUARTERS = pd.date_range("2011-03-31", periods=24, freq="QE")


def built_panel(seed: int = 0, firms: int = 60) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Firms whose translation response to the dollar is a fixed firm trait, and no other FX effect."""
    rng = np.random.default_rng(seed)
    moves = rng.normal(0, 0.06, len(QUARTERS))
    moves[[3, 9, 15, 21]] = [0.15, -0.12, 0.2, 0.14]
    ptax = 2.0 * np.cumprod(1 + np.r_[0.0, moves[1:]])
    macro = pd.DataFrame({"quarter": QUARTERS, "PTAX": ptax})
    shock = pd.Series(np.r_[np.nan, moves[1:]], index=QUARTERS)
    beta = rng.normal(0, 1, firms)
    rows = []
    for i in range(firms):
        for quarter in QUARTERS:
            s = shock[quarter]
            translation = 0.0 if np.isnan(s) else beta[i] * s * 1e8 + rng.normal(0, 1e5)
            rows.append({
                "cd_cvm": i, "quarter": quarter, "sector": f"sector {i % 3}", "suspect_q": False,
                "fin_result_q": -1e6 + rng.normal(0, 1e5), "oci_translation_q": translation, "oci_hedge_q": 0.0,
                "equity_lag": 1e8, "total_assets_lag": 3e8, "fx_loans_short_lag": 0.0, "fx_loans_long_lag": 0.0,
                "debt_short_lag": rng.uniform(1e7, 5e7), "debt_long_lag": rng.uniform(1e7, 5e7),
            })
    return pd.DataFrame(rows), macro


def test_the_signed_predictor_finds_a_firm_trait_and_passes():
    panel, macro = built_panel()
    rows = et.prepare(panel, macro)
    quarters = [QUARTERS[i] for i in (9, 15, 21)]
    table = et.run(rows, quarters)
    assert (table["signed"] > 0.9).all()
    assert (table["shipped"].abs() < 0.5).all()


def test_no_row_of_the_scored_quarter_reaches_the_training():
    panel, macro = built_panel(seed=1)
    rows = et.prepare(panel, macro)
    quarter = QUARTERS[15]
    before = et.predict(rows, quarter)
    changed = rows.copy()
    later = changed["quarter"] >= quarter
    changed.loc[later, ["target", "fin", "oci_fx"]] = 999.0
    after = et.predict(changed, quarter)
    for name in ["pass_through", "signed", "sector", "firm_mean"]:
        assert np.allclose(before[name].to_numpy(), after[name].to_numpy(), equal_nan=True), name


def test_the_firm_rule_drops_small_equity_unknown_classes_and_suspect_rows():
    panel, macro = built_panel(firms=4)
    panel.loc[(panel["cd_cvm"] == 0), "equity_lag"] = 1e6  # below R$ 10 million
    panel.loc[(panel["cd_cvm"] == 1), "oci_hedge_q"] = np.nan  # unknown classes
    panel.loc[(panel["cd_cvm"] == 2), "suspect_q"] = True
    rows = et.prepare(panel, macro)
    assert set(rows["cd_cvm"]) == {3}


def test_a_firm_with_few_quarters_gets_a_zero_slope():
    train = pd.DataFrame({"cd_cvm": [1] * 7 + [2] * 8, "s": [0.1] * 15, "oci_fx": [0.05] * 15})
    slopes = et.firm_slopes(train)
    assert slopes[1] == 0.0
    assert slopes[2] == pytest.approx(0.5)


def test_the_pass_rule():
    table = pd.DataFrame({
        "firms": [100] * 9, "signed": [0.5] * 9, "shipped": [0.0] * 9,
        "sector": [0.2, 0.3, 0.25, 0.3, 0.3, 0.3, 0.3, 0.6, 0.6],
    })
    table["gain"] = table["signed"] - table[["shipped", "sector"]].max(axis=1)
    table["wins"] = (table["signed"] > table["shipped"]) & (table["signed"] > table["sector"])
    result = et.verdict(table)
    assert result["wins"] == 7 and result["mean_gain"] == pytest.approx(0.15)
    assert result["passed"] == (result["t"] >= 2)
    table.loc[0, "firms"] = 40
    assert not et.verdict(table)["passed"]  # fewer than 50 firms in a quarter
