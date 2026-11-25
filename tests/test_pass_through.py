"""The step 4 pass-through regression on built data. No download."""
import numpy as np
import pandas as pd
import pytest

from src.evaluation import pass_through as pt

QUARTERS = pd.date_range("2016-03-31", "2024-12-31", freq="QE")


def built_rows(theta: float = 0.5, seed: int = 0, firms: int = 40) -> pd.DataFrame:
    """y = firm effect + quarter effect + θ x + noise."""
    rng = np.random.default_rng(seed)
    moves = pd.Series(rng.normal(0, 0.06, len(QUARTERS)), index=QUARTERS)
    common = pd.Series(rng.normal(0, 0.002, len(QUARTERS)), index=QUARTERS)
    rows = []
    for i in range(firms):
        level, exposure = rng.normal(-0.01, 0.005), rng.uniform(0.05, 0.4)
        for quarter in QUARTERS:
            x = -exposure * moves[quarter]
            rows.append({"cd_cvm": i, "quarter": quarter, "s": moves[quarter], "x": x,
                         "y": level + common[quarter] + theta * x + rng.normal(0, 0.001)})
    return pd.DataFrame(rows)


def test_two_way_fixed_effects_recover_theta():
    rows = built_rows(theta=0.5)
    assert pt.fit(rows)[0] == pytest.approx(0.5, abs=0.02)


def test_the_model_beats_the_firm_mean_when_the_exposure_matters():
    table, scored = pt.run(built_rows(theta=0.8))
    assert len(table) == 12
    assert table["win"].sum() >= 11
    assert 1 - scored["error_model"].sum() / scored["error_baseline"].sum() > 0.5


def test_no_row_of_the_scored_quarter_reaches_the_training():
    rows = built_rows(seed=2)
    quarter = pd.Timestamp("2023-06-30")
    before = pt.score_quarter(rows, quarter)
    changed = rows.copy()
    changed.loc[changed["quarter"] > quarter, "y"] = 5.0
    changed.loc[changed["quarter"] == quarter, "y"] += 1.0  # only the outcome of q itself
    after = pt.score_quarter(changed, quarter)
    assert np.allclose(before["model"], after["model"]) and np.allclose(before["baseline"], after["baseline"])


def test_the_kill_rule():
    table = pd.DataFrame({"win": [True] * 9 + [False] * 3})
    scored = pd.DataFrame({"error_model": [0.85], "error_baseline": [1.0]})
    assert not pt.verdict(table, scored, {"low": 0.3, "high": 0.7})["killed"]
    assert pt.verdict(table, scored, {"low": -0.1, "high": 0.7})["killed"]  # contains 0
    assert pt.verdict(table, scored, {"low": 0.6, "high": 1.1})["killed"]  # contains 1
    assert pt.verdict(table.assign(win=[True] * 8 + [False] * 4), scored, {"low": 0.3, "high": 0.7})["killed"]
    assert pt.verdict(table, scored.assign(error_model=[0.95]), {"low": 0.3, "high": 0.7})["killed"]
