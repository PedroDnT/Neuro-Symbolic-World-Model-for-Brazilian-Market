"""Step 3 of docs/IMPROVEMENTS.md: the signed first-hop event test.

In each quarter since 2015 with a PTAX move above 10 %, rank the firms by the filed equity effect
(financial result + translation + hedge in other comprehensive income, over the equity of the
quarter before) and score 4 predictors by their Spearman correlation with it. Each predictor uses
only the quarters before the scored quarter. The method and the pass rule were fixed in the
tracker before the first run.

Run: python -m src.evaluation.event_test   (needs data/filed_panel.csv and data/macro_quarterly.csv)
Writes data/event_test.csv, 1 row for each event quarter.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_ingestion.download import DATA_DIR

EVENT_MOVE = 0.10
FIRST_EVENT = pd.Timestamp("2015-01-01")
TAX = 0.34
EQUITY_SHARE, EQUITY_FLOOR = 0.05, 10e6  # equity_lag >= 5 % of total assets and >= R$ 10 million
MIN_FIRMS = 50
MIN_FIRM_QUARTERS = 8
WINSOR = (0.01, 0.99)
PASS_WINS, PASS_GAIN, PASS_T = 7, 0.15, 2.0
PREDICTORS = ["shipped", "pass_through", "signed", "sector", "firm_mean"]


def prepare(panel: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    """The rows that meet the firm rule, with the shock and the scaled variables."""
    shock = macro.set_index("quarter")["PTAX"].pct_change().rename("s")
    rows = panel.merge(shock, left_on="quarter", right_index=True, how="inner")
    keep = rows[["fin_result_q", "oci_translation_q", "oci_hedge_q", "equity_lag", "total_assets_lag"]].notna().all(axis=1)
    keep &= ~rows["suspect_q"].astype(bool)
    keep &= (rows["equity_lag"] >= EQUITY_SHARE * rows["total_assets_lag"]) & (rows["equity_lag"] >= EQUITY_FLOOR)
    rows = rows[keep & rows["s"].notna()].copy()
    equity = rows["equity_lag"]
    rows["target"] = (rows["fin_result_q"] + rows["oci_translation_q"] + rows["oci_hedge_q"]) / equity
    rows["fin"] = rows["fin_result_q"] / equity
    rows["oci_fx"] = (rows["oci_translation_q"] + rows["oci_hedge_q"]) / equity
    rows["fx_exposure"] = -(rows["fx_loans_short_lag"].fillna(0) + rows["fx_loans_long_lag"].fillna(0)) * rows["s"] / equity
    rows["debt_effect"] = -(1 - TAX) * (rows["debt_short_lag"].fillna(0) + rows["debt_long_lag"].fillna(0)) * rows["s"] / equity
    return rows.reset_index(drop=True)


def winsorize(values: pd.Series) -> pd.Series:
    low, high = values.quantile(list(WINSOR))
    return values.clip(low, high)


def pass_through(train: pd.DataFrame) -> float:
    """Slope of the scaled financial result on the scaled FX exposure, with quarter fixed effects."""
    frame = pd.DataFrame({"y": winsorize(train["fin"]), "x": winsorize(train["fx_exposure"]), "q": train["quarter"]})
    demeaned = frame[["y", "x"]] - frame.groupby("q")[["y", "x"]].transform("mean")
    denominator = (demeaned["x"] ** 2).sum()
    return float((demeaned["x"] * demeaned["y"]).sum() / denominator) if denominator > 0 else 0.0


def firm_slopes(train: pd.DataFrame) -> pd.Series:
    """Each firm's slope through the origin of its scaled translation and hedge results on s."""
    frame = pd.DataFrame({"cd_cvm": train["cd_cvm"], "s": train["s"], "o": winsorize(train["oci_fx"])})
    grouped = frame.assign(so=frame["s"] * frame["o"], ss=frame["s"] ** 2).groupby("cd_cvm")
    sums = grouped[["so", "ss"]].sum().join(grouped.size().rename("n"))
    slopes = (sums["so"] / sums["ss"]).where((sums["n"] >= MIN_FIRM_QUARTERS) & (sums["ss"] > 0), 0.0)
    return slopes.fillna(0.0)


def sector_medians(train: pd.DataFrame) -> tuple[pd.Series, float]:
    """Median of target / s per sector over the training quarters with a large move."""
    events = train[train["s"].abs() > EVENT_MOVE]
    response = events["target"] / events["s"]
    return response.groupby(events["sector"]).median(), float(response.median())


def firm_means(train: pd.DataFrame) -> pd.Series:
    last = train.sort_values("quarter").groupby("cd_cvm").tail(4)
    return last.groupby("cd_cvm")["target"].mean()


def predict(rows: pd.DataFrame, quarter: pd.Timestamp) -> pd.DataFrame:
    """The predictions of each predictor for the firms of `quarter`, from earlier quarters only."""
    train = rows[rows["quarter"] < quarter]
    test = rows[rows["quarter"] == quarter].copy()
    theta = pass_through(train)
    slopes = firm_slopes(train)
    sectors, overall = sector_medians(train)
    test["shipped"] = test["debt_effect"]
    test["pass_through"] = theta * test["fx_exposure"]
    test["signed"] = test["pass_through"] + test["cd_cvm"].map(slopes).fillna(0.0) * test["s"]
    test["sector"] = test["sector"].map(sectors).fillna(overall) * test["s"]
    test["firm_mean"] = test["cd_cvm"].map(firm_means(train))
    test["theta"] = theta
    return test


def spearman(predicted: pd.Series, target: pd.Series) -> float:
    both = pd.DataFrame({"p": predicted, "t": target}).dropna()
    if len(both) < 3 or both["p"].nunique() < 2:
        return 0.0  # a constant predictor has no rank information
    return float(both["p"].rank().corr(both["t"].rank()))


def event_quarters(macro: pd.DataFrame) -> list[pd.Timestamp]:
    shock = macro.set_index("quarter")["PTAX"].pct_change()
    return [q for q, move in shock.items() if q >= FIRST_EVENT and abs(move) > EVENT_MOVE]


def run(rows: pd.DataFrame, quarters: list[pd.Timestamp]) -> pd.DataFrame:
    results = []
    for quarter in quarters:
        test = predict(rows, quarter)
        result = {"quarter": quarter.date(), "s": float(test["s"].iloc[0]) if len(test) else np.nan,
                  "firms": len(test), "theta": float(test["theta"].iloc[0]) if len(test) else np.nan}
        for name in PREDICTORS:
            result[name] = spearman(test[name], test["target"])
        result["firm_mean_firms"] = int(test["firm_mean"].notna().sum())
        results.append(result)
    table = pd.DataFrame(results)
    table["gain"] = table["signed"] - table[["shipped", "sector"]].max(axis=1)
    table["wins"] = (table["signed"] > table["shipped"]) & (table["signed"] > table["sector"])
    return table


def verdict(table: pd.DataFrame) -> dict:
    gains = table["gain"]
    t = float(gains.mean() / (gains.std(ddof=1) / np.sqrt(len(gains)))) if len(gains) > 1 and gains.std(ddof=1) > 0 else 0.0
    enough = bool((table["firms"] >= MIN_FIRMS).all())
    wins, mean_gain = int(table["wins"].sum()), float(gains.mean())
    passed = enough and wins >= PASS_WINS and mean_gain >= PASS_GAIN and t >= PASS_T
    return {"quarters": len(table), "wins": wins, "mean_gain": mean_gain, "t": t, "enough_firms": enough, "passed": passed}


def main() -> pd.DataFrame:
    panel = pd.read_csv(DATA_DIR / "filed_panel.csv", parse_dates=["quarter"])
    macro = pd.read_csv(DATA_DIR / "macro_quarterly.csv", parse_dates=["quarter"])
    quarters = event_quarters(macro)
    table = run(prepare(panel, macro), quarters)
    table.to_csv(DATA_DIR / "event_test.csv", index=False)
    pd.set_option("display.width", 200)
    print(table.round(3).to_string(index=False))
    print("verdict:", {k: round(v, 3) if isinstance(v, float) else v for k, v in verdict(table).items()})
    return table


if __name__ == "__main__":
    main()
