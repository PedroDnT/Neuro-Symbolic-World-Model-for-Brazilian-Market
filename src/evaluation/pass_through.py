"""Step 4 of docs/IMPROVEMENTS.md: the pass-through of the dollar to the financial result.

For firms with a filed foreign-currency split, regress the financial result (over the total
assets of the quarter before) on the FX exposure (FX loans of the quarter before times the PTAX
move, same scale), with firm and quarter fixed effects. Score it out of sample in the 12 quarters
2022-2024 against each firm's own mean. The method and the kill rule were fixed in the tracker
before the first run.

Run: python -m src.evaluation.pass_through   (needs data/filed_panel.csv and data/macro_quarterly.csv)
Writes data/pass_through.csv, 1 row for each test quarter.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_ingestion.download import DATA_DIR

TEST_START, TEST_END = pd.Timestamp("2022-01-01"), pd.Timestamp("2024-12-31")
WINSOR = (0.01, 0.99)
MIN_TRAIN_QUARTERS = 4
TOLERANCE = 1e-12
BOOTSTRAP_DRAWS = 200
KILL_SKILL, KILL_WINS = 0.10, 9


def prepare(panel: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    shock = macro.set_index("quarter")["PTAX"].pct_change().rename("s")
    rows = panel.merge(shock, left_on="quarter", right_index=True, how="inner")
    fx = rows["fx_loans_short_lag"].fillna(0) + rows["fx_loans_long_lag"].fillna(0)
    keep = rows["fin_result_q"].notna() & (rows["total_assets_lag"] > 0) & ~rows["suspect_q"].astype(bool)
    keep &= (fx > 0) & rows["s"].notna()
    rows = rows[keep].copy()
    assets = rows["total_assets_lag"]
    rows["y"] = rows["fin_result_q"] / assets
    rows["x"] = -fx[keep] * rows["s"] / assets
    return rows[["cd_cvm", "quarter", "s", "y", "x"]].reset_index(drop=True)


def two_way_demean(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Remove firm and quarter means by alternating projections (works on an unbalanced panel)."""
    values = frame[columns].astype(float).copy()
    for _ in range(1000):
        before = values.copy()
        values -= values.groupby(frame["cd_cvm"]).transform("mean")
        values -= values.groupby(frame["quarter"]).transform("mean")
        if float((values - before).abs().to_numpy().max()) < TOLERANCE:
            break
    return values


def fit(frame: pd.DataFrame) -> tuple[float, np.ndarray, np.ndarray]:
    """θ, and the demeaned x and residuals (for the clustered standard error)."""
    demeaned = two_way_demean(frame, ["y", "x"])
    x, y = demeaned["x"].to_numpy(), demeaned["y"].to_numpy()
    theta = float((x * y).sum() / (x * x).sum())
    return theta, x, y - theta * x


def clustered_se(frame: pd.DataFrame, x: np.ndarray, residual: np.ndarray) -> float:
    """Standard error of θ, clustered by quarter."""
    scores = pd.Series(x * residual).groupby(frame["quarter"].to_numpy()).sum()
    clusters = len(scores)
    correction = clusters / (clusters - 1) if clusters > 1 else 1.0
    return float(np.sqrt(correction * (scores ** 2).sum()) / (x * x).sum())


def limits(train: pd.DataFrame) -> dict:
    return {column: tuple(train[column].quantile(list(WINSOR))) for column in ("y", "x")}


def clip(frame: pd.DataFrame, bounds: dict) -> pd.DataFrame:
    return frame.assign(**{column: frame[column].clip(*bounds[column]) for column in bounds})


def score_quarter(rows: pd.DataFrame, quarter: pd.Timestamp) -> pd.DataFrame:
    """Model and baseline predictions for the firms of `quarter`, from earlier rows only."""
    train = rows[rows["quarter"] < quarter]
    bounds = limits(train)
    train = clip(train, bounds)
    theta = fit(train)[0]
    stats = train.groupby("cd_cvm").agg(n=("y", "size"), mean_y=("y", "mean"), mean_x=("x", "mean"))
    stats = stats[stats["n"] >= MIN_TRAIN_QUARTERS]
    test = clip(rows[rows["quarter"] == quarter], bounds).join(stats, on="cd_cvm", how="inner")
    test["baseline"] = test["mean_y"]
    test["model"] = test["mean_y"] + theta * (test["x"] - test["mean_x"])
    test["theta"] = theta
    return test


def run(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    quarters = sorted(q for q in rows["quarter"].unique() if TEST_START <= q <= TEST_END)
    scored = pd.concat([score_quarter(rows, q) for q in quarters], ignore_index=True)
    scored["error_model"] = (scored["y"] - scored["model"]) ** 2
    scored["error_baseline"] = (scored["y"] - scored["baseline"]) ** 2
    table = scored.groupby("quarter").agg(
        firms=("y", "size"), s=("s", "first"), theta=("theta", "first"),
        sse_model=("error_model", "sum"), sse_baseline=("error_baseline", "sum"),
    ).reset_index()
    table["skill"] = 1 - table["sse_model"] / table["sse_baseline"]
    table["win"] = table["sse_model"] < table["sse_baseline"]
    return table, scored


def interval(rows: pd.DataFrame, seed: int = 0) -> dict:
    full = clip(rows[rows["quarter"] <= TEST_END], limits(rows[rows["quarter"] <= TEST_END]))
    theta, x, residual = fit(full)
    se = clustered_se(full, x, residual)
    rng = np.random.default_rng(seed)
    firms = full["cd_cvm"].unique()
    groups = {firm: frame for firm, frame in full.groupby("cd_cvm")}
    draws = []
    for _ in range(BOOTSTRAP_DRAWS):
        chosen = rng.choice(firms, size=len(firms), replace=True)
        sample = pd.concat([groups[f].assign(cd_cvm=k) for k, f in enumerate(chosen)], ignore_index=True)
        draws.append(fit(sample)[0])
    return {"theta": theta, "se": se, "low": theta - 1.96 * se, "high": theta + 1.96 * se,
            "boot_low": float(np.percentile(draws, 2.5)), "boot_high": float(np.percentile(draws, 97.5)),
            "firms": int(len(firms)), "rows": int(len(full))}


def verdict(table: pd.DataFrame, scored: pd.DataFrame, theta: dict) -> dict:
    skill = float(1 - scored["error_model"].sum() / scored["error_baseline"].sum())
    wins = int(table["win"].sum())
    contains = theta["low"] <= 0 <= theta["high"] or theta["low"] <= 1 <= theta["high"]
    killed = skill < KILL_SKILL or wins < KILL_WINS or contains
    return {"skill": skill, "wins": wins, "quarters": len(table), "interval_contains_0_or_1": contains, "killed": killed}


def main() -> pd.DataFrame:
    panel = pd.read_csv(DATA_DIR / "filed_panel.csv", parse_dates=["quarter"])
    macro = pd.read_csv(DATA_DIR / "macro_quarterly.csv", parse_dates=["quarter"])
    rows = prepare(panel, macro)
    table, scored = run(rows)
    theta = interval(rows)
    table.to_csv(DATA_DIR / "pass_through.csv", index=False, date_format="%Y-%m-%d")
    pd.set_option("display.width", 200)
    print(table.round(4).to_string(index=False))
    print("theta:", {k: round(v, 3) if isinstance(v, float) else v for k, v in theta.items()})
    print("verdict:", {k: round(v, 3) if isinstance(v, float) else v for k, v in verdict(table, scored, theta).items()})
    return table


if __name__ == "__main__":
    main()
