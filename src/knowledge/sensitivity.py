"""The share of a company's debt that follows the short-term rate (CDI or SELIC).

First source: the extraction of the debt note (data/debt_profile.csv). Basis "extracted".
Fallback: a regression of the implied interest rate of the company on SELIC, in the train
split. Basis "estimate". Financial expense also holds exchange-rate effects, so the
estimate is rough. With no source the share is empty, and the shock rule does not run.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_ingestion.download import DATA_DIR

MIN_POINTS = 20


def regression_share(history: pd.DataFrame, selic: pd.Series, train_end: str) -> float | None:
    """Slope of the implied rate (percent per year) on SELIC (percent per year), cut to 0..1."""
    frame = history.set_index("quarter").sort_index()
    gross_debt = (frame["debt_short"] + frame["debt_long"]).rolling(4).mean()
    implied = frame["finexp_ttm"].abs() / gross_debt * 100
    joined = pd.DataFrame({"implied": implied, "selic": selic.rolling(4).mean()}).dropna()
    joined = joined[(joined.index <= pd.Timestamp(train_end)) & np.isfinite(joined["implied"])]
    if len(joined) < MIN_POINTS or joined["selic"].var() == 0:
        return None
    slope = joined["implied"].cov(joined["selic"]) / joined["selic"].var()
    return float(min(1.0, max(0.0, slope)))


def floating_shares(universe: pd.DataFrame, train_end: str) -> pd.DataFrame:
    """1 row for each universe company: floating_share and its basis."""
    statements = pd.read_csv(DATA_DIR / "cvm_quarterly.csv", parse_dates=["quarter"])
    macro = pd.read_csv(DATA_DIR / "macro_quarterly.csv", parse_dates=["quarter"], index_col="quarter")
    profile_path = DATA_DIR / "debt_profile.csv"
    extracted = {}
    if profile_path.exists():
        profile = pd.read_csv(profile_path).dropna(subset=["floating_share"])
        extracted = dict(zip(profile["cd_cvm"], profile["floating_share"]))
    rows = []
    for company in universe.itertuples():
        if company.cd_cvm in extracted:
            share, basis = float(extracted[company.cd_cvm]), "extracted"
        else:
            history = statements[statements["cd_cvm"] == company.cd_cvm]
            share = regression_share(history, macro["SELIC"], train_end)
            basis = "estimate" if share is not None else "none"
        rows.append({"cd_cvm": company.cd_cvm, "ticker": company.ticker, "floating_share": share, "basis": basis})
    return pd.DataFrame(rows)
