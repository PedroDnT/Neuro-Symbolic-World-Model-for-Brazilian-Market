"""Fetch SELIC, IPCA and PTAX from the BCB time series system (SGS).

Output: data/macro_quarterly.csv, one row for each calendar quarter end.

    SELIC  SGS 432  target rate, % per year, last value of the quarter
    IPCA   SGS 433  monthly change in %, compounded over the 3 months of the quarter
    PTAX   SGS 1    USD sell rate in BRL, last value of the quarter

Run: python -m src.data_ingestion.fetch_macro
"""
from __future__ import annotations

import time

import pandas as pd
from bcb import sgs

from src.data_ingestion.download import DATA_DIR
from src.protocol import data_fetch_error

MODULE = "src/data_ingestion/fetch_macro.py"
SERIES = {"SELIC": 432, "IPCA": 433, "PTAX": 1}
START, END = "2010-01-01", "2024-12-31"
# BCB accepts 10 years at most for each request on a daily series.
WINDOW_YEARS = 8


def _windows(start: str, end: str, years: int) -> list[tuple[str, str]]:
    first, last = pd.Timestamp(start), pd.Timestamp(end)
    windows = []
    while first <= last:
        stop = min(first + pd.DateOffset(years=years) - pd.Timedelta(days=1), last)
        windows.append((first.strftime("%Y-%m-%d"), stop.strftime("%Y-%m-%d")))
        first = stop + pd.Timedelta(days=1)
    return windows


def fetch_series(name: str, code: int, start: str = START, end: str = END) -> pd.Series:
    frames = []
    for window_start, window_end in _windows(start, end, WINDOW_YEARS):
        last_error: Exception | None = None
        for attempt in range(4):
            try:
                frame = sgs.get({name: code}, start=window_start, end=window_end)
                if frame.empty:
                    raise ValueError("empty DataFrame")
                frames.append(frame)
                break
            except Exception as exc:  # any failure is retried, then raised below
                last_error = exc
                time.sleep(2**attempt)
        else:
            raise data_fetch_error(MODULE, last_error) from last_error
    series = pd.concat(frames)[name].astype(float)
    return series[~series.index.duplicated()].sort_index()


def build_quarterly(selic: pd.Series, ipca: pd.Series, ptax: pd.Series) -> pd.DataFrame:
    months = ipca.resample("QE").count()
    if not (months == 3).all():
        raise ValueError(f"IPCA does not have 3 months in each quarter: {months[months != 3].to_dict()}")
    quarterly = pd.DataFrame(
        {
            "SELIC": selic.resample("QE").last(),
            "IPCA": ((1 + ipca / 100).resample("QE").prod() - 1) * 100,
            "PTAX": ptax.resample("QE").last(),
        }
    ).ffill()
    quarterly.index.name = "quarter"
    if quarterly.isna().any().any():
        raise ValueError("macro table has empty values after forward fill")
    return quarterly


def main() -> pd.DataFrame:
    series = {name: fetch_series(name, code) for name, code in SERIES.items()}
    quarterly = build_quarterly(series["SELIC"], series["IPCA"], series["PTAX"])
    DATA_DIR.mkdir(exist_ok=True)
    quarterly.to_csv(DATA_DIR / "macro_quarterly.csv", date_format="%Y-%m-%d")
    print(f"macro_quarterly.csv: {len(quarterly)} quarters, {quarterly.index[0].date()} to {quarterly.index[-1].date()}")
    print(quarterly.describe().loc[["min", "mean", "max"]].round(3).to_string())
    return quarterly


if __name__ == "__main__":
    main()
