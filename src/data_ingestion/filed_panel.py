"""Quarterly filed outcomes for the pass-through and event tests (docs/IMPROVEMENTS.md, step 2).

For every company with consolidated statements, from the raw CVM archives:
    fin_result_q       financial result (DRE 3.06), the flow of the quarter
    net_income_q       net income (DRE 3.11)
    oci_q              other comprehensive income (DRA 4.02)
    oci_translation_q  translation of foreign operations (DRA 4.02.x lines, by label)
    oci_hedge_q        hedge results in other comprehensive income (DRA 4.02.x lines, by label)
Flow of a quarter: the year to date of that quarter's filing minus the year to date of the previous
quarter's filing (Q1 = its year to date; Q4 = DFP annual minus the 9-month year to date of the Q3
ITR). So the 4 quarters always add up to the annual value, and a restatement shows in the quarter
in which it was filed. Generic tax lines ("efeito fiscal") cannot be assigned, so they stay in
oci_q only.

The stocks (assets, cash, financial investments, equity, debt, the filed foreign-currency split)
come from data/cvm_quarterly.csv, consolidated rows only. Financial companies are left out.

Writes data/filed_panel.csv, and data/dra_label_map.csv (each comprehensive-income label, its
class and how often it occurs) so that the label rules can be reviewed.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from src.data_ingestion.download import DATA_DIR
from src.data_ingestion.fetch_cvm import (
    DFP_YEARS, ITR_YEARS, KEY, _clean, _last_version, _read, _single, normalize_text, unit_fixes,
)
from src.data_ingestion.universe import FINANCIAL_SECTORS, normalize_sector, register

HEDGE = re.compile(r"hedge|derivativ|cobertura|protecao|swap")
TRANSLATION = re.compile(
    r"convers|camb(ia(l|is)|io|\.).*(exterior|investid|investiment|controlad|operac)|"
    r"(exterior|investid).*cambia|moeda estrangeira|hiperinflac"
)
# Days covered by the year to date of a Q1, Q2 and Q3 filing, and by an annual filing.
YEAR_TO_DATE = {1: (80, 100), 2: (170, 190), 3: (260, 285), 4: (350, 380)}
DRE_CODES = {"3.06": "fin_result_q", "3.11": "net_income_q"}
STOCKS = [
    "total_assets", "cash", "financial_investments", "equity", "debt_short", "debt_long",
    "fx_loans_short", "fx_loans_long",
]


def classify(label: str) -> str:
    """The class of 1 comprehensive-income line: hedge, translation or other."""
    text = normalize_text(label)
    if HEDGE.search(text):
        return "hedge"
    if TRANSLATION.search(text):
        return "translation"
    return "other"


def quarter_flows(rows: pd.DataFrame) -> pd.DataFrame:
    """The flow of each quarter for each company and account: a difference of year-to-date values.

    `rows` holds the chosen filing of each company and date (see fetch_cvm._last_version), with
    columns cd_cvm, quarter, doc, CD_CONTA, DS_CONTA, DT_INI_EXERC, DT_FIM_EXERC and value.
    """
    frame = rows.assign(days=(pd.to_datetime(rows["DT_FIM_EXERC"]) - pd.to_datetime(rows["DT_INI_EXERC"])).dt.days)
    frame = _single(frame, ["cd_cvm", "quarter", "CD_CONTA", "doc", "days"])
    frame["q"] = frame["quarter"].dt.quarter
    in_range = pd.Series(False, index=frame.index)
    for q, (low, high) in YEAR_TO_DATE.items():
        doc = "dfp" if q == 4 else "itr"
        in_range |= (frame["q"] == q) & (frame["doc"] == doc) & frame["days"].between(low, high)
    ytd = frame[in_range]
    keys = ["cd_cvm", "quarter", "CD_CONTA"]
    previous = ytd[ytd["q"] < 4][keys + ["value"]].assign(quarter=lambda f: f["quarter"] + pd.offsets.QuarterEnd(1))
    flows = ytd.merge(previous, on=keys, how="left", suffixes=("", "_before"))
    first = flows["q"] == 1
    flows["value"] = flows["value"].where(first, flows["value"] - flows["value_before"])
    flows = flows[first | flows["value_before"].notna()]
    return flows[["cd_cvm", "quarter", "CD_CONTA", "DS_CONTA", "value"]].reset_index(drop=True)


def oci_parts(flows: pd.DataFrame) -> pd.DataFrame:
    """Translation and hedge sums of each company and quarter, from DRA 4.02.x lines.

    A line counts only when no ancestor line of the same filing is counted too, so a total
    and its parts are never added together.
    """
    lines = flows[flows["CD_CONTA"].str.startswith("4.02.")].copy()
    lines["kind"] = lines["DS_CONTA"].map(classify)
    lines = lines[lines["kind"] != "other"]
    counted = lines.groupby(["cd_cvm", "quarter"])["CD_CONTA"].agg(set)

    def has_counted_ancestor(row) -> bool:
        codes = counted[(row.cd_cvm, row.quarter)]
        parts = row.CD_CONTA.split(".")
        return any(".".join(parts[:k]) in codes for k in range(2, len(parts)))

    if len(lines):
        lines = lines[~np.array([has_counted_ancestor(row) for row in lines.itertuples()], dtype=bool)]
    sums = lines.pivot_table(index=["cd_cvm", "quarter"], columns="kind", values="value", aggfunc="sum")
    return sums.reindex(columns=["translation", "hedge"]).rename(
        columns={"translation": "oci_translation_q", "hedge": "oci_hedge_q"}
    )


def _flow_rows(statement: str, keep, fixes: set) -> pd.DataFrame:
    frames = [_read("dfp", year, statement, "con", keep) for year in DFP_YEARS]
    frames += [_read("itr", year, statement, "con", keep) for year in ITR_YEARS]
    raw = pd.concat(frames, ignore_index=True)
    raw = raw[raw["ORDEM_EXERC"] == "ÚLTIMO"]
    accounts = {code: (code, "") for code in raw["CD_CONTA"].unique()}
    return _last_version(_clean(raw, accounts, fixes))


def build() -> tuple[pd.DataFrame, pd.DataFrame]:
    balance = [_read("dfp", year, "BPA", "con", {"1"}) for year in DFP_YEARS]
    balance += [_read("itr", year, "BPA", "con", {"1"}) for year in ITR_YEARS]
    fixes = unit_fixes(balance)

    dre = quarter_flows(_flow_rows("DRE", set(DRE_CODES), fixes))
    dre = dre.pivot_table(index=["cd_cvm", "quarter"], columns="CD_CONTA", values="value", aggfunc="first").rename(columns=DRE_CODES)
    dra = quarter_flows(_flow_rows("DRA", lambda codes: codes.str.startswith("4.02"), fixes))
    total = dra[dra["CD_CONTA"] == "4.02"].set_index(["cd_cvm", "quarter"])["value"].rename("oci_q")
    flows = dre.join(total, how="left").join(oci_parts(dra), how="left")

    statements = pd.read_csv(DATA_DIR / "cvm_quarterly.csv", parse_dates=["quarter"])
    if "financial_investments" not in statements:
        raise ValueError("data/cvm_quarterly.csv is older than the code: run python -m src.data_ingestion.fetch_cvm")
    stocks = statements[statements["scope"] == "con"].set_index(KEY)[STOCKS]
    panel = stocks.join(flows, how="inner").reset_index()
    sectors = register().assign(sector=lambda r: r["SETOR_ATIV"].map(normalize_sector))[["cd_cvm", "sector"]]
    panel = panel.merge(sectors, on="cd_cvm", how="left")
    panel = panel[~panel["sector"].isin(FINANCIAL_SECTORS)].sort_values(KEY)

    labels = dra[dra["CD_CONTA"].str.startswith("4.02.")]
    label_map = labels["DS_CONTA"].map(normalize_text).value_counts().rename_axis("label").reset_index(name="count")
    label_map["kind"] = label_map["label"].map(classify)
    return panel, label_map


def main() -> pd.DataFrame:
    panel, label_map = build()
    panel.to_csv(DATA_DIR / "filed_panel.csv", index=False, date_format="%Y-%m-%d")
    label_map.to_csv(DATA_DIR / "dra_label_map.csv", index=False)
    print(f"filed_panel.csv: {len(panel)} rows, {panel['cd_cvm'].nunique()} non-financial companies, "
          f"{panel['quarter'].min().date()} to {panel['quarter'].max().date()}")
    print("filled share:", panel[["fin_result_q", "oci_q", "oci_translation_q", "oci_hedge_q", "fx_loans_short"]].notna().mean().round(3).to_dict())
    print("label rows by class:", label_map.groupby("kind")["count"].sum().to_dict())
    return panel


if __name__ == "__main__":
    main()
