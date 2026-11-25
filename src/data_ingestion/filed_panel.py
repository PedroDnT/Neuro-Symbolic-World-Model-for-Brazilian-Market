"""Quarterly filed outcomes for the pass-through and event tests (docs/IMPROVEMENTS.md, step 2).

For every non-financial company, from the raw CVM archives:
    fin_result_q       financial result (DRE 3.06), the flow of the quarter
    net_income_q       net income (DRE 3.11)
    oci_q              other comprehensive income (DRA 4.02)
    oci_translation_q  translation of foreign operations (DRA 4.02.x lines, by label)
    oci_hedge_q        hedge results in other comprehensive income (DRA 4.02.x lines, by label)
    oci_tax_q          tax lines in other comprehensive income (DRA 4.02.x lines, by label)
    fiscal_q           fiscal quarter, 1 to 4, from the start of the fiscal year
    restated_q         the year-to-date difference of 3.06, 4.02, translation or hedge differs from
                       the filed 3-month value, which is used instead (fiscal Q2 and Q3 only)
    suspect_q          a fiscal Q4 placeholder (annual 3.06 of 0 after a nonzero 9 months), or a jump
                       of total assets by a factor of 100 or more (a unit error)
    <stock>_lag        each stock at the end of the previous calendar quarter (empty after a gap)

Flow of a fiscal quarter: fiscal Q1 = the year to date of its filing. Fiscal Q2 and Q3 = the filed
3-month value when the filing has it, otherwise the year to date minus the year to date of the
previous fiscal quarter of the same fiscal year. Fiscal Q4 = the DFP annual value minus the 9-month
year to date of the fiscal Q3 ITR; it cannot be checked against a filed value. A company that files
a year from April to March gets correct quarters too.

The comprehensive-income lines are classified inside each filing first, and the class sums are
differenced, because companies renumber their 4.02.x lines between filings. A class that a filing
with lines below 4.02 does not show is 0; a filing with a nonzero 4.02 total only gives unknown
(empty) classes. Tax lines are a class of their own. Some companies file hedge lines net of tax and others
gross, so the hedge class mixes both; it also holds interest-rate and commodity hedges.
Translation moved into profit on a disposal is class other.

Scope: consolidated statements; a company that never files them (it has no subsidiaries) uses its
individual statements. Stocks come from data/cvm_quarterly.csv, the latest filed version, not the
version known at that date. Financial companies and foreign issuers (cd_cvm 80000 and above) are
left out, and so are 3 financial holdings that the register puts in a non-financial sector.
A company that files consolidated statements is in the panel only in quarters where it does. A
listed holding and its listed subsidiary can both be in the panel. The filed foreign-currency split (fx_loans_*) is optional in the chart of accounts, so a
0 there does not prove that a company has no foreign-currency debt.

Writes data/filed_panel.csv, and data/dra_label_map.csv (each comprehensive-income label, its
class and how often it occurs) so that the label rules can be reviewed.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from src.data_ingestion.download import DATA_DIR
from src.data_ingestion.fetch_cvm import (
    DFP_YEARS, ITR_YEARS, KEY, RESTATED, _clean, _last_version, _read, _single, normalize_text, unit_fixes,
)
from src.data_ingestion.universe import FINANCIAL_SECTORS, normalize_sector, register

HYPERINFLATION = re.compile(r"hiperinfl")
TAX = re.compile(r"imposto|tribut|fisca|\bir\b|\bcs\b|irpj|csll|diferid")
# A net-of-tax clause ("liquido de impostos", "liq. dos efeitos tributarios", "sem efeito
# tributario") describes another line; it does not make that line a tax line.
NET_OF_TAX = re.compile(r"\bliq(uid[oa]s?)?\.?\s*(de|do|dos|da|das)\b[^,;()]*|\(?sem efeito\s*tribut\w*\)?")
# FX results on debt in other comprehensive income exist only under hedge accounting.
HEDGE = re.compile(
    r"hedge|derivativ|cobertura|protecao|swap|"
    r"cambia\w* (de|sobre) (os )?(contratos|emprestimos|financiamentos|dividas)"
)
TRANSLATION = re.compile(
    r"convers|(camb(ia(l|is)|io|il|\.)|combial|cancial).*(exterior|investid|investiment|controlad|operac)|"
    r"(exterior|investid).*cambia|moeda estrangeira|diferencas (de cambio|cambia)"
)
# Translation moved into profit on a disposal is not an FX move of the quarter, and companies
# give it a different label from 1 filing to the next.
RECYCLING = re.compile(r"reclassific|transferencia|realizad|realizac|baixa|alienac")
# Days covered by the year to date of a fiscal Q1, Q2 and Q3 filing, and by an annual filing.
YEAR_TO_DATE = {1: (80, 100), 2: (170, 190), 3: (260, 285), 4: (350, 380)}
THREE_MONTHS = (80, 100)
FOREIGN_ISSUER = 80000  # CVM codes from 80000 up belong to foreign issuers
# Financial holdings that the CVM register puts in a non-financial sector ("Sem Setor Principal").
FINANCIAL_HOLDINGS = {16772: "BNDESPAR", 26964: "REAG Capital Holding", 16705: "YBYRA Capital"}
SUSPECT_ASSET_JUMP = 100.0  # total assets / total assets of the quarter before, either way
# account code -> (column, start of the expected label without accents)
DRE_ACCOUNTS = {"3.06": ("fin_result_q", "resultado financeiro"), "3.11": ("net_income_q", "lucro")}
OCI_TOTAL = ("4.02", "outros resultados abrangentes")
PARTS = {"translation": "oci_translation_q", "hedge": "oci_hedge_q", "tax": "oci_tax_q"}
STOCKS = [
    "total_assets", "cash", "financial_investments", "equity", "debt_short", "debt_long",
    "fx_loans_short", "fx_loans_long",
]


def classify(label: str) -> str:
    """The class of 1 comprehensive-income line: tax, hedge, translation or other."""
    text = normalize_text(label)
    if HYPERINFLATION.search(text):
        return "other"  # IAS 29 restatement follows the inflation of the country, not PTAX
    if TAX.search(NET_OF_TAX.sub("", text)):
        return "tax"
    if HEDGE.search(text):
        return "hedge"
    if TRANSLATION.search(text):
        return "other" if RECYCLING.search(text) else "translation"
    return "other"


def year_to_date(rows: pd.DataFrame) -> pd.DataFrame:
    """The year-to-date rows of each filing, with the fiscal year start and the fiscal quarter.

    `rows` holds the chosen filing of each company and date (see fetch_cvm._last_version), with
    columns cd_cvm, quarter, doc, CD_CONTA, DS_CONTA, DT_INI_EXERC, DT_FIM_EXERC and value.
    The year to date of a filing starts at the earliest period start of that filing.
    """
    frame = rows.assign(fy_start=pd.to_datetime(rows["DT_INI_EXERC"]), end=pd.to_datetime(rows["DT_FIM_EXERC"]))
    frame = frame[frame["end"] == frame["quarter"]]
    frame = frame[frame["fy_start"] == frame.groupby(KEY)["fy_start"].transform("min")].copy()
    days = (frame["end"] - frame["fy_start"]).dt.days
    frame["fiscal_q"] = 0
    for q, (low, high) in YEAR_TO_DATE.items():
        frame.loc[days.between(low, high), "fiscal_q"] = q
    annual = frame["doc"] == "dfp"
    valid = (annual & (frame["fiscal_q"] == 4)) | (~annual & frame["fiscal_q"].between(1, 3))
    # Some filings repeat a row; rows that disagree give an empty value (fetch_cvm._single).
    return _single(frame[valid], KEY + ["CD_CONTA"]).dropna(subset=["value"])


def three_months(rows: pd.DataFrame) -> pd.DataFrame:
    """The filed 3-month rows of the fiscal Q2 and Q3 filings (they start after the fiscal year)."""
    frame = rows.assign(start=pd.to_datetime(rows["DT_INI_EXERC"]), end=pd.to_datetime(rows["DT_FIM_EXERC"]))
    frame = frame[(frame["doc"] == "itr") & (frame["end"] == frame["quarter"])]
    later = frame["start"] > frame.groupby(KEY)["start"].transform("min")
    frame = frame[later & (frame["end"] - frame["start"]).dt.days.between(*THREE_MONTHS)]
    return _single(frame, KEY + ["CD_CONTA"]).dropna(subset=["value"])


def oci_items(ytd: pd.DataFrame) -> pd.DataFrame:
    """The comprehensive-income items of each filing: the 4.02 total and the class sums.

    A 4.02.x line counts only when no ancestor line of the same filing counts too, so a total
    and its parts are never added together. In a filing with lines below 4.02, a class that it
    does not show is 0. A filing with a nonzero 4.02 total only gives empty classes: they are unknown.
    """
    code, label = OCI_TOTAL
    total = ytd[(ytd["CD_CONTA"] == code) & ytd["DS_CONTA"].map(normalize_text).str.startswith(label)]
    lines = ytd[ytd["CD_CONTA"].str.startswith(code + ".")].copy()
    detailed = pd.MultiIndex.from_frame(lines[KEY].drop_duplicates())
    lines["kind"] = lines["DS_CONTA"].map(classify)
    lines = lines[lines["kind"] != "other"]
    counted = lines.groupby(KEY)["CD_CONTA"].agg(set)

    def has_counted_ancestor(row) -> bool:
        codes = counted[(row.cd_cvm, row.quarter)]
        parts = row.CD_CONTA.split(".")
        return any(".".join(parts[:k]) in codes for k in range(2, len(parts)))

    if len(lines):
        lines = lines[~np.array([has_counted_ancestor(row) for row in lines.itertuples()], dtype=bool)]
    filing = ["cd_cvm", "quarter", "fy_start", "fiscal_q"]
    sums = lines.pivot_table(index=KEY, columns="kind", values="value", aggfunc="sum")
    items = total.set_index(KEY)[filing[2:] + ["value"]].rename(columns={"value": "oci_q"})
    items = items.join(sums.reindex(columns=list(PARTS)).rename(columns=PARTS), how="left")
    shown = items.index.isin(detailed) | (items["oci_q"] == 0)  # no comprehensive income at all
    items.loc[shown] = items.loc[shown].fillna(dict.fromkeys(PARTS.values(), 0.0))
    return items.reset_index().melt(id_vars=filing, var_name="item", value_name="value")


def dre_items(ytd: pd.DataFrame) -> pd.DataFrame:
    """The year to date of each checked DRE account of each filing, in long form."""
    labels = ytd["DS_CONTA"].map(normalize_text)
    expected = ytd["CD_CONTA"].map(lambda code: DRE_ACCOUNTS.get(code, ("", "\x00"))[1])
    rows = ytd[[label.startswith(start) for label, start in zip(labels, expected)]]
    rows = rows.assign(item=rows["CD_CONTA"].map(lambda code: DRE_ACCOUNTS[code][0]))
    return rows[["cd_cvm", "quarter", "fy_start", "fiscal_q", "item", "value"]]


def quarter_flows(items: pd.DataFrame) -> pd.DataFrame:
    """The flow of each fiscal quarter: a difference of year-to-date values of the same item.

    `items` has 1 row for each filing and item: cd_cvm, quarter, fy_start, fiscal_q, item, value.
    The previous filing is the one of the same company and fiscal year, 1 fiscal quarter earlier.
    """
    keys = ["cd_cvm", "quarter", "item"]
    if items.duplicated(keys).any():
        raise ValueError(f"more than 1 year to date for a filing and item: {items[items.duplicated(keys)].head()}")
    match = ["cd_cvm", "fy_start", "fiscal_q", "item"]
    previous = items[match + ["value"]].assign(fiscal_q=lambda f: f["fiscal_q"] + 1)
    flows = items.merge(previous, on=match, how="left", suffixes=("", "_before"))
    first = flows["fiscal_q"] == 1
    flows["value"] = flows["value"].where(first, flows["value"] - flows["value_before"])
    flows = flows[first | flows["value_before"].notna()]
    return flows[["cd_cvm", "quarter", "fiscal_q", "item", "value"]].reset_index(drop=True)


def restated(flows: pd.DataFrame, filed: pd.DataFrame) -> pd.Series:
    """True for a company and quarter whose derived flow differs from the filed 3-month row."""
    both = flows.merge(filed, on=["cd_cvm", "quarter", "item"], suffixes=("", "_filed"))
    gap = (both["value"] - both["value_filed"]).abs()
    size = np.maximum(both["value"].abs(), both["value_filed"].abs())
    both["restated_q"] = gap > np.maximum(RESTATED * size, 1000.0)
    return both.groupby(KEY)["restated_q"].any()


def _scopes() -> pd.DataFrame:
    """Each company's scope: consolidated, or individual when it never files consolidated."""
    statements = pd.read_csv(DATA_DIR / "cvm_quarterly.csv", parse_dates=["quarter"])
    if "financial_investments" not in statements:
        raise ValueError("data/cvm_quarterly.csv is older than the code: run python -m src.data_ingestion.fetch_cvm")
    consolidated = set(statements.loc[statements["scope"] == "con", "cd_cvm"])
    scope = statements["cd_cvm"].map(lambda code: "con" if code in consolidated else "ind")
    return statements[statements["scope"] == scope]


def _flow_rows(statement: str, keep, fixes: set, consolidated: set) -> pd.DataFrame:
    frames = []
    for scope in ("con", "ind"):
        for doc, years in (("dfp", DFP_YEARS), ("itr", ITR_YEARS)):
            frames += [_read(doc, year, statement, scope, keep).assign(scope=scope) for year in years]
    raw = pd.concat(frames, ignore_index=True)
    raw = raw[raw["ORDEM_EXERC"] == "ÚLTIMO"]
    company = raw["CD_CVM"].astype(int)
    raw = raw[(raw["scope"] == "con") == company.isin(consolidated)]
    accounts = {code: (code, "") for code in raw["CD_CONTA"].unique()}
    return _last_version(_clean(raw, accounts, fixes))


def combine(derived: pd.DataFrame, filed: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """The flows to use, and the restatement flag of each company and quarter.

    In fiscal Q2 and Q3 the filed 3-month value of an item is used when it exists, because all its
    lines come from 1 filing. Elsewhere the difference of year-to-date values is used.
    `derived` is the output of quarter_flows; `filed` holds the 3-month items of the same filings.
    """
    keys = ["cd_cvm", "quarter", "item"]
    both = derived.merge(filed[keys + ["fiscal_q", "value"]], on=keys, how="outer", suffixes=("", "_filed"))
    both["fiscal_q"] = both["fiscal_q"].fillna(both["fiscal_q_filed"]).astype(int)
    checked = both["item"].isin(["fin_result_q", "oci_q", "oci_translation_q", "oci_hedge_q"])
    flag = restated(both.loc[checked, keys + ["value"]].dropna(), both.loc[checked, keys + ["value_filed"]].dropna().rename(columns={"value_filed": "value"}))
    both["value"] = both["value_filed"].where(both["value_filed"].notna(), both["value"])
    return both[keys + ["fiscal_q", "value"]].dropna(subset=["value"]), flag


def build() -> tuple[pd.DataFrame, pd.DataFrame]:
    fixes: set = set()
    for scope in ("con", "ind"):  # the unit check compares filings of 1 scope
        balance = [_read("dfp", year, "BPA", scope, {"1"}) for year in DFP_YEARS]
        balance += [_read("itr", year, "BPA", scope, {"1"}) for year in ITR_YEARS]
        fixes |= unit_fixes(balance)
    statements = _scopes()
    consolidated = set(statements.loc[statements["scope"] == "con", "cd_cvm"])

    dre_rows = _flow_rows("DRE", set(DRE_ACCOUNTS), fixes, consolidated)
    dra_rows = _flow_rows("DRA", lambda codes: codes.str.startswith(OCI_TOTAL[0]), fixes, consolidated)
    dre_ytd, dra_ytd = year_to_date(dre_rows), year_to_date(dra_rows)
    dre_flows = quarter_flows(dre_items(dre_ytd))
    derived = pd.concat([dre_flows, quarter_flows(oci_items(dra_ytd))], ignore_index=True)

    filed = []
    for rows, ytd, items in ((dre_rows, dre_ytd, dre_items), (dra_rows, dra_ytd, oci_items)):
        filing = ytd[KEY + ["fy_start", "fiscal_q"]].drop_duplicates(KEY)
        months = three_months(rows).merge(filing, on=KEY)
        filed.append(items(months[months["fiscal_q"].isin([2, 3])]))
    flows, flag = combine(derived, pd.concat(filed, ignore_index=True))

    wide = flows.pivot(index=KEY, columns="item", values="value")
    wide = wide.join(flows.groupby(KEY)["fiscal_q"].first()).join(flag, how="left")
    wide["restated_q"] = wide["restated_q"].fillna(False).astype(bool)

    stocks = statements.set_index(KEY)[["scope"] + STOCKS]
    lagged = stocks[STOCKS].add_suffix("_lag").reset_index()
    lagged["quarter"] = lagged["quarter"] + pd.offsets.QuarterEnd(1)
    panel = stocks.join(wide, how="inner").reset_index().merge(lagged, on=KEY, how="left")

    # A fiscal Q4 whose annual financial result is 0 after a nonzero 9 months is a placeholder,
    # and a jump of total assets by a factor of 100 or more is a unit error.
    annual = dre_items(dre_ytd)
    annual = annual[(annual["item"] == "fin_result_q") & (annual["fiscal_q"] == 4) & (annual["value"] == 0)]
    placeholder = pd.MultiIndex.from_frame(annual[KEY])
    jump = panel["total_assets"] / panel["total_assets_lag"]
    panel["suspect_q"] = pd.MultiIndex.from_frame(panel[KEY]).isin(placeholder) & panel["fin_result_q"].ne(0)
    panel["suspect_q"] |= (jump > SUSPECT_ASSET_JUMP) | (jump < 1 / SUSPECT_ASSET_JUMP)

    sectors = register().assign(sector=lambda r: r["SETOR_ATIV"].map(normalize_sector))[["cd_cvm", "sector"]]
    panel = panel.merge(sectors, on="cd_cvm", how="left")
    keep = ~panel["sector"].isin(FINANCIAL_SECTORS) & (panel["cd_cvm"] < FOREIGN_ISSUER)
    keep &= ~panel["cd_cvm"].isin(FINANCIAL_HOLDINGS)
    panel = panel[keep].sort_values(KEY)

    labels = dra_ytd[dra_ytd["CD_CONTA"].str.startswith(OCI_TOTAL[0] + ".")]
    label_map = labels["DS_CONTA"].map(normalize_text).value_counts().rename_axis("label").reset_index(name="count")
    label_map["kind"] = label_map["label"].map(classify)
    return panel, label_map


def main() -> pd.DataFrame:
    panel, label_map = build()
    panel.to_csv(DATA_DIR / "filed_panel.csv", index=False, date_format="%Y-%m-%d")
    label_map.to_csv(DATA_DIR / "dra_label_map.csv", index=False)
    print(f"filed_panel.csv: {len(panel)} rows, {panel['cd_cvm'].nunique()} non-financial companies, "
          f"{panel['quarter'].min().date()} to {panel['quarter'].max().date()}")
    print("filled share:", panel[["fin_result_q", "oci_q", "equity_lag", "fx_loans_short_lag"]].notna().mean().round(3).to_dict())
    print("nonzero share:", (panel[["oci_translation_q", "oci_hedge_q", "oci_tax_q"]].fillna(0) != 0).mean().round(3).to_dict())
    print("scope:", panel.groupby("scope")["cd_cvm"].nunique().to_dict(), "| fiscal quarter:", panel["fiscal_q"].value_counts().sort_index().to_dict())
    print("restated quarters (filed 3-month values used):", int(panel["restated_q"].sum()), "| suspect quarters:", int(panel["suspect_q"].sum()))
    print("unknown classes (total-only filings):", int(panel["oci_translation_q"].isna().sum()))
    print("label rows by class:", label_map.groupby("kind")["count"].sum().to_dict())
    return panel


if __name__ == "__main__":
    main()
