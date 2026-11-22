"""Build the quarterly statement table from the CVM public archives (DFP and ITR).

DFP holds the fiscal year end, from 2010. ITR holds the other 3 quarter ends, from 2011.
The table keeps the last filed version, calendar quarter-end dates, and the industrial
chart of accounts only. Banks and insurers use other charts: their labels do not match,
so their rows do not pass.

Scope: the consolidated statement. A company with no subsidiary files only the individual
statement, so the individual statement is used for a date that has no consolidated one.
The `scope` column says which one each row holds ("con" or "ind").

A value is never guessed. These source errors are handled:
- A row with a label that does not match its account code is dropped.
- A filing declared in units (ESCALA_MOEDA = UNIDADE) whose own comparative column is
  1/1000 of the value filed for that date is in thousands. Its values are multiplied by
  1000, only when this proof exists. Each case is printed.
- When 2 rows of 1 filing give different values for 1 account and period, the value stays empty.
- A row with zero total assets counts as absent (some companies file an all-zero form).
- A financial expense (3.06.02) of 0 or above is a placeholder or a swapped line, so it
  stays empty.
- A trailing-12-month value stays empty when the comparative year to date of the filing
  differs from the value first filed for that period (a restatement): the formula would
  mix 2 bases.
A flow with a missing part stays empty.

Output: data/cvm_quarterly.csv (values in reais)
Run:    python -m src.data_ingestion.fetch_cvm
"""
from __future__ import annotations

import unicodedata
import zipfile

import numpy as np
import pandas as pd

from src.data_ingestion.download import CVM_BASE, DATA_DIR, RAW_DIR, download

MODULE = "src/data_ingestion/fetch_cvm.py"
DFP_YEARS = range(2010, 2025)
ITR_YEARS = range(2011, 2025)

# account code -> (column name, start of the expected label without accents)
BALANCE = {
    "1": ("total_assets", "ativo total"),
    "1.01": ("current_assets", "ativo circulante"),
    "1.01.01": ("cash", "caixa e equivalentes de caixa"),
    "2.01": ("current_liabilities", "passivo circulante"),
    "2.01.04": ("debt_short", "emprestimos e financiamentos"),
    "2.02": ("noncurrent_liabilities", "passivo nao circulante"),
    "2.02.01": ("debt_long", "emprestimos e financiamentos"),
    "2.03": ("equity", "patrimonio liquido"),
}
FLOWS = {
    "3.05": ("ebit_ttm", "resultado antes do resultado financeiro e dos tributos"),
    "3.06.02": ("finexp_ttm", "despesas financeiras"),
    "3.11": ("net_income_ttm", "lucro"),
}
MODEL_COLUMNS = ["total_assets", "current_liabilities", "noncurrent_liabilities", "equity"]
NEEDED = {
    "CNPJ_CIA", "DT_REFER", "VERSAO", "DENOM_CIA", "CD_CVM", "MOEDA", "ESCALA_MOEDA",
    "ORDEM_EXERC", "DT_INI_EXERC", "DT_FIM_EXERC", "CD_CONTA", "DS_CONTA", "VL_CONTA",
}
KEY = ["cd_cvm", "quarter"]
FILING = ["cd_cvm", "quarter", "doc", "versao"]
UNIT_RATIO = (0.00099, 0.00101)  # comparative / value filed for that date, for a filing in thousands
RESTATED = 0.01  # relative difference above which a year to date counts as restated


def normalize_text(text: str) -> str:
    plain = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode()
    return " ".join(plain.lower().split())


def _read(doc: str, year: int, statement: str, scope: str, codes: set[str]) -> pd.DataFrame:
    archive_name = f"{doc}_cia_aberta_{year}.zip"
    path = download(f"{CVM_BASE}/DOC/{doc.upper()}/DADOS/{archive_name}", RAW_DIR / "cvm" / archive_name, MODULE)
    parts = []
    member = f"{doc}_cia_aberta_{statement}_{scope}_{year}.csv"
    with zipfile.ZipFile(path) as archive, archive.open(member) as handle:
        reader = pd.read_csv(
            handle, sep=";", encoding="latin1", dtype=str, chunksize=250_000, usecols=lambda name: name in NEEDED
        )
        for chunk in reader:
            parts.append(chunk[chunk["CD_CONTA"].isin(codes)])
    frame = pd.concat(parts, ignore_index=True)
    frame["doc"] = doc
    return frame


def _clean(frame: pd.DataFrame, accounts: dict, unit_fixes: set | None = None) -> pd.DataFrame:
    frame = frame[frame["MOEDA"] == "REAL"].copy()
    frame["cd_cvm"] = frame["CD_CVM"].astype(int)
    frame["quarter"] = pd.to_datetime(frame["DT_REFER"])
    frame = frame[frame["quarter"].dt.is_quarter_end]
    frame["versao"] = frame["VERSAO"].astype(int)
    scale = frame["ESCALA_MOEDA"].map({"MIL": 1000.0, "UNIDADE": 1.0})
    if scale.isna().any():
        raise ValueError(f"unknown ESCALA_MOEDA: {sorted(frame.loc[scale.isna(), 'ESCALA_MOEDA'].unique())}")
    if unit_fixes:
        filing = pd.MultiIndex.from_frame(frame[FILING])
        proven = filing.isin(list(unit_fixes)) & (frame["ESCALA_MOEDA"] == "UNIDADE").to_numpy()
        scale = scale.where(~proven, 1000.0)
    frame["value"] = pd.to_numeric(frame["VL_CONTA"]) * scale
    frame["column"] = frame["CD_CONTA"].map(lambda code: accounts[code][0])
    expected = frame["CD_CONTA"].map(lambda code: accounts[code][1])
    labels = frame["DS_CONTA"].map(normalize_text)
    matches = [label.startswith(start) for label, start in zip(labels, expected)]
    return frame[matches]


def unit_fixes(balance_frames: list[pd.DataFrame]) -> set:
    """Filings declared in units whose own comparative proves that they are in thousands.

    The comparative column (PENÚLTIMO) of total assets must be 1/1000 of the total assets
    that the chosen filing of the comparative date reports.
    """
    raw = pd.concat(balance_frames, ignore_index=True)
    assets = _clean(raw[raw["CD_CONTA"] == "1"], BALANCE)
    chosen = _last_version(assets[assets["ORDEM_EXERC"] == "ÚLTIMO"])
    stored = chosen.groupby(KEY)["value"].first()
    comparative = assets[(assets["ORDEM_EXERC"] == "PENÚLTIMO") & (assets["ESCALA_MOEDA"] == "UNIDADE")].copy()
    comparative["date"] = pd.to_datetime(comparative["DT_FIM_EXERC"])
    reference = stored.reindex(pd.MultiIndex.from_arrays([comparative["cd_cvm"], comparative["date"]])).to_numpy()
    ratio = comparative["value"].to_numpy() / reference
    proven = comparative[(ratio > UNIT_RATIO[0]) & (ratio < UNIT_RATIO[1])]
    fixes = set(map(tuple, proven[FILING].drop_duplicates().itertuples(index=False)))
    for cd_cvm, quarter, doc, versao in sorted(fixes):
        print(f"unit fix: cd_cvm {cd_cvm} {doc} {quarter.date()} version {versao} is in thousands, not units")
    return fixes


def _single(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """1 row for each key. When the rows of a key disagree on the value, the value is empty."""
    grouped = frame.groupby(keys, as_index=False)
    first = grouped.first()
    distinct = grouped["value"].nunique(dropna=False)["value"].to_numpy()
    first.loc[distinct > 1, "value"] = np.nan
    return first


def _last_version(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep 1 filing for each company and date: DFP before ITR, then the highest version."""
    filings = frame[KEY + ["doc", "versao"]].drop_duplicates()
    filings["rank"] = filings["doc"].map({"itr": 0, "dfp": 1})
    best = filings.sort_values(["rank", "versao"]).groupby(KEY).tail(1)
    return frame.merge(best[KEY + ["doc", "versao"]], on=KEY + ["doc", "versao"])


def build_balance(frames: list[pd.DataFrame], fixes: set) -> pd.DataFrame:
    raw = pd.concat(frames, ignore_index=True)
    raw = raw[raw["ORDEM_EXERC"] == "ÚLTIMO"]
    raw = _last_version(_clean(raw, BALANCE, fixes))
    cells = _single(raw, KEY + ["column"])
    wide = cells.pivot(index=KEY, columns="column", values="value")
    meta = raw.groupby(KEY).agg(
        cnpj=("CNPJ_CIA", "first"), company=("DENOM_CIA", "first"), doc=("doc", "first"), versao=("versao", "first")
    )
    meta["cnpj"] = meta["cnpj"].str.replace(r"\D", "", regex=True)
    table = meta.join(wide).dropna(subset=MODEL_COLUMNS)
    return table[table["total_assets"] > 0]


def _year_to_date(frame: pd.DataFrame) -> pd.DataFrame:
    """For each company, date and account, keep the row with the earliest period start."""
    frame = frame.assign(start=pd.to_datetime(frame["DT_INI_EXERC"]), end=pd.to_datetime(frame["DT_FIM_EXERC"]))
    earliest = frame.groupby(KEY + ["column"])["start"].transform("min")
    return _single(frame[frame["start"] == earliest], KEY + ["column"])


def _flows(frames: list[pd.DataFrame], fixes: set) -> pd.DataFrame:
    """Clean flow rows. A financial expense of 0 or above is not an expense, so it is removed."""
    frame = _clean(pd.concat(frames, ignore_index=True), FLOWS, fixes)
    placeholder = (frame["column"] == "finexp_ttm") & (frame["value"] >= 0)
    return frame[~placeholder]


def build_flows(dfp_frames: list[pd.DataFrame], itr_frames: list[pd.DataFrame], fixes: set) -> pd.DataFrame:
    """Trailing 12 months for each flow account.

    DFP date:  the annual value.
    ITR date:  year to date - year to date of the year before + annual value of the
               fiscal year before. Both year-to-date values come from the same filing.
               Empty when the year to date of the year before was restated.
    """
    dfp = _flows(dfp_frames, fixes)
    dfp = _last_version(dfp[dfp["ORDEM_EXERC"] == "ÚLTIMO"])
    annual = _year_to_date(dfp)
    annual = annual[(annual["end"] - annual["start"]).dt.days.between(350, 380)]
    annual = annual[KEY + ["column", "value"]].rename(columns={"quarter": "fy_end", "value": "annual"})

    itr = _last_version(_flows(itr_frames, fixes))
    current = _year_to_date(itr[itr["ORDEM_EXERC"] == "ÚLTIMO"])[KEY + ["column", "start", "value"]]
    before = _year_to_date(itr[itr["ORDEM_EXERC"] == "PENÚLTIMO"])[KEY + ["column", "value"]]
    merged = current.merge(before, on=KEY + ["column"], suffixes=("", "_before"))
    merged["fy_end"] = merged["start"] - pd.Timedelta(days=1)
    merged = merged.merge(annual, on=["cd_cvm", "fy_end", "column"])
    merged["ttm"] = merged["value"] - merged["value_before"] + merged["annual"]

    # The year to date first filed 1 year earlier, to detect a restated comparative.
    first_filed = current[KEY + ["column", "value"]].rename(columns={"quarter": "year_before", "value": "first_filed"})
    merged["year_before"] = merged["quarter"] - pd.DateOffset(years=1)
    merged = merged.merge(first_filed, on=["cd_cvm", "year_before", "column"], how="left")
    gap = (merged["value_before"] - merged["first_filed"]).abs()
    size = np.maximum(merged["value_before"].abs(), merged["first_filed"].abs())
    restated = merged["first_filed"].notna() & (gap > np.maximum(RESTATED * size, 1000.0))
    merged.loc[restated, "ttm"] = np.nan

    from_dfp = annual.rename(columns={"fy_end": "quarter", "annual": "ttm"})
    ttm = pd.concat([merged[KEY + ["column", "ttm"]], from_dfp[KEY + ["column", "ttm"]]], ignore_index=True)
    ttm = ttm.drop_duplicates(subset=KEY + ["column"], keep="last")
    print(f"flows: {int(restated.sum())} trailing values left empty after a restatement", flush=True)
    wide = ttm.pivot(index=KEY, columns="column", values="ttm")
    # A trailing financial expense of 0 or above is not an expense: the parts do not fit.
    wide["finexp_ttm"] = wide["finexp_ttm"].where(wide["finexp_ttm"] < 0)
    return wide


def build_scope(scope: str) -> pd.DataFrame:
    """The statement table of 1 scope: "con" (consolidated) or "ind" (individual)."""
    asset_codes = {code for code in BALANCE if code.startswith("1")}
    liability_codes = {code for code in BALANCE if code.startswith("2")}
    balance_frames, dfp_flows, itr_flows = [], [], []
    for doc, years, flows in (("dfp", DFP_YEARS, dfp_flows), ("itr", ITR_YEARS, itr_flows)):
        for year in years:
            balance_frames.append(_read(doc, year, "BPA", scope, asset_codes))
            balance_frames.append(_read(doc, year, "BPP", scope, liability_codes))
            flows.append(_read(doc, year, "DRE", scope, set(FLOWS)))
            print(f"read {scope} {doc} {year}", flush=True)
    fixes = unit_fixes(balance_frames)
    table = build_balance(balance_frames, fixes).join(build_flows(dfp_flows, itr_flows, fixes))
    table.insert(0, "scope", scope)
    return table


def main() -> pd.DataFrame:
    consolidated = build_scope("con")
    individual = build_scope("ind")
    # Row-level choice: a row never mixes the 2 scopes.
    only_individual = individual[~individual.index.isin(consolidated.index)]
    table = pd.concat([consolidated, only_individual])
    table = table.reset_index().sort_values(KEY)
    DATA_DIR.mkdir(exist_ok=True)
    table.to_csv(DATA_DIR / "cvm_quarterly.csv", index=False, date_format="%Y-%m-%d")
    residual = (
        table["total_assets"] - table["current_liabilities"] - table["noncurrent_liabilities"] - table["equity"]
    ).abs() / table["total_assets"].abs()
    print(
        f"cvm_quarterly.csv: {len(table)} rows, {table['cd_cvm'].nunique()} companies, "
        f"{table['quarter'].min().date()} to {table['quarter'].max().date()}"
    )
    print(f"identity residual / assets: median {residual.median():.2e}, share above 1e-6: {(residual > 1e-6).mean():.4f}")
    print("rows by scope:", table["scope"].value_counts().to_dict())
    print("empty share by column:", table.isna().mean().round(3).to_dict())
    return table


if __name__ == "__main__":
    main()
