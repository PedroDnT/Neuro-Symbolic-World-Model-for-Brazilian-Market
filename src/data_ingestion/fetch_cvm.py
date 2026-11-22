"""Build the quarterly statement table from the CVM public archives (DFP and ITR).

DFP holds the fiscal year end, from 2010. ITR holds the other 3 quarter ends, from 2011.
The table keeps the last filed version, calendar quarter-end dates, and the industrial
chart of accounts only. Banks and insurers use other charts: their labels do not match,
so their rows do not pass.

Scope: the consolidated statement. A company with no subsidiary files only the individual
statement, so the individual statement is used for a date that has no consolidated one.
The `scope` column says which one each row holds ("con" or "ind").

A value is never guessed. A row with a label that does not match its account code is
dropped. A flow with a missing part stays empty.

Output: data/cvm_quarterly.csv (values in reais)
Run:    python -m src.data_ingestion.fetch_cvm
"""
from __future__ import annotations

import unicodedata
import zipfile

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


def _clean(frame: pd.DataFrame, accounts: dict) -> pd.DataFrame:
    frame = frame[frame["MOEDA"] == "REAL"].copy()
    frame["cd_cvm"] = frame["CD_CVM"].astype(int)
    frame["quarter"] = pd.to_datetime(frame["DT_REFER"])
    frame = frame[frame["quarter"].dt.is_quarter_end]
    frame["versao"] = frame["VERSAO"].astype(int)
    scale = frame["ESCALA_MOEDA"].map({"MIL": 1000.0, "UNIDADE": 1.0})
    if scale.isna().any():
        raise ValueError(f"unknown ESCALA_MOEDA: {sorted(frame.loc[scale.isna(), 'ESCALA_MOEDA'].unique())}")
    frame["value"] = pd.to_numeric(frame["VL_CONTA"]) * scale
    frame["column"] = frame["CD_CONTA"].map(lambda code: accounts[code][0])
    expected = frame["CD_CONTA"].map(lambda code: accounts[code][1])
    labels = frame["DS_CONTA"].map(normalize_text)
    matches = [label.startswith(start) for label, start in zip(labels, expected)]
    return frame[matches]


def _last_version(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep 1 filing for each company and date: DFP before ITR, then the highest version."""
    filings = frame[KEY + ["doc", "versao"]].drop_duplicates()
    filings["rank"] = filings["doc"].map({"itr": 0, "dfp": 1})
    best = filings.sort_values(["rank", "versao"]).groupby(KEY).tail(1)
    return frame.merge(best[KEY + ["doc", "versao"]], on=KEY + ["doc", "versao"])


def build_balance(frames: list[pd.DataFrame]) -> pd.DataFrame:
    raw = pd.concat(frames, ignore_index=True)
    raw = raw[raw["ORDEM_EXERC"] == "ÚLTIMO"]
    raw = _last_version(_clean(raw, BALANCE))
    wide = raw.pivot_table(index=KEY, columns="column", values="value", aggfunc="first")
    meta = raw.groupby(KEY).agg(
        cnpj=("CNPJ_CIA", "first"), company=("DENOM_CIA", "first"), doc=("doc", "first"), versao=("versao", "first")
    )
    meta["cnpj"] = meta["cnpj"].str.replace(r"\D", "", regex=True)
    table = meta.join(wide)
    return table.dropna(subset=MODEL_COLUMNS)


def _year_to_date(frame: pd.DataFrame) -> pd.DataFrame:
    """For each company, date and account, keep the row with the earliest period start."""
    frame = frame.assign(start=pd.to_datetime(frame["DT_INI_EXERC"]), end=pd.to_datetime(frame["DT_FIM_EXERC"]))
    return frame.sort_values("start").groupby(KEY + ["column"], as_index=False).first()


def build_flows(dfp_frames: list[pd.DataFrame], itr_frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Trailing 12 months for each flow account.

    DFP date:  the annual value.
    ITR date:  year to date - year to date of the year before + annual value of the
               fiscal year before. Both year-to-date values come from the same filing.
    """
    dfp = _clean(pd.concat(dfp_frames, ignore_index=True), FLOWS)
    dfp = _last_version(dfp[dfp["ORDEM_EXERC"] == "ÚLTIMO"])
    annual = _year_to_date(dfp)
    annual = annual[(annual["end"] - annual["start"]).dt.days.between(350, 380)]
    annual = annual[KEY + ["column", "value"]].rename(columns={"quarter": "fy_end", "value": "annual"})

    itr = _last_version(_clean(pd.concat(itr_frames, ignore_index=True), FLOWS))
    current = _year_to_date(itr[itr["ORDEM_EXERC"] == "ÚLTIMO"])[KEY + ["column", "start", "value"]]
    before = _year_to_date(itr[itr["ORDEM_EXERC"] == "PENÚLTIMO"])[KEY + ["column", "value"]]
    merged = current.merge(before, on=KEY + ["column"], suffixes=("", "_before"))
    merged["fy_end"] = merged["start"] - pd.Timedelta(days=1)
    merged = merged.merge(annual, on=["cd_cvm", "fy_end", "column"])
    merged["ttm"] = merged["value"] - merged["value_before"] + merged["annual"]

    from_dfp = annual.rename(columns={"fy_end": "quarter", "annual": "ttm"})
    ttm = pd.concat([merged[KEY + ["column", "ttm"]], from_dfp[KEY + ["column", "ttm"]]], ignore_index=True)
    ttm = ttm.drop_duplicates(subset=KEY + ["column"], keep="last")
    return ttm.pivot(index=KEY, columns="column", values="ttm")


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
    table = build_balance(balance_frames).join(build_flows(dfp_flows, itr_flows))
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
