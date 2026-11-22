"""Select the universe: non-financial B3 companies with full history, by 2024 volume.

Steps:
1. Sum the 2024 cash-market volume of each ticker from the B3 COTAHIST file.
2. Map each ticker to a company. First source: the CVM FCA ticker table. Some companies
   wrote a wrong ticker there, so the second source is the B3 list of companies, which
   gives the CVM code of each issuer code (the first 4 characters of a ticker).
3. Keep companies outside the financial sectors of the CVM register.
4. Keep companies that have all quarters of the window in data/cvm_quarterly.csv.
5. Rank by mean quarter volume and keep the top N.

Output: data/universe.csv
Run:    python -m src.data_ingestion.universe
"""
from __future__ import annotations

import base64
import json
import re
import time
import zipfile
from collections import defaultdict

import pandas as pd
import requests

from src.data_ingestion.download import CVM_BASE, DATA_DIR, RAW_DIR, download
from src.data_ingestion.fetch_cvm import MODEL_COLUMNS
from src.protocol import data_fetch_error

MODULE = "src/data_ingestion/universe.py"
B3_COMPANIES = "https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetInitialCompanies/"
TICKER_PATTERN = re.compile(r"^[A-Z0-9]{4}\d{1,2}$")
TOP_N = 100
FIRST_QUARTER, LAST_QUARTER = "2010-12-31", "2024-12-31"
VOLUME_YEAR = 2024
IDENTITY_TOLERANCE = 1e-6

FINANCIAL_SECTORS = {
    "Bancos",
    "Securitização de Recebíveis",
    "Arrendamento Mercantil",
    "Seguradoras e Corretoras",
    "Intermediação Financeira",
    "Crédito Imobiliário",
    "Bolsas de Valores/Mercadorias e Futuros",
    "Factoring",
}
# The register abbreviates some sector names in its "Emp. Adm. Part. - <sector>" labels.
SECTOR_ALIASES = {
    "Const. Civil, Mat. Const. e Decoração": "Construção Civil, Mat. Constr. e Decoração",
    "Máqs., Equip., Veíc. e Peças": "Máquinas, Equipamentos, Veículos e Peças",
    "Serviços médicos": "Serviços Médicos",
    "Sem Setor Principal": "Emp. Adm. Participações",
}
SHARE_TYPES = {"Ações Ordinárias", "Ações Preferenciais", "Units"}
# COTAHIST market codes of a standard lot of shares. 02 is the normal code. A company in
# distress trades under its own code: 05 sanctioned, 06 concordata, 07 out-of-court
# recovery, 08 judicial recovery, 09 special temporary administration, 11 intervention.
# Without these codes the universe loses the companies in judicial recovery.
SHARE_BDI_CODES = {b"02", b"05", b"06", b"07", b"08", b"09", b"11"}


def normalize_sector(raw: str) -> str:
    """Map a holding-company label to the sector that it holds."""
    sector = str(raw).strip()
    match = re.match(r"^Emp\. Adm\. Part\.\s*-\s*(.+)$", sector)
    if match:
        sector = match.group(1).strip()
    return SECTOR_ALIASES.get(sector, sector)


def quarter_ends() -> pd.DatetimeIndex:
    return pd.date_range(FIRST_QUARTER, LAST_QUARTER, freq="QE")


def ticker_volume() -> pd.Series:
    """Mean quarter volume in reais for each ticker: cash market, standard lot."""
    name = f"COTAHIST_A{VOLUME_YEAR}.ZIP"
    path = download(f"https://bvmf.bmfbovespa.com.br/InstDados/SerHist/{name}", RAW_DIR / "b3" / name, MODULE)
    totals: dict[str, float] = defaultdict(float)
    with zipfile.ZipFile(path) as archive, archive.open(archive.namelist()[0]) as handle:
        for line in handle:
            # TIPREG 01 = quote, TPMERC 010 = cash market, CODBDI = standard lot of a share.
            if line[0:2] != b"01" or line[10:12] not in SHARE_BDI_CODES or line[24:27] != b"010":
                continue
            totals[line[12:24].decode("latin1").strip()] += int(line[170:188]) / 100
    if not totals:
        raise ValueError("no cash-market record in the COTAHIST file")
    return (pd.Series(totals, name="volume") / 4).sort_values(ascending=False)


def register() -> pd.DataFrame:
    path = download(f"{CVM_BASE}/CAD/DADOS/cad_cia_aberta.csv", RAW_DIR / "cvm" / "cad_cia_aberta.csv", MODULE)
    frame = pd.read_csv(path, sep=";", encoding="latin1", dtype=str)
    frame["cd_cvm"] = frame["CD_CVM"].astype(int)
    frame["cnpj"] = frame["CNPJ_CIA"].str.replace(r"\D", "", regex=True)
    # A CNPJ can have more than 1 register row. Keep the active one, then the newest.
    frame["active"] = (frame["SIT"] == "ATIVO").astype(int)
    frame = frame.sort_values(["active", "DT_REG"]).drop_duplicates("cd_cvm", keep="last")
    return frame[["cd_cvm", "cnpj", "DENOM_SOCIAL", "DENOM_COMERC", "SETOR_ATIV", "SIT"]]


def ticker_map(companies: pd.DataFrame) -> pd.DataFrame:
    """Ticker -> company, from the FCA table of traded securities."""
    name = f"fca_cia_aberta_{VOLUME_YEAR}.zip"
    path = download(f"{CVM_BASE}/DOC/FCA/DADOS/{name}", RAW_DIR / "cvm" / name, MODULE)
    with zipfile.ZipFile(path) as archive:
        with archive.open(f"fca_cia_aberta_valor_mobiliario_{VOLUME_YEAR}.csv") as handle:
            frame = pd.read_csv(handle, sep=";", encoding="latin1", dtype=str)
    frame = frame[frame["Valor_Mobiliario"].isin(SHARE_TYPES) & frame["Codigo_Negociacao"].notna()]
    frame = frame.assign(
        ticker=frame["Codigo_Negociacao"].str.strip().str.upper(),
        cnpj=frame["CNPJ_Companhia"].str.replace(r"\D", "", regex=True),
    )
    # Drop a code that is not a ticker (some companies wrote their CVM code or "ADR").
    frame = frame[frame["ticker"].map(lambda code: bool(TICKER_PATTERN.match(code)))]
    active = companies.sort_values("SIT").drop_duplicates("cnpj", keep="first")
    frame = frame.merge(active[["cnpj", "cd_cvm"]], on="cnpj")
    return frame[["ticker", "cd_cvm"]].drop_duplicates()


def b3_issuers() -> dict[str, int]:
    """Issuer code (first 4 characters of a ticker) -> CVM code, from the B3 list of companies."""
    cache = RAW_DIR / "b3" / "listed_companies.json"
    if cache.exists():
        records = json.loads(cache.read_text())
    else:
        records, page, pages = [], 1, 1
        while page <= pages:
            query = json.dumps({"language": "pt-br", "pageNumber": page, "pageSize": 120}).encode()
            last_error: Exception | None = None
            for attempt in range(4):
                try:
                    response = requests.get(
                        B3_COMPANIES + base64.b64encode(query).decode(),
                        timeout=60,
                        headers={"User-Agent": "Mozilla/5.0"},
                    )
                    response.raise_for_status()
                    body = response.json()
                    break
                except (requests.RequestException, ValueError) as exc:
                    last_error = exc
                    time.sleep(2**attempt)
            else:
                raise data_fetch_error(MODULE, last_error) from last_error
            records += body["results"]
            pages = body["page"]["totalPages"]
            page += 1
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(records))
    owners: dict[str, set[int]] = defaultdict(set)
    for record in records:
        code, cvm = record.get("issuingCompany"), str(record.get("codeCVM") or "")
        if code and cvm.isdigit():
            owners[code.strip().upper()].add(int(cvm))
    # An issuer code with more than 1 CVM code is ambiguous, so it is not used.
    return {code: next(iter(codes)) for code, codes in owners.items() if len(codes) == 1}


def company_volume(volume: pd.Series, tickers: pd.DataFrame, issuers: dict[str, int]) -> pd.DataFrame:
    """Sum the tickers of each company. Report how each ticker found its company."""
    exact = dict(zip(tickers["ticker"], tickers["cd_cvm"]))
    stems = tickers.assign(stem=tickers["ticker"].str[:4]).groupby("stem")["cd_cvm"].nunique()
    stem_owner = tickers.assign(stem=tickers["ticker"].str[:4]).drop_duplicates("stem").set_index("stem")["cd_cvm"]
    rows = []
    for ticker, value in volume.items():
        if ticker in exact:
            rows.append((ticker, exact[ticker], value, "fca_ticker"))
        elif ticker[:4] in issuers:
            rows.append((ticker, issuers[ticker[:4]], value, "b3_issuer"))
        elif stems.get(ticker[:4], 0) == 1:
            rows.append((ticker, stem_owner[ticker[:4]], value, "fca_stem"))
        else:
            rows.append((ticker, None, value, "no_match"))
    frame = pd.DataFrame(rows, columns=["ticker", "cd_cvm", "volume", "basis"])
    matched = frame.dropna(subset=["cd_cvm"]).astype({"cd_cvm": int})
    main = matched.sort_values("volume", ascending=False).drop_duplicates("cd_cvm")
    total = matched.groupby("cd_cvm")["volume"].sum().rename("volume_mean_quarter")
    out = main.set_index("cd_cvm")[["ticker", "basis"]].join(total).reset_index()
    out.attrs["no_match"] = frame[frame["basis"] == "no_match"].head(15)
    return out


def full_history(statements: pd.DataFrame) -> pd.Series:
    """True for a company that has all quarters with a balanced sheet and positive assets."""
    quarters = quarter_ends()
    frame = statements[statements["quarter"].isin(quarters)].copy()
    residual = (
        frame["total_assets"] - frame["current_liabilities"] - frame["noncurrent_liabilities"] - frame["equity"]
    ).abs() / frame["total_assets"].abs()
    frame = frame[(residual <= IDENTITY_TOLERANCE) & (frame["total_assets"] > 0)]
    frame = frame.dropna(subset=MODEL_COLUMNS)
    counts = frame.groupby("cd_cvm")["quarter"].nunique()
    return counts == len(quarters)


def main() -> pd.DataFrame:
    statements = pd.read_csv(DATA_DIR / "cvm_quarterly.csv", parse_dates=["quarter"])
    companies = register()
    volume = company_volume(ticker_volume(), ticker_map(companies), b3_issuers())
    table = volume.merge(companies, on="cd_cvm")
    table["sector"] = table["SETOR_ATIV"].map(normalize_sector)
    table["financial"] = table["sector"].isin(FINANCIAL_SECTORS)
    complete = full_history(statements)
    table["full_history"] = table["cd_cvm"].map(complete).fillna(False).astype(bool)
    table = table.sort_values("volume_mean_quarter", ascending=False).reset_index(drop=True)

    eligible = table[~table["financial"] & table["full_history"]]
    universe = eligible.head(TOP_N).copy()
    universe = universe.rename(
        columns={"DENOM_SOCIAL": "company", "DENOM_COMERC": "trade_name", "SETOR_ATIV": "sector_raw"}
    )
    universe = universe[
        ["cd_cvm", "cnpj", "ticker", "company", "trade_name", "sector", "sector_raw", "volume_mean_quarter", "basis"]
    ]
    universe = universe.sort_values("ticker").reset_index(drop=True)
    universe.to_csv(DATA_DIR / "universe.csv", index=False)

    top = table.head(150)
    print(f"universe.csv: N = {len(universe)} companies ({len(eligible)} eligible), {len(quarter_ends())} quarters")
    print(f"ticker match basis in universe: {universe['basis'].value_counts().to_dict()}")
    print("top 150 by volume: financial =", int(top["financial"].sum()),
          "| no full history =", int((~top["financial"] & ~top["full_history"]).sum()))
    excluded = top[~top["financial"] & ~top["full_history"]].head(25)
    print("largest companies without full history:", ", ".join(excluded["ticker"]))
    print("largest tickers with no company match:", ", ".join(volume.attrs["no_match"]["ticker"]))
    print("sectors:", universe["sector"].value_counts().to_dict())
    return universe


if __name__ == "__main__":
    main()
