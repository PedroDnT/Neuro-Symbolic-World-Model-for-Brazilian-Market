"""Ownership and related-party links between companies of the universe.

Source: the structured tables of the CVM reference form (FRE) of 2021, the last year of
the train split. A link is kept only when both ends are companies of the universe.

    ownership       a shareholder CNPJ, or an investee CNPJ, is a universe company
    related_party   the related party is a universe company, by its CNPJ or by its
                    exact normalized name

There is no fuzzy match. A party that does not match is counted and dropped.

Output: data/fre_edges.csv
Run:    python -m src.data_ingestion.fetch_fre
"""
from __future__ import annotations

import re
import zipfile

import pandas as pd

from src.data_ingestion.download import CVM_BASE, DATA_DIR, RAW_DIR, download
from src.data_ingestion.fetch_cvm import normalize_text

MODULE = "src/data_ingestion/fetch_fre.py"
FRE_YEAR = 2021


def load_universe() -> pd.DataFrame:
    frame = pd.read_csv(DATA_DIR / "universe.csv", dtype={"cnpj": str})
    frame["cnpj"] = frame["cnpj"].str.zfill(14)
    return frame


def company_key(name: object) -> str:
    """Company name for an exact match: no accents, no punctuation, no legal suffix."""
    if pd.isna(name):
        return ""
    text = " ".join(re.sub(r"[^a-z0-9 ]", " ", normalize_text(name)).split())
    return re.sub(r"\s+(s a|sa)$", "", text)


def _digits(series: pd.Series) -> pd.Series:
    return series.fillna("").str.replace(r"\D", "", regex=True)


def _table(archive: zipfile.ZipFile, name: str) -> pd.DataFrame:
    """A FRE table, with the highest version of each company only."""
    with archive.open(f"fre_cia_aberta_{name}_{FRE_YEAR}.csv") as handle:
        frame = pd.read_csv(handle, sep=";", encoding="latin1", dtype=str)
    frame["filer"] = _digits(frame["CNPJ_Companhia"])
    frame["versao"] = frame["Versao"].astype(int)
    return frame[frame["versao"] == frame.groupby("filer")["versao"].transform("max")]


def build_edges(universe: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    by_cnpj = dict(zip(universe["cnpj"], universe["cd_cvm"]))
    by_name: dict[str, int] = {}
    for column in ("company", "trade_name"):
        for name, code in zip(universe[column], universe["cd_cvm"]):
            if company_key(name):
                by_name.setdefault(company_key(name), code)

    archive_name = f"fre_cia_aberta_{FRE_YEAR}.zip"
    path = download(f"{CVM_BASE}/DOC/FRE/DADOS/{archive_name}", RAW_DIR / "cvm" / archive_name, MODULE)
    rows = []
    with zipfile.ZipFile(path) as archive:
        holders = _table(archive, "posicao_acionaria")
        holders = holders[holders["filer"].isin(by_cnpj)].assign(other=lambda d: _digits(d["CPF_CNPJ_Acionista"]))
        for _, row in holders[holders["other"].isin(by_cnpj)].iterrows():
            share = pd.to_numeric(row["Percentual_Total_Acoes_Circulacao"], errors="coerce")
            rows.append(("ownership", by_cnpj[row["other"]], by_cnpj[row["filer"]], share, "cnpj", "posicao_acionaria"))

        investees = _table(archive, "participacao_sociedade")
        investees = investees[investees["filer"].isin(by_cnpj)].assign(other=lambda d: _digits(d["CNPJ"]).str.zfill(14))
        for _, row in investees[investees["other"].isin(by_cnpj)].iterrows():
            share = pd.to_numeric(row["Participacao_Emissor"], errors="coerce")
            rows.append(("ownership", by_cnpj[row["filer"]], by_cnpj[row["other"]], share, "cnpj", "participacao_sociedade"))

        related = _table(archive, "transacao_parte_relacionada")
        related = related[related["filer"].isin(by_cnpj)]
        document = _digits(related["Documento_Parte_Relacionada"])
        name_key = related["Parte_Relacionada"].map(company_key)
        counts = {"related_rows": len(related), "by_cnpj": 0, "by_name": 0, "no_match": 0}
        for (_, row), doc, key in zip(related.iterrows(), document, name_key):
            if doc in by_cnpj:
                target, basis = by_cnpj[doc], "cnpj"
            elif key in by_name:
                target, basis = by_name[key], "name"
            else:
                counts["no_match"] += 1
                continue
            counts[f"by_{basis}"] += 1
            evidence = " ".join(str(row["Objeto_Contrato"]).split())[:160]
            rows.append(("related_party", by_cnpj[row["filer"]], target, 1.0, basis, evidence))

    edges = pd.DataFrame(rows, columns=["relation", "src_cd_cvm", "dst_cd_cvm", "weight", "basis", "evidence"])
    edges = edges[edges["src_cd_cvm"] != edges["dst_cd_cvm"]]
    # Weight: the largest filed share in percent for ownership, the number of filed
    # transactions for a related party.
    grouped = edges.groupby(["relation", "src_cd_cvm", "dst_cd_cvm"], as_index=False).agg(
        weight=("weight", "max"), rows=("weight", "size"), basis=("basis", "first"), evidence=("evidence", "first")
    )
    is_related = grouped["relation"] == "related_party"
    grouped.loc[is_related, "weight"] = grouped.loc[is_related, "rows"].astype(float)
    grouped.insert(0, "year", FRE_YEAR)
    return grouped, counts


def main() -> pd.DataFrame:
    universe = load_universe()
    edges, counts = build_edges(universe)
    edges.to_csv(DATA_DIR / "fre_edges.csv", index=False)
    ticker = dict(zip(universe["cd_cvm"], universe["ticker"]))
    print(f"fre_edges.csv: {len(edges)} links between universe companies: {edges['relation'].value_counts().to_dict()}")
    print(f"related-party rows of universe companies: {counts}")
    for relation, part in edges.groupby("relation"):
        pairs = [f"{ticker[s]}>{ticker[d]}" for s, d in zip(part["src_cd_cvm"], part["dst_cd_cvm"])]
        print(f"  {relation}: {', '.join(pairs[:40])}")
    return edges


if __name__ == "__main__":
    main()
