"""Turn the extracted facts into supplier links and a debt profile for each company.

Outputs:
    data/extracted_edges.csv  supplier links between 2 universe companies
    data/debt_profile.csv     share of debt that follows the CDI rate, for each company

A supplier or customer name is matched to a universe company by its exact normalized
name. The normalization removes accents, punctuation, a country tag such as "(Brasil)",
the legal suffix, and a leading "Grupo" or "Sistema". It writes "companhia" as "cia".
There is no similarity score: a name that does not match is counted and dropped.

Run: python -m src.extraction.aggregate
"""
from __future__ import annotations

import json
import re

import pandas as pd

from src.data_ingestion.download import DATA_DIR
from src.data_ingestion.fetch_fre import company_key, load_universe
from src.extraction.pipeline import FACTS, RUNS

GENERIC_FIRST_WORDS = {"grupo", "sistema"}
FLOATING_INDEXERS = {"CDI", "SELIC"}
TOTAL_TOLERANCE = (0.8, 1.2)
SUPPLIER_YEAR = 2021


def party_key(name: object) -> str:
    words = company_key(re.sub(r"\([^)]*\)", " ", str(name))).split()
    words = ["cia" if word == "companhia" else word for word in words]
    while words and words[0] in GENERIC_FIRST_WORDS:
        words = words[1:]
    return " ".join(words)


def _read(path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def universe_keys(universe: pd.DataFrame) -> dict[str, int]:
    """Name key -> company. A key that 2 companies share is ambiguous, so it is not used."""
    owners: dict[str, set[int]] = {}
    for column in ("company", "trade_name"):
        for name, code in zip(universe[column], universe["cd_cvm"]):
            if pd.notna(name) and party_key(name):
                owners.setdefault(party_key(name), set()).add(int(code))
    return {key: next(iter(codes)) for key, codes in owners.items() if len(codes) == 1}


def supplier_edges(facts: list[dict], universe: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    keys = universe_keys(universe)
    counts = {"facts": 0, "matched": 0, "own_name": 0, "no_match": 0}
    rows = []
    for fact in facts:
        if fact["task"] != "suppliers":
            continue
        counts["facts"] += 1
        other = keys.get(party_key(fact["counterparty"]))
        if other is None:
            counts["no_match"] += 1
        elif other == fact["cd_cvm"]:
            counts["own_name"] += 1
        else:
            counts["matched"] += 1
            pair = (other, fact["cd_cvm"]) if fact["role"] == "supplier" else (fact["cd_cvm"], other)
            rows.append({"src_cd_cvm": pair[0], "dst_cd_cvm": pair[1], "evidence": fact["quote"][:160]})
    edges = pd.DataFrame(rows, columns=["src_cd_cvm", "dst_cd_cvm", "evidence"])
    edges = edges.groupby(["src_cd_cvm", "dst_cd_cvm"], as_index=False).agg(
        rows=("evidence", "size"), evidence=("evidence", "first")
    )
    edges["weight"] = edges["rows"].astype(float)
    edges.insert(0, "relation", "supplier")
    edges.insert(0, "year", SUPPLIER_YEAR)
    edges["basis"] = "name"
    return edges, counts


def debt_profile(facts: list[dict], runs: list[dict]) -> pd.DataFrame:
    """For each company: the share of debt at a floating domestic rate, with a consistency status.

    floating_share counts a line with the CDI or SELIC indexer, or a line that the text
    says is swapped to CDI. The share is given only when the sum of the lines is within
    20 % of the total debt that the notes state, or when the notes state no total.
    """
    summary = {run["cd_cvm"]: run for run in runs if run["task"] == "debt"}
    rows = []
    for cd_cvm, run in summary.items():
        slices = [f for f in facts if f["task"] == "debt" and f["cd_cvm"] == cd_cvm and (f.get("amount") or 0) > 0]
        amount_sum = sum(item["amount"] for item in slices)
        floating = sum(i["amount"] for i in slices if i["indexer"] in FLOATING_INDEXERS or i.get("hedged_to_cdi"))
        contract = sum(i["amount"] for i in slices if i["indexer"] in FLOATING_INDEXERS)
        total = run.get("summary", {}).get("total_debt")
        if run["status"] != "ok" or amount_sum <= 0:
            status = "no_amounts"
        elif not total:
            status = "ok_no_stated_total"
        elif TOTAL_TOLERANCE[0] <= amount_sum / total <= TOTAL_TOLERANCE[1]:
            status = "ok"
        else:
            status = "sum_differs_from_total"
        usable = status.startswith("ok")
        rows.append(
            {
                "cd_cvm": cd_cvm,
                "ticker": run["ticker"],
                "floating_share": floating / amount_sum if usable else None,
                "contract_cdi_share": contract / amount_sum if usable else None,
                "amount_sum": amount_sum,
                "total_debt": total,
                "unit": run.get("summary", {}).get("unit"),
                "lines": len(slices),
                "status": status,
            }
        )
    return pd.DataFrame(rows).sort_values("ticker") if rows else pd.DataFrame()


def main() -> None:
    facts, runs = _read(FACTS), _read(RUNS)
    universe = load_universe()
    edges, counts = supplier_edges(facts, universe)
    edges.to_csv(DATA_DIR / "extracted_edges.csv", index=False)
    profile = debt_profile(facts, runs)
    profile.to_csv(DATA_DIR / "debt_profile.csv", index=False)

    ticker = dict(zip(universe["cd_cvm"], universe["ticker"]))
    status = pd.Series([run["status"] for run in runs]).value_counts().to_dict() if runs else {}
    print(f"runs: {len(runs)} {status} | facts kept: {len(facts)} | facts dropped: {sum(r['dropped'] for r in runs)}")
    print(f"supplier facts: {counts} -> {len(edges)} links between universe companies")
    print("  " + ", ".join(f"{ticker[s]}>{ticker[d]}" for s, d in zip(edges["src_cd_cvm"], edges["dst_cd_cvm"])))
    if len(profile):
        usable = profile[profile["floating_share"].notna()]
        print(f"debt profile: {profile['status'].value_counts().to_dict()}")
        print(
            f"  floating share for {len(usable)} companies: median {usable['floating_share'].median():.2f}, "
            f"contract CDI share median {usable['contract_cdi_share'].median():.2f}"
        )


if __name__ == "__main__":
    main()
