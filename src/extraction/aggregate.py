"""Turn the extracted facts into supplier links and a debt profile for each company.

Outputs:
    data/extracted_edges.csv  supplier links between 2 universe companies. Weight 1 for each
                              pair. `by_cutoff` is true when the quote is also in the version
                              of the form that CVM received by 2021-12-31 (the end of the
                              train split), so the model may use the link.
    data/debt_profile.csv     shares of debt by rate (floating, IPCA, foreign currency)

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
from src.extraction import sources, validate
from src.extraction.pipeline import FACTS, RUNS, TASKS

GENERIC_FIRST_WORDS = {"grupo", "sistema"}
FLOATING_INDEXERS = {"CDI", "SELIC"}
FX_INDEXERS = {"USD", "EUR", "OTHER_FX"}
# The first extraction schema had 1 field for the rate and the currency, so the model often
# gave "FIXED" or "OTHER" to a fixed-rate or SOFR line in US dollars. Such a line counts as
# foreign currency when its own quote names a currency code.
FX_CODE = re.compile(r"(?<![A-Za-z])(USD|EUR|GBP|CAD|AUD|JPY|CHF|CNY)(?![A-Za-z])|US\$")


def is_fx(item: dict) -> bool:
    """A foreign-currency line that is not swapped to CDI."""
    if item.get("hedged_to_cdi"):
        return False
    if item["indexer"] in FX_INDEXERS:
        return True
    return item["indexer"] in ("FIXED", "OTHER") and FX_CODE.search(item["quote"]) is not None
TOTAL_TOLERANCE = (0.8, 1.2)
SUPPLIER_YEAR = 2021
TRAIN_END = "2021-12-31"
PROFILE_COLUMNS = [
    "cd_cvm", "ticker", "floating_share", "contract_cdi_share", "ipca_share", "fx_share",
    "amount_sum", "total_debt", "unit", "lines", "lines_without_amount_in_quote", "status",
]


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


def _cutoff_text(cd_cvm: int, cache: dict) -> list[dict]:
    """Section 7 of the 2021 reference form, in the last version received by the train end."""
    if cd_cvm not in cache:
        task = TASKS["suppliers"]
        document = sources.document_link(task["doc"], task["year"], cd_cvm, received_by=TRAIN_END)
        sections = []
        if document is not None:
            sections = [vars(section) for section in sources.fre_sections(sources.fetch(document))]
        cache[cd_cvm] = sections
    return cache[cd_cvm]


def supplier_edges(facts: list[dict], universe: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    keys = universe_keys(universe)
    counts = {"facts": 0, "matched": 0, "own_name": 0, "no_match": 0}
    rows, cutoff_cache = [], {}
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
            by_cutoff = validate.find_location(fact["quote"], _cutoff_text(fact["cd_cvm"], cutoff_cache)) is not None
            rows.append(
                {
                    "src_cd_cvm": pair[0],
                    "dst_cd_cvm": pair[1],
                    "dependence": fact["dependence"],
                    "by_cutoff": by_cutoff,
                    "evidence": fact["quote"][:160],
                }
            )
    edges = pd.DataFrame(rows, columns=["src_cd_cvm", "dst_cd_cvm", "dependence", "by_cutoff", "evidence"])
    rank = {"high": 3, "moderate": 2, "low": 1, "not_stated": 0}
    edges = edges.assign(dependence_rank=edges["dependence"].map(rank)).sort_values("dependence_rank", ascending=False)
    edges = edges.groupby(["src_cd_cvm", "dst_cd_cvm"], as_index=False).agg(
        rows=("evidence", "size"),
        dependence=("dependence", "first"),
        by_cutoff=("by_cutoff", "any"),
        evidence=("evidence", "first"),
    )
    edges["weight"] = 1.0
    edges.insert(0, "relation", "supplier")
    edges.insert(0, "year", SUPPLIER_YEAR)
    edges["basis"] = "name"
    return edges, counts


def debt_profile(facts: list[dict], runs: list[dict]) -> pd.DataFrame:
    """For each company: the shares of debt by rate, with a consistency status.

    A line counts only when its amount is 1 of the numbers of its quote.
    floating_share  CDI or SELIC indexer, or a line that the text says is swapped to CDI
    ipca_share      IPCA indexer and not swapped to CDI
    fx_share        a foreign currency and not swapped to CDI (see is_fx)
    The shares are given only when the sum of the lines is within 20 % of the total debt
    that the notes state, or when the notes state no total.
    """
    summary = {run["cd_cvm"]: run for run in runs if run["task"] == "debt"}
    rows = []
    for cd_cvm, run in summary.items():
        lines = [f for f in facts if f["task"] == "debt" and f["cd_cvm"] == cd_cvm and (f.get("amount") or 0) > 0]
        slices = [line for line in lines if validate.amount_in_quote(line)]
        amount_sum = sum(item["amount"] for item in slices)

        def share_of(test) -> float:
            return sum(item["amount"] for item in slices if test(item)) / amount_sum if amount_sum > 0 else float("nan")

        floating = share_of(lambda i: i["indexer"] in FLOATING_INDEXERS or i.get("hedged_to_cdi"))
        contract = share_of(lambda i: i["indexer"] in FLOATING_INDEXERS)
        ipca = share_of(lambda i: i["indexer"] == "IPCA" and not i.get("hedged_to_cdi"))
        fx = share_of(is_fx)
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
                "floating_share": floating if usable else None,
                "contract_cdi_share": contract if usable else None,
                "ipca_share": ipca if usable else None,
                "fx_share": fx if usable else None,
                "amount_sum": amount_sum,
                "total_debt": total,
                "unit": run.get("summary", {}).get("unit"),
                "lines": len(slices),
                "lines_without_amount_in_quote": len(lines) - len(slices),
                "status": status,
            }
        )
    return pd.DataFrame(rows, columns=PROFILE_COLUMNS).sort_values("ticker")


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
    print(f"supplier facts: {counts} -> {len(edges)} links between universe companies, {int(edges['by_cutoff'].sum())} disclosed by {TRAIN_END}")
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
