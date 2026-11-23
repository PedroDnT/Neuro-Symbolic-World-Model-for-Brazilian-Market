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
import math
import re

import pandas as pd

from dataclasses import asdict
from pathlib import Path

from src.data_ingestion.download import DATA_DIR, RAW_DIR
from src.data_ingestion.fetch_fre import company_key, load_universe
from src.extraction import sources, validate
from src.extraction.pipeline import FACTS, RUNS, TASKS

GENERIC_FIRST_WORDS = {"grupo", "sistema"}
FLOATING_INDEXERS = {"CDI", "SELIC"}
FX_INDEXERS = {"USD", "EUR", "OTHER_FX"}
# The first extraction schema had 1 field for the rate and the currency, so the model often
# gave "FIXED" or "OTHER" to a fixed-rate or SOFR line in US dollars. Such a line counts as
# foreign currency when its own quote names a currency (a code such as USD, or Dólar, Euro,
# Iene), a foreign reference rate (SOFR, LIBOR), an export advance in foreign currency (ACC,
# ACE) or "moeda estrangeira". The word "bond" alone does not count: it names no currency.
FX_CODE = re.compile(
    r"(?<![A-Za-z])(USD|EUR|GBP|CAD|AUD|JPY|CHF|CNY|SOFR|LIBOR|ACC|ACE|D[óo]lar(?:es)?|Euros?|Ienes?)(?![A-Za-z])"
    r"|US\$|moeda estrangeira",
    re.I,
)


def is_foreign(item: dict) -> bool:
    """A line in a foreign currency, swapped to CDI or not."""
    if item["indexer"] in FX_INDEXERS:
        return True
    return item["indexer"] in ("FIXED", "OTHER") and FX_CODE.search(item["quote"]) is not None


def names_currency(item: dict) -> bool:
    """A line whose type or quote names a foreign currency, whatever its indexer. A dollar loan
    that the model indexed CDI (swapped, or at a CDI cost) counts here. Used by the FX
    cross-check, which compares with the filed split by currency of the loan."""
    return item["indexer"] in FX_INDEXERS or FX_CODE.search(item["quote"]) is not None


def is_fx(item: dict) -> bool:
    """A foreign-currency line that is not swapped to CDI."""
    return not item.get("hedged_to_cdi") and is_foreign(item)
TOTAL_TOLERANCE = (0.8, 1.2)
# Reviewed manual corrections of extracted facts, each with a verbatim quote of its source.
CORRECTIONS = Path(__file__).with_name("corrections.json")
CORRECTIONS_APPLIED = DATA_DIR / "corrections_applied.csv"
APPLIED_COLUMNS = [
    "id", "cd_cvm", "ticker", "task", "field", "old", "new", "fact_quote",
    "reason", "evidence_doc", "evidence_location", "evidence_quote",
]
# The filed split "Em Moeda Estrangeira" against the foreign-currency lines of the note.
FX_CHECK_BAND = (0.8, 1.2)
# The unit label of the model is wrong for some tables ("BRL" for a table in thousands), so
# only a stated thousand or million counts. Else the unit is the factor that puts the line
# sum closest to the filed debt, and that factor counts only inside this band.
STATED_UNITS = {"BRL_thousand": 1e3, "BRL_million": 1e6}
UNIT_FACTORS = (1.0, 1e3, 1e6)
INFERRED_UNIT_BAND = (0.5, 2.0)
SUPPLIER_YEAR = 2021
TRAIN_END = "2021-12-31"
PROFILE_COLUMNS = [
    "cd_cvm", "ticker", "floating_share", "contract_cdi_share", "ipca_share", "fx_share",
    "floating_debt_brl", "ipca_debt_brl", "fx_debt_brl", "filed_debt_brl", "sum_over_filed",
    "fx_lines_brl", "filed_fx_brl", "fx_check",
    "amount_sum", "total_debt", "unit", "unit_factor", "unit_basis",
    "lines", "lines_without_amount_in_quote", "status",
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


def source_sections(evidence: dict) -> list[dict]:
    """The full text of a cached source document, as sections for validate.find_location."""
    path = RAW_DIR / "rad" / f"{evidence['doc']}_{evidence['id_doc']}.zip"
    if not path.exists():
        raise FileNotFoundError(f"{path} is not cached: run python -m src.extraction.pipeline for this company first")
    found = sources.dfp_pages(path) if evidence["doc"] == "dfp" else sources.fre_sections(path)
    return [asdict(section) for section in found]


def apply_corrections(facts: list[dict], corrections: list[dict], sections_of=source_sections) -> tuple[list[dict], list[dict]]:
    """Apply reviewed corrections. Return the facts (copies) and 1 record for each changed field.

    A correction must quote its source verbatim, and it must match exactly the number of facts
    in its "expect" field. Else the run stops: a correction that no longer applies, or applies
    only in part, is never skipped without notice.
    """
    facts = [dict(fact) for fact in facts]
    applied = []
    for correction in corrections:
        evidence = correction["evidence"]
        location = validate.find_location(evidence["quote"], sections_of(evidence))
        if location is None:
            raise ValueError(f"correction {correction['id']}: the evidence quote is not in {evidence['doc']} {evidence['id_doc']}")
        hits = [
            fact for fact in facts
            if fact["cd_cvm"] == correction["cd_cvm"] and fact["task"] == correction["task"]
            and all(fact.get(field) in allowed for field, allowed in correction["match"].items())
        ]
        if len(hits) != correction["expect"]:
            raise ValueError(
                f"correction {correction['id']}: it matches {len(hits)} facts, not {correction['expect']}; update or remove it"
            )
        for fact in hits:
            original = {field: fact.get(field) for field in correction["set"]}
            fact.update(correction["set"])
            fact["corrected_by"] = correction["id"]
            fact["original"] = original
            for field, new in correction["set"].items():
                applied.append({
                    "id": correction["id"], "cd_cvm": correction["cd_cvm"], "ticker": correction["ticker"],
                    "task": correction["task"], "field": field, "old": original[field], "new": new,
                    "fact_quote": fact["quote"][:160], "reason": correction["reason"],
                    "evidence_doc": f"{evidence['doc']} {evidence['id_doc']}", "evidence_location": location,
                    "evidence_quote": evidence["quote"],
                })
    return facts, applied


def load_facts(write_log: bool = False) -> list[dict]:
    """The extracted facts with the reviewed corrections applied."""
    corrections = json.loads(CORRECTIONS.read_text()) if CORRECTIONS.exists() else []
    facts, applied = apply_corrections(_read(FACTS), corrections)
    if write_log:
        pd.DataFrame(applied, columns=APPLIED_COLUMNS).to_csv(CORRECTIONS_APPLIED, index=False)
    return facts


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


def unit_factor(unit: str | None, amount_sum: float, filed_debt: float) -> tuple[float | None, str]:
    """The factor from the unit of the debt table to reais, and its basis.

    A stated thousand or million is used as it is. Else the factor of UNIT_FACTORS that puts
    the line sum closest to the filed debt is used, only when the ratio is inside
    INFERRED_UNIT_BAND. Without a filed debt, or outside the band, the unit is unclear (None).
    """
    if unit in STATED_UNITS:
        return STATED_UNITS[unit], "stated"
    if not amount_sum > 0 or not filed_debt > 0:
        return None, "unclear"
    factor = min(UNIT_FACTORS, key=lambda f: abs(math.log(amount_sum * f / filed_debt)))
    ratio = amount_sum * factor / filed_debt
    if INFERRED_UNIT_BAND[0] <= ratio <= INFERRED_UNIT_BAND[1]:
        return factor, "inferred"
    return None, "unclear"


def fx_check(fx_lines: float, filed_fx: float) -> str:
    """Compare the foreign-currency lines of the note with the filed split "Em Moeda Estrangeira".

    Many companies leave the split at 0 even with foreign-currency debt, so a filed 0 means
    that the check cannot run, not that the company has no such debt.
    """
    if not filed_fx > 0:
        return "no_filed_split"
    ratio = fx_lines / filed_fx
    if ratio < FX_CHECK_BAND[0]:
        return "fx_lines_missing"
    if ratio > FX_CHECK_BAND[1]:
        return "fx_above_filed"
    return "ok"


def debt_profile(
    facts: list[dict], runs: list[dict], filed_debt: dict[int, float], filed_fx: dict[int, float] | None = None
) -> pd.DataFrame:
    """For each company: its debt by rate, as shares and as amounts in reais, with a status.

    A line counts only when its amount is 1 of the numbers of its quote.
    floating  CDI or SELIC indexer, or a line that the text says is swapped to CDI
    ipca      IPCA indexer and not swapped to CDI
    fx        a foreign currency and not swapped to CDI (see is_fx)
    `filed_debt` is the filed debt in reais (accounts 2.01.04 + 2.02.01) at the date of the
    note. The profile is usable only when the unit is known (see unit_factor) and the sum of
    the lines is within 20 % of the total that the note states, or of the filed debt, or
    when the note states no total. The amounts in reais are what the shock uses: the
    filed accounts can hold leases, and a note can list its lines before swap lines.
    """
    summary = {run["cd_cvm"]: run for run in runs if run["task"] == "debt"}
    rows = []
    for cd_cvm, run in summary.items():
        lines = [f for f in facts if f["task"] == "debt" and f["cd_cvm"] == cd_cvm and (f.get("amount") or 0) > 0]
        slices = [line for line in lines if validate.amount_in_quote(line)]
        amount_sum = sum(item["amount"] for item in slices)

        def amount_of(test) -> float:
            return sum(item["amount"] for item in slices if test(item))

        floating = amount_of(lambda i: i["indexer"] in FLOATING_INDEXERS or i.get("hedged_to_cdi"))
        contract = amount_of(lambda i: i["indexer"] in FLOATING_INDEXERS)
        ipca = amount_of(lambda i: i["indexer"] == "IPCA" and not i.get("hedged_to_cdi"))
        fx = amount_of(is_fx)
        foreign = amount_of(names_currency)
        unit = run.get("summary", {}).get("unit")
        total = run.get("summary", {}).get("total_debt")
        filed = float(filed_debt.get(cd_cvm, float("nan")))
        factor, unit_basis = unit_factor(unit, amount_sum, filed)
        sum_over_filed = amount_sum * factor / filed if factor and filed > 0 else float("nan")
        if run["status"] != "ok" or amount_sum <= 0:
            status = "no_amounts"
        elif factor is None:
            status = "unit_unclear"
        elif not total:
            status = "ok_no_stated_total"
        elif TOTAL_TOLERANCE[0] <= amount_sum / total <= TOTAL_TOLERANCE[1]:
            status = "ok"
        elif TOTAL_TOLERANCE[0] <= sum_over_filed <= TOTAL_TOLERANCE[1]:
            status = "ok_filed_total"
        else:
            status = "sum_differs_from_total"
        usable = status.startswith("ok")

        def share(amount: float):
            return amount / amount_sum if usable else None

        def brl(amount: float):
            return amount * factor if usable else None

        fx_filed = float((filed_fx or {}).get(cd_cvm, float("nan")))

        rows.append(
            {
                "cd_cvm": cd_cvm,
                "ticker": run["ticker"],
                "floating_share": share(floating),
                "contract_cdi_share": share(contract),
                "ipca_share": share(ipca),
                "fx_share": share(fx),
                "floating_debt_brl": brl(floating),
                "ipca_debt_brl": brl(ipca),
                "fx_debt_brl": brl(fx),
                "filed_debt_brl": filed,
                "sum_over_filed": sum_over_filed,
                "fx_lines_brl": brl(foreign),
                "filed_fx_brl": fx_filed,
                "fx_check": fx_check(foreign * factor, fx_filed) if usable else "no_profile",
                "amount_sum": amount_sum,
                "total_debt": total,
                "unit": unit,
                "unit_factor": factor,
                "unit_basis": unit_basis,
                "lines": len(slices),
                "lines_without_amount_in_quote": len(lines) - len(slices),
                "status": status,
            }
        )
    return pd.DataFrame(rows, columns=PROFILE_COLUMNS).sort_values("ticker")


def filed_at(date: str) -> tuple[dict[int, float], dict[int, float]]:
    """Filed debt in reais of each company at a date: total (accounts 2.01.04 + 2.02.01) and
    the split "Em Moeda Estrangeira" (2.01.04.01.02 + 2.02.01.01.02)."""
    statements = pd.read_csv(DATA_DIR / "cvm_quarterly.csv", parse_dates=["quarter"])
    if "fx_loans_short" not in statements:
        raise ValueError("data/cvm_quarterly.csv is older than the code: run python -m src.data_ingestion.fetch_cvm")
    last = statements[statements["quarter"] == pd.Timestamp(date)]
    debt = last["debt_short"].fillna(0) + last["debt_long"].fillna(0)
    fx = last["fx_loans_short"].fillna(0) + last["fx_loans_long"].fillna(0)
    return dict(zip(last["cd_cvm"], debt)), dict(zip(last["cd_cvm"], fx))


def main() -> None:
    facts, runs = load_facts(write_log=True), _read(RUNS)
    universe = load_universe()
    edges, counts = supplier_edges(facts, universe)
    edges.to_csv(DATA_DIR / "extracted_edges.csv", index=False)
    profile = debt_profile(facts, runs, *filed_at(f"{TASKS['debt']['year']}-12-31"))
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
        print(f"  FX cross-check: {profile['fx_check'].value_counts().to_dict()}")
        corrected = pd.read_csv(CORRECTIONS_APPLIED)
        print(f"  corrections applied: {len(corrected)} fields, " + ", ".join(sorted(set(corrected["id"]))))
        inferred = profile[profile["unit_basis"] == "inferred"]
        print("  inferred units: " + ", ".join(
            f"{row.ticker} x{row.unit_factor:,.0f} (sum / filed debt {row.sum_over_filed:.2f})" for row in inferred.itertuples()
        ))


if __name__ == "__main__":
    main()
