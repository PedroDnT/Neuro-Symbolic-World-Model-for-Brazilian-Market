"""Step 6 of docs/IMPROVEMENTS.md: a worst-case bound on firm-to-firm contagion.

For the 4 largest dollar rises since 2015, compute each universe company's direct FX loss, then
let losses travel along the known links (ownership, supplier, related party) with worst-case
amounts, to a fixed point. The result is the second-round loss as a share of the direct loss.
The method was fixed in the tracker before the first run.

Run: python -m src.evaluation.contagion_bound
Writes data/contagion_bound.csv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_ingestion.download import DATA_DIR
from src.data_ingestion.fetch_cvm import _clean, _last_version, _read

BALANCE_DATE = pd.Timestamp("2024-12-31")
FIRST_SHOCK = pd.Timestamp("2015-01-01")
SHOCK_COUNT = 4
THETAS = (0.665, 1.0)  # step 4 pass-through, and full pass-through as an upper case
USABLE = {"ok", "ok_no_stated_total", "ok_filed_total"}
RECEIVABLES = {"1.01.03": ("receivables", "contas a receber")}
TOLERANCE = 1.0  # reais
LINK_TYPES = ["ownership", "supplier", "related_party"]


def largest_rises(macro: pd.DataFrame) -> pd.Series:
    shock = macro.set_index("quarter")["PTAX"].pct_change()
    return shock[shock.index >= FIRST_SHOCK].nlargest(SHOCK_COUNT)


def fx_debt(universe: pd.Series, profile: pd.DataFrame, stocks: pd.DataFrame, unknown_as_total: bool = False) -> pd.Series:
    """FX debt of each company: the audited debt-note amount when usable, else the filed split.

    With `unknown_as_total`, a company with neither (no usable amount, split 0) gets its total
    filed debt: the worst case. This rule was added after RE3; the registered rule gives 0.
    """
    audited = profile[profile["status"].isin(USABLE)].set_index("cd_cvm")["fx_debt_brl"].dropna()
    split = stocks["fx_loans_short"].fillna(0) + stocks["fx_loans_long"].fillna(0)
    total = stocks["debt_short"].fillna(0) + stocks["debt_long"].fillna(0)
    values = []
    for c in universe:
        value = audited.get(c, split.get(c, 0.0))
        if unknown_as_total and c not in audited.index and not split.get(c, 0.0) > 0:
            value = total.get(c, 0.0)
        values.append(value)
    return pd.Series(values, index=universe, dtype=float)


def links(fre: pd.DataFrame, suppliers: pd.DataFrame) -> pd.DataFrame:
    """1 row for each exposure: `exposed` loses when `source` is in distress (or owns a part of it)."""
    ownership = fre[fre["relation"] == "ownership"]
    related = fre[fre["relation"] == "related_party"]
    rows = [
        pd.DataFrame({"type": "ownership", "exposed": ownership["src_cd_cvm"], "source": ownership["dst_cd_cvm"], "stake": ownership["weight"] / 100}),
        pd.DataFrame({"type": "supplier", "exposed": suppliers["src_cd_cvm"], "source": suppliers["dst_cd_cvm"], "stake": np.nan}),
        pd.DataFrame({"type": "related_party", "exposed": related["src_cd_cvm"], "source": related["dst_cd_cvm"], "stake": np.nan}),
        pd.DataFrame({"type": "related_party", "exposed": related["dst_cd_cvm"], "source": related["src_cd_cvm"], "stake": np.nan}),
    ]
    return pd.concat(rows, ignore_index=True).drop_duplicates(["type", "exposed", "source"])


def propagate(direct: pd.Series, equity: pd.Series, receivables: pd.Series, edges: pd.DataFrame) -> tuple[pd.Series, dict]:
    """Total loss of each company at the fixed point, and the added loss by link type.

    Distress = loss / equity, between 0 and 1. A company with no equity left (equity <= 0) and a
    loss is fully distressed (worst case). Supplier and related-party links share 1 pool: a
    company loses its receivables × the largest distress among all its trade counterparties, so
    it never loses more than its receivables. The added losses by type are scaled so that they
    add up to the second round after the cap at equity.
    """
    cap = equity.clip(lower=0.0)
    insolvent = ~(equity > 0)
    loss = direct.clip(upper=cap)
    trade = edges[edges["type"] != "ownership"]
    owned = edges[edges["type"] == "ownership"]
    extra = pd.Series(0.0, index=direct.index)
    for _ in range(10_000):
        distress = (loss / equity.where(equity > 0)).clip(0, 1).fillna(0.0)
        distress[insolvent & ((direct + extra) > 0)] = 1.0
        own = owned.assign(amount=owned["stake"] * owned["source"].map(loss).fillna(0.0))
        own_add = own.groupby("exposed")["amount"].sum().reindex(direct.index).fillna(0.0).astype(float)
        linked = trade.assign(d=trade["source"].map(distress).fillna(0.0))
        top = linked.sort_values("d").groupby("exposed").tail(1).set_index("exposed")
        trade_add = (top["d"].astype(float) * top.index.map(receivables).fillna(0.0).astype(float)).reindex(direct.index).fillna(0.0).astype(float)
        extra = own_add + trade_add
        new = (direct + extra).clip(upper=cap)
        if float((new - loss).abs().max()) < TOLERANCE:
            loss = new
            break
        loss = new
    else:
        raise RuntimeError("the contagion fixed point did not converge in 10,000 rounds")
    raw = {"ownership": own_add, "supplier": trade_add.where(top["type"].reindex(direct.index) == "supplier", 0.0),
           "related_party": trade_add.where(top["type"].reindex(direct.index) == "related_party", 0.0)}
    added = loss - direct.clip(upper=cap)
    scale = (added / extra.where(extra > 0)).fillna(0.0)
    shares = {kind: float((series * scale).sum()) for kind, series in raw.items()}
    return loss, shares


def receivables_2024(stocks: pd.DataFrame) -> pd.Series:
    """Trade receivables (1.01.03) of each company at 2024-12-31, in the company's scope."""
    parts = []
    for scope in ("con", "ind"):
        raw = _read("dfp", BALANCE_DATE.year, "BPA", scope, set(RECEIVABLES))
        raw = raw[raw["ORDEM_EXERC"] == "ÚLTIMO"].assign(scope=scope)
        parts.append(_last_version(_clean(raw, RECEIVABLES)).assign(scope=scope))
    frame = pd.concat(parts, ignore_index=True)
    frame = frame[frame["quarter"] == BALANCE_DATE]
    wanted = stocks.reset_index()[["cd_cvm", "scope"]]
    chosen = frame.merge(wanted, on=["cd_cvm", "scope"]).groupby("cd_cvm")["value"].first()
    above = chosen[chosen > stocks["total_assets"].reindex(chosen.index)]
    if len(above):
        raise ValueError(f"receivables above total assets (a unit error): {above.to_dict()}")
    return chosen


def main() -> pd.DataFrame:
    universe = pd.read_csv(DATA_DIR / "universe.csv")["cd_cvm"]
    statements = pd.read_csv(DATA_DIR / "cvm_quarterly.csv", parse_dates=["quarter"])
    stocks = statements[statements["quarter"] == BALANCE_DATE].drop_duplicates("cd_cvm").set_index("cd_cvm")
    missing = sorted(set(universe) - set(stocks.index))
    universe = universe[universe.isin(stocks.index)].reset_index(drop=True)
    stocks = stocks.reindex(universe)
    edges = links(pd.read_csv(DATA_DIR / "fre_edges.csv"), pd.read_csv(DATA_DIR / "extracted_edges.csv"))
    profile = pd.read_csv(DATA_DIR / "debt_profile.csv")
    receivables = receivables_2024(stocks.dropna(subset=["scope"]))
    equity = stocks["equity"]
    rows = []
    macro = pd.read_csv(DATA_DIR / "macro_quarterly.csv", parse_dates=["quarter"])
    for rule, unknown_as_total in (("registered", False), ("unknown_as_total", True)):
        exposure = fx_debt(universe, profile, stocks, unknown_as_total)
        exposure.index = stocks.index
        for quarter, s in largest_rises(macro).items():
            for theta in THETAS:
                direct = (theta * exposure * s).clip(lower=0.0)
                loss, added = propagate(direct, equity, receivables, edges)
                capped = float(direct.clip(upper=equity.clip(lower=0)).sum())
                second = float(loss.sum()) - capped
                rows.append({"fx_rule": rule, "quarter": quarter.date(), "s": s, "theta": theta,
                             "direct_brl_bn": float(direct.sum()) / 1e9, "direct_capped_brl_bn": capped / 1e9,
                             "second_round_brl_bn": second / 1e9, "second_over_direct_capped": second / capped,
                             "second_over_direct": second / float(direct.sum()),
                             **{f"{k}_brl_bn": v / 1e9 for k, v in added.items()}})
    table = pd.DataFrame(rows)
    table.to_csv(DATA_DIR / "contagion_bound.csv", index=False)
    pd.set_option("display.width", 200)
    print(f"left out, no 2024 Q4 balance: {missing}")
    print(f"companies: {len(universe)}, with receivables: {receivables.notna().sum()}, links: {edges['type'].value_counts().to_dict()}")
    print(table.round(4).to_string(index=False))
    return table


if __name__ == "__main__":
    main()
