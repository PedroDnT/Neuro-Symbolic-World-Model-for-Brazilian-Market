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


def fx_debt(universe: pd.Series, profile: pd.DataFrame, stocks: pd.DataFrame) -> pd.Series:
    """FX debt of each company: the audited debt-note amount when usable, else the filed split."""
    audited = profile[profile["status"].isin(USABLE)].set_index("cd_cvm")["fx_debt_brl"].dropna()
    split = stocks["fx_loans_short"].fillna(0) + stocks["fx_loans_long"].fillna(0)
    return pd.Series([audited.get(c, split.get(c, 0.0)) for c in universe], index=universe, dtype=float)


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
    """Total loss of each company at the fixed point, and the added loss by link type."""
    cap = equity.clip(lower=0.0)
    loss = direct.clip(upper=cap)
    trade = edges[edges["type"] != "ownership"]
    owned = edges[edges["type"] == "ownership"]
    for _ in range(10_000):
        distress = (loss / equity.where(equity > 0)).clip(0, 1).fillna(1.0).where(loss > 0, 0.0)
        added = {}
        own = owned.assign(amount=owned["stake"] * owned["source"].map(loss))
        added["ownership"] = own.groupby("exposed")["amount"].sum()
        for kind in ("supplier", "related_party"):
            part = trade[trade["type"] == kind]
            worst = part.assign(d=part["source"].map(distress)).groupby("exposed")["d"].max()
            added[kind] = worst * worst.index.map(receivables).fillna(0.0)
        extra = sum((series.reindex(direct.index).fillna(0.0) for series in added.values()), pd.Series(0.0, index=direct.index))
        new = (direct + extra).clip(upper=cap)
        if float((new - loss).abs().max()) < TOLERANCE:
            loss = new
            break
        loss = new
    shares = {kind: float(series.reindex(direct.index).fillna(0.0).sum()) for kind, series in added.items()}
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
    exposure = fx_debt(universe, pd.read_csv(DATA_DIR / "debt_profile.csv"), stocks)
    exposure.index = stocks.index
    receivables = receivables_2024(stocks.dropna(subset=["scope"]))
    equity = stocks["equity"]
    rows = []
    macro = pd.read_csv(DATA_DIR / "macro_quarterly.csv", parse_dates=["quarter"])
    for quarter, s in largest_rises(macro).items():
        for theta in THETAS:
            direct = (theta * exposure * s).clip(lower=0.0)
            loss, added = propagate(direct, equity, receivables, edges)
            capped = float(direct.clip(upper=equity.clip(lower=0)).sum())
            second = float(loss.sum()) - capped
            rows.append({"quarter": quarter.date(), "s": s, "theta": theta, "direct_brl_bn": capped / 1e9,
                         "second_round_brl_bn": second / 1e9, "second_over_direct": second / capped,
                         **{f"{k}_brl_bn": v / 1e9 for k, v in added.items()}})
    table = pd.DataFrame(rows)
    table.to_csv(DATA_DIR / "contagion_bound.csv", index=False)
    pd.set_option("display.width", 200)
    print(f"left out, no 2024 Q4 balance: {missing}")
    print(f"companies: {len(universe)}, with FX debt: {int((exposure > 0).sum())}, with receivables: {receivables.notna().sum()}, links: {edges['type'].value_counts().to_dict()}")
    print(table.round(4).to_string(index=False))
    return table


if __name__ == "__main__":
    main()
