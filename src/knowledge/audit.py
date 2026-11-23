"""The audit trail of 1 company in a stress scenario.

The trail is an ordered list of records. Each record says where its numbers come from:
    input      the shock that the user gave
    filed      numbers and links that the company filed
    extracted  text that the extraction read from a filing, with its quote
    rule       arithmetic of a stated accounting rule on the numbers above
    model      output of a model (the seasonal ridge base path, or the neural engine)
    estimate   a value that a regression estimated
    none       no source: the channel gives no effect

Use:
    scenario = simulate({"SELIC_delta": 3.0})
    print(Auditor(scenario).render("PETR4"))
"""
from __future__ import annotations

import json
import pandas as pd

import torch

from src.knowledge import rules
from src.knowledge.graph import GRAPH_PATH
from src.simulator.stress_test import ACCOUNTS, Scenario
from src.simulator.transmission import TAX

TOP_NEIGHBOURS = 5
MAX_LINKS = 6


def brl(value: float) -> str:
    if value != value:  # NaN
        return "unknown"
    if abs(value) >= 1e9:
        return f"R$ {value / 1e9:,.2f} bn"
    return f"R$ {value / 1e6:,.1f} mn"



def share(value: float) -> str:
    return "unknown" if pd.isna(value) else f"{value:.0%}"


def _rank(value) -> str:
    """A rank, or "unranked" when the engine has no result for the company."""
    return "unranked" if pd.isna(value) else str(int(value))

class Auditor:
    def __init__(self, scenario: Scenario):
        self.scenario = scenario
        self.meta = scenario.meta
        self.index = {ticker: position for position, ticker in enumerate(self.meta["tickers"])}
        self.cd_cvm = dict(zip(self.meta["tickers"], self.meta["cd_cvm"]))
        self.table = scenario.table().set_index("ticker")
        graph = json.loads(GRAPH_PATH.read_text()) if GRAPH_PATH.exists() else {"nodes": [], "edges": []}
        self.names = {node["id"]: node["properties"].get("content", node["id"]) for node in graph["nodes"]}
        self.edges = graph["edges"]

    def _links(self, node: str, kind: str) -> list[dict]:
        return [e for e in self.edges if e["type"] == kind and node in (e["source_id"], e["target_id"])]

    def trail(self, ticker: str) -> list[dict]:
        scenario, node, code = self.scenario, self.index[ticker], self.cd_cvm[ticker]
        row = self.table.loc[ticker]
        records: list[dict] = []

        def add(kind: str, basis: str, text: str) -> None:
            records.append({"step": len(records) + 1, "kind": kind, "basis": basis, "text": text})

        shock = scenario.shock
        add(
            "shock",
            "input",
            f"SELIC {shock.SELIC_delta:+.2f} pp, IPCA {shock.IPCA_delta:+.2f} pp, PTAX {shock.PTAX_delta_pct:+.1f} %, "
            f"held for {scenario.horizon} quarters after {self.meta['quarters'][-1]}",
        )
        last = dict(zip(ACCOUNTS, scenario.last[node].tolist()))
        add("last filed balance", "filed", ", ".join(f"{name} {brl(value)}" for name, value in last.items()))

        exposure = scenario.exposures[code]
        add(
            "debt exposure",
            exposure.profile_basis if exposure.profile_basis != "none" else exposure.floating_basis,
            f"gross debt {brl(exposure.gross_debt)} (filed accounts 2.01.04 + 2.02.01); "
            f"floating {brl(exposure.floating_debt)} ({share(exposure.floating_share)}) [{exposure.floating_basis}], "
            f"IPCA {brl(exposure.ipca_debt)} ({share(exposure.ipca_share)}), "
            f"foreign currency not swapped {brl(exposure.fx_debt)} ({share(exposure.fx_share)}) [{exposure.profile_basis}]; "
            f"source: {exposure.amount_source}",
        )
        for effect in scenario.effects[code]:
            add(
                f"transmission, quarter +{effect.quarter}",
                "rule",
                f"extra interest {brl(effect.interest)}, IPCA accrual {brl(effect.ipca_accrual)}, "
                f"FX revaluation {brl(effect.fx_revaluation)}; after {TAX:.0%} tax, equity {brl(float(effect.delta[3]))}",
            )
        for quarter in range(scenario.horizon):
            add(
                f"hybrid path, quarter +{quarter + 1}",
                "model",
                f"equity base {brl(float(scenario.hybrid_base[quarter, node, 3]))}, "
                f"shock {brl(float(scenario.hybrid_shock[quarter, node, 3]))}",
            )

        final = dict(zip(ACCOUNTS, scenario.hybrid_shock[-1, node].tolist()))
        results = rules.balance_rules(final, "model")
        results.append(rules.equity_loss(float(scenario.hybrid_base[-1, node, 3]), final["equity"], last["equity"]))
        if code in scenario.flows.index:
            flows = scenario.flows.loc[code]
            results.append(rules.interest_coverage(flows["ebit_ttm"], flows["finexp_ttm"]))
        coverage = scenario.coverage_under_shock()[node]
        results.append(coverage)
        for result in results:
            add("rule", result.basis, result.line())
        if row["breaching_suppliers"]:
            add("contagion", "rule", f"a supplier with high dependence breaches coverage under the shock: {row['breaching_suppliers']}")

        company = f"company:{ticker}"
        for edge in self._links(company, "SUPPLIES")[:MAX_LINKS]:
            properties = edge["properties"].get("properties", edge["properties"])
            role = "supplier" if edge["target_id"] == company else "customer"
            other = edge["source_id"] if role == "supplier" else edge["target_id"]
            add(
                f"{role} link",
                "extracted",
                f"{self.names.get(other, other)} ({properties.get('dependence')} dependence): "
                f"\"{properties.get('quote')}\" [{properties.get('location')}]",
            )
        for edge in self._links(company, "HAS_LENDER")[:MAX_LINKS]:
            properties = edge["properties"].get("properties", edge["properties"])
            add(
                "lender link",
                "extracted",
                f"{self.names.get(edge['target_id'], edge['target_id'])}, indexer {properties.get('indexer')}: "
                f"\"{properties.get('quote')}\" [{properties.get('location')}]",
            )
        for kind in ("OWNS", "RELATED_PARTY"):
            for edge in self._links(company, kind)[:MAX_LINKS]:
                other = edge["target_id"] if edge["source_id"] == company else edge["source_id"]
                direction = "to" if edge["source_id"] == company else "from"
                add(f"{kind.lower()} link", "filed", f"{direction} {self.names.get(other, other)}, weight {edge['weight']:g}")

        add(
            f"neural engine ({scenario.neural_variant})",
            "model",
            f"equity base {brl(float(row['neural_equity_base']))}, shock {brl(float(row['neural_equity_shock']))}, "
            f"change {row['neural_change_pct']:+.2f} % of filed equity, neural rank {_rank(row['neural_rank'])}. "
            "This engine does not beat the mean-change forecast on the test split.",
        )
        edge_index, weights = scenario.attention
        incoming = (edge_index[1] == node) & (edge_index[0] != node)
        # A pair can have more than 1 edge (1 for each relation): sum the weight of each neighbour.
        total = torch.zeros(len(self.meta["tickers"])).index_add_(0, edge_index[0][incoming], weights[incoming])
        for source in total.argsort(descending=True)[:TOP_NEIGHBOURS].tolist():
            if total[source] <= 0:
                break
            neighbour = self.meta["tickers"][source]
            add(
                "graph neighbour",
                "model",
                f"{neighbour}: attention weight {float(total[source]):.3f}. "
                "An attention weight is a model weight, not proof of a causal link.",
            )
        return records

    def render(self, ticker: str) -> str:
        row = self.table.loc[ticker]
        flag = " | FILED EQUITY NOT POSITIVE" if row["equity_not_positive"] else ""
        if pd.isna(row["rank"]):
            result = "no hybrid rank: an input of the shock effect is unknown (see the debt exposure line)"
        else:
            change = row["hybrid_change_pct"]
            effect = "not computed" if pd.isna(change) else f"{change:+.2f} % of filed equity"
            ranked = int(self.table["rank"].notna().sum())
            result = f"hybrid rank {int(row['rank'])} of {ranked} ranked companies | equity change caused by the shock: {effect}"
        header = f"{ticker} | {row['company']} | {row['sector']}\n{result}{flag}"
        lines = [f"{record['step']:>2}. [{record['basis']:9s}] {record['kind']}: {record['text']}" for record in self.trail(ticker)]
        return header + "\n" + "\n".join(lines)
