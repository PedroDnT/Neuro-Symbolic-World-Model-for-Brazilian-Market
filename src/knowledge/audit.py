"""The audit trail of 1 company in a stress scenario.

The trail is an ordered list of records. Each record says where its numbers come from:
    input      the shock that the user gave
    filed      numbers and links that the company filed
    extracted  text that the extraction read from a filing, with its quote
    model      output of the neural model
    estimate   a value that a regression estimated

Use:
    scenario = simulate({"SELIC_delta": 3.0})
    print(Auditor(scenario).render("PETR4"))
"""
from __future__ import annotations

import json

import pandas as pd

from src.data_ingestion.download import DATA_DIR
from src.data_ingestion.fetch_fre import load_universe
from src.knowledge import rules
from src.knowledge.graph import GRAPH_PATH
from src.knowledge.sensitivity import floating_shares
from src.simulator.stress_test import ACCOUNTS, Scenario

TOP_NEIGHBOURS = 5
MAX_LINKS = 6


def brl(value: float) -> str:
    if abs(value) >= 1e9:
        return f"R$ {value / 1e9:,.2f} bn"
    return f"R$ {value / 1e6:,.1f} mn"


class Auditor:
    def __init__(self, scenario: Scenario):
        self.scenario = scenario
        self.meta = scenario.meta
        self.index = {ticker: position for position, ticker in enumerate(self.meta["tickers"])}
        universe = load_universe()
        self.cd_cvm = dict(zip(universe["ticker"], universe["cd_cvm"]))
        statements = pd.read_csv(DATA_DIR / "cvm_quarterly.csv", parse_dates=["quarter"])
        last_quarter = pd.Timestamp(self.meta["quarters"][-1])
        self.flows = statements[statements["quarter"] == last_quarter].set_index("cd_cvm")
        shares = floating_shares(universe, self.meta["train_end"])
        self.shares = shares.set_index("ticker")
        self.table = scenario.table().set_index("ticker")
        graph = json.loads(GRAPH_PATH.read_text()) if GRAPH_PATH.exists() else {"nodes": [], "edges": []}
        self.names = {node["id"]: node["properties"].get("content", node["id"]) for node in graph["nodes"]}
        self.edges = graph["edges"]

    def _links(self, node: str, kind: str) -> list[dict]:
        return [e for e in self.edges if e["type"] == kind and node in (e["source_id"], e["target_id"])]

    def trail(self, ticker: str) -> list[dict]:
        scenario, node = self.scenario, self.index[ticker]
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

        for quarter in range(scenario.horizon):
            base, shocked = scenario.base[quarter, node], scenario.shocked[quarter, node]
            add(
                f"quarter +{quarter + 1}",
                "model",
                f"equity base {brl(float(base[3]))}, shock {brl(float(shocked[3]))} | "
                f"assets base {brl(float(base[0]))}, shock {brl(float(shocked[0]))}",
            )

        final = dict(zip(ACCOUNTS, scenario.shocked[-1, node].tolist()))
        results = rules.balance_rules(final, "model")
        results.append(rules.equity_loss(float(scenario.base[-1, node, 3]), final["equity"]))
        flows = self.flows.loc[self.cd_cvm[ticker]] if self.cd_cvm[ticker] in self.flows.index else None
        if flows is not None:
            results.append(rules.interest_coverage(flows["ebit_ttm"], flows["finexp_ttm"]))
            share = self.shares.loc[ticker]
            results.append(
                rules.interest_coverage_under_shock(
                    flows["ebit_ttm"],
                    flows["finexp_ttm"],
                    flows["debt_short"] + flows["debt_long"],
                    share["floating_share"],
                    shock.SELIC_delta,
                    share["basis"],
                )
            )
        for result in results:
            add("rule", result.basis, result.line())

        company = f"company:{ticker}"
        for edge in self._links(company, "SUPPLIES")[:MAX_LINKS]:
            properties = edge["properties"]
            role = "supplier" if edge["target_id"] == company else "customer"
            other = edge["source_id"] if role == "supplier" else edge["target_id"]
            add(
                f"{role} link",
                "extracted",
                f"{self.names.get(other, other)} ({properties.get('dependence')} dependence): "
                f"\"{properties.get('quote')}\" [{properties.get('location')}]",
            )
        for edge in self._links(company, "HAS_LENDER")[:MAX_LINKS]:
            properties = edge["properties"]
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

        edge_index, weights = scenario.attention
        incoming = (edge_index[1] == node) & (edge_index[0] != node)
        order = weights[incoming].argsort(descending=True)[:TOP_NEIGHBOURS]
        for source, weight in zip(edge_index[0][incoming][order].tolist(), weights[incoming][order].tolist()):
            neighbour = self.meta["tickers"][source]
            change = self.table.loc[neighbour, "equity_change_pct"]
            add(
                "graph neighbour",
                "model",
                f"{neighbour}: attention weight {weight:.3f}, its equity change under the shock {change:+.2f} %. "
                "An attention weight is a model weight, not proof of a causal link.",
            )
        return records

    def render(self, ticker: str) -> str:
        row = self.table.loc[ticker]
        header = (
            f"{ticker} | {row['company']} | {row['sector']}\n"
            f"rank {int(row['rank'])} of {len(self.table)} | equity change against the base path: "
            f"{row['equity_change_pct']:+.2f} %"
        )
        lines = [f"{record['step']:>2}. [{record['basis']:9s}] {record['kind']}: {record['text']}" for record in self.trail(ticker)]
        return header + "\n" + "\n".join(lines)
