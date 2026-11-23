"""Build the Knowledge Graph with Semantica (`semantica.context.ContextGraph`).

Nodes:  company, sector, macro, lender, external_party
Edges:  IN_SECTOR      company -> sector         CVM register
        OWNS           owner -> company          FRE 2021 tables, weight = share in percent
        RELATED_PARTY  company -> company        FRE 2021 tables
        SUPPLIES       supplier -> customer      extraction, with its quote and document
        HAS_LENDER     company -> lender         extraction, with its quote and document
        EXPOSED_TO     company -> macro:SELIC    weight = floating share of debt, with its basis

Each edge has a `basis` property: "filed", "extracted" or "estimate".

Output: results/knowledge_graph.json. View it with: semantica-explorer --graph results/knowledge_graph.json
Run:    python -m src.knowledge.graph
"""
from __future__ import annotations

import json

import pandas as pd
from semantica.context import ContextGraph

from src.data_ingestion.download import DATA_DIR, RESULTS_DIR
from src.data_ingestion.fetch_fre import load_universe
from src.extraction.aggregate import load_facts, party_key, universe_keys
from src.knowledge.sensitivity import floating_shares

MACROS = {"SELIC": "SELIC target rate", "IPCA": "IPCA inflation", "PTAX": "USD/BRL exchange rate"}
GRAPH_PATH = RESULTS_DIR / "knowledge_graph.json"


def _provenance(fact: dict) -> dict:
    document = fact["document"]
    return {
        "basis": "corrected" if fact.get("corrected_by") else "extracted",
        "correction": fact.get("corrected_by", ""),
        "quote": fact["quote"],
        "location": fact["location"],
        "document_id": document["id_doc"],
        "document_link": document["link"],
        "model": fact["model"],
    }


def build() -> ContextGraph:
    universe = load_universe()
    meta = json.loads((DATA_DIR / "meta.json").read_text())
    node_of = {int(code): f"company:{ticker}" for code, ticker in zip(universe["cd_cvm"], universe["ticker"])}
    keys = universe_keys(universe)
    graph = ContextGraph()

    for name, label in MACROS.items():
        graph.add_node(f"macro:{name}", "macro", content=label)
    for sector in sorted(universe["sector"].unique()):
        graph.add_node(f"sector:{sector}", "sector", content=sector)
    for company in universe.itertuples():
        graph.add_node(
            node_of[company.cd_cvm], "company", content=company.company, ticker=company.ticker, cd_cvm=int(company.cd_cvm)
        )
        graph.add_edge(node_of[company.cd_cvm], f"sector:{company.sector}", "IN_SECTOR", basis="filed")

    links = pd.read_csv(DATA_DIR / "fre_edges.csv")
    kinds = {"ownership": "OWNS", "related_party": "RELATED_PARTY"}
    for link in links.itertuples():
        graph.add_edge(
            node_of[link.src_cd_cvm],
            node_of[link.dst_cd_cvm],
            kinds[link.relation],
            weight=float(link.weight),
            basis="filed",
            source=f"FRE {link.year}",
            evidence=str(link.evidence),
        )

    def party_node(name: str, kind: str) -> str:
        """A universe company keeps its company node. Another party gets its own node."""
        code = keys.get(party_key(name))
        if code is not None:
            return node_of[code]
        node_id = f"{kind}:{party_key(name) or name.strip().lower()}"
        graph.add_node(node_id, kind, content=name)
        return node_id

    for fact in load_facts():
        company = node_of.get(fact["cd_cvm"])
        if company is None:
            continue
        if fact["task"] == "suppliers":
            other = party_node(fact["counterparty"], "external_party")
            if other == company:
                continue
            source, target = (other, company) if fact["role"] == "supplier" else (company, other)
            graph.add_edge(
                source, target, "SUPPLIES", dependence=fact["dependence"], what=fact["description"], **_provenance(fact)
            )
        elif fact.get("lender"):
            lender = party_node(fact["lender"], "lender")
            if lender != company:
                graph.add_edge(
                    company,
                    lender,
                    "HAS_LENDER",
                    indexer=fact["indexer"],
                    amount=fact.get("amount"),
                    **_provenance(fact),
                )

    for share in floating_shares(universe, meta["train_end"]).dropna(subset=["floating_share"]).itertuples():
        graph.add_edge(
            node_of[share.cd_cvm], "macro:SELIC", "EXPOSED_TO", weight=float(share.floating_share), basis=share.basis
        )
    return graph


def main() -> ContextGraph:
    graph = build()
    RESULTS_DIR.mkdir(exist_ok=True)
    graph.save_to_file(GRAPH_PATH)
    saved = json.loads(GRAPH_PATH.read_text())
    nodes = pd.Series([node["type"] for node in saved["nodes"]]).value_counts().to_dict()
    edges = pd.Series([edge["type"] for edge in saved["edges"]]).value_counts().to_dict()
    print(f"knowledge_graph.json: nodes {nodes} | edges {edges}")
    return graph


if __name__ == "__main__":
    main()
