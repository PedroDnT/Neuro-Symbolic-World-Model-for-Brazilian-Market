"""LangGraph pipeline that reads supplier links and debt indexers from filing text.

Graph:  fetch -> locate -> extract -> validate -> write
        A failed validation goes back to extract 1 time, with the rejected quotes.

Tasks:
    suppliers  FRE 2021, section 7  -> named suppliers and customers
    debt       DFP 2024, notes      -> debt by interest-rate indexer, with lenders

Outputs:
    data/extracted_facts.jsonl  1 line for each kept fact, with its quote and document
    data/extraction_runs.jsonl  1 line for each company and task: status, counts, cost

Run the pilot:  python -m src.extraction.pipeline --tickers BRKM5 UGPA3 USIM5 WEGE3 GOLL4
Run all:        python -m src.extraction.pipeline --all
No cost:        add --dry-run to stop before the model call
"""
from __future__ import annotations

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

from src.data_ingestion.download import DATA_DIR
from src.data_ingestion.fetch_fre import load_universe
from src.extraction import llm, sources, validate
from src.extraction.schemas import (
    COUNTERPARTY_INSTRUCTIONS,
    DEBT_INSTRUCTIONS,
    CounterpartyExtraction,
    DebtExtraction,
)

TASKS = {
    "suppliers": {
        "doc": "fre",
        "year": 2021,
        "schema": CounterpartyExtraction,
        "instructions": COUNTERPARTY_INSTRUCTIONS,
        "items": "facts",
        "kind": "role",
        "check": validate.validate_counterparties,
    },
    "debt": {
        "doc": "dfp",
        "year": 2024,
        "schema": DebtExtraction,
        "instructions": DEBT_INSTRUCTIONS,
        "items": "slices",
        "kind": "indexer",
        "check": validate.validate_debt,
    },
}
MAX_ATTEMPTS = 2
FACTS = DATA_DIR / "extracted_facts.jsonl"
RUNS = DATA_DIR / "extraction_runs.jsonl"
FINAL_STATUS = {"ok", "no_document", "no_sections", "refused"}
WRITE_LOCK = threading.Lock()


class State(TypedDict, total=False):
    cd_cvm: int
    ticker: str
    task: str
    dry_run: bool
    document: dict
    sections: list[dict]
    extraction: dict
    facts: list[dict]
    dropped: list[dict]
    attempts: int
    cost_usd: float
    status: str


def fetch_node(state: State) -> dict:
    task = TASKS[state["task"]]
    document = sources.document_link(task["doc"], task["year"], state["cd_cvm"])
    if document is None:
        return {"status": "no_document"}
    document["path"] = str(sources.fetch(document))
    return {"document": document}


def locate_node(state: State) -> dict:
    path = Path(state["document"]["path"])
    if state["task"] == "suppliers":
        found = sources.locate_supplier_sections(sources.fre_sections(path))
    else:
        found = sources.locate_debt_pages(sources.dfp_pages(path))
    if not found:
        return {"status": "no_sections", "sections": []}
    return {"sections": [asdict(section) for section in found]}


def extract_node(state: State) -> dict:
    if state.get("dry_run"):
        return {"status": "dry_run"}
    task = TASKS[state["task"]]
    text = "\n\n".join(f"[{section['location']}]\n{section['text']}" for section in state["sections"])
    rejected = [item["quote"] for item in state.get("dropped", [])]
    if rejected:
        text += (
            "\n\nThese quotes from your last answer are not in the text. "
            "Copy each quote exactly from the text, or remove the item:\n"
            + "\n".join(f"- {quote}" for quote in rejected)
        )
    parsed, cost = llm.extract(task["schema"], task["instructions"], text)
    update = {"attempts": state.get("attempts", 0) + 1, "cost_usd": state.get("cost_usd", 0.0) + cost}
    if parsed is None:
        return {**update, "status": "refused"}
    return {**update, "extraction": parsed.model_dump()}


def validate_node(state: State) -> dict:
    task = TASKS[state["task"]]
    kept, dropped = task["check"](state["extraction"], state["sections"])

    def key(fact: dict) -> tuple:
        return (validate.normalize(fact["quote"]), fact[task["kind"]])

    facts = list(state.get("facts", []))
    known = {key(fact) for fact in facts}
    facts += [fact for fact in kept if key(fact) not in known]
    return {"facts": facts, "dropped": dropped}


def write_node(state: State) -> dict:
    status = state.get("status") or "ok"
    if status == "dry_run":
        return {"status": status}
    task = TASKS[state["task"]]
    identity = {"cd_cvm": state["cd_cvm"], "ticker": state["ticker"], "task": state["task"]}
    document = {name: value for name, value in state.get("document", {}).items() if name != "path"}
    facts = state.get("facts", [])
    sections = state.get("sections", [])
    extraction = state.get("extraction", {})
    run = {
        **identity,
        "status": status,
        "kept": len(facts),
        "dropped": len(state.get("dropped", [])),
        "attempts": state.get("attempts", 0),
        "cost_usd": round(state.get("cost_usd", 0.0), 4),
        "sections": len(sections),
        "chars": sum(len(section["text"]) for section in sections),
        "summary": {name: value for name, value in extraction.items() if name != task["items"]},
        "document": document,
    }
    with WRITE_LOCK:  # the facts of a run and its log line are written together
        with open(FACTS, "a") as handle:
            for fact in facts:
                record = {**identity, **fact, "document": document, "model": llm.MODEL}
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        with open(RUNS, "a") as handle:
            handle.write(json.dumps(run, ensure_ascii=False) + "\n")
    return {"status": status}


def after_fetch(state: State) -> Literal["locate", "write"]:
    return "locate" if "document" in state else "write"


def after_locate(state: State) -> Literal["extract", "write"]:
    return "extract" if state.get("sections") else "write"


def after_extract(state: State) -> Literal["validate", "write"]:
    return "validate" if "extraction" in state and "status" not in state else "write"


def after_validate(state: State) -> Literal["extract", "write"]:
    return "extract" if state["dropped"] and state["attempts"] < MAX_ATTEMPTS else "write"


def build_graph():
    graph = StateGraph(State)
    graph.add_node("fetch", fetch_node)
    graph.add_node("locate", locate_node)
    graph.add_node("extract", extract_node, retry_policy=RetryPolicy(max_attempts=2))
    graph.add_node("validate", validate_node)
    graph.add_node("write", write_node)
    graph.add_edge(START, "fetch")
    graph.add_conditional_edges("fetch", after_fetch)
    graph.add_conditional_edges("locate", after_locate)
    graph.add_conditional_edges("extract", after_extract)
    graph.add_conditional_edges("validate", after_validate)
    graph.add_edge("write", END)
    return graph.compile()


def finished_runs() -> set[tuple[int, str]]:
    if not RUNS.exists():
        return set()
    runs = [json.loads(line) for line in RUNS.read_text().splitlines() if line.strip()]
    return {(run["cd_cvm"], run["task"]) for run in runs if run["status"] in FINAL_STATUS}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--tickers", nargs="*", default=[])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--tasks", nargs="*", default=list(TASKS), choices=list(TASKS))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)

    universe = load_universe()
    companies = universe if args.all else universe[universe["ticker"].isin(args.tickers)]
    graph = build_graph()
    finished = finished_runs()
    for task in args.tasks:  # read each yearly index 1 time, before the threads start
        sources._index(TASKS[task]["doc"], TASKS[task]["year"])
    over_budget = threading.Event()

    def run(company, task: str) -> str:
        if over_budget.is_set():
            return f"{company.ticker} {task}: skipped, the spending cap is reached"
        start = {"cd_cvm": int(company.cd_cvm), "ticker": company.ticker, "task": task, "dry_run": args.dry_run}
        try:
            final = graph.invoke(start)
        except llm.BudgetExceeded as exc:
            over_budget.set()
            return f"{company.ticker} {task}: STOP, {exc}"
        except Exception as exc:
            # The failure is reported and the other companies go on. The pair is not
            # marked as finished, so the next run tries it again.
            return f"{company.ticker} {task}: FAILED, {type(exc).__name__}: {str(exc)[:200]}"
        sections = final.get("sections", [])
        return (
            f"{company.ticker} {task}: {final.get('status')} | sections {len(sections)}, "
            f"chars {sum(len(s['text']) for s in sections)} | kept {len(final.get('facts', []))}, "
            f"dropped {len(final.get('dropped', []))}, attempts {final.get('attempts', 0)}, "
            f"cost US$ {final.get('cost_usd', 0.0):.3f}"
        )

    jobs = [
        (company, task)
        for company in companies.itertuples()
        for task in args.tasks
        if args.dry_run or (company.cd_cvm, task) not in finished
    ]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for line in pool.map(lambda job: run(*job), jobs):
            print(line, flush=True)
    print("cost ledger:", llm.read_ledger())


if __name__ == "__main__":
    main()
