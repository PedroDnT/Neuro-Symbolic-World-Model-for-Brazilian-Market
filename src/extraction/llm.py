"""The model call of the extraction, with a spending cap.

Provider: OpenAI, Responses API, structured output into a Pydantic model
(`responses.parse` with `text_format`). The key is read from OPENAI_API_KEY.

Settings (environment variables):
    EXTRACTION_MODEL    default gpt-5.1
    EXTRACTION_EFFORT   reasoning effort, default medium
    EXTRACTION_CAP_USD  spending cap for all runs together, default 25

The cost ledger is data/extraction_cost.json. It adds the token usage of each call. Before
a call is sent, its worst-case cost is reserved. The call is refused when the money spent
plus the money reserved can pass the cap, so parallel calls cannot pass it together.
A call that fails after it was sent (a reply that cannot be parsed, a timeout) is booked
at its worst-case cost, because the provider can bill it. So the ledger can only overstate.
The SDK does not retry: 1 reservation covers exactly 1 request.

A reply that the token limit cut (status "incomplete") raises IncompleteReply. It is not a
refusal: the pipeline reports the pair as failed, and the next run tries it again.
"""
from __future__ import annotations

import json
import os
import threading

from pydantic import BaseModel

from src.data_ingestion.download import DATA_DIR

MODEL = os.environ.get("EXTRACTION_MODEL", "gpt-5.1")
EFFORT = os.environ.get("EXTRACTION_EFFORT", "medium")
CAP_USD = float(os.environ.get("EXTRACTION_CAP_USD", "25"))
# Reasoning tokens count against this limit. With 16,000 a long debt table was cut and
# the reply could not be parsed.
MAX_OUTPUT_TOKENS = 32_000
TIMEOUT_SECONDS = 300.0
# US$ for each 1 million tokens (input, output). Standard price of the model page,
# read 2026-10-04. Cached input is billed lower, so this ledger can only overstate.
PRICES = {"gpt-5.1": (1.25, 10.00)}
LEDGER = DATA_DIR / "extraction_cost.json"

_lock = threading.Lock()
_reserved = 0.0


class BudgetExceeded(RuntimeError):
    pass


class IncompleteReply(RuntimeError):
    pass


def _price(model: str) -> tuple[float, float]:
    if model not in PRICES:
        raise ValueError(f"no price for model {model}: add it to PRICES in src/extraction/llm.py")
    return PRICES[model]


def read_ledger() -> dict:
    if LEDGER.exists():
        return json.loads(LEDGER.read_text())
    return {"spent_usd": 0.0, "calls": 0, "input_tokens": 0, "output_tokens": 0}


def _reserve(worst: float) -> None:
    global _reserved
    with _lock:
        spent = read_ledger()["spent_usd"]
        if spent + _reserved + worst > CAP_USD:
            raise BudgetExceeded(
                f"spent US$ {spent:.2f}, reserved US$ {_reserved:.2f}; "
                f"the next call can cost US$ {worst:.2f}; cap US$ {CAP_USD:.2f}"
            )
        _reserved += worst


def _settle(worst: float, usage) -> float:
    """Release the reservation and book the cost. `usage` is None when the call failed:
    then the worst case is booked, because the provider can bill a failed call."""
    global _reserved
    with _lock:
        _reserved -= worst
        ledger = read_ledger()
        if usage is None:
            cost = worst
            ledger["failed_calls"] = ledger.get("failed_calls", 0) + 1
        else:
            price_in, price_out = _price(MODEL)
            cost = (usage.input_tokens * price_in + usage.output_tokens * price_out) / 1e6
            ledger["input_tokens"] += usage.input_tokens
            ledger["output_tokens"] += usage.output_tokens
        ledger["spent_usd"] += cost
        ledger["calls"] += 1
        LEDGER.write_text(json.dumps(ledger, indent=1))
        return cost


def extract(schema: type[BaseModel], instructions: str, text: str) -> tuple[BaseModel | None, float]:
    """Return the parsed result and the cost of the call. The result is None on a refusal.

    Raise IncompleteReply when the token limit cut the reply."""
    import openai

    price_in, price_out = _price(MODEL)
    # Worst case: about 2 characters for each input token, and the full output limit.
    worst = (len(instructions + text) / 2 * price_in + MAX_OUTPUT_TOKENS * price_out) / 1e6
    _reserve(worst)
    usage = None
    try:
        client = openai.OpenAI(timeout=TIMEOUT_SECONDS, max_retries=0)
        response = client.responses.parse(
            model=MODEL,
            instructions=instructions,
            input=[{"role": "user", "content": text}],
            text_format=schema,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            reasoning={"effort": EFFORT},
            store=False,
        )
        usage = response.usage
    finally:
        cost = _settle(worst, usage)
    if response.status == "incomplete":
        raise IncompleteReply(f"the reply was cut: {response.incomplete_details}")
    return response.output_parsed, cost
