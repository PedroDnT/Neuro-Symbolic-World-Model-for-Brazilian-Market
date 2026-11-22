"""Check each extracted fact against the source text.

A fact is kept only when its quote is in the text that the model read. The check ignores
case, white space and hyphens, because PDF text breaks lines and words. It does not
accept a different word. A fact that fails is dropped and counted, never corrected.
"""
from __future__ import annotations

import re
import unicodedata

MIN_QUOTE_CHARS = 20


def normalize(text: str) -> str:
    plain = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"[\s\-­]+", "", plain)


def find_location(quote: str, sections: list[dict]) -> str | None:
    """The location of the first section that holds the quote. None when no section holds it."""
    needle = normalize(quote)
    if len(needle) < MIN_QUOTE_CHARS:
        return None
    for section in sections:
        if needle in normalize(section["text"]):
            return section["location"]
    return None


def _check(items: list[dict], sections: list[dict], extra) -> tuple[list[dict], list[dict]]:
    kept, dropped = [], []
    for item in items:
        location = find_location(item["quote"], sections)
        reason = "quote not in source text" if location is None else extra(item)
        if reason:
            dropped.append({**item, "reason": reason})
        else:
            kept.append({**item, "location": location})
    return kept, dropped


def validate_counterparties(extraction: dict, sections: list[dict]) -> tuple[list[dict], list[dict]]:
    def extra(item: dict) -> str | None:
        name = normalize(item["counterparty"])
        if len(name) < 2:
            return "empty counterparty"
        if name not in normalize(item["quote"]):
            return "quote does not name the counterparty"
        return None

    return _check(extraction["facts"], sections, extra)


def validate_debt(extraction: dict, sections: list[dict]) -> tuple[list[dict], list[dict]]:
    def extra(item: dict) -> str | None:
        amount = item.get("amount")
        if amount is not None and amount < 0:
            return "negative amount"
        return None

    return _check(extraction["slices"], sections, extra)
