"""Check each extracted fact against the source text.

A fact is kept only when its quote is in the text that the model read. The check ignores
case, white space, hyphens and dashes, because PDF text breaks lines and words (PDFium
writes U+FFFE for a hyphen at a line end). It does not accept a different word. A debt
amount must also be 1 of the numbers of its own quote. A fact that fails is dropped and
counted, never corrected.
"""
from __future__ import annotations

import re
import unicodedata

MIN_QUOTE_CHARS = 20
# White space, hyphen, soft hyphen, U+FFFE, and the dashes U+2010 to U+2015.
IGNORED = re.compile("[\\s\\-\u00ad\ufffe\u2010-\u2015]+")
# A number in Brazilian format: 1.234.567 or 1.234,5 or 1234 or 12,5.
NUMBER = re.compile(r"\d{1,3}(?:\.\d{3})+(?:,\d+)?|\d+(?:,\d+)?")
AMOUNT_TOLERANCE = 0.5


def normalize(text: str) -> str:
    plain = unicodedata.normalize("NFKC", text).casefold()
    return IGNORED.sub("", plain)


def quote_numbers(quote: str) -> list[float]:
    return [float(match.replace(".", "").replace(",", ".")) for match in NUMBER.findall(quote)]


def amount_in_quote(item: dict) -> bool:
    """True when the amount is null, or equal to 1 number of the quote."""
    amount = item.get("amount")
    if amount is None:
        return True
    return any(abs(amount - number) <= AMOUNT_TOLERANCE for number in quote_numbers(item["quote"]))


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
        if not amount_in_quote(item):
            return "amount not in quote"
        return None

    return _check(extraction["slices"], sections, extra)
