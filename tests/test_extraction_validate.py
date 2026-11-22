"""Offline tests of the quote check. No download and no model call."""
from src.extraction.validate import find_location, validate_counterparties, validate_debt

SECTIONS = [
    {
        "location": "page 1",
        "text": "A Companhia adquire nafta da Petrobras, sua principal forne-\ncedora de matéria-prima.",
    },
    {"location": "page 2", "text": "Debêntures 5ª emissão   CDI + 1,20% a.a.   1.250.000"},
]


def fact(counterparty: str, quote: str) -> dict:
    return {
        "counterparty": counterparty,
        "role": "supplier",
        "description": "nafta",
        "dependence": "high",
        "quote": quote,
    }


def test_quote_with_a_broken_word_is_found():
    assert find_location("nafta da Petrobras, sua principal fornecedora de matéria-prima", SECTIONS) == "page 1"


def test_quote_with_other_spacing_is_found():
    assert find_location("Debêntures 5ª emissão CDI + 1,20% a.a. 1.250.000", SECTIONS) == "page 2"


def test_invented_quote_is_not_found():
    assert find_location("A Companhia adquire eteno da Braskem em contrato de longo prazo", SECTIONS) is None


def test_short_quote_is_rejected():
    assert find_location("Petrobras", SECTIONS) is None


def test_fact_is_kept_with_the_real_location():
    kept, dropped = validate_counterparties(
        {"facts": [fact("Petrobras", "A Companhia adquire nafta da Petrobras")]}, SECTIONS
    )
    assert dropped == []
    assert kept[0]["location"] == "page 1"


def test_fact_with_invented_quote_is_dropped():
    kept, dropped = validate_counterparties(
        {"facts": [fact("Braskem", "A Companhia adquire eteno da Braskem em contrato")]}, SECTIONS
    )
    assert kept == []
    assert dropped[0]["reason"] == "quote not in source text"


def test_quote_must_name_the_counterparty():
    kept, dropped = validate_counterparties(
        {"facts": [fact("Vale", "A Companhia adquire nafta da Petrobras")]}, SECTIONS
    )
    assert kept == []
    assert dropped[0]["reason"] == "quote does not name the counterparty"


def test_negative_debt_amount_is_dropped():
    item = {
        "indexer": "CDI",
        "amount": -5.0,
        "lender": None,
        "instrument": "debentures",
        "quote": "Debêntures 5ª emissão CDI + 1,20% a.a. 1.250.000",
    }
    kept, dropped = validate_debt({"slices": [item]}, SECTIONS)
    assert kept == []
    assert dropped[0]["reason"] == "negative amount"


def test_debt_amount_must_be_a_number_of_the_quote():
    item = {
        "indexer": "CDI",
        "amount": 1250.0,
        "lender": None,
        "instrument": "debentures",
        "quote": "Debêntures 5ª emissão CDI + 1,20% a.a. 1.250.000",
    }
    kept, dropped = validate_debt({"slices": [item]}, SECTIONS)
    assert kept == []
    assert dropped[0]["reason"] == "amount not in quote"
    kept, _ = validate_debt({"slices": [{**item, "amount": 1250000.0}]}, SECTIONS)
    assert len(kept) == 1


def test_pdf_line_break_hyphen_and_dash_are_ignored():
    sections = [{"location": "p", "text": "em dólares norte￾americanos – Companhia Siderúrgica Nacional"}]
    assert find_location("em dólares norte-americanos - Companhia Siderúrgica Nacional", sections) == "p"
