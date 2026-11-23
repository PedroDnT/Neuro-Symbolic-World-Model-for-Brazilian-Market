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


def debt_item(amount: float, quote: str) -> dict:
    return {"indexer": "USD", "amount": amount, "lender": None, "instrument": "bonds", "quote": quote}


def test_debt_amount_can_be_the_sum_of_current_and_non_current():
    sections = [{"location": "p", "text": "Bonds em dólares norte-americanos 12.345 67.890"}]
    kept, _ = validate_debt({"slices": [debt_item(80235.0, "Bonds em dólares norte-americanos 12.345 67.890")]}, sections)
    assert len(kept) == 1


def test_debt_amount_must_match_exactly():
    sections = [{"location": "p", "text": "Bonds em dólares norte-americanos 12.345 67.890"}]
    _, dropped = validate_debt({"slices": [debt_item(12345.4, "Bonds em dólares norte-americanos 12.345 67.890")]}, sections)
    assert dropped[0]["reason"] == "amount not in quote"


def test_a_rate_is_not_read_as_an_amount_and_a_final_number_is_read():
    from src.extraction.validate import quote_numbers

    assert quote_numbers("CDI + 1,20% a.a. saldo de 1.250.000.") == [1250000.0]
    assert quote_numbers("taxa de 3,5 % e saldo de 300.000, vencimento") == [300000.0]


def test_a_fixed_rate_dollar_line_counts_as_foreign_currency():
    from src.extraction.aggregate import is_fx

    assert is_fx({"indexer": "FIXED", "quote": "Bonds USD Fixo 5,0% 52.396.445"})
    assert is_fx({"indexer": "OTHER", "quote": "Pré-pagamento USD SOFR 6,3% 776.999"})
    assert not is_fx({"indexer": "FIXED", "quote": "BNDES BRL Fixo 4.020"})
    assert not is_fx({"indexer": "OTHER", "quote": "Bonds USD Fixo 5,0% 52.396.445", "hedged_to_cdi": True})
    assert not is_fx({"indexer": "CDI", "quote": "NCE BRL CDI swap de USD 103.027"})


def test_a_sofr_or_export_advance_line_counts_as_foreign_currency():
    from src.extraction.aggregate import is_fx

    assert is_fx({"indexer": "OTHER", "quote": "Pré Pgto Exportação SOFR+ 3,23% Jul/25 61.752"})
    assert is_fx({"indexer": "FIXED", "quote": "ACC FIXO 6,06% Set/29 3.176"})
    assert not is_fx({"indexer": "FIXED", "quote": "Bonds 2030 9.418"})  # names no currency
    assert is_fx({"indexer": "FIXED", "quote": "Para capital de giro ACE (desconto de recebível) 45.553"})
    assert is_fx({"indexer": "FIXED", "quote": "Finame Dólar 89.659 94.313"})
    assert not is_fx({"indexer": "OTHER", "quote": "Capital de giro Euribor + 1% 10.000"})  # a rate, not a currency
    assert not is_fx({"indexer": "OTHER", "quote": "Debêntures BRL CDI/IPCA 15,4% 9.859.547 accrued"})
