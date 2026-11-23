"""Reviewed corrections of extracted facts. No download and no model call."""
import json

import pytest

from src.extraction.aggregate import CORRECTIONS, apply_corrections

SOURCE = [{"location": "page 140", "text": "A exposição relativa à captação de recursos em moeda estrangeira está coberta por operações financeiras de swap."}]


def correction(**overrides) -> dict:
    base = {
        "id": "test-fix", "ticker": "T1", "cd_cvm": 1, "task": "debt",
        "match": {"indexer": ["USD"]}, "expect": 1, "set": {"hedged_to_cdi": True}, "reason": "test",
        "evidence": {"doc": "dfp", "id_doc": "1", "quote": "coberta por operações financeiras de swap"},
    }
    return {**base, **overrides}


DOCUMENT = {"doc": "dfp", "id_doc": "1", "link": "https://example.invalid/1"}


def facts() -> list[dict]:
    return [
        {"cd_cvm": 1, "task": "debt", "indexer": "USD", "hedged_to_cdi": False, "quote": "Dólar US$ 4.101.588", "document": DOCUMENT},
        {"cd_cvm": 1, "task": "debt", "indexer": "CDI", "hedged_to_cdi": False, "quote": "NCE CDI 100", "document": DOCUMENT},
    ]


def test_a_correction_changes_only_the_matching_facts_and_is_recorded():
    original = facts()
    fixed, applied = apply_corrections(original, [correction()], sections_of=lambda evidence: SOURCE)
    assert fixed[0]["hedged_to_cdi"] is True and fixed[0]["corrected_by"] == "test-fix"
    assert fixed[0]["original"] == {"hedged_to_cdi": False}
    assert fixed[1]["hedged_to_cdi"] is False and "corrected_by" not in fixed[1]
    assert original[0]["hedged_to_cdi"] is False  # the input stays as it was
    assert len(applied) == 1 and applied[0]["evidence_location"] == "page 140"


def test_a_correction_that_matches_no_fact_stops_the_run():
    with pytest.raises(ValueError, match="matches 0 facts, not 1"):
        apply_corrections(facts(), [correction(cd_cvm=2)], sections_of=lambda evidence: SOURCE)


def test_a_correction_that_matches_only_part_of_its_facts_stops_the_run():
    # A rerun relabels 1 of 2 dollar lines: the correction must not apply to the other alone.
    two = correction(expect=2)
    with pytest.raises(ValueError, match="matches 1 facts, not 2"):
        apply_corrections(facts(), [two], sections_of=lambda evidence: SOURCE)


def test_a_correction_without_its_quote_in_the_source_stops_the_run():
    stale = correction(evidence={"doc": "dfp", "id_doc": "1", "quote": "a sentence that the filing does not hold"})
    with pytest.raises(ValueError, match="evidence quote is not in"):
        apply_corrections(facts(), [stale], sections_of=lambda evidence: SOURCE)


def test_the_correction_file_is_well_formed():
    entries = json.loads(CORRECTIONS.read_text())
    assert len({entry["id"] for entry in entries}) == len(entries)
    for entry in entries:
        assert {"id", "ticker", "cd_cvm", "task", "reason", "evidence"} <= set(entry)
        assert {"doc", "id_doc", "quote"} <= set(entry["evidence"])
        if "add" in entry:
            assert entry["task"] == "debt" and not {"match", "set", "expect"} & set(entry)
            assert {"indexer", "amount", "hedged_to_cdi", "lender", "instrument", "quote"} <= set(entry["add"])
        else:
            assert {"match", "expect", "set"} <= set(entry)
            assert all(isinstance(allowed, list) for allowed in entry["match"].values())


PAGE_196 = [{"location": "page 196", "text": "Consolidado Dívida Moeda Estrangeira (6.685.532) (5.570.174) (7.209.096) (8.848.017) "
             "A Companhia possui proteção contra variação cambial adversa de 100% dos financiamentos"}]


def addition(**line) -> dict:
    base_line = {
        "indexer": "OTHER_FX", "amount": 6685532.0, "hedged_to_cdi": True, "lender": None,
        "instrument": "Dívida Moeda Estrangeira", "quote": "Dívida Moeda Estrangeira (6.685.532) (5.570.174)",
    }
    return {
        "id": "test-add", "ticker": "T1", "cd_cvm": 1, "task": "debt", "add": {**base_line, **line}, "reason": "test",
        "evidence": {"doc": "dfp", "id_doc": "1", "quote": "proteção contra variação cambial adversa de 100% dos financiamentos"},
    }


def test_an_added_line_passes_the_extraction_check_and_is_recorded():
    fixed, applied = apply_corrections(facts(), [addition()], sections_of=lambda evidence: PAGE_196)
    added = fixed[-1]
    assert len(fixed) == 3 and added["corrected_by"] == "test-add" and added["model"] == "correction"
    assert added["location"] == "page 196" and added["hedged_to_cdi"] is True
    assert added["document"] == DOCUMENT  # the full document of the debt run, with its link
    assert applied[0]["field"] == "added"


def test_an_added_line_with_an_invented_quote_stops_the_run():
    with pytest.raises(ValueError, match="quote not in source text"):
        apply_corrections(facts(), [addition(quote="Dívida Moeda Estrangeira (9.999.999) inventada")], sections_of=lambda e: PAGE_196)


def test_an_added_line_whose_amount_is_not_in_its_quote_stops_the_run():
    with pytest.raises(ValueError, match="amount not in quote"):
        apply_corrections(facts(), [addition(amount=6685.532)], sections_of=lambda evidence: PAGE_196)


def test_an_added_line_that_the_extraction_already_holds_stops_the_run():
    held = facts() + [{"cd_cvm": 1, "task": "debt", "indexer": "USD", "quote": "Dívida Moeda Estrangeira (6.685.532) (5.570.174)", "document": DOCUMENT}]
    with pytest.raises(ValueError, match="already holds this line"):
        apply_corrections(held, [addition()], sections_of=lambda evidence: PAGE_196)


def test_an_added_line_from_another_document_stops_the_run():
    other = addition()
    other["evidence"] = {**other["evidence"], "id_doc": "2"}
    with pytest.raises(ValueError, match="is not the document of the company's debt lines"):
        apply_corrections(facts(), [other], sections_of=lambda evidence: PAGE_196)
