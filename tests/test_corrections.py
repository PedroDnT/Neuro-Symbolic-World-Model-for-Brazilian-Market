"""Reviewed corrections of extracted facts. No download and no model call."""
import json

import pytest

from src.extraction.aggregate import CORRECTIONS, apply_corrections

SOURCE = [{"location": "page 140", "text": "A exposição relativa à captação de recursos em moeda estrangeira está coberta por operações financeiras de swap."}]


def correction(**overrides) -> dict:
    base = {
        "id": "test-fix", "ticker": "T1", "cd_cvm": 1, "task": "debt",
        "match": {"indexer": ["USD"]}, "set": {"hedged_to_cdi": True}, "reason": "test",
        "evidence": {"doc": "dfp", "id_doc": "1", "quote": "coberta por operações financeiras de swap"},
    }
    return {**base, **overrides}


def facts() -> list[dict]:
    return [
        {"cd_cvm": 1, "task": "debt", "indexer": "USD", "hedged_to_cdi": False, "quote": "Dólar US$ 4.101.588"},
        {"cd_cvm": 1, "task": "debt", "indexer": "CDI", "hedged_to_cdi": False, "quote": "NCE CDI 100"},
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
    with pytest.raises(ValueError, match="matches no fact"):
        apply_corrections(facts(), [correction(cd_cvm=2)], sections_of=lambda evidence: SOURCE)


def test_a_correction_without_its_quote_in_the_source_stops_the_run():
    stale = correction(evidence={"doc": "dfp", "id_doc": "1", "quote": "a sentence that the filing does not hold"})
    with pytest.raises(ValueError, match="evidence quote is not in"):
        apply_corrections(facts(), [stale], sections_of=lambda evidence: SOURCE)


def test_the_correction_file_is_well_formed():
    entries = json.loads(CORRECTIONS.read_text())
    assert len({entry["id"] for entry in entries}) == len(entries)
    for entry in entries:
        assert {"id", "ticker", "cd_cvm", "task", "match", "set", "reason", "evidence"} <= set(entry)
        assert {"doc", "id_doc", "quote"} <= set(entry["evidence"])
        assert all(isinstance(allowed, list) for allowed in entry["match"].values())
