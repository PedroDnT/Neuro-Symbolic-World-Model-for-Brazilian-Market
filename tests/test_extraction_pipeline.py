"""Offline tests of the extraction control flow. No network and no model call."""
from types import SimpleNamespace

import pytest

from src.extraction import llm, pipeline


def test_only_the_last_attempt_counts(monkeypatch):
    first = {"cd_cvm": 1, "facts": [{"quote": "old answer line", "indexer": "CDI", "amount": 1.0}]}
    kept = [{"quote": "new answer line", "indexer": "CDI", "amount": 1.0}]
    monkeypatch.setitem(pipeline.TASKS["debt"], "check", lambda extraction, sections: (kept, []))
    state = {**first, "task": "debt", "extraction": {"slices": []}, "sections": []}
    assert pipeline.validate_node(state)["facts"] == kept


class FakeResponses:
    def __init__(self, response=None, error=None):
        self.response, self.error = response, error

    def parse(self, **kwargs):
        if self.error:
            raise self.error
        return self.response


def fake_openai(monkeypatch, responses):
    import openai

    monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: SimpleNamespace(responses=responses))


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "LEDGER", tmp_path / "ledger.json")
    return tmp_path / "ledger.json"


def test_a_failed_call_is_booked_at_its_worst_case(monkeypatch, ledger):
    fake_openai(monkeypatch, FakeResponses(error=ValueError("cut JSON")))
    with pytest.raises(ValueError):
        llm.extract(pipeline.TASKS["debt"]["schema"], "instructions", "text")
    booked = llm.read_ledger()
    assert booked["calls"] == 1 and booked["failed_calls"] == 1 and booked["spent_usd"] > 0


def test_an_incomplete_reply_raises_and_is_not_a_refusal(monkeypatch, ledger):
    usage = SimpleNamespace(input_tokens=1000, output_tokens=16000)
    response = SimpleNamespace(status="incomplete", incomplete_details="max_output_tokens", usage=usage, output_parsed=None)
    fake_openai(monkeypatch, FakeResponses(response=response))
    with pytest.raises(llm.IncompleteReply):
        llm.extract(pipeline.TASKS["debt"]["schema"], "instructions", "text")
    assert llm.read_ledger()["output_tokens"] == 16000


def status_error(status: int):
    import httpx
    import openai

    response = httpx.Response(status, request=httpx.Request("POST", "https://example.invalid/v1/responses"))
    return openai.APIStatusError("error", response=response, body=None)


def test_a_client_error_costs_nothing_and_is_not_retried(monkeypatch, ledger):
    error = status_error(401)
    fake_openai(monkeypatch, FakeResponses(error=error))
    with pytest.raises(Exception):
        llm.extract(pipeline.TASKS["debt"]["schema"], "instructions", "text")
    assert llm.read_ledger()["spent_usd"] == 0
    assert llm.worth_retry(error) is False
    assert llm.worth_retry(status_error(500)) is True


def test_redo_replaces_the_old_result_only_with_a_final_one(monkeypatch, tmp_path):
    import json

    monkeypatch.setattr(pipeline, "FACTS", tmp_path / "facts.jsonl")
    monkeypatch.setattr(pipeline, "RUNS", tmp_path / "runs.jsonl")
    old = {"cd_cvm": 1, "ticker": "X", "task": "debt", "status": "ok"}
    pipeline.RUNS.write_text(json.dumps(old) + "\n")
    pipeline.FACTS.write_text(json.dumps({"cd_cvm": 1, "task": "debt", "quote": "old"}) + "\n")
    state = {"cd_cvm": 1, "ticker": "X", "task": "debt", "redo": True}
    pipeline.write_node({**state, "status": "truncated"})
    assert "old" in pipeline.FACTS.read_text() and len(pipeline.RUNS.read_text().splitlines()) == 1
    pipeline.write_node({**state, "facts": [{"quote": "new"}]})
    facts = [json.loads(line)["quote"] for line in pipeline.FACTS.read_text().splitlines()]
    runs = [json.loads(line)["status"] for line in pipeline.RUNS.read_text().splitlines()]
    assert facts == ["new"] and runs == ["ok"]
