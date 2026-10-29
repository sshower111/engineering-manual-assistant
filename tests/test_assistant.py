import json

from fakes import FakeGenerator, FakeRetriever, chunk

from manual_assistant.assistant import REFUSAL, Assistant
from manual_assistant.query_log import QueryLogger


def make_assistant(hits, generator, tmp_path, min_score=0.4):
    logger = QueryLogger(tmp_path / "queries.jsonl")
    return Assistant(FakeRetriever(hits), generator, logger, top_k=5, min_score=min_score)


def read_log(tmp_path):
    return [json.loads(line) for line in (tmp_path / "queries.jsonl").read_text().splitlines()]


def test_answers_with_citations_and_confidence(tmp_path):
    generator = FakeGenerator()
    assistant = make_assistant([chunk("a", 0.71), chunk("b", 0.55)], generator, tmp_path)

    response = assistant.ask("How do I fix a face?")

    assert response.answer == "Use the Fixed Boundary Condition."
    assert [c.source for c in response.citations] == ["u/a"]
    assert response.citations[0].section == "FEM ConstraintFixed > Usage"
    assert response.confidence == 0.71


def test_refuses_without_calling_llm_when_scores_are_low(tmp_path):
    generator = FakeGenerator()
    assistant = make_assistant([chunk("a", 0.30)], generator, tmp_path, min_score=0.4)

    response = assistant.ask("What does a commercial license cost?")

    assert response.answer == REFUSAL
    assert response.citations == []
    assert response.confidence == 0.0
    assert generator.calls == []
    assert read_log(tmp_path)[0]["refusal_reason"] == "low_retrieval_score"


def test_refuses_when_nothing_is_retrieved(tmp_path):
    response = make_assistant([], FakeGenerator(), tmp_path).ask("anything?")
    assert response.answer == REFUSAL


def test_refuses_when_model_says_context_is_insufficient(tmp_path):
    assistant = make_assistant([chunk("a", 0.8)], FakeGenerator(answerable=False), tmp_path)

    response = assistant.ask("Unrelated question?")

    assert response.answer == REFUSAL
    assert response.citations == []
    assert read_log(tmp_path)[0]["refusal_reason"] == "model_unanswerable"


def test_every_query_is_logged_as_json(tmp_path):
    assistant = make_assistant([chunk("a", 0.71), chunk("b", 0.55)], FakeGenerator(), tmp_path)
    assistant.ask("How do I fix a face?")
    assistant.ask("Second question?")

    records = read_log(tmp_path)
    assert len(records) == 2
    first = records[0]
    assert first["question"] == "How do I fix a face?"
    assert first["retrieved"] == [
        {"id": "a", "score": 0.71, "url": "u/a"},
        {"id": "b", "score": 0.55, "url": "u/b"},
    ]
    assert first["cited_ids"] == ["a"]
    assert first["answer"] == "Use the Fixed Boundary Condition."
    assert first["latency_ms"] >= first["retrieval_ms"] >= 0
    assert first["model"] == "fake-model"
    assert "timestamp" in first


def test_long_snippets_are_truncated_on_a_word_boundary(tmp_path):
    long_text = "word " * 200
    assistant = make_assistant([chunk("a", 0.9, long_text)], FakeGenerator(), tmp_path)
    snippet = assistant.ask("question?").citations[0].snippet
    assert len(snippet) <= 301
    assert snippet.endswith("word…")
