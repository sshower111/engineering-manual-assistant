import pytest
from fakes import FakeGenerator, FakeRetriever, chunk
from fastapi.testclient import TestClient

from manual_assistant.api import create_app, require_index
from manual_assistant.assistant import Assistant
from manual_assistant.query_log import QueryLogger
from manual_assistant.store import VectorStore


def client(hits, tmp_path):
    assistant = Assistant(
        FakeRetriever(hits), FakeGenerator(), QueryLogger(tmp_path / "q.jsonl"), 5, 0.4
    )
    return TestClient(create_app(assistant))


def test_ask_returns_validated_json(tmp_path):
    with client([chunk("a", 0.7)], tmp_path) as c:
        response = c.post("/ask", json={"question": "How do I fix a face?"})

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"answer", "citations", "confidence"}
    assert body["citations"][0] == {
        "source": "u/a",
        "section": "FEM ConstraintFixed > Usage",
        "snippet": "Some documentation text.",
    }


def test_ask_rejects_invalid_question(tmp_path):
    with client([], tmp_path) as c:
        assert c.post("/ask", json={"question": ""}).status_code == 422
        assert c.post("/ask", json={}).status_code == 422


def test_index_page_and_health(tmp_path):
    with client([], tmp_path) as c:
        assert "Engineering Manual Assistant" in c.get("/").text
        assert c.get("/health").json() == {"status": "ok"}


def test_empty_index_fails_at_startup(fake_embedder):
    store = VectorStore(None, "test_empty", fake_embedder)
    with pytest.raises(RuntimeError, match="manual_assistant.ingest"):
        require_index(store, "test_empty")
