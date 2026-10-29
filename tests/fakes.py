"""Test doubles for the retriever and the LLM."""

from types import SimpleNamespace

from pydantic import BaseModel

from manual_assistant.llm import Generation
from manual_assistant.models import RetrievedChunk


def chunk(id: str, score: float, text: str = "Some documentation text.") -> RetrievedChunk:
    return RetrievedChunk(
        id=id, text=text, score=score, title="FEM ConstraintFixed", section="Usage", url=f"u/{id}"
    )


class FakeRetriever:
    def __init__(self, hits: list[RetrievedChunk]) -> None:
        self.hits = hits

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        return self.hits[:top_k]


class FakeGenerator:
    """Answers with the first chunk, or refuses when ``answerable`` is False."""

    def __init__(self, answerable: bool = True) -> None:
        self.answerable = answerable
        self.calls: list[str] = []

    def generate(self, question: str, chunks: list[RetrievedChunk]) -> Generation:
        self.calls.append(question)
        if not self.answerable:
            return Generation(False, "", [], "fake-model")
        return Generation(
            True, "Use the Fixed Boundary Condition.", chunks[:1], "fake-model", 10, 5
        )


class FakeAnthropicClient:
    """Mimics ``client.beta.messages.parse`` / ``client.messages.parse`` and records requests."""

    def __init__(self, parsed: BaseModel | None, stop_reason: str = "end_turn") -> None:
        self.requests: list[dict] = []
        self._response = SimpleNamespace(
            parsed_output=parsed,
            stop_reason=stop_reason,
            model="claude-test",
            usage=SimpleNamespace(input_tokens=100, output_tokens=20),
        )
        self.messages = SimpleNamespace(parse=self._parse)
        self.beta = SimpleNamespace(messages=self.messages)

    def _parse(self, **kwargs):
        self.requests.append(kwargs)
        return self._response
