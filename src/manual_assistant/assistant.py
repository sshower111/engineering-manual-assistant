"""The RAG pipeline: retrieve, decide whether to answer, generate, log."""

import time
from dataclasses import dataclass, field
from typing import Protocol

from manual_assistant.llm import AnswerGenerator
from manual_assistant.models import RetrievedChunk
from manual_assistant.query_log import QueryLogger
from manual_assistant.schemas import AskResponse, Citation

REFUSAL = "I don't know. The indexed documentation does not cover this question."


class Retriever(Protocol):
    def search(self, query: str, top_k: int) -> list[RetrievedChunk]: ...


@dataclass
class AskResult:
    """The API response plus the details the evaluation needs."""

    response: AskResponse
    hits: list[RetrievedChunk]
    refusal_reason: str | None = None
    record: dict = field(default_factory=dict)


def snippet(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "…"


class Assistant:
    def __init__(
        self,
        retriever: Retriever,
        generator: AnswerGenerator,
        logger: QueryLogger,
        top_k: int,
        min_score: float,
        snippet_chars: int = 300,
    ) -> None:
        self._retriever = retriever
        self._generator = generator
        self._logger = logger
        self._top_k = top_k
        self._min_score = min_score
        self._snippet_chars = snippet_chars

    def ask(self, question: str) -> AskResponse:
        return self.ask_with_details(question).response

    def ask_with_details(self, question: str) -> AskResult:
        start = time.perf_counter()
        hits = self._retriever.search(question, self._top_k)
        retrieval_ms = (time.perf_counter() - start) * 1000

        record = {
            "question": question,
            "retrieved": [{"id": h.id, "score": h.score, "url": h.url} for h in hits],
            "retrieval_ms": round(retrieval_ms, 1),
        }

        if not hits or hits[0].score < self._min_score:
            response = AskResponse(answer=REFUSAL, citations=[], confidence=0.0)
            record["refusal_reason"] = "low_retrieval_score"
        else:
            generation = self._generator.generate(question, hits)
            record.update(
                model=generation.model,
                input_tokens=generation.input_tokens,
                output_tokens=generation.output_tokens,
            )
            if generation.answerable:
                response = AskResponse(
                    answer=generation.answer,
                    citations=[
                        Citation(
                            source=c.url,
                            section=f"{c.title} > {c.section}",
                            snippet=snippet(c.text, self._snippet_chars),
                        )
                        for c in generation.cited_chunks
                    ],
                    confidence=max((c.score for c in generation.cited_chunks), default=0.0),
                )
                record["cited_ids"] = [c.id for c in generation.cited_chunks]
            else:
                response = AskResponse(answer=REFUSAL, citations=[], confidence=0.0)
                record["refusal_reason"] = "model_unanswerable"

        record["answer"] = response.answer
        record["confidence"] = response.confidence
        record["latency_ms"] = round((time.perf_counter() - start) * 1000, 1)
        self._logger.log(record)
        return AskResult(response, hits, record.get("refusal_reason"), record)
