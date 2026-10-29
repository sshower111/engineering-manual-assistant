"""Scoring for the evaluation set: retrieval hits, key facts, refusals."""

import json
import re
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path

from manual_assistant.models import RetrievedChunk


@dataclass(frozen=True)
class EvalQuestion:
    id: str
    question: str
    answerable: bool
    expected_sources: list[str]  # substrings of a chunk's URL or "title > section"
    key_facts: list[str]  # each fact may list alternatives separated by "|"
    corpus: str | None = None


def load_questions(path: Path) -> list[EvalQuestion]:
    with path.open(encoding="utf-8") as f:
        return [EvalQuestion(**json.loads(line)) for line in f if line.strip()]


def normalize(text: str) -> str:
    """Lowercase, treat underscores as spaces and collapse whitespace."""
    return re.sub(r"\s+", " ", text.lower().replace("_", " ")).strip()


def fact_present(fact: str, answer: str) -> bool:
    answer = normalize(answer)
    return any(normalize(alternative) in answer for alternative in fact.split("|"))


def first_hit_rank(expected_sources: list[str], hits: list[RetrievedChunk]) -> int | None:
    """1-based rank of the first retrieved chunk matching an expected source, if any."""
    expected = [normalize(e) for e in expected_sources]
    for rank, hit in enumerate(hits, start=1):
        label = normalize(f"{hit.url} | {hit.title} > {hit.section}")
        if any(e in label for e in expected):
            return rank
    return None


@dataclass
class QuestionResult:
    id: str
    answerable: bool
    refused: bool
    hit_rank: int | None = None
    top_score: float = 0.0
    facts_found: int = 0
    facts_total: int = 0
    judge_correct: bool | None = None
    answer: str = ""
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    refusal_reason: str | None = None
    missing_facts: list[str] = field(default_factory=list)

    @property
    def facts_correct(self) -> bool:
        return not self.refused and self.facts_found == self.facts_total


def score_question(
    question: EvalQuestion,
    hits: list[RetrievedChunk],
    answer: str,
    refused: bool,
) -> QuestionResult:
    missing = [] if refused else [f for f in question.key_facts if not fact_present(f, answer)]
    return QuestionResult(
        id=question.id,
        answerable=question.answerable,
        refused=refused,
        hit_rank=first_hit_rank(question.expected_sources, hits) if question.answerable else None,
        top_score=hits[0].score if hits else 0.0,
        facts_found=0 if refused else len(question.key_facts) - len(missing),
        facts_total=len(question.key_facts),
        answer=answer,
        missing_facts=missing,
    )


def _rate(values: list[bool]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


def summarize(results: list[QuestionResult], with_answers: bool = True) -> dict:
    answerable = [r for r in results if r.answerable]
    unanswerable = [r for r in results if not r.answerable]
    summary = {
        "questions": len(results),
        "answerable": len(answerable),
        "unanswerable": len(unanswerable),
        "hit_rate": _rate([r.hit_rank is not None for r in answerable]),
        "hit_rate_at_1": _rate([r.hit_rank == 1 for r in answerable]),
        "mrr": round(statistics.fmean(1 / r.hit_rank if r.hit_rank else 0 for r in answerable), 3)
        if answerable
        else None,
        "correct_refusal_rate": _rate([r.refused for r in unanswerable]),
        "false_refusal_rate": _rate([r.refused for r in answerable]),
    }
    if with_answers:
        judged = [r.judge_correct for r in answerable if r.judge_correct is not None]
        summary.update(
            answer_correct_rate=_rate([r.facts_correct for r in answerable]),
            key_fact_recall=round(
                sum(r.facts_found for r in answerable)
                / max(1, sum(r.facts_total for r in answerable)),
                3,
            ),
            judge_correct_rate=_rate(judged),
            mean_latency_ms=round(statistics.fmean(r.latency_ms for r in results), 1),
            input_tokens=sum(r.input_tokens for r in results),
            output_tokens=sum(r.output_tokens for r in results),
        )
    return summary


def results_to_json(summary: dict, results: list[QuestionResult]) -> str:
    return json.dumps({"summary": summary, "results": [asdict(r) for r in results]}, indent=2)
