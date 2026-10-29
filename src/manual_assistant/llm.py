"""Answer generation with Claude, constrained to the retrieved context."""

from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel, Field

from manual_assistant.models import RetrievedChunk

SYSTEM_PROMPT = """\
You answer questions about CAD and FEA software documentation (FreeCAD, CalculiX).

Rules:
- Use only the documentation excerpts in the user message. Do not use outside knowledge.
- If the excerpts do not contain the answer, set "answerable" to false and leave the \
answer empty. A partial match on keywords is not enough.
- Cite every excerpt you used by its id (for example "C2") in "cited_ids".
- Be concise: a few sentences or a short list. Keep exact names of menus, keywords and \
parameters as written in the excerpts."""


class LLMAnswer(BaseModel):
    """What the model must return (enforced with structured outputs)."""

    answerable: bool = Field(description="Whether the excerpts contain the answer")
    answer: str = Field(description="The answer, empty if not answerable")
    cited_ids: list[str] = Field(description="Ids of the excerpts used, e.g. ['C1', 'C3']")


@dataclass(frozen=True)
class Generation:
    answerable: bool
    answer: str
    cited_chunks: list[RetrievedChunk]
    model: str
    input_tokens: int = 0
    output_tokens: int = 0


class AnswerGenerator(Protocol):
    def generate(self, question: str, chunks: list[RetrievedChunk]) -> Generation: ...


def excerpt_id(index: int) -> str:
    return f"C{index + 1}"


def build_user_message(question: str, chunks: list[RetrievedChunk]) -> str:
    excerpts = "\n\n".join(
        f'<excerpt id="{excerpt_id(i)}" source="{c.title} > {c.section}">\n{c.text}\n</excerpt>'
        for i, c in enumerate(chunks)
    )
    return f"<documentation>\n{excerpts}\n</documentation>\n\nQuestion: {question}"


class ClaudeAnswerGenerator:
    def __init__(self, model: str, max_tokens: int, effort: str, client: Any = None) -> None:
        if client is None:
            import anthropic

            client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        self._client = client
        self._model = model
        self._max_tokens = max_tokens
        self._effort = effort

    def generate(self, question: str, chunks: list[RetrievedChunk]) -> Generation:
        response = self._client.beta.messages.parse(
            model=self._model,
            max_tokens=self._max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": build_user_message(question, chunks)}],
            output_format=LLMAnswer,
            output_config={"effort": self._effort},
            # If a safety classifier declines, the API retries on a fallback model.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        usage = response.usage
        parsed: LLMAnswer | None = response.parsed_output
        if response.stop_reason == "refusal" or parsed is None:
            return Generation(
                False, "", [], response.model, usage.input_tokens, usage.output_tokens
            )

        by_id = {excerpt_id(i): c for i, c in enumerate(chunks)}
        cited = [by_id[i] for i in dict.fromkeys(parsed.cited_ids) if i in by_id]
        answerable = parsed.answerable and bool(parsed.answer.strip())
        return Generation(
            answerable=answerable,
            answer=parsed.answer.strip() if answerable else "",
            cited_chunks=cited if answerable else [],
            model=response.model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
        )


JUDGE_PROMPT = """\
You grade answers from a documentation assistant. Given a question, the key facts a \
correct answer must contain, and the assistant's answer, decide whether the answer states \
every key fact (wording may differ) and contains nothing that contradicts them."""


class JudgeVerdict(BaseModel):
    correct: bool
    reason: str = Field(description="One sentence explaining the verdict")


class ClaudeJudge:
    """Optional LLM-as-judge for answer correctness (a second opinion on key-fact matching)."""

    def __init__(self, model: str, client: Any = None) -> None:
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self._client = client
        self._model = model

    def judge(self, question: str, key_facts: list[str], answer: str) -> JudgeVerdict:
        facts = "\n".join(f"- {fact.replace('|', ' or ')}" for fact in key_facts)
        response = self._client.messages.parse(
            model=self._model,
            max_tokens=2000,
            system=JUDGE_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": f"Question: {question}\n\nKey facts:\n{facts}\n\nAnswer:\n{answer}",
                }
            ],
            output_format=JudgeVerdict,
            output_config={"effort": "low"},
        )
        return response.parsed_output or JudgeVerdict(correct=False, reason="no verdict")
