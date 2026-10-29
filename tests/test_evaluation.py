import json
from pathlib import Path

from fakes import chunk

from manual_assistant.evaluation import (
    EvalQuestion,
    fact_present,
    first_hit_rank,
    load_questions,
    score_question,
    summarize,
)
from manual_assistant.models import RetrievedChunk

QUESTIONS_FILE = Path(__file__).parents[1] / "eval" / "questions.jsonl"


def hit(url: str, section: str = "Usage", score: float = 0.5) -> RetrievedChunk:
    return RetrievedChunk(id=url, text="", score=score, title="T", section=section, url=url)


def question(answerable=True, facts=("Gmsh",), sources=("FEM_Install",)) -> EvalQuestion:
    return EvalQuestion("q1", "Which mesher?", answerable, list(sources), list(facts))


def test_fact_matching_is_case_insensitive_and_supports_alternatives():
    assert fact_present("gmsh", "FreeCAD uses GMSH by default.")
    assert fact_present("10 m|10 meter", "It is a 10 meter deep pocket.")
    assert not fact_present("Netgen", "FreeCAD uses Gmsh.")


def test_fact_matching_treats_underscores_as_spaces():
    # pdf text says "OMP NUM THREADS"; a model may write OMP_NUM_THREADS.
    assert fact_present("OMP NUM THREADS", "Set OMP_NUM_THREADS to the core count.")


def test_first_hit_rank_matches_url_or_section():
    hits = [hit("https://w/Placement"), hit("https://x/ccx.pdf#page=427", "Input > *CLOAD")]
    assert first_hit_rank(["*CLOAD"], hits) == 2
    assert first_hit_rank(["Placement"], hits) == 1
    assert first_hit_rank(["PartDesign_Pad"], hits) is None


def test_score_answered_question():
    result = score_question(
        question(facts=["Gmsh", "Netgen"]), [hit("https://w/FEM_Install")], "Gmsh.", False
    )
    assert result.hit_rank == 1
    assert (result.facts_found, result.facts_total) == (1, 2)
    assert result.missing_facts == ["Netgen"]
    assert not result.facts_correct


def test_refused_answerable_question_scores_no_facts():
    result = score_question(question(), [hit("https://w/FEM_Install")], "I don't know.", True)
    assert result.facts_found == 0 and not result.facts_correct


def test_summary_metrics():
    results = [
        score_question(question(), [hit("u/other"), hit("u/FEM_Install")], "Gmsh", False),
        score_question(question(), [hit("u/other")], "", True),
        score_question(question(answerable=False, facts=(), sources=()), [hit("u/x")], "", True),
        score_question(question(answerable=False, facts=(), sources=()), [hit("u/x")], "Hi", False),
    ]
    summary = summarize(results)
    assert summary["hit_rate"] == 0.5
    assert summary["hit_rate_at_1"] == 0.0
    assert summary["mrr"] == 0.25  # (1/2 + 0) / 2
    assert summary["answer_correct_rate"] == 0.5
    assert summary["correct_refusal_rate"] == 0.5
    assert summary["false_refusal_rate"] == 0.5


def test_question_file_is_well_formed():
    questions = load_questions(QUESTIONS_FILE)
    assert len(questions) == 50
    assert len({q.id for q in questions}) == 50
    unanswerable = [q for q in questions if not q.answerable]
    assert len(unanswerable) == 5
    for q in questions:
        if q.answerable:
            assert q.expected_sources and q.key_facts, q.id
    # Lines are valid JSON objects with the documented keys.
    first = json.loads(QUESTIONS_FILE.read_text().splitlines()[0])
    assert set(first) == {
        "id",
        "question",
        "answerable",
        "corpus",
        "expected_sources",
        "key_facts",
    }


def test_chunk_fake_is_usable_with_scoring():
    assert score_question(question(), [chunk("a", 0.9)], "Gmsh", False).top_score == 0.9
