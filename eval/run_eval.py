"""Run the evaluation set against one or more chunking configurations.

Examples:
    # Build the three indexes, then evaluate retrieval only (no API key needed)
    python eval/run_eval.py --build --retrieval-only

    # Full evaluation with answers from Claude and an LLM judge
    python eval/run_eval.py --judge

Each configuration is "size:overlap" in characters and uses its own Chroma
collection (manuals_c<size>). Results go to eval/results/.
"""

import argparse
import logging
from pathlib import Path

from manual_assistant.assistant import Assistant
from manual_assistant.config import get_settings
from manual_assistant.embeddings import SentenceTransformerEmbedder
from manual_assistant.evaluation import (
    QuestionResult,
    load_questions,
    results_to_json,
    score_question,
    summarize,
)
from manual_assistant.ingest import build_index
from manual_assistant.llm import ClaudeAnswerGenerator, ClaudeJudge
from manual_assistant.query_log import QueryLogger
from manual_assistant.store import VectorStore

EVAL_DIR = Path(__file__).parent
DEFAULT_CONFIGS = ["400:80", "800:150", "1200:200"]


def evaluate(args, size: int, overlap: int, embedder) -> tuple[dict, list[QuestionResult]]:
    settings = get_settings()
    collection = f"manuals_c{size}"
    if args.build:
        build_index(settings, size, overlap, collection)
    store = VectorStore(settings.chroma_dir, collection, embedder)
    questions = load_questions(EVAL_DIR / "questions.jsonl")
    results = []

    if args.retrieval_only:
        # Refusal is decided by the retrieval threshold alone.
        for q in questions:
            hits = store.search(q.question, settings.top_k)
            refused = not hits or hits[0].score < args.min_score
            results.append(score_question(q, hits, "", refused))
        return summarize(results, with_answers=False), results

    generator = ClaudeAnswerGenerator(
        settings.llm_model, settings.llm_max_tokens, settings.llm_effort
    )
    logger = QueryLogger(EVAL_DIR / "results" / f"queries_c{size}.jsonl")
    assistant = Assistant(store, generator, logger, settings.top_k, args.min_score)
    judge = ClaudeJudge(settings.judge_model) if args.judge else None

    for q in questions:
        detail = assistant.ask_with_details(q.question)
        refused = detail.refusal_reason is not None
        result = score_question(q, detail.hits, detail.response.answer, refused)
        result.refusal_reason = detail.refusal_reason
        result.latency_ms = detail.record["latency_ms"]
        result.input_tokens = detail.record.get("input_tokens", 0)
        result.output_tokens = detail.record.get("output_tokens", 0)
        if judge and q.answerable and not refused:
            result.judge_correct = judge.judge(q.question, q.key_facts, result.answer).correct
        results.append(result)
        status = "refused" if refused else f"{result.facts_found}/{result.facts_total} facts"
        print(f"  {q.id} rank={result.hit_rank} top={result.top_score:.2f} {status}")
    return summarize(results), results


COLUMNS = [
    ("hit_rate", "Hit rate@5"),
    ("hit_rate_at_1", "Hit@1"),
    ("mrr", "MRR"),
    ("answer_correct_rate", "Answer correct"),
    ("key_fact_recall", "Key-fact recall"),
    ("judge_correct_rate", "Judge correct"),
    ("correct_refusal_rate", "Correct refusals"),
    ("false_refusal_rate", "False refusals"),
    ("mean_latency_ms", "Latency (ms)"),
]


def markdown_table(rows: list[tuple[str, dict]]) -> str:
    columns = [(key, title) for key, title in COLUMNS if any(key in s for _, s in rows)]
    lines = [
        "| Chunks | " + " | ".join(title for _, title in columns) + " |",
        "|---" * (len(columns) + 1) + "|",
    ]
    for name, summary in rows:
        cells = ["—" if summary.get(key) is None else str(summary[key]) for key, _ in columns]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS, help="size:overlap")
    parser.add_argument("--build", action="store_true", help="(re)build the indexes first")
    parser.add_argument("--retrieval-only", action="store_true", help="skip the LLM")
    parser.add_argument("--judge", action="store_true", help="also grade answers with an LLM")
    parser.add_argument("--min-score", type=float, default=settings.min_score)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    out_dir = EVAL_DIR / "results"
    out_dir.mkdir(exist_ok=True)
    embedder = SentenceTransformerEmbedder(settings.embedding_model)
    mode = "retrieval" if args.retrieval_only else "full"

    rows = []
    for config in args.configs:
        size, overlap = (int(v) for v in config.split(":"))
        name = f"{size} / {overlap}"
        print(f"\n== chunk size {size}, overlap {overlap}")
        summary, results = evaluate(args, size, overlap, embedder)
        summary.update(chunk_size=size, chunk_overlap=overlap, min_score=args.min_score)
        (out_dir / f"{mode}_c{size}.json").write_text(results_to_json(summary, results))
        rows.append((name, summary))

    table = markdown_table(rows)
    (out_dir / f"{mode}_summary.md").write_text(table + "\n")
    print("\n" + table)


if __name__ == "__main__":
    main()
