"""Print the top retrieved chunks for a query (retrieval only, no LLM).

Usage: python -m manual_assistant.search "How do I fix a face in FEM?"
"""

import argparse

from manual_assistant.config import get_settings
from manual_assistant.embeddings import SentenceTransformerEmbedder
from manual_assistant.store import VectorStore


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="+")
    parser.add_argument("--top-k", type=int, default=settings.top_k)
    parser.add_argument("--collection", default=settings.collection_name)
    args = parser.parse_args()

    store = VectorStore(
        settings.chroma_dir, args.collection, SentenceTransformerEmbedder(settings.embedding_model)
    )
    for query in args.query:
        print(f"\n=== {query}")
        for hit in store.search(query, args.top_k):
            snippet = hit.text[:160].replace("\n", " ")
            print(
                f"{hit.score:.3f}  {hit.title} > {hit.section}\n       {hit.url}\n       {snippet}…"
            )


if __name__ == "__main__":
    main()
