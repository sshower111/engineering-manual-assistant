"""Clean, chunk and index the downloaded documents.

Usage: python -m manual_assistant.ingest [--chunk-size 800] [--chunk-overlap 150]
"""

import argparse
import json
import logging
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from manual_assistant.chunking import chunk_section
from manual_assistant.cleaning import pdf_pages_to_sections, read_pdf, wiki_html_to_sections
from manual_assistant.config import get_settings
from manual_assistant.embeddings import SentenceTransformerEmbedder
from manual_assistant.models import Chunk, Section
from manual_assistant.sources import freecad_page_url
from manual_assistant.store import VectorStore

CALCULIX_TITLE = "CalculiX CrunchiX User's Manual"
log = logging.getLogger(__name__)


def load_freecad_sections(raw_dir: Path) -> list[Section]:
    sections = []
    for path in sorted(raw_dir.glob("*.json")):
        page = json.loads(path.read_text(encoding="utf-8"))
        title = page["title"]
        url = freecad_page_url(title.replace(" ", "_"))
        sections.extend(wiki_html_to_sections(page["text"], title, url))
    return sections


def load_calculix_sections(raw_dir: Path) -> list[Section]:
    pdf = raw_dir / "ccx.pdf"
    if not pdf.exists():
        return []
    url = json.loads((raw_dir / "source.json").read_text(encoding="utf-8"))["url"]
    pages, outline = read_pdf(str(pdf))
    return pdf_pages_to_sections(pages, outline, CALCULIX_TITLE, url)


def load_sections(data_dir: Path) -> list[Section]:
    raw = data_dir / "raw"
    return load_freecad_sections(raw / "freecad") + load_calculix_sections(raw / "calculix")


# Chunks shorter than this are usually just a heading with no content.
MIN_CHUNK_CHARS = 60


def build_chunks(sections: list[Section], size: int, overlap: int) -> list[Chunk]:
    chunks: dict[str, Chunk] = {}
    for section in sections:
        for chunk in chunk_section(section, size, overlap):
            if len(chunk.text) >= MIN_CHUNK_CHARS:
                chunks[chunk.id] = chunk  # dedupe identical chunks
    return list(chunks.values())


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-size", type=int, default=settings.chunk_size)
    parser.add_argument("--chunk-overlap", type=int, default=settings.chunk_overlap)
    parser.add_argument("--collection", default=settings.collection_name)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    sections = load_sections(settings.data_dir)
    processed = settings.data_dir / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    with (processed / "sections.jsonl").open("w", encoding="utf-8") as f:
        for section in sections:
            f.write(json.dumps(asdict(section)) + "\n")

    chunks = build_chunks(sections, args.chunk_size, args.chunk_overlap)
    log.info("sections per source: %s", dict(Counter(s.source for s in sections)))
    log.info("chunks: %d (size=%d, overlap=%d)", len(chunks), args.chunk_size, args.chunk_overlap)

    store = VectorStore(
        settings.chroma_dir, args.collection, SentenceTransformerEmbedder(settings.embedding_model)
    )
    store.reset()
    store.add(chunks)
    log.info("indexed %d chunks into collection %r", store.count(), args.collection)


if __name__ == "__main__":
    main()
