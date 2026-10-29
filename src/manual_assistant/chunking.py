"""Split section text into overlapping chunks."""

import hashlib
import re

from manual_assistant.models import Chunk, Section


def split_text(text: str, size: int, overlap: int) -> list[str]:
    """Split ``text`` into chunks of at most ``size`` characters.

    Chunks break on whitespace, and consecutive chunks share roughly ``overlap``
    characters of trailing context. A single word longer than ``size`` becomes its
    own chunk rather than being cut.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must be >= 0 and smaller than size")

    words = text.split()
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = start
        length = 0
        while end < len(words) and (end == start or length + 1 + len(words[end]) <= size):
            length += len(words[end]) + (1 if end > start else 0)
            end += 1
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break

        # Step back from `end` until about `overlap` characters are repeated,
        # but always move forward by at least one word.
        next_start = end
        repeated = 0
        while next_start - 1 > start and repeated + len(words[next_start - 1]) + 1 <= overlap:
            next_start -= 1
            repeated += len(words[next_start]) + 1
        start = next_start
    return chunks


def chunk_id(url: str, index: int, text: str) -> str:
    digest = hashlib.sha1(f"{url}|{index}|{text}".encode()).hexdigest()
    return digest[:16]


def chunk_section(section: Section, size: int, overlap: int) -> list[Chunk]:
    return [
        Chunk(
            id=chunk_id(section.url, i, piece),
            text=piece,
            source=section.source,
            title=section.title,
            section=section.section,
            url=section.url,
        )
        for i, piece in enumerate(split_text(section.text, size, overlap))
    ]


def normalize_whitespace(text: str) -> str:
    """Collapse runs of spaces while keeping paragraph breaks."""
    text = re.sub(r"[ \t ]+", " ", text)
    text = re.sub(r"\s*\n\s*\n\s*", "\n\n", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    return text.strip()
