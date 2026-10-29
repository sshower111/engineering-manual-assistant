import pytest

from manual_assistant.chunking import chunk_section, normalize_whitespace, split_text
from manual_assistant.models import Section


def test_short_text_is_one_chunk():
    assert split_text("a short sentence", size=100, overlap=10) == ["a short sentence"]


def test_empty_text_has_no_chunks():
    assert split_text("   ", size=100, overlap=10) == []


def test_chunks_respect_size_and_cover_all_words():
    words = [f"word{i}" for i in range(200)]
    chunks = split_text(" ".join(words), size=60, overlap=15)
    assert all(len(c) <= 60 for c in chunks)
    seen = {w for c in chunks for w in c.split()}
    assert seen == set(words)


def test_consecutive_chunks_overlap():
    text = " ".join(f"w{i:03d}" for i in range(100))
    chunks = split_text(text, size=50, overlap=20)
    for first, second in zip(chunks, chunks[1:], strict=False):
        shared = set(first.split()) & set(second.split())
        # Words are 4 chars + a space, so 20 characters of overlap is 4 words.
        assert len(shared) == 4
        assert first.split()[-1] in shared


def test_zero_overlap_has_no_repeats():
    text = " ".join(f"w{i:03d}" for i in range(100))
    chunks = split_text(text, size=50, overlap=0)
    all_words = [w for c in chunks for w in c.split()]
    assert len(all_words) == 100


def test_oversized_word_becomes_own_chunk():
    chunks = split_text("tiny " + "x" * 30 + " end", size=10, overlap=2)
    assert "x" * 30 in chunks


@pytest.mark.parametrize(("size", "overlap"), [(0, 0), (10, 10), (10, -1)])
def test_invalid_parameters(size, overlap):
    with pytest.raises(ValueError):
        split_text("text", size=size, overlap=overlap)


def test_chunk_section_keeps_metadata_and_unique_ids():
    section = Section("freecad", "Placement", "Usage", "https://x/Placement#Usage", "a " * 300)
    chunks = chunk_section(section, size=100, overlap=20)
    assert len(chunks) > 1
    assert len({c.id for c in chunks}) == len(chunks)
    assert all(c.url == section.url and c.section == "Usage" for c in chunks)


def test_normalize_whitespace_keeps_paragraphs():
    assert normalize_whitespace("a  b\nc\n\n\n d\t e") == "a b c\n\nd e"
