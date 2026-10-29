from manual_assistant.ingest import build_chunks
from manual_assistant.models import Section
from manual_assistant.store import VectorStore

SECTIONS = [
    Section("freecad", "FEM ConstraintFixed", "Usage", "u1", "fix a face so it cannot move"),
    Section("freecad", "PartDesign Pad", "Usage", "u2", "pad extrudes a sketch into a solid"),
    Section("calculix", "CCX", "*CLOAD", "u3", "concentrated force applied at a node"),
]


def test_search_returns_best_match_first(fake_embedder):
    store = VectorStore(None, "test_search", fake_embedder)
    store.reset()
    store.add(build_chunks(SECTIONS, size=200, overlap=20))
    assert store.count() == 3

    hits = store.search("how do I extrude a sketch with pad", top_k=2)
    assert hits[0].url == "u2"
    assert hits[0].score > hits[1].score
    assert 0.0 <= hits[1].score <= 1.0


def test_reindexing_is_idempotent(fake_embedder):
    store = VectorStore(None, "test_idempotent", fake_embedder)
    store.reset()
    chunks = build_chunks(SECTIONS + SECTIONS, size=200, overlap=20)
    store.add(chunks)
    store.add(chunks)
    assert store.count() == 3
