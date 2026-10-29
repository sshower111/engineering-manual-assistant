from manual_assistant.ingest import build_chunks
from manual_assistant.models import Section
from manual_assistant.store import VectorStore

SECTIONS = [
    Section(
        "freecad",
        "FEM ConstraintFixed",
        "Usage",
        "u1",
        "Select a face and fix it so that it cannot move in any direction during the analysis.",
    ),
    Section(
        "freecad",
        "PartDesign Pad",
        "Usage",
        "u2",
        "The pad tool extrudes a sketch or a face of the body into a solid along a straight path.",
    ),
    Section(
        "calculix",
        "CCX",
        "*CLOAD",
        "u3",
        "This keyword applies a concentrated force to a node that is not fixed by a constraint.",
    ),
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


def test_heading_only_chunks_are_dropped():
    sections = [Section("calculix", "CCX", "Boundary conditions", "u", "Boundary conditions 6.7.1")]
    assert build_chunks(sections, size=200, overlap=20) == []


def test_collection_uses_high_hnsw_search_ef(fake_embedder):
    """Chroma's default search_ef (10) silently missed the best match on the real corpus."""
    store = VectorStore(None, "test_config", fake_embedder)
    store.reset()
    assert store._collection.metadata["hnsw:search_ef"] >= 100
