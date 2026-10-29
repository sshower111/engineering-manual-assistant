"""Chroma-backed vector store: index chunks and retrieve the closest ones."""

from pathlib import Path

import chromadb

from manual_assistant.embeddings import Embedder
from manual_assistant.models import Chunk, RetrievedChunk

# Chroma's default HNSW search_ef is 10, which misses obvious best matches even on a
# few thousand chunks. A larger value costs little at this scale and keeps recall high.
COLLECTION_METADATA = {"hnsw:space": "cosine", "hnsw:search_ef": 200, "hnsw:construction_ef": 200}


def embedding_text(chunk: Chunk) -> str:
    """Text that gets embedded: the chunk prefixed with where it comes from."""
    return f"{chunk.title} - {chunk.section}\n{chunk.text}"


class VectorStore:
    def __init__(self, chroma_dir: Path | None, collection_name: str, embedder: Embedder) -> None:
        if chroma_dir is None:
            client = chromadb.EphemeralClient()
        else:
            client = chromadb.PersistentClient(path=str(chroma_dir))
        self._client = client
        self._name = collection_name
        self._embedder = embedder
        self._collection = client.get_or_create_collection(
            collection_name, metadata=COLLECTION_METADATA
        )

    def reset(self) -> None:
        self._client.delete_collection(self._name)
        self._collection = self._client.get_or_create_collection(
            self._name, metadata=COLLECTION_METADATA
        )

    def count(self) -> int:
        return self._collection.count()

    def add(self, chunks: list[Chunk], batch_size: int = 256) -> None:
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            self._collection.upsert(
                ids=[c.id for c in batch],
                documents=[c.text for c in batch],
                metadatas=[c.metadata() for c in batch],
                embeddings=self._embedder.embed([embedding_text(c) for c in batch]),
            )

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        result = self._collection.query(
            query_embeddings=self._embedder.embed([query]),
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for chunk_id, text, meta, distance in zip(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
            strict=True,
        ):
            hits.append(
                RetrievedChunk(
                    id=chunk_id,
                    text=text,
                    score=round(1.0 - distance, 4),
                    title=meta["title"],
                    section=meta["section"],
                    url=meta["url"],
                )
            )
        return hits
