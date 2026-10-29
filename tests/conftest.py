import hashlib
import math
import re

import pytest


class FakeEmbedder:
    """Deterministic bag-of-words embedder: shared words mean similar vectors."""

    dim = 256

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vector = [0.0] * self.dim
            for word in re.findall(r"[a-z0-9*]+", text.lower()):
                bucket = int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim
                vector[bucket] += 1.0
            norm = math.sqrt(sum(v * v for v in vector)) or 1.0
            vectors.append([v / norm for v in vector])
        return vectors


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()
