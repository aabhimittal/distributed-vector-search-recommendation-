"""Unit tests for the embedding wrapper (model mocked to avoid downloads)."""

from __future__ import annotations

import numpy as np

from app import embeddings


class _FakeModel:
    """Deterministic stand-in for a SentenceTransformer."""

    def get_sentence_embedding_dimension(self) -> int:
        return 4

    def encode(self, texts, **_kwargs):
        # Map text length to a 4-d vector; values are irrelevant, shape matters.
        return np.array([[float(len(t)), 1.0, 2.0, 3.0] for t in texts])


def test_embed_texts_shape(monkeypatch):
    monkeypatch.setattr(embeddings, "get_model", lambda: _FakeModel())
    vectors = embeddings.embed_texts(["hello", "world!"])
    assert len(vectors) == 2
    assert all(len(v) == 4 for v in vectors)


def test_embed_text_single(monkeypatch):
    monkeypatch.setattr(embeddings, "get_model", lambda: _FakeModel())
    vector = embeddings.embed_text("abc")
    assert isinstance(vector, list)
    assert len(vector) == 4
