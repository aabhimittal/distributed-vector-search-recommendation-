"""Sentence-transformer embedding model wrapper.

The model is expensive to load, so it is instantiated once per process and
shared by the API (text search) and the Celery worker (ingestion).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Sequence

from app.config import get_settings
from app.logging_conf import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_model():
    """Load and cache the sentence-transformer model for this process."""
    # Imported lazily so that modules which only need helpers (e.g. tests that
    # mock embeddings) don't pay the heavy import cost.
    from sentence_transformers import SentenceTransformer

    settings = get_settings()
    logger.info("loading_embedding_model", extra={"model": settings.embedding_model})
    model = SentenceTransformer(settings.embedding_model)
    logger.info(
        "embedding_model_loaded",
        extra={
            "model": settings.embedding_model,
            "dim": model.get_sentence_embedding_dimension(),
        },
    )
    return model


def embed_texts(texts: Sequence[str], batch_size: int | None = None) -> list[list[float]]:
    """Embed a sequence of texts into normalized dense vectors.

    Vectors are L2-normalized so that cosine distance in Qdrant behaves as a
    proper similarity measure.
    """
    settings = get_settings()
    model = get_model()
    vectors = model.encode(
        list(texts),
        batch_size=batch_size or settings.batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return vectors.tolist()


def embed_text(text: str) -> list[float]:
    """Embed a single text into a normalized dense vector."""
    return embed_texts([text])[0]
