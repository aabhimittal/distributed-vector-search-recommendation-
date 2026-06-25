"""Qdrant vector-store access layer.

Encapsulates collection management, upserts, and similarity queries so the API,
worker, and ingestion all share one consistent implementation.
"""

from __future__ import annotations

import uuid
from functools import lru_cache
from typing import Iterable, Sequence

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.config import get_settings
from app.embeddings import embed_texts
from app.logging_conf import get_logger

logger = get_logger(__name__)

# Stable namespace so a given external item_id always maps to the same point id.
_NAMESPACE = uuid.UUID("6f8d4b1e-2c3a-4f5b-9a7e-1d2c3b4a5e6f")


def point_id_for(item_id: str) -> str:
    """Deterministically map an arbitrary external item id to a Qdrant point id.

    Using uuid5 keeps point ids valid (Qdrant requires int or UUID) and makes
    re-ingesting the same item idempotent (it overwrites the same point).
    """
    return str(uuid.uuid5(_NAMESPACE, str(item_id)))


@lru_cache(maxsize=1)
def get_client() -> QdrantClient:
    """Return a cached Qdrant client for this process."""
    settings = get_settings()
    return QdrantClient(url=settings.qdrant_url, timeout=30.0)


def ensure_collection(recreate: bool = False) -> None:
    """Create the collection if it does not already exist."""
    settings = get_settings()
    client = get_client()
    exists = client.collection_exists(settings.qdrant_collection)
    if exists and recreate:
        client.delete_collection(settings.qdrant_collection)
        exists = False
    if not exists:
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=qmodels.VectorParams(
                size=settings.embedding_dim,
                distance=qmodels.Distance.COSINE,
            ),
        )
        logger.info(
            "collection_created",
            extra={
                "collection": settings.qdrant_collection,
                "dim": settings.embedding_dim,
            },
        )


def _to_point(record: dict, vector: Sequence[float]) -> qmodels.PointStruct:
    item_id = str(record["item_id"])
    payload = {
        "item_id": item_id,
        "title": record.get("title"),
        "text": record.get("text"),
    }
    return qmodels.PointStruct(
        id=point_id_for(item_id),
        vector=list(vector),
        payload=payload,
    )


def upsert_items(records: Iterable[dict], vectors: Sequence[Sequence[float]] | None = None) -> int:
    """Upsert items into the collection.

    Each record must contain `item_id` and `text` (and optionally `title`).
    If `vectors` is not supplied, embeddings are computed from each record's text.
    Returns the number of points upserted.
    """
    settings = get_settings()
    client = get_client()
    records = list(records)
    if not records:
        return 0

    if vectors is None:
        vectors = embed_texts([r["text"] for r in records])
    if len(vectors) != len(records):
        raise ValueError("Number of vectors does not match number of records.")

    points = [_to_point(rec, vec) for rec, vec in zip(records, vectors)]
    client.upsert(collection_name=settings.qdrant_collection, points=points)
    return len(points)


def get_item(item_id: str) -> dict | None:
    """Retrieve a single stored item's payload, or None if it does not exist."""
    settings = get_settings()
    client = get_client()
    points = client.retrieve(
        collection_name=settings.qdrant_collection,
        ids=[point_id_for(item_id)],
        with_payload=True,
    )
    if not points:
        return None
    return points[0].payload


def recommend_by_id(item_id: str, k: int) -> list[dict] | None:
    """Return the top-k items most similar to the given item.

    Uses Qdrant's recommend-by-point query so the stored vector is reused (no
    re-embedding). The query item itself is excluded from the results. Returns
    None if the item id is unknown.
    """
    settings = get_settings()
    client = get_client()
    pid = point_id_for(item_id)

    # Confirm the item exists so we can return a 404 rather than empty results.
    if get_item(item_id) is None:
        return None

    response = client.query_points(
        collection_name=settings.qdrant_collection,
        query=qmodels.RecommendQuery(
            recommend=qmodels.RecommendInput(positive=[pid])
        ),
        limit=k + 1,  # +1 because the query item may come back; we drop it below.
        with_payload=True,
    )
    return _format_hits(response.points, exclude_item_id=item_id, k=k)


def search_by_text(text: str, k: int) -> list[dict]:
    """Embed free text and return the top-k most similar stored items."""
    settings = get_settings()
    client = get_client()
    vector = embed_texts([text])[0]
    response = client.query_points(
        collection_name=settings.qdrant_collection,
        query=vector,
        limit=k,
        with_payload=True,
    )
    return _format_hits(response.points, exclude_item_id=None, k=k)


def _format_hits(points, exclude_item_id: str | None, k: int) -> list[dict]:
    results: list[dict] = []
    for p in points:
        payload = p.payload or {}
        if exclude_item_id is not None and payload.get("item_id") == exclude_item_id:
            continue
        results.append(
            {
                "item_id": payload.get("item_id"),
                "title": payload.get("title"),
                "text": payload.get("text"),
                "score": p.score,
            }
        )
        if len(results) >= k:
            break
    return results


def count() -> int:
    """Return the number of points currently in the collection."""
    settings = get_settings()
    client = get_client()
    return client.count(collection_name=settings.qdrant_collection).count


def healthy() -> bool:
    """Return True if Qdrant is reachable."""
    try:
        get_client().get_collections()
        return True
    except Exception:  # noqa: BLE001 - health check must never raise
        return False
