"""API route definitions."""

from __future__ import annotations

import redis as redis_lib
from fastapi import APIRouter, HTTPException, Query

from app import vector_store
from app.config import get_settings
from app.recommender import ItemNotFoundError, recommend_similar, search_text
from app.schemas import (
    EnqueueResponse,
    HealthResponse,
    ItemIn,
    ItemOut,
    RecommendResponse,
    SimilarItem,
)
from app.worker.tasks import ingest_item

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """Liveness/readiness check for Qdrant and Redis."""
    settings = get_settings()
    qdrant_ok = vector_store.healthy()
    try:
        redis_ok = redis_lib.Redis.from_url(settings.redis_url).ping()
    except Exception:  # noqa: BLE001
        redis_ok = False
    status = "ok" if (qdrant_ok and redis_ok) else "degraded"
    return HealthResponse(status=status, qdrant=qdrant_ok, redis=bool(redis_ok))


@router.get(
    "/items/{item_id}/similar",
    response_model=RecommendResponse,
    tags=["recommendations"],
)
def similar_items(
    item_id: str,
    k: int = Query(default=None, ge=1, le=100, description="Number of neighbors to return."),
) -> RecommendResponse:
    """Return the top-k items most similar to the given item (default 5)."""
    settings = get_settings()
    top_k = k or settings.default_top_k
    try:
        results, latency_ms = recommend_similar(item_id, top_k)
    except ItemNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return RecommendResponse(
        query_item_id=item_id,
        results=[SimilarItem(**r) for r in results],
        search_latency_ms=round(latency_ms, 3),
    )


@router.get("/items/{item_id}", response_model=ItemOut, tags=["items"])
def get_item(item_id: str) -> ItemOut:
    """Return a stored item's payload."""
    payload = vector_store.get_item(item_id)
    if payload is None:
        raise HTTPException(status_code=404, detail=f"Item not found: {item_id}")
    return ItemOut(**payload)


@router.post(
    "/items",
    response_model=EnqueueResponse,
    status_code=202,
    tags=["items"],
)
def add_item(item: ItemIn) -> EnqueueResponse:
    """Accept an item for asynchronous embedding + ingestion via Celery."""
    record = {"item_id": item.item_id, "title": item.title, "text": item.text}
    task = ingest_item.delay(record)
    return EnqueueResponse(task_id=task.id, item_id=item.item_id)


@router.post("/search", response_model=RecommendResponse, tags=["recommendations"])
def search(
    q: str = Query(..., min_length=1, description="Free-text query."),
    k: int = Query(default=None, ge=1, le=100),
) -> RecommendResponse:
    """Embed a free-text query and return the top-k most similar items."""
    settings = get_settings()
    top_k = k or settings.default_top_k
    results, latency_ms = search_text(q, top_k)
    return RecommendResponse(
        query_text=q,
        results=[SimilarItem(**r) for r in results],
        search_latency_ms=round(latency_ms, 3),
    )
