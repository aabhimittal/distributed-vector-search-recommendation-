"""Pydantic request/response models for the API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ItemIn(BaseModel):
    """Payload for adding/updating a single item via the async ingestion path."""

    item_id: str = Field(..., description="Stable, caller-supplied item identifier.")
    title: str | None = Field(default=None, description="Human-readable item title.")
    text: str = Field(..., description="Text used to compute the item embedding.")


class ItemOut(BaseModel):
    """A stored item as returned by the API."""

    item_id: str
    title: str | None = None
    text: str | None = None


class SimilarItem(ItemOut):
    """A neighbor returned by a similarity query, with its similarity score."""

    score: float = Field(..., description="Cosine similarity to the query item (higher = closer).")


class RecommendResponse(BaseModel):
    """Response for the top-k similarity endpoints."""

    query_item_id: str | None = Field(
        default=None, description="The item the recommendations are based on (if any)."
    )
    query_text: str | None = Field(
        default=None, description="The free-text query used (for /search)."
    )
    results: list[SimilarItem]
    search_latency_ms: float = Field(
        ..., description="Wall-clock latency of the vector-store query in milliseconds."
    )


class EnqueueResponse(BaseModel):
    """Response when an item is accepted for asynchronous ingestion."""

    task_id: str
    item_id: str
    status: str = "queued"


class HealthResponse(BaseModel):
    """Liveness/readiness response."""

    status: str
    qdrant: bool
    redis: bool
