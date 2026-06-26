"""Recommendation orchestration with per-call latency measurement."""

from __future__ import annotations

import time

from app import vector_store
from app.logging_conf import get_logger

logger = get_logger(__name__)


class ItemNotFoundError(Exception):
    """Raised when a recommendation is requested for an unknown item id."""

    def __init__(self, item_id: str):
        self.item_id = item_id
        super().__init__(f"Item not found: {item_id}")


def _timed(fn, **log_fields):
    """Run `fn`, returning (result, elapsed_ms) and logging the search latency."""
    start = time.perf_counter()
    result = fn()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    logger.info("vector_search", extra={"search_latency_ms": round(elapsed_ms, 3), **log_fields})
    return result, elapsed_ms


def recommend_similar(item_id: str, k: int) -> tuple[list[dict], float]:
    """Return (top-k similar items, search_latency_ms) for the given item.

    Raises ItemNotFoundError if the item id is unknown.
    """
    results, elapsed_ms = _timed(
        lambda: vector_store.recommend_by_id(item_id, k),
        op="recommend_by_id",
        item_id=item_id,
        k=k,
    )
    if results is None:
        raise ItemNotFoundError(item_id)
    return results, elapsed_ms


def search_text(text: str, k: int) -> tuple[list[dict], float]:
    """Return (top-k items matching free text, search_latency_ms)."""
    results, elapsed_ms = _timed(
        lambda: vector_store.search_by_text(text, k),
        op="search_by_text",
        k=k,
    )
    return results, elapsed_ms
