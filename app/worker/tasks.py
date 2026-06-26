"""Celery tasks for asynchronous embedding ingestion.

These tasks embed item text and upsert the resulting vectors into Qdrant. They
back both the bulk seed (`ingest_items`) and the live "add an item" API path
(`ingest_item`), so new items get their embeddings updated asynchronously.
"""

from __future__ import annotations

from app import vector_store
from app.embeddings import embed_texts
from app.logging_conf import get_logger
from app.worker.celery_app import celery_app

logger = get_logger(__name__)


@celery_app.task(name="ingest.items", bind=True, max_retries=3, default_retry_delay=5)
def ingest_items(self, records: list[dict]) -> dict:
    """Embed and upsert a batch of item records.

    Each record must contain `item_id` and `text` (optionally `title`).
    """
    try:
        vector_store.ensure_collection()
        vectors = embed_texts([r["text"] for r in records])
        n = vector_store.upsert_items(records, vectors=vectors)
        logger.info("ingest_items_done", extra={"count": n})
        return {"upserted": n}
    except Exception as exc:  # noqa: BLE001 - retry transient failures
        logger.error("ingest_items_failed", extra={"error": str(exc)})
        raise self.retry(exc=exc)


@celery_app.task(name="ingest.item", bind=True, max_retries=3, default_retry_delay=5)
def ingest_item(self, record: dict) -> dict:
    """Embed and upsert a single item record."""
    try:
        vector_store.ensure_collection()
        n = vector_store.upsert_items([record])
        logger.info("ingest_item_done", extra={"item_id": record.get("item_id")})
        return {"upserted": n, "item_id": record.get("item_id")}
    except Exception as exc:  # noqa: BLE001 - retry transient failures
        logger.error(
            "ingest_item_failed",
            extra={"item_id": record.get("item_id"), "error": str(exc)},
        )
        raise self.retry(exc=exc)
