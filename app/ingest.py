"""Bulk ingestion: load a HuggingFace text dataset, embed it, upsert to Qdrant.

The loader is dataset-agnostic. The dataset name, config, split, and the
columns used for the item id / title / embedded text are all configurable via
settings (env / .env), so any text dataset can be ingested without code changes.
"""

from __future__ import annotations

import time
from typing import Iterator

from app import vector_store
from app.config import Settings, get_settings
from app.embeddings import embed_texts
from app.logging_conf import configure_logging, get_logger

logger = get_logger(__name__)


def _coerce_text(value) -> str:
    """Flatten a dataset cell into a single string (lists are space-joined)."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return " ".join(_coerce_text(v) for v in value)
    return str(value)


def record_from_row(row: dict, index: int, settings: Settings) -> dict | None:
    """Build an ingestion record {item_id, title, text} from a dataset row.

    Falls back to the row index for the id when the configured id column is
    missing/empty. Returns None when there is no usable text to embed.
    """
    raw_id = row.get(settings.id_column)
    item_id = str(raw_id) if raw_id not in (None, "") else f"{settings.dataset_config or 'row'}-{index}"

    title = _coerce_text(row.get(settings.title_column)) or None

    parts = [_coerce_text(row.get(col)) for col in settings.text_column_list]
    text = " ".join(p for p in parts if p).strip()
    if not text:
        return None

    return {"item_id": item_id, "title": title, "text": text}


def _iter_records(settings: Settings) -> Iterator[dict]:
    """Stream the dataset and yield cleaned ingestion records, capped to max_items."""
    from datasets import load_dataset

    logger.info(
        "loading_dataset",
        extra={
            "dataset": settings.dataset_name,
            "config": settings.dataset_config,
            "split": settings.dataset_split,
            "max_items": settings.max_items,
        },
    )
    ds = load_dataset(
        settings.dataset_name,
        settings.dataset_config or None,
        split=settings.dataset_split,
        streaming=True,
        trust_remote_code=True,
    )

    yielded = 0
    for index, row in enumerate(ds):
        record = record_from_row(row, index, settings)
        if record is None:
            continue
        yield record
        yielded += 1
        if yielded >= settings.max_items:
            break


def run(settings: Settings | None = None, recreate: bool = False) -> int:
    """Run the full bulk ingestion pipeline. Returns the number of items ingested."""
    settings = settings or get_settings()
    vector_store.ensure_collection(recreate=recreate)

    total = 0
    batch: list[dict] = []
    start = time.perf_counter()

    def flush(records: list[dict]) -> int:
        if not records:
            return 0
        vectors = embed_texts([r["text"] for r in records])
        return vector_store.upsert_items(records, vectors=vectors)

    for record in _iter_records(settings):
        batch.append(record)
        if len(batch) >= settings.batch_size:
            n = flush(batch)
            total += n
            logger.info("ingest_progress", extra={"ingested": total})
            batch = []

    total += flush(batch)

    elapsed = time.perf_counter() - start
    logger.info(
        "ingest_complete",
        extra={
            "ingested": total,
            "elapsed_s": round(elapsed, 2),
            "items_per_s": round(total / elapsed, 1) if elapsed > 0 else None,
            "collection_count": vector_store.count(),
        },
    )
    return total


if __name__ == "__main__":  # pragma: no cover
    configure_logging()
    run()
