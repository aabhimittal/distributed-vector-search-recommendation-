"""Celery application wired to Redis as broker and result backend."""

from __future__ import annotations

from celery import Celery

from app.config import get_settings
from app.logging_conf import configure_logging

settings = get_settings()

celery_app = Celery(
    "recsys",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["app.worker.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    result_expires=3600,
)


@celery_app.on_after_configure.connect
def _setup_logging(sender, **_kwargs):  # noqa: ANN001
    configure_logging()
