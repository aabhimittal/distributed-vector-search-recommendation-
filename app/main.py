"""FastAPI application entrypoint with request-latency logging."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from app import vector_store
from app.api.routes import router
from app.logging_conf import configure_logging, get_logger

configure_logging()
logger = get_logger("app.request")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ensure the Qdrant collection exists on startup (best-effort)."""
    try:
        vector_store.ensure_collection()
    except Exception as exc:  # noqa: BLE001 - API should still start if Qdrant is down
        logger.error("startup_ensure_collection_failed", extra={"error": str(exc)})
    yield


app = FastAPI(
    title="Distributed Vector Search & Recommendation System",
    description="Item-to-item recommendations powered by sentence-transformer "
    "embeddings and a Qdrant vector database.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def log_latency(request: Request, call_next):
    """Log every request's total latency as structured JSON."""
    start = time.perf_counter()
    response = await call_next(request)
    latency_ms = (time.perf_counter() - start) * 1000.0
    response.headers["X-Process-Time-Ms"] = f"{latency_ms:.3f}"
    logger.info(
        "request",
        extra={
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "latency_ms": round(latency_ms, 3),
        },
    )
    return response


app.include_router(router)


@app.get("/", tags=["ops"])
def root() -> dict:
    """Service banner."""
    return {
        "service": "distributed-vector-search-recommendation",
        "docs": "/docs",
        "health": "/health",
    }
