# Distributed Vector Search & Recommendation System

An item-to-item recommendation API: given an item ID, it returns the top-5 most
**textually similar** items. Built as an MLE-architecture reference combining
dense sentence-transformer embeddings, a specialized vector database, an
asynchronous ingestion worker, and latency profiling under load.

```
HF dataset ──► ingest (batch embed) ──► Qdrant (cosine, 384-d)
                                            ▲            │
POST /items ──► Celery task (Redis) ────────┘            │ recommend-by-id
                                                         ▼
                    GET /items/{id}/similar?k=5 ──► FastAPI ──► top-5 JSON
```

## Stack

| Concern            | Choice                                                        |
| ------------------ | ------------------------------------------------------------- |
| Embeddings (ML)    | `sentence-transformers/all-MiniLM-L6-v2` (384-d, cosine)      |
| Vector database    | **Qdrant** (self-hosted, no API key)                          |
| API                | FastAPI + Uvicorn                                             |
| Async ingestion    | **Celery** with **Redis** broker/backend                     |
| Dataset            | HuggingFace — Amazon product metadata (configurable)          |
| Load testing       | **Locust** (headless, CSV percentile output)                  |

## Components

| Path                       | Responsibility                                                   |
| -------------------------- | --------------------------------------------------------------- |
| `app/embeddings.py`        | Process-singleton sentence-transformer; `embed_texts()`         |
| `app/vector_store.py`      | Qdrant collection mgmt, upsert, recommend-by-id, text search    |
| `app/recommender.py`       | Orchestration + per-call vector-search latency timing           |
| `app/api/routes.py`        | HTTP endpoints                                                  |
| `app/main.py`              | FastAPI app + request-latency middleware                        |
| `app/worker/`              | Celery app + ingestion tasks                                    |
| `app/ingest.py`            | Bulk HF dataset loader → embed → upsert                         |
| `scripts/seed.py`          | CLI to run the bulk ingestion                                   |
| `loadtest/locustfile.py`   | Load test driving `/similar` and `/search`                      |

## Quickstart (Docker)

```bash
cp .env.example .env            # optional: tweak dataset / model
make up                         # start qdrant, redis, api, worker
make seed                       # bulk-ingest the dataset into Qdrant
curl localhost:8000/health
```

Then query recommendations (grab a real `item_id` from a search first):

```bash
# Discover some items
curl -s "localhost:8000/search?q=antivirus&k=5" | jq

# Top-5 similar to an item
curl -s "localhost:8000/items/<item_id>/similar?k=5" | jq
```

Interactive docs: <http://localhost:8000/docs>.

## API

| Method & path                     | Description                                              |
| --------------------------------- | ------------------------------------------------------- |
| `GET  /health`                    | Liveness + Qdrant/Redis reachability                    |
| `GET  /items/{item_id}/similar`   | **Top-k similar items** (default `k=5`); 404 if unknown |
| `GET  /items/{item_id}`           | Fetch a stored item's payload                           |
| `POST /items`                     | Add/update an item → enqueues a Celery task (`202`)     |
| `POST /search?q=...`              | Free-text query → top-k similar items                   |

Every similarity response includes `search_latency_ms` (the pure vector-store
query time), and every response carries an `X-Process-Time-Ms` header.

### Asynchronous ingestion

`POST /items` does not embed inline. It enqueues a Celery task on Redis; the
worker embeds the text and upserts it into Qdrant, so embeddings are updated
asynchronously as new items arrive:

```bash
curl -X POST localhost:8000/items \
  -H 'content-type: application/json' \
  -d '{"item_id":"demo-1","title":"Demo","text":"a cheap blue umbrella"}'
# -> 202 {"task_id": "...", "item_id": "demo-1", "status": "queued"}
```

## Dataset configuration

The ingestion pipeline is dataset-agnostic. Defaults target Amazon product
metadata; override via `.env` to use any text dataset:

```env
DATASET_NAME=McAuley-Lab/Amazon-Reviews-2023
DATASET_CONFIG=raw_meta_Software
DATASET_SPLIT=full
TEXT_COLUMNS=title,description,features   # concatenated to form embedded text
TITLE_COLUMN=title
ID_COLUMN=parent_asin
MAX_ITEMS=5000                            # cap for a fast, bounded seed
```

## Latency profiling & load testing

Latency is observed at two layers:

- **Request latency** — JSON-logged per request by middleware (`latency_ms`).
- **Vector-search latency** — timed separately in `recommender.py` and returned
  as `search_latency_ms`.

Run a load test (API must be up and seeded):

```bash
make loadtest        # 50 users, 10/s ramp, 60s, headless
```

Locust writes `loadtest/results_stats.csv` with the **p50 / p95 / p99** latency
profile per endpoint. The runner first discovers real item IDs via `/search`,
then drives `/items/{id}/similar` and `/search` concurrently.

## Local development (without Docker)

```bash
make install
# Bring up just the infra:
docker compose up -d qdrant redis
# API:
uvicorn app.main:app --reload
# Worker (separate shell):
celery -A app.worker.celery_app.celery_app worker --loglevel=info
# Seed (separate shell):
python -m scripts.seed --recreate
```

## Tests

```bash
make test     # pytest
```

Unit tests mock the embedding model and external services (Qdrant/Redis/Celery),
so they run with no infrastructure — this is what CI executes
(`.github/workflows/ci.yml`).

## Out of scope / future work

- **Visual** similarity (image/CLIP embeddings) — currently text-only.
- Auth, rate limiting, API horizontal scaling, and a managed (Pinecone) backend.
