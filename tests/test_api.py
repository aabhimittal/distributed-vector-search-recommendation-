"""API tests with external services (Qdrant/Redis/Celery) mocked out."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app import vector_store
from app.api import routes
from app.recommender import ItemNotFoundError


@pytest.fixture
def client(monkeypatch):
    # Don't touch Qdrant on startup.
    monkeypatch.setattr(vector_store, "ensure_collection", lambda *a, **k: None)

    from app.main import app

    with TestClient(app) as c:
        yield c


def test_similar_items_ok(client, monkeypatch):
    def fake_recommend(item_id, k):
        results = [
            {"item_id": "x", "title": "X", "text": "x text", "score": 0.91},
            {"item_id": "y", "title": "Y", "text": "y text", "score": 0.88},
        ]
        return results, 1.234

    monkeypatch.setattr(routes, "recommend_similar", fake_recommend)

    resp = client.get("/items/a1/similar?k=2")
    assert resp.status_code == 200
    body = resp.json()
    assert body["query_item_id"] == "a1"
    assert len(body["results"]) == 2
    assert body["results"][0]["item_id"] == "x"
    assert body["search_latency_ms"] == 1.234


def test_similar_items_not_found(client, monkeypatch):
    def fake_recommend(item_id, k):
        raise ItemNotFoundError(item_id)

    monkeypatch.setattr(routes, "recommend_similar", fake_recommend)

    resp = client.get("/items/missing/similar")
    assert resp.status_code == 404


def test_add_item_enqueues_task(client, monkeypatch):
    class FakeTask:
        id = "task-123"

    monkeypatch.setattr(routes.ingest_item, "delay", lambda record: FakeTask())

    resp = client.post(
        "/items",
        json={"item_id": "new1", "title": "New", "text": "a brand new item"},
    )
    assert resp.status_code == 202
    body = resp.json()
    assert body["task_id"] == "task-123"
    assert body["item_id"] == "new1"
    assert body["status"] == "queued"


def test_search_ok(client, monkeypatch):
    monkeypatch.setattr(
        routes,
        "search_text",
        lambda q, k: ([{"item_id": "z", "title": "Z", "text": "z", "score": 0.5}], 2.0),
    )
    resp = client.post("/search", params={"q": "shoes", "k": 1})
    assert resp.status_code == 200
    body = resp.json()
    assert body["query_text"] == "shoes"
    assert body["results"][0]["item_id"] == "z"


def test_get_item_not_found(client, monkeypatch):
    monkeypatch.setattr(vector_store, "get_item", lambda item_id: None)
    resp = client.get("/items/nope")
    assert resp.status_code == 404
