"""Unit tests for vector-store helpers that don't require a live Qdrant."""

from __future__ import annotations

import uuid

from app import vector_store


def test_point_id_is_deterministic():
    assert vector_store.point_id_for("abc") == vector_store.point_id_for("abc")


def test_point_id_is_valid_uuid_and_distinct():
    pid_a = vector_store.point_id_for("a")
    pid_b = vector_store.point_id_for("b")
    # Valid UUID strings (Qdrant requires int or UUID point ids).
    uuid.UUID(pid_a)
    uuid.UUID(pid_b)
    assert pid_a != pid_b


def test_format_hits_excludes_query_and_limits_k():
    class Hit:
        def __init__(self, item_id, score):
            self.payload = {"item_id": item_id, "title": item_id, "text": item_id}
            self.score = score

    hits = [Hit("self", 1.0), Hit("x", 0.9), Hit("y", 0.8), Hit("z", 0.7)]
    out = vector_store._format_hits(hits, exclude_item_id="self", k=2)
    assert [r["item_id"] for r in out] == ["x", "y"]
    assert out[0]["score"] == 0.9
