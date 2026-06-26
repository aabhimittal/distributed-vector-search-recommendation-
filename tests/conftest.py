"""Shared test fixtures and environment isolation."""

from __future__ import annotations

import os

import pytest

# Ensure tests never accidentally read a developer's local .env or hit real
# services; point everything at unreachable-but-well-formed defaults.
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")


@pytest.fixture
def sample_records():
    return [
        {"item_id": "a1", "title": "Red Running Shoes", "text": "lightweight red running shoes"},
        {"item_id": "b2", "title": "Blue Hiking Boots", "text": "durable blue hiking boots"},
        {"item_id": "c3", "title": "Wireless Headphones", "text": "noise cancelling headphones"},
    ]
