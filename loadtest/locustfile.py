"""Locust load test for the recommendation API.

Exercises the two read paths under concurrent load so we can capture a latency
profile (p50/p95/p99):

  * GET /items/{id}/similar  -- item-to-item recommendations
  * POST /search             -- free-text embedding search

Run headless with CSV output (see the Makefile `loadtest` target):

    locust -f loadtest/locustfile.py --headless \\
        -u 50 -r 10 -t 60s --host http://localhost:8000 \\
        --csv loadtest/results

Then inspect loadtest/results_stats.csv for the percentile breakdown.
"""

from __future__ import annotations

import random

from locust import HttpUser, between, events, task

# A handful of generic query phrases to vary the embedding search load.
QUERIES = [
    "wireless headphones",
    "office software license",
    "kids educational game",
    "photo editing tool",
    "antivirus protection",
    "video conferencing app",
    "data backup utility",
    "language learning course",
    "tax preparation software",
    "music production suite",
]

# Populated on test start by discovering real item ids from the API.
ITEM_IDS: list[str] = []


@events.test_start.add_listener
def _discover_item_ids(environment, **_kwargs):
    """Seed ITEM_IDS by issuing a few text searches against the running API."""
    host = environment.host or "http://localhost:8000"
    import urllib.parse
    import urllib.request

    found: set[str] = set()
    for q in QUERIES:
        try:
            url = f"{host}/search?{urllib.parse.urlencode({'q': q, 'k': 20})}"
            with urllib.request.urlopen(url, timeout=10) as resp:  # noqa: S310
                import json

                data = json.load(resp)
            for item in data.get("results", []):
                if item.get("item_id"):
                    found.add(item["item_id"])
        except Exception:  # noqa: BLE001 - best-effort discovery
            continue
    ITEM_IDS.extend(sorted(found))
    print(f"[loadtest] discovered {len(ITEM_IDS)} item ids for /similar load")


class RecsysUser(HttpUser):
    """Simulated user hitting the recommendation endpoints."""

    wait_time = between(0.1, 0.5)

    @task(3)
    def similar(self):
        if not ITEM_IDS:
            return
        item_id = random.choice(ITEM_IDS)
        self.client.get(
            f"/items/{item_id}/similar?k=5",
            name="/items/[id]/similar",
        )

    @task(1)
    def search(self):
        q = random.choice(QUERIES)
        self.client.post(
            "/search",
            params={"q": q, "k": 5},
            name="/search",
        )
