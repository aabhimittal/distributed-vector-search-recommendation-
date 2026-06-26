.PHONY: help install up down logs seed test loadtest fmt

HOST ?= http://localhost:8000

help:
	@echo "Targets:"
	@echo "  install   Install Python dependencies"
	@echo "  up        Start qdrant, redis, api, worker (docker compose)"
	@echo "  down      Stop and remove the stack"
	@echo "  logs      Tail service logs"
	@echo "  seed      Bulk-ingest the dataset into Qdrant (run inside the api container)"
	@echo "  test      Run pytest"
	@echo "  loadtest  Run the Locust load test headless and write CSV results"

install:
	pip install -r requirements.txt

up:
	docker compose up -d --build

down:
	docker compose down

logs:
	docker compose logs -f

# Run the seed inside the api container so it shares the model cache + network.
seed:
	docker compose exec api python -m scripts.seed --recreate

# Local seed (requires QDRANT_URL/REDIS_URL reachable from the host).
seed-local:
	python -m scripts.seed --recreate

test:
	pytest -q

loadtest:
	locust -f loadtest/locustfile.py --headless \
		-u 50 -r 10 -t 60s --host $(HOST) \
		--csv loadtest/results
	@echo "Latency profile written to loadtest/results_stats.csv"
