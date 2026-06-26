"""CLI entrypoint for the initial bulk ingestion.

Usage:
    python -m scripts.seed              # ingest using .env / defaults
    python -m scripts.seed --recreate   # drop and rebuild the collection first
"""

from __future__ import annotations

import argparse

from app.ingest import run
from app.logging_conf import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed the Qdrant collection from a HF dataset.")
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and recreate the collection before ingesting.",
    )
    args = parser.parse_args()

    configure_logging()
    count = run(recreate=args.recreate)
    print(f"Ingested {count} items.")


if __name__ == "__main__":
    main()
