"""Unit tests for dataset-row normalization in the ingestion pipeline."""

from __future__ import annotations

from app.config import Settings
from app.ingest import record_from_row


def _settings(**overrides) -> Settings:
    base = dict(
        id_column="parent_asin",
        title_column="title",
        text_columns="title,description,features",
        dataset_config="raw_meta_Software",
    )
    base.update(overrides)
    return Settings(**base)


def test_record_concatenates_text_columns():
    settings = _settings()
    row = {
        "parent_asin": "B001",
        "title": "Cool App",
        "description": "Does cool things",
        "features": ["fast", "secure"],
    }
    rec = record_from_row(row, 0, settings)
    assert rec["item_id"] == "B001"
    assert rec["title"] == "Cool App"
    assert "Cool App" in rec["text"]
    assert "Does cool things" in rec["text"]
    # List-valued columns are flattened.
    assert "fast" in rec["text"] and "secure" in rec["text"]


def test_record_falls_back_to_index_id():
    settings = _settings()
    row = {"parent_asin": None, "title": "No Id Item", "description": "x"}
    rec = record_from_row(row, 7, settings)
    assert rec["item_id"] == "raw_meta_Software-7"


def test_record_skips_when_no_text():
    settings = _settings()
    row = {"parent_asin": "B002", "title": "", "description": "", "features": []}
    assert record_from_row(row, 0, settings) is None
