"""Application configuration loaded from environment variables / .env."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for the API, worker, and ingestion."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Services
    qdrant_url: str = Field(default="http://localhost:6333")
    redis_url: str = Field(default="redis://localhost:6379/0")

    # Vector store
    qdrant_collection: str = Field(default="items")

    # Embedding model
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    embedding_dim: int = Field(default=384)

    # Dataset (HuggingFace)
    dataset_name: str = Field(default="McAuley-Lab/Amazon-Reviews-2023")
    dataset_config: str = Field(default="raw_meta_Software")
    dataset_split: str = Field(default="full")
    text_columns: str = Field(default="title,description,features")
    title_column: str = Field(default="title")
    id_column: str = Field(default="parent_asin")
    max_items: int = Field(default=5000)

    # Ingestion
    batch_size: int = Field(default=256)

    # Defaults
    default_top_k: int = Field(default=5)

    @property
    def text_column_list(self) -> list[str]:
        """Parsed list of text columns to concatenate for embedding."""
        return [c.strip() for c in self.text_columns.split(",") if c.strip()]


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
