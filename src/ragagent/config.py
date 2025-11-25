from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=(".env", ".env.local"), env_file_encoding="utf-8", extra="ignore")

    # App
    app_env: str = "development"
    log_level: str = "INFO"

    # OpenAI
    openai_api_key: str | None = None

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "documents"
    qdrant_vector_size: int = 1024
    qdrant_distance: str = "cosine"

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "neo4j"

    # Embeddings service
    embeddings_endpoint: str | None = None

    # OCR / Parsing
    ocr_engine: str = "easyocr"
    ocr_languages: str = "en"


settings = Settings()  # Load on import for convenience
