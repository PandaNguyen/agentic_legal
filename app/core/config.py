from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "legal-mvp-starter"
    app_env: str = "local"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"
    log_file: str = "log.log"

    openai_api_key: str | None = None
    openai_model: str = "gpt-4.1-mini"
    openai_llm_model: str | None = None
    openai_base_url: str | None = None
    openai_api_style: str = "responses"

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "legal_chunks_hybrid_v1"
    qdrant_collection_hybrid: str = "legal_chunks_hybrid_v1"
    qdrant_embedding_dim: int = 1024
    dense_embedding_model: str = "mainguyen9/vietlegal-harrier-0.6b"
    dense_embedding_device: str | None = None
    dense_embedding_batch_size: int = 32
    sparse_embedding_model: str = "Qdrant/bm25"
    bm25_k: float = 1.2
    bm25_b: float = 0.75
    bm25_language: str = "none"
    ingest_checkpoint_db: str = "data/processed/hybrid_ingest_checkpoint.sqlite3"
    ingest_pipeline_version: str = "hybrid_v1"

    enable_web_search: bool = False
    firecrawl_api_key: str | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()
