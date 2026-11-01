from functools import lru_cache
from pathlib import Path

from pydantic import (
    Field,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Application settings."""

    app_name: str = "rag_backend"
    app_version: str = "0.1.0"
    app_env: str = "development"

    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    groq_dev_api_key: str = Field(..., env="GROQ_DEV_API_KEY")
    tavily_api_key: str = Field(..., env="TAVILY_API_KEY")
    pinecone_api_key: str = Field(..., env="PINECONE_API_KEY")
    pinecone_index_name: str = Field(..., env="PINECONE_INDEX_NAME")
    pinecone_namespace: str = Field(..., env="PINECONE_NAMESPACE")
    embedding_model: str = Field(..., env="EMBEDDING_MODEL")
    groq_model: str = Field(..., env="GROQ_MODEL")
    top_k: int = Field(5, env="TOP_K")
    max_retries: int = Field(3, env="MAX_RETRIES")
    admin_api_key: str = Field(..., env="ADMIN_API_KEY")

    audit_db_path: str = str(BASE_DIR / "data" / "audit_logs.db")
    upload_dir: str = str(BASE_DIR / "data" / "uploads")
    sample_kb_dir: str = str(BASE_DIR / "data" / "sample_kb")

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache()
def get_settings() -> Settings:
    return Settings()
