"""Application settings, read from environment variables (prefix ``MA_``)."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MA_", env_file=".env", extra="ignore")

    data_dir: Path = Path("data")
    chroma_dir: Path = Path("data/chroma")
    collection_name: str = "manuals"

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    chunk_size: int = 800  # characters
    chunk_overlap: int = 150  # characters

    top_k: int = 5
    # Cosine similarity below which we refuse to answer.
    min_score: float = 0.35

    llm_model: str = "claude-sonnet-5-5"
    llm_max_tokens: int = 800

    log_file: Path = Path("logs/queries.jsonl")


def get_settings() -> Settings:
    return Settings()
