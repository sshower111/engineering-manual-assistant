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
    # Refuse without calling the LLM when the best cosine similarity is below this.
    # 0.45 is the highest value with no false refusals in the retrieval-only eval sweep.
    min_score: float = 0.45

    llm_model: str = "claude-sonnet-5-5"
    llm_max_tokens: int = 4000  # includes thinking tokens
    llm_effort: str = "low"  # short factual answers don't need deep thinking

    judge_model: str = "claude-haiku-5-5"

    snippet_chars: int = 300

    log_file: Path = Path("logs/queries.jsonl")


def get_settings() -> Settings:
    return Settings()
