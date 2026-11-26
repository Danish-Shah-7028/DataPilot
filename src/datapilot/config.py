from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

GROQ_BASE_URL = "https://api.groq.com/openai/v1"


class Settings(BaseSettings):
    """Runtime configuration, read from environment variables and an optional .env file."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    llm_provider: Literal["groq", "openai"] = "groq"
    llm_model: str = "llama-3.3-70b-versatile"
    llm_temperature: float = 0.0
    llm_timeout_seconds: float = 60.0
    llm_max_retries: int = 5
    llm_cache_enabled: bool = False

    groq_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    openai_base_url: str | None = None

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    rag_full_schema_max_columns: int = 40

    sql_row_limit: int = 5000
    sql_timeout_seconds: float = 15.0
    sandbox_timeout_seconds: float = 30.0
    sandbox_table_row_cap: int = 200_000
    ml_max_rows: int = 50_000

    critic_max_retries: int = 3
    critic_llm_review: bool = True
    confidence_threshold: float = 0.5

    data_dir: Path = Path("data")
    cache_dir: Path = Path(".cache")
    log_dir: Path = Path("logs")
    reports_dir: Path = Path("reports")
    models_dir: Path = Path("models")

    @property
    def llm_cache_path(self) -> Path:
        return self.cache_dir / "llm_cache.sqlite"

    @property
    def confidence_model_path(self) -> Path:
        return self.models_dir / "confidence.joblib"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
