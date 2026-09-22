"""Application configuration.

12-factor configuration: every knob is an environment variable, parsed and
validated once at process start by `pydantic-settings`. Secrets live in `.env`
(git-ignored) and are never hardcoded — the application fails fast at startup
if a required secret is missing.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field, SecretStr, computed_field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

Device = Literal["cpu", "cuda", "mps"]
LLMProvider = Literal["openai", "ollama"]

# Ollama's OpenAI-compatible endpoint ignores the key, but the SDK requires one.
OLLAMA_PLACEHOLDER_KEY = "ollama"


class Settings(BaseSettings):
    """Strongly-typed runtime configuration, sourced from the environment."""

    model_config = SettingsConfigDict(
        env_file=(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        validate_default=True,
    )

    # ----------------------------- Service --------------------------------- #
    app_name: str = "rag-showcase"
    environment: Literal["local", "dev", "staging", "prod"] = "local"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = Field(default=8000, ge=1, le=65535)
    cors_origins: str = "http://localhost:3000"

    # ------------------------------ Qdrant --------------------------------- #
    qdrant_host: str = "localhost"
    qdrant_port: int = Field(default=6333, ge=1, le=65535)
    qdrant_collection: str = "rag_showcase"
    qdrant_api_key: SecretStr | None = None
    qdrant_timeout_seconds: float = Field(default=10.0, gt=0)

    # ------------------------------- LLM ----------------------------------- #
    # Both providers speak the OpenAI chat-completions protocol, so the service
    # code is provider-agnostic: only the client's base_url, key and model
    # differ. Resolve them through `llm_*` below, never through `openai_*`.
    llm_provider: LLMProvider = "ollama"

    # -- OpenAI (LLM_PROVIDER=openai) -- #
    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str | None = None

    # -- Ollama (LLM_PROVIDER=ollama), local and key-less -- #
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_model: str = "llama3.1:8b"

    # -- shared generation parameters -- #
    openai_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    openai_max_tokens: int = Field(default=800, ge=1, le=8192)

    # ----------------------------- ML models ------------------------------- #
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = Field(default=384, ge=1)
    embedding_batch_size: int = Field(default=64, ge=1, le=512)
    reranker_model: str = "BAAI/bge-reranker-base"
    reranker_max_length: int = Field(default=512, ge=64, le=2048)
    model_device: Device | None = None

    # ----------------------------- Retrieval ------------------------------- #
    retrieval_candidates: int = Field(default=10, ge=1, le=100)
    rerank_top_n: int = Field(default=3, ge=1, le=50)
    max_top_k: int = Field(default=10, ge=1, le=50)
    score_threshold: float | None = None

    # ------------------------------ Chunking ------------------------------- #
    chunk_size: int = Field(default=500, ge=64, le=4096)
    chunk_overlap: int = Field(default=50, ge=0, le=1024)
    documents_dir: Path = Path("documents")
    ingest_upsert_batch_size: int = Field(default=128, ge=1, le=1024)

    # ------------------------------ Timeouts ------------------------------- #
    retrieval_timeout_seconds: float = Field(default=15.0, gt=0)
    generation_timeout_seconds: float = Field(default=60.0, gt=0)
    request_timeout_seconds: float = Field(default=90.0, gt=0)

    # ------------------------------ Langfuse ------------------------------- #
    langfuse_enabled: bool = True
    langfuse_public_key: SecretStr | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_host: str = "https://cloud.langfuse.com"

    # --------------------------- Normalisation ----------------------------- #
    @field_validator(
        "qdrant_api_key",
        "openai_api_key",
        "openai_base_url",
        "model_device",
        "score_threshold",
        "langfuse_public_key",
        "langfuse_secret_key",
        mode="before",
    )
    @classmethod
    def _empty_string_to_none(cls, value: Any) -> Any:
        """Treat `KEY=` in a dotenv file as "unset" rather than as an empty value."""
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("openai_api_key", mode="before")
    @classmethod
    def _reject_placeholder_key(cls, value: Any) -> Any:
        """Refuse the template key outright — it only ever produces 401s."""
        if isinstance(value, str) and value.strip() == "sk-replace-me":
            msg = (
                "OPENAI_API_KEY is still the .env.example placeholder. Set a real key, "
                "or run locally with LLM_PROVIDER=ollama."
            )
            raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def _provider_requirements(self) -> Settings:
        """Fail fast when the selected provider is missing its credentials."""
        if self.llm_provider == "openai" and not self.openai_api_key:
            msg = (
                "LLM_PROVIDER=openai requires OPENAI_API_KEY. Set it in .env, "
                "or switch to LLM_PROVIDER=ollama for local inference."
            )
            raise ValueError(msg)
        return self

    @field_validator("documents_dir")
    @classmethod
    def _resolve_documents_dir(cls, value: Path) -> Path:
        return value if value.is_absolute() else (PROJECT_ROOT / value).resolve()

    # ----------------------------- Invariants ------------------------------ #
    @field_validator("chunk_overlap")
    @classmethod
    def _overlap_smaller_than_chunk(cls, value: int, info: Any) -> int:
        chunk_size = info.data.get("chunk_size")
        if chunk_size is not None and value >= chunk_size:
            msg = f"CHUNK_OVERLAP ({value}) must be smaller than CHUNK_SIZE ({chunk_size})"
            raise ValueError(msg)
        return value

    @field_validator("rerank_top_n")
    @classmethod
    def _top_n_within_candidates(cls, value: int, info: Any) -> int:
        candidates = info.data.get("retrieval_candidates")
        if candidates is not None and value > candidates:
            msg = f"RERANK_TOP_N ({value}) cannot exceed RETRIEVAL_CANDIDATES ({candidates})"
            raise ValueError(msg)
        return value

    # ---------------------------- Derived views ---------------------------- #
    @computed_field  # type: ignore[prop-decorator]
    @property
    def qdrant_url(self) -> str:
        return f"http://{self.qdrant_host}:{self.qdrant_port}"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def allowed_origins(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def llm_model(self) -> str:
        """Model id passed to the chat-completions API for the active provider."""
        return self.ollama_model if self.llm_provider == "ollama" else self.openai_model

    @computed_field  # type: ignore[prop-decorator]
    @property
    def llm_base_url(self) -> str | None:
        """Base URL for the active provider (None = the OpenAI default host)."""
        if self.llm_provider == "ollama":
            return self.ollama_base_url
        return self.openai_base_url

    # Deliberately NOT a computed field: computed fields are included in
    # `repr()` and `model_dump()`, which would print the API key in logs.
    @property
    def llm_api_key(self) -> str:
        """Credential for the active provider; Ollama needs a non-empty dummy."""
        if self.llm_provider == "ollama":
            return OLLAMA_PLACEHOLDER_KEY
        # `_provider_requirements` guarantees the key exists for the OpenAI path.
        return self.openai_api_key.get_secret_value() if self.openai_api_key else ""

    @computed_field  # type: ignore[prop-decorator]
    @property
    def langfuse_configured(self) -> bool:
        """Tracing is only attempted when a full credential pair is present."""
        return bool(self.langfuse_enabled and self.langfuse_public_key and self.langfuse_secret_key)

    def secret(self, value: SecretStr | None) -> str | None:
        """Unwrap an optional secret without sprinkling `get_secret_value()` everywhere."""
        return value.get_secret_value() if value is not None else None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton (cached; safe to call anywhere)."""
    return Settings()  # all values are sourced from the environment / .env


SettingsDep = Annotated[Settings, "injected application settings"]
