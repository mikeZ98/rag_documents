"""Configuration contract tests: fail fast, never hardcode secrets."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError
from src.config import Settings, get_settings


def _settings(**overrides: Any) -> Settings:
    defaults: dict[str, Any] = {"openai_api_key": "sk-unit-test"}
    return Settings(**{**defaults, **overrides})


def test_missing_api_key_fails_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_placeholder_api_key_is_rejected() -> None:
    with pytest.raises(ValidationError, match="placeholder"):
        _settings(openai_api_key="sk-replace-me")


def test_secrets_are_not_leaked_by_repr_or_dump() -> None:
    settings = _settings(llm_provider="openai", openai_api_key="sk-super-secret")

    # Neither the repr nor a serialised dump may carry the credential — the
    # resolved `llm_api_key` must stay a plain property, not a computed field.
    assert "sk-super-secret" not in repr(settings)
    assert "sk-super-secret" not in str(settings.model_dump())
    assert settings.llm_api_key == "sk-super-secret"
    assert settings.secret(settings.openai_api_key) == "sk-super-secret"
    assert settings.secret(None) is None


def test_empty_env_values_are_treated_as_unset() -> None:
    settings = _settings(qdrant_api_key="", openai_base_url="  ", score_threshold="")

    assert settings.qdrant_api_key is None
    assert settings.openai_base_url is None
    assert settings.score_threshold is None


def test_chunk_overlap_must_be_smaller_than_chunk_size() -> None:
    with pytest.raises(ValidationError, match="must be smaller than CHUNK_SIZE"):
        _settings(chunk_size=500, chunk_overlap=500)


def test_rerank_top_n_cannot_exceed_candidate_pool() -> None:
    with pytest.raises(ValidationError, match="cannot exceed RETRIEVAL_CANDIDATES"):
        _settings(retrieval_candidates=5, rerank_top_n=10)


def test_derived_values() -> None:
    settings = _settings(
        qdrant_host="vectors.internal",
        qdrant_port=6333,
        cors_origins="http://a.test, http://b.test ,",
    )

    assert settings.qdrant_url == "http://vectors.internal:6333"
    assert settings.allowed_origins == ["http://a.test", "http://b.test"]


def test_documents_dir_is_resolved_against_the_project_root() -> None:
    assert _settings(documents_dir="documents").documents_dir.is_absolute()


def test_tracing_requires_a_full_credential_pair() -> None:
    assert _settings(langfuse_enabled=True).langfuse_configured is False
    assert (
        _settings(
            langfuse_enabled=True, langfuse_public_key="pk", langfuse_secret_key="sk"
        ).langfuse_configured
        is True
    )
    assert (
        _settings(
            langfuse_enabled=False, langfuse_public_key="pk", langfuse_secret_key="sk"
        ).langfuse_configured
        is False
    )


def test_settings_singleton_is_cached() -> None:
    assert get_settings() is get_settings()


# --------------------------------------------------------------------------- #
# LLM provider toggle
# --------------------------------------------------------------------------- #
def test_ollama_is_the_default_provider_and_needs_no_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The suite pins LLM_PROVIDER=openai; drop it to observe the real default.
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    settings = Settings(_env_file=None, openai_api_key=None)

    assert settings.llm_provider == "ollama"
    assert settings.llm_model == "llama3.1:8b"
    assert settings.llm_base_url == "http://localhost:11434/v1"
    # The SDK refuses an empty credential, so a dummy is supplied for Ollama.
    assert settings.llm_api_key == "ollama"


def test_ollama_endpoint_and_model_are_configurable() -> None:
    settings = _settings(
        llm_provider="ollama",
        ollama_base_url="http://gpu-box.internal:11434/v1",
        ollama_model="mistral",
    )

    assert settings.llm_base_url == "http://gpu-box.internal:11434/v1"
    assert settings.llm_model == "mistral"


def test_openai_provider_resolves_to_openai_settings() -> None:
    settings = _settings(llm_provider="openai", openai_model="gpt-4o", openai_base_url=None)

    assert settings.llm_model == "gpt-4o"
    assert settings.llm_base_url is None  # the SDK falls back to the hosted API
    assert settings.llm_api_key == "sk-unit-test"


def test_openai_gateway_base_url_is_honoured() -> None:
    settings = _settings(llm_provider="openai", openai_base_url="https://gateway.corp/v1")

    assert settings.llm_base_url == "https://gateway.corp/v1"


def test_openai_provider_without_a_key_fails_fast() -> None:
    with pytest.raises(ValidationError, match="requires OPENAI_API_KEY"):
        Settings(_env_file=None, llm_provider="openai", openai_api_key=None)


def test_switching_to_ollama_tolerates_a_missing_openai_key() -> None:
    settings = Settings(_env_file=None, llm_provider="ollama", openai_api_key=None)

    assert settings.llm_api_key == "ollama"
    assert settings.openai_api_key is None
