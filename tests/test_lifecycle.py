"""Process lifecycle: model loading, client wiring, startup/shutdown, tracing setup.

The HTTP tests inject a ready-made `AppContext`; these cover the code that
builds and tears it down. `sentence_transformers` and `langfuse` are replaced
by stand-in modules, so nothing is downloaded and torch is never imported.
"""

from __future__ import annotations

import logging
import sys
import types
from collections.abc import Iterator
from typing import Any

import httpx
import pytest
from pydantic import SecretStr
from src.config import Settings
from src.main import AppContext, Span, Tracing, configure_logging, create_app, lifespan
from src.retriever import build_qdrant_client, load_embedding_model, load_reranker

from .conftest import (
    FakeEncoder,
    FakeOpenAIClient,
    FakeQdrantClient,
    FakeReranker,
    parse_sse,
)


# --------------------------------------------------------------------------- #
# Model loaders
# --------------------------------------------------------------------------- #
class _FakeSentenceTransformer:
    dimension = 384

    def __init__(self, name: str, *, device: str) -> None:
        self.name = name
        self.device = device

    def get_embedding_dimension(self) -> int:
        return self.dimension

    def encode(self, *args: Any, **kwargs: Any) -> Any:
        return [0.0] * self.dimension


class _LegacySentenceTransformer(_FakeSentenceTransformer):
    """sentence-transformers < 6 only had the old method name."""

    get_embedding_dimension = None  # type: ignore[assignment]

    def get_sentence_embedding_dimension(self) -> int:
        return self.dimension


class _FakeCrossEncoder:
    def __init__(self, name: str, *, max_length: int, device: str) -> None:
        self.name = name
        self.max_length = max_length
        self.device = device

    def predict(self, *args: Any, **kwargs: Any) -> Any:
        return []


def _install_sentence_transformers(
    monkeypatch: pytest.MonkeyPatch, encoder: type[_FakeSentenceTransformer]
) -> None:
    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = encoder  # type: ignore[attr-defined]
    module.CrossEncoder = _FakeCrossEncoder  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)


@pytest.fixture
def cpu_settings(settings: Settings) -> Settings:
    return settings.model_copy(update={"model_device": "cpu"})


@pytest.mark.parametrize("encoder", [_FakeSentenceTransformer, _LegacySentenceTransformer])
def test_embedding_model_loads_on_the_configured_device(
    monkeypatch: pytest.MonkeyPatch,
    cpu_settings: Settings,
    encoder: type[_FakeSentenceTransformer],
) -> None:
    _install_sentence_transformers(monkeypatch, encoder)

    model: Any = load_embedding_model(cpu_settings)

    assert model.name == cpu_settings.embedding_model
    assert model.device == "cpu"


def test_embedding_dimension_mismatch_fails_at_load_time(
    monkeypatch: pytest.MonkeyPatch, cpu_settings: Settings
) -> None:
    _install_sentence_transformers(monkeypatch, _FakeSentenceTransformer)

    with pytest.raises(ValueError, match="EMBEDDING_DIMENSION=768 does not match"):
        load_embedding_model(cpu_settings.model_copy(update={"embedding_dimension": 768}))


def test_reranker_honours_max_length_and_device(
    monkeypatch: pytest.MonkeyPatch, cpu_settings: Settings
) -> None:
    _install_sentence_transformers(monkeypatch, _FakeSentenceTransformer)

    model: Any = load_reranker(cpu_settings)

    assert (model.name, model.max_length, model.device) == (
        cpu_settings.reranker_model,
        cpu_settings.reranker_max_length,
        "cpu",
    )


def test_qdrant_client_is_built_from_settings(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    # The real constructor probes the server version over HTTP, so record the
    # arguments instead. Runtime protocol conformance of the real client is
    # proven by tests/test_pipeline_local_qdrant.py.
    recorded: dict[str, Any] = {}
    monkeypatch.setattr("src.retriever.AsyncQdrantClient", lambda **kwargs: recorded.update(kwargs))
    secured = settings.model_copy(
        update={"qdrant_api_key": SecretStr("qdrant-secret"), "qdrant_timeout_seconds": 7.9}
    )

    build_qdrant_client(secured)

    assert recorded == {
        "host": secured.qdrant_host,
        "port": secured.qdrant_port,
        "api_key": "qdrant-secret",
        "timeout": 7,
        "prefer_grpc": False,
    }


# --------------------------------------------------------------------------- #
# Lifespan
# --------------------------------------------------------------------------- #
async def test_lifespan_wires_the_context_and_releases_every_client(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    qdrant = FakeQdrantClient()
    llm = FakeOpenAIClient()
    logged: list[Settings] = []
    monkeypatch.setattr("src.main.get_settings", lambda: settings)
    monkeypatch.setattr("src.main.configure_logging", logged.append)
    monkeypatch.setattr("src.main.load_embedding_model", lambda _s: FakeEncoder())
    monkeypatch.setattr("src.main.load_reranker", lambda _s: FakeReranker())
    monkeypatch.setattr("src.main.build_qdrant_client", lambda _s: qdrant)
    monkeypatch.setattr("src.main.build_llm_client", lambda _s: llm)
    app = create_app(settings)

    async with lifespan(app):
        context = app.state.context
        assert isinstance(context, AppContext)
        assert context.tracing.enabled is False
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            assert (await client.get("/health")).json()["status"] == "ok"

    assert logged == [settings]
    assert qdrant.closed and llm.closed
    assert app.state.context is None


def test_unloadable_model_aborts_startup(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    monkeypatch.setattr("src.main.get_settings", lambda: settings)
    monkeypatch.setattr("src.main.configure_logging", lambda _s: None)
    monkeypatch.setattr("src.main.load_embedding_model", lambda _s: settings.embedding_model)
    monkeypatch.setattr("src.main.load_reranker", lambda _s: FakeReranker())
    monkeypatch.setattr("src.main.build_qdrant_client", lambda _s: FakeQdrantClient())
    monkeypatch.setattr("src.main.build_llm_client", lambda _s: FakeOpenAIClient())

    async def start() -> None:
        async with lifespan(create_app(settings)):
            pass  # pragma: no cover - startup must fail first

    import asyncio

    with pytest.raises(ValueError, match="expected a loaded model"):
        asyncio.run(start())


@pytest.fixture
def restore_root_logging() -> Iterator[None]:
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield
    root.handlers[:] = handlers
    root.setLevel(level)


@pytest.mark.usefixtures("restore_root_logging")
def test_configure_logging_applies_the_level_and_quiets_httpx(settings: Settings) -> None:
    configure_logging(settings.model_copy(update={"log_level": "DEBUG"}))

    assert logging.getLogger().level == logging.DEBUG
    assert logging.getLogger("httpx").level == logging.WARNING


# --------------------------------------------------------------------------- #
# Tracing setup and failure isolation
# --------------------------------------------------------------------------- #
class _RecordingLangfuse:
    instances: list[_RecordingLangfuse] = []  # noqa: RUF012 - test-local registry

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.flushed = self.shut_down = False
        _RecordingLangfuse.instances.append(self)

    def flush(self) -> None:
        self.flushed = True

    def shutdown(self) -> None:
        self.shut_down = True


def _install_langfuse(monkeypatch: pytest.MonkeyPatch, factory: Any) -> None:
    module = types.ModuleType("langfuse")
    module.Langfuse = factory  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "langfuse", module)


@pytest.fixture
def traced_settings(settings: Settings) -> Settings:
    return settings.model_copy(
        update={
            "langfuse_enabled": True,
            "langfuse_public_key": SecretStr("pk-lf-test"),
            "langfuse_secret_key": SecretStr("sk-lf-test"),
        }
    )


async def test_tracing_starts_with_credentials_and_flushes_on_shutdown(
    monkeypatch: pytest.MonkeyPatch, traced_settings: Settings
) -> None:
    _RecordingLangfuse.instances.clear()
    _install_langfuse(monkeypatch, _RecordingLangfuse)

    tracing = Tracing.from_settings(traced_settings)
    await tracing.shutdown()

    assert tracing.enabled
    client = _RecordingLangfuse.instances[0]
    assert client.kwargs["public_key"] == "pk-lf-test"
    assert client.kwargs["environment"] == traced_settings.environment
    assert client.flushed and client.shut_down


def test_tracing_initialisation_failure_degrades_to_untraced(
    monkeypatch: pytest.MonkeyPatch, traced_settings: Settings
) -> None:
    def broken(**_kwargs: Any) -> Any:
        msg = "invalid host"
        raise RuntimeError(msg)

    _install_langfuse(monkeypatch, broken)

    assert Tracing.from_settings(traced_settings).enabled is False


class _ExplodingObservation:
    trace_id = "trace-x"

    def start_observation(self, **_kwargs: Any) -> Any:
        raise RuntimeError

    def update(self, **_kwargs: Any) -> None:
        raise RuntimeError

    def end(self) -> None:
        raise RuntimeError


def test_span_swallows_every_exporter_failure() -> None:
    span = Span(_ExplodingObservation())

    child = span.child("retrieval")
    span.update(output="x")
    span.end()

    assert span.trace_id == "trace-x"
    assert child.trace_id is None  # degraded to a no-op span


async def test_stream_maps_an_unexpected_failure_to_a_terminal_error_event(
    client: httpx.AsyncClient, openai_client: FakeOpenAIClient
) -> None:
    openai_client.stream_error = RuntimeError("socket closed: secret detail")

    async with client.stream("POST", "/rag/stream", json={"query": "notice period"}) as response:
        body = "".join([chunk async for chunk in response.aiter_text()])

    name, data = parse_sse(body)[-1]
    assert name == "error"
    assert '"internal_error"' in data
    assert "secret detail" not in body
