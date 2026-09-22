"""Shared fixtures and in-memory doubles.

The suite never touches Qdrant, OpenAI, HuggingFace or Langfuse: the vector
store and the LLM are replaced by deterministic fakes, while the production
code paths (validation, two-stage retrieval, SSE framing, error mapping,
timeouts) run for real.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from typing import Any, cast

# Configuration must exist before `src.config` is imported anywhere. These are
# assignments, not `setdefault`: a developer's real OPENAI_API_KEY must never
# leak into the suite, and the provider under test must never depend on the
# machine's environment.
os.environ["OPENAI_API_KEY"] = "sk-test-not-a-real-key"
os.environ["LLM_PROVIDER"] = "openai"
os.environ["LANGFUSE_ENABLED"] = "false"
os.environ["QDRANT_COLLECTION"] = "test_collection"
os.environ["ENVIRONMENT"] = "local"
os.environ["LOG_LEVEL"] = "WARNING"

import httpx
import pytest
from openai import AsyncOpenAI
from pydantic import SecretStr
from qdrant_client import AsyncQdrantClient
from src.config import Settings
from src.main import AppContext, Tracing, create_app
from src.retriever import TwoStageRetriever

# --------------------------------------------------------------------------- #
# Corpus used by the fake vector store
# --------------------------------------------------------------------------- #
CORPUS: list[dict[str, Any]] = [
    {
        "text": "The agreement may be terminated by either party with a notice period "
        "of thirty (30) days, served in writing.",
        "source": "contract.pdf",
        "page": 4,
        "chunk_index": 0,
    },
    {
        "text": "Payment terms: invoices are due within fourteen (14) days of receipt.",
        "source": "contract.pdf",
        "page": 5,
        "chunk_index": 1,
    },
    {
        "text": "Annual leave entitlement is twenty-six (26) working days per calendar year.",
        "source": "handbook.pdf",
        "page": 12,
        "chunk_index": 0,
    },
    {
        "text": "Employees must report security incidents to the SOC within one hour.",
        "source": "handbook.pdf",
        "page": 31,
        "chunk_index": 2,
    },
]


# --------------------------------------------------------------------------- #
# Fake vector store
# --------------------------------------------------------------------------- #
@dataclass
class FakePoint:
    id: str
    score: float
    payload: dict[str, Any]


@dataclass
class FakeQueryResponse:
    points: list[FakePoint]


@dataclass
class FakeCollectionInfo:
    points_count: int


class FakeQdrantClient:
    """Minimal async stand-in for `AsyncQdrantClient`."""

    def __init__(
        self,
        corpus: list[dict[str, Any]] | None = None,
        *,
        points_count: int | None = 42,
        fail: Exception | None = None,
    ) -> None:
        self.corpus = corpus if corpus is not None else CORPUS
        self.points_count = points_count
        self.fail = fail
        self.calls: list[dict[str, Any]] = []
        self.closed = False

    async def query_points(self, **kwargs: Any) -> FakeQueryResponse:
        self.calls.append(kwargs)
        if self.fail is not None:
            raise self.fail
        limit = int(kwargs.get("limit", 10))
        points = [
            FakePoint(id=f"p{index}", score=round(0.9 - index * 0.05, 4), payload=dict(item))
            for index, item in enumerate(self.corpus[:limit])
        ]
        return FakeQueryResponse(points=points)

    async def get_collection(self, collection_name: str) -> FakeCollectionInfo:
        if self.points_count is None:
            msg = f"collection {collection_name} not found"
            raise RuntimeError(msg)
        return FakeCollectionInfo(points_count=self.points_count)

    async def close(self) -> None:
        self.closed = True


# --------------------------------------------------------------------------- #
# Fake models
# --------------------------------------------------------------------------- #
class FakeEncoder:
    """Deterministic 4-dimensional "embeddings" — no torch involved."""

    def __init__(self) -> None:
        self.calls: list[Any] = []

    def encode(self, sentences: Any, **kwargs: Any) -> Any:
        self.calls.append(sentences)
        if isinstance(sentences, str):
            return [float(len(sentences) % 7), 0.1, 0.2, 0.3]
        return [[float(len(text) % 7), 0.1, 0.2, 0.3] for text in sentences]


class FakeReranker:
    """Scores pairs by token overlap, so ordering is meaningful and testable."""

    def __init__(self) -> None:
        self.calls: list[Any] = []

    def predict(self, sentences: Any, **kwargs: Any) -> Any:
        self.calls.append(sentences)
        scores = []
        for query, passage in sentences:
            query_terms = {token.lower().strip(".,()") for token in query.split()}
            passage_terms = {token.lower().strip(".,()") for token in passage.split()}
            overlap = len(query_terms & passage_terms)
            scores.append(float(overlap) / max(len(query_terms), 1))
        return scores


# --------------------------------------------------------------------------- #
# Fake OpenAI client
# --------------------------------------------------------------------------- #
@dataclass
class FakeDelta:
    content: str | None = None


@dataclass
class FakeStreamChoice:
    delta: FakeDelta
    finish_reason: str | None = None


@dataclass
class FakeUsage:
    prompt_tokens: int = 128
    completion_tokens: int = 16


@dataclass
class FakeStreamChunk:
    choices: list[FakeStreamChoice] = field(default_factory=list)
    usage: FakeUsage | None = None


@dataclass
class FakeMessage:
    content: str


@dataclass
class FakeChoice:
    message: FakeMessage
    finish_reason: str = "stop"


@dataclass
class FakeCompletion:
    choices: list[FakeChoice]
    usage: FakeUsage | None = None


class FakeCompletions:
    def __init__(self, owner: FakeOpenAIClient) -> None:
        self._owner = owner

    async def create(self, **kwargs: Any) -> Any:
        self._owner.calls.append(kwargs)
        if self._owner.error is not None:
            raise self._owner.error
        if kwargs.get("stream"):
            return self._owner.stream()
        return FakeCompletion(
            choices=[FakeChoice(message=FakeMessage(content="".join(self._owner.tokens)))],
            usage=FakeUsage(),
        )


class FakeChat:
    def __init__(self, owner: FakeOpenAIClient) -> None:
        self.completions = FakeCompletions(owner)


class FakeOpenAIClient:
    """Async stand-in for `AsyncOpenAI` covering buffered and streamed calls."""

    def __init__(
        self,
        tokens: list[str] | None = None,
        *,
        error: Exception | None = None,
        stream_error: Exception | None = None,
        delay: float = 0.0,
    ) -> None:
        self.tokens = tokens or ["The notice ", "period is ", "30 days [1]."]
        self.error = error
        self.stream_error = stream_error
        self.delay = delay
        self.calls: list[dict[str, Any]] = []
        self.chat = FakeChat(self)
        self.closed = False

    async def stream(self) -> AsyncIterator[FakeStreamChunk]:
        import asyncio

        for token in self.tokens:
            if self.delay:
                await asyncio.sleep(self.delay)
            yield FakeStreamChunk(choices=[FakeStreamChoice(delta=FakeDelta(content=token))])
        if self.stream_error is not None:
            raise self.stream_error
        yield FakeStreamChunk(
            choices=[FakeStreamChoice(delta=FakeDelta(content=None), finish_reason="stop")]
        )
        yield FakeStreamChunk(choices=[], usage=FakeUsage())

    async def close(self) -> None:
        self.closed = True


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def _reset_sse_app_status() -> Iterator[None]:
    """sse-starlette caches a loop-bound exit event; clear it between tests."""
    from sse_starlette import sse as sse_module

    app_status = getattr(sse_module, "AppStatus", None)
    if app_status is not None:
        app_status.should_exit_event = None
    yield
    if app_status is not None:
        app_status.should_exit_event = None


@pytest.fixture
def settings() -> Settings:
    return Settings(
        llm_provider="openai",
        openai_api_key=SecretStr("sk-test-not-a-real-key"),
        qdrant_collection="test_collection",
        langfuse_enabled=False,
        retrieval_candidates=4,
        rerank_top_n=3,
        retrieval_timeout_seconds=5.0,
        generation_timeout_seconds=5.0,
        request_timeout_seconds=5.0,
    )


@pytest.fixture
def ollama_settings() -> Settings:
    """Settings for the default, key-less local provider."""
    return Settings(
        llm_provider="ollama",
        openai_api_key=None,
        ollama_base_url="http://localhost:11434/v1",
        ollama_model="llama3.1:8b",
        qdrant_collection="test_collection",
        langfuse_enabled=False,
        retrieval_candidates=4,
        rerank_top_n=3,
    )


@pytest.fixture
def qdrant() -> FakeQdrantClient:
    return FakeQdrantClient()


@pytest.fixture
def openai_client() -> FakeOpenAIClient:
    return FakeOpenAIClient()


@pytest.fixture
def retriever(settings: Settings, qdrant: FakeQdrantClient) -> TwoStageRetriever:
    return TwoStageRetriever(
        client=cast(AsyncQdrantClient, qdrant),
        encoder=FakeEncoder(),
        reranker=FakeReranker(),
        settings=settings,
    )


@pytest.fixture
def context(
    settings: Settings, retriever: TwoStageRetriever, openai_client: FakeOpenAIClient
) -> AppContext:
    return AppContext(
        settings=settings,
        retriever=retriever,
        openai=cast(AsyncOpenAI, openai_client),
        tracing=Tracing(None),
    )


@pytest.fixture
def client(settings: Settings, context: AppContext) -> Iterator[httpx.AsyncClient]:
    """ASGI client with the lifespan bypassed and the context injected."""
    app = create_app(settings)
    app.state.context = context
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    yield httpx.AsyncClient(transport=transport, base_url="http://testserver")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def parse_sse(body: str) -> list[tuple[str, str]]:
    """Parse a raw SSE body into `(event, data)` pairs."""
    events: list[tuple[str, str]] = []
    event_name = "message"
    data_lines: list[str] = []
    for line in body.splitlines():
        if line.startswith("event:"):
            event_name = line.removeprefix("event:").strip()
        elif line.startswith("data:"):
            data_lines.append(line.removeprefix("data:").strip())
        elif not line:
            if data_lines:
                events.append((event_name, "\n".join(data_lines)))
            event_name = "message"
            data_lines = []
    if data_lines:
        events.append((event_name, "\n".join(data_lines)))
    return events
