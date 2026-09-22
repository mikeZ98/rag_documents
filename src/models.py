"""Pydantic v2 schemas: the public contract of the service.

Everything crossing the process boundary — HTTP request bodies, HTTP
responses and SSE event payloads — is declared here so the wire format has a
single source of truth (and FastAPI can derive an accurate OpenAPI document).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

# Query text: trimmed, non-trivial, and bounded to keep the embedding cost and
# the prompt-injection surface predictable.
QueryText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=3, max_length=500),
]


class SSEEventType(StrEnum):
    """Names of the Server-Sent Events emitted by `POST /rag/stream`."""

    SOURCES = "sources"
    TOKEN = "token"
    DONE = "done"
    ERROR = "error"


# --------------------------------------------------------------------------- #
# Requests
# --------------------------------------------------------------------------- #
class QueryRequest(BaseModel):
    """Inbound RAG question."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "query": "What is the notice period for terminating the agreement?",
                    "k": 3,
                    "session_id": "demo-session-1",
                }
            ]
        },
    )

    query: QueryText = Field(description="Natural-language question, 3-500 characters.")
    k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of reranked passages to ground the answer in.",
    )
    session_id: str | None = Field(
        default=None,
        max_length=128,
        description="Client-supplied conversation id; forwarded to Langfuse as session_id.",
    )


# --------------------------------------------------------------------------- #
# Retrieval results
# --------------------------------------------------------------------------- #
class RetrievedChunk(BaseModel):
    """A passage that survived both retrieval stages."""

    model_config = ConfigDict(frozen=True)

    text: str = Field(description="Verbatim chunk text used to ground the answer.")
    source: str = Field(description="Originating file name, e.g. `handbook.pdf`.")
    page: int = Field(ge=1, description="1-indexed page number inside the source PDF.")
    chunk_index: int = Field(ge=0, description="Position of the chunk within its page.")
    dense_score: float = Field(description="Stage-1 cosine similarity from Qdrant.")
    rerank_score: float = Field(description="Stage-2 cross-encoder relevance logit.")

    def citation(self) -> str:
        """Short human/LLM-readable citation label."""
        return f"{self.source} p.{self.page}"


class SourceReference(BaseModel):
    """Chunk projection sent to clients (text truncated to a preview)."""

    source: str
    page: int
    chunk_index: int
    dense_score: float
    rerank_score: float
    preview: str = Field(description="First 280 characters of the grounding passage.")

    @classmethod
    def from_chunk(cls, chunk: RetrievedChunk, preview_chars: int = 280) -> SourceReference:
        preview = chunk.text.strip()
        if len(preview) > preview_chars:
            preview = preview[:preview_chars].rstrip() + "…"
        return cls(
            source=chunk.source,
            page=chunk.page,
            chunk_index=chunk.chunk_index,
            dense_score=round(chunk.dense_score, 6),
            rerank_score=round(chunk.rerank_score, 6),
            preview=preview,
        )


# --------------------------------------------------------------------------- #
# Responses
# --------------------------------------------------------------------------- #
class TokenUsage(BaseModel):
    """LLM token accounting, mirrored into the Langfuse generation span."""

    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class SearchResponse(BaseModel):
    """Result of retrieval-only calls (`POST /rag/search`)."""

    query: str
    sources: list[SourceReference]
    candidates_considered: int = Field(
        ge=0, description="How many dense candidates the reranker scored."
    )
    retrieval_ms: float = Field(ge=0)


class AnswerResponse(BaseModel):
    """Result of buffered generation (`POST /rag/answer`)."""

    query: str
    answer: str
    sources: list[SourceReference]
    usage: TokenUsage
    retrieval_ms: float = Field(ge=0)
    generation_ms: float = Field(ge=0)
    model: str
    trace_id: str | None = None


class HealthResponse(BaseModel):
    """Liveness/readiness payload (`GET /health`)."""

    status: Literal["ok", "degraded"]
    version: str
    environment: str
    collection: str
    collection_ready: bool
    indexed_points: int | None = None
    embedding_model: str
    reranker_model: str
    llm_provider: str
    llm_model: str
    llm_endpoint: str
    tracing_enabled: bool
    detail: str | None = None


class ErrorResponse(BaseModel):
    """Sanitised error body — internal exception details never reach the client."""

    error: str = Field(description="Stable, machine-readable error code.")
    detail: str = Field(description="Safe, human-readable explanation.")
    request_id: str | None = None


# --------------------------------------------------------------------------- #
# SSE event payloads
# --------------------------------------------------------------------------- #
class SourcesEvent(BaseModel):
    """First event of a stream: what the answer will be grounded in."""

    type: Literal[SSEEventType.SOURCES] = SSEEventType.SOURCES
    query: str
    sources: list[SourceReference]
    candidates_considered: int
    retrieval_ms: float
    trace_id: str | None = None


class TokenEvent(BaseModel):
    """One LLM delta."""

    type: Literal[SSEEventType.TOKEN] = SSEEventType.TOKEN
    text: str


class DoneEvent(BaseModel):
    """Terminal event of a successful stream."""

    type: Literal[SSEEventType.DONE] = SSEEventType.DONE
    usage: TokenUsage
    generation_ms: float
    finish_reason: str | None = None


class ErrorEvent(BaseModel):
    """Terminal event of a failed stream (HTTP status is already 200 by then)."""

    type: Literal[SSEEventType.ERROR] = SSEEventType.ERROR
    error: str
    detail: str
