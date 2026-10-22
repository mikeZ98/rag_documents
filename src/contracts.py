"""Stage-transition contracts: what each pipeline stage hands to the next.

`src.models` owns the *wire* format; this module owns the *internal* hand-offs:

    ingestion   extract  ──DocumentExtraction──▶ chunk ──IndexBatch──▶ upsert
    serving     retrieve ──RetrievalResult─────▶ generate

Every contract is `extra="forbid"`, frozen and strict, and validates the
invariants the next stage silently relies on (vector width, rerank order,
page provenance). A violation is a bug in the producing stage, so it fails
loudly at the boundary rather than surfacing later as a Qdrant 400 or a
mis-cited answer. Foreign runtime objects (models, SDK clients) cross the
retriever boundary as `InstanceOf[...]` of the protocols declared here.
"""

from __future__ import annotations

import math
import uuid
from typing import Annotated, Any, Protocol, Self, runtime_checkable

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    InstanceOf,
    StringConstraints,
    field_validator,
    model_validator,
)
from qdrant_client.http import models as rest

from src.models import Ranking, RetrievedChunk

# uuid5 namespace for this project — keeps generated point ids stable across runs.
POINT_NAMESPACE = uuid.UUID("6f1c2d3e-8a4b-5c6d-9e0f-1a2b3c4d5e6f")

NonEmptyStr = Annotated[str, StringConstraints(min_length=1)]


class StageContract(BaseModel):
    """Base for every internal stage hand-off."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


# --------------------------------------------------------------------------- #
# Runtime collaborators — structural, so tests inject fakes without torch.
# --------------------------------------------------------------------------- #
@runtime_checkable
class TextEncoder(Protocol):
    """Minimal surface of `sentence_transformers.SentenceTransformer` we rely on.

    The signature is intentionally permissive: `sentence-transformers` renames
    and overloads these parameters between minor releases, and we only ever
    call them with the keywords used in `src.retriever` / `src.ingest`.
    """

    def encode(self, *args: Any, **kwargs: Any) -> Any: ...


@runtime_checkable
class PairScorer(Protocol):
    """Minimal surface of `sentence_transformers.CrossEncoder` we rely on."""

    def predict(self, *args: Any, **kwargs: Any) -> Any: ...


class CollectionStats(Protocol):
    """The part of Qdrant's `CollectionInfo` the health probe reads."""

    @property
    def points_count(self) -> int | None: ...


@runtime_checkable
class VectorStore(Protocol):
    """The slice of `AsyncQdrantClient` the retriever uses — and all a fake must implement."""

    async def query_points(
        self,
        collection_name: str,
        query: list[float],
        *,
        limit: int,
        with_payload: bool,
        score_threshold: float | None,
    ) -> rest.QueryResponse: ...

    async def get_collection(self, collection_name: str) -> CollectionStats: ...

    async def close(self) -> None: ...


class RetrieverComponents(StageContract):
    """What a `TwoStageRetriever` is wired with, checked at construction."""

    store: InstanceOf[VectorStore]
    encoder: InstanceOf[TextEncoder]
    reranker: InstanceOf[PairScorer]

    @field_validator("encoder", "reranker", mode="before")
    @classmethod
    def _not_a_model_id(cls, value: object) -> object:
        # `str.encode` satisfies `TextEncoder` structurally, so passing the model
        # *name* instead of the loaded model would otherwise slip through.
        if isinstance(value, str | bytes):
            msg = f"expected a loaded model, got the identifier {value!r}"
            raise ValueError(msg)
        return value


# --------------------------------------------------------------------------- #
# Ingestion
# --------------------------------------------------------------------------- #
class PageText(StageContract):
    """Normalised text of a single PDF page."""

    source: NonEmptyStr
    page: int = Field(ge=1, description="1-indexed, as printed in the document.")
    text: NonEmptyStr


class Chunk(StageContract):
    """A splitter-produced passage with full provenance."""

    text: NonEmptyStr
    source: NonEmptyStr
    page: int = Field(ge=1)
    chunk_index: int = Field(ge=0)

    @property
    def point_id(self) -> str:
        """Deterministic id so repeated ingestion updates rather than duplicates."""
        return str(uuid.uuid5(POINT_NAMESPACE, f"{self.source}:{self.page}:{self.chunk_index}"))


class DocumentExtraction(StageContract):
    """Extraction → chunking: everything one PDF contributed, and what failed."""

    source: NonEmptyStr
    pages: tuple[PageText, ...] = ()
    failed_pages: int = Field(default=0, ge=0)
    status: NonEmptyStr = "ok"

    @model_validator(mode="after")
    def _pages_belong_to_this_document(self) -> Self:
        numbers = [page.page for page in self.pages]
        if any(page.source != self.source for page in self.pages):
            msg = f"every page of {self.source!r} must carry that source"
            raise ValueError(msg)
        if numbers != sorted(set(numbers)):
            msg = f"page numbers of {self.source!r} must be unique and ascending: {numbers}"
            raise ValueError(msg)
        return self


class IndexBatch(StageContract):
    """Embedding → upsert: chunks paired one-to-one with finite, fixed-width vectors."""

    chunks: tuple[Chunk, ...]
    vectors: tuple[tuple[float, ...], ...]
    dimension: int = Field(ge=1)

    @model_validator(mode="after")
    def _vectors_match_chunks_and_collection(self) -> Self:
        if len(self.vectors) != len(self.chunks):
            msg = f"{len(self.chunks)} chunks but {len(self.vectors)} vectors"
            raise ValueError(msg)
        for index, vector in enumerate(self.vectors):
            if len(vector) != self.dimension:
                msg = (
                    f"vector {index} has {len(vector)} dimensions; the collection expects "
                    f"{self.dimension} (EMBEDDING_DIMENSION)"
                )
                raise ValueError(msg)
            if not all(math.isfinite(value) for value in vector):
                msg = f"vector {index} contains NaN or infinity"
                raise ValueError(msg)
        ids = [chunk.point_id for chunk in self.chunks]
        if len(set(ids)) != len(ids):
            msg = "duplicate (source, page, chunk_index) would overwrite points in one batch"
            raise ValueError(msg)
        return self


# --------------------------------------------------------------------------- #
# Serving
# --------------------------------------------------------------------------- #
class RetrievalResult(StageContract):
    """Retrieval → generation: ranked passages plus per-stage timings."""

    chunks: tuple[RetrievedChunk, ...]
    candidates_considered: int = Field(ge=0)
    dense_ms: float = Field(ge=0)
    rerank_ms: float = Field(ge=0)
    ranking: Ranking = "cross_encoder"

    @model_validator(mode="after")
    def _ranked_subset_of_candidates(self) -> Self:
        if len(self.chunks) > self.candidates_considered:
            msg = (
                f"{len(self.chunks)} chunks cannot come from "
                f"{self.candidates_considered} candidates"
            )
            raise ValueError(msg)
        by_dense = self.ranking == "dense_fallback"
        scores = [chunk.dense_score if by_dense else chunk.rerank_score for chunk in self.chunks]
        if scores != sorted(scores, reverse=True):
            kind = "dense" if by_dense else "rerank"
            msg = f"chunks must be ordered by descending {kind} score (citation [1] = best)"
            raise ValueError(msg)
        return self

    @property
    def total_ms(self) -> float:
        return self.dense_ms + self.rerank_ms
