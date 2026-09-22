"""Two-stage retrieval: dense recall from Qdrant, cross-encoder precision on top.

Stage 1 (recall)    — a bi-encoder embeds the query once; Qdrant returns the
                      top-N nearest chunks by cosine similarity. Cheap, ANN,
                      O(log n): it can scan millions of vectors.
Stage 2 (precision) — a cross-encoder re-scores every (query, passage) pair
                      with full cross-attention and keeps the best `top_n`.
                      Expensive, O(candidates), but dramatically better at
                      separating "lexically close" from "actually answers it".

Both models are CPU-bound PyTorch graphs, so every call into them is pushed
onto a worker thread via `asyncio.to_thread` — the event loop never blocks.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from qdrant_client import AsyncQdrantClient

from src.config import Settings
from src.models import RetrievedChunk

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps import time low
    from sentence_transformers import CrossEncoder, SentenceTransformer

logger = logging.getLogger(__name__)

PAYLOAD_TEXT_KEY = "text"
PAYLOAD_SOURCE_KEY = "source"
PAYLOAD_PAGE_KEY = "page"
PAYLOAD_CHUNK_KEY = "chunk_index"


# --------------------------------------------------------------------------- #
# Structural typing — lets tests inject trivial fakes without importing torch.
# --------------------------------------------------------------------------- #
@runtime_checkable
class TextEncoder(Protocol):
    """Minimal surface of `sentence_transformers.SentenceTransformer` we rely on.

    The signature is intentionally permissive: `sentence-transformers` renames
    and overloads these parameters between minor releases, and we only ever
    call them with the keywords below.
    """

    def encode(self, *args: Any, **kwargs: Any) -> Any: ...


@runtime_checkable
class PairScorer(Protocol):
    """Minimal surface of `sentence_transformers.CrossEncoder` we rely on."""

    def predict(self, *args: Any, **kwargs: Any) -> Any: ...


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """Outcome of a two-stage retrieval pass, with per-stage timings."""

    chunks: list[RetrievedChunk]
    candidates_considered: int
    dense_ms: float
    rerank_ms: float

    @property
    def total_ms(self) -> float:
        return self.dense_ms + self.rerank_ms


class RetrievalError(RuntimeError):
    """Raised when the vector store cannot serve a query."""


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
def resolve_device(preferred: str | None = None) -> str:
    """Pick an inference device, honouring an explicit override.

    Apple Silicon (`mps`) is deliberately *not* auto-selected for these small
    models: the host-to-GPU copy dominates the matmul at this batch size, and
    MPS lacks kernels for a few transformer ops. `MODEL_DEVICE=mps` forces it.
    """
    if preferred:
        return preferred
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a hard dependency
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_embedding_model(settings: Settings) -> SentenceTransformer:
    """Load the bi-encoder used for both ingestion and query embedding."""
    from sentence_transformers import SentenceTransformer

    device = resolve_device(settings.model_device)
    logger.info("Loading embedding model %s on %s", settings.embedding_model, device)
    model = SentenceTransformer(settings.embedding_model, device=device)
    # `get_sentence_embedding_dimension` was renamed in sentence-transformers 6;
    # support both so the service runs on either major version.
    read_dimension = getattr(model, "get_embedding_dimension", None) or (
        model.get_sentence_embedding_dimension
    )
    dimension = int(read_dimension() or 0)
    if dimension != settings.embedding_dimension:
        msg = (
            f"EMBEDDING_DIMENSION={settings.embedding_dimension} does not match "
            f"{settings.embedding_model} (actual: {dimension}). Fix the config or the "
            f"collection will be built with the wrong vector size."
        )
        raise ValueError(msg)
    return model


def load_reranker(settings: Settings) -> CrossEncoder:
    """Load the cross-encoder used for stage-2 reranking."""
    from sentence_transformers import CrossEncoder

    device = resolve_device(settings.model_device)
    logger.info("Loading reranker %s on %s", settings.reranker_model, device)
    model: CrossEncoder = CrossEncoder(
        settings.reranker_model,
        max_length=settings.reranker_max_length,
        device=device,
    )
    return model


def build_qdrant_client(settings: Settings) -> AsyncQdrantClient:
    """Construct the async Qdrant client from settings."""
    return AsyncQdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        api_key=settings.secret(settings.qdrant_api_key),
        timeout=int(settings.qdrant_timeout_seconds),
        prefer_grpc=False,
    )


# --------------------------------------------------------------------------- #
# Retriever
# --------------------------------------------------------------------------- #
class TwoStageRetriever:
    """Dense retrieval followed by cross-encoder reranking.

    The class owns no lifecycle: clients and models are injected, which keeps
    it trivially unit-testable and lets the FastAPI lifespan decide when the
    expensive objects are created and torn down.
    """

    def __init__(
        self,
        *,
        client: AsyncQdrantClient,
        encoder: TextEncoder,
        reranker: PairScorer,
        settings: Settings,
    ) -> None:
        self._client = client
        self._encoder = encoder
        self._reranker = reranker
        self._settings = settings

    # ------------------------------ Stage 1 -------------------------------- #
    async def embed_query(self, query: str) -> list[float]:
        """Embed a single query off the event loop, L2-normalised for cosine."""
        vector = await asyncio.to_thread(
            self._encoder.encode,
            query,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [float(value) for value in vector]

    async def _dense_search(self, vector: list[float], limit: int) -> list[RetrievedChunk]:
        try:
            response = await self._client.query_points(
                collection_name=self._settings.qdrant_collection,
                query=vector,
                limit=limit,
                with_payload=True,
                score_threshold=self._settings.score_threshold,
            )
        except Exception as exc:
            msg = f"Qdrant query failed for collection '{self._settings.qdrant_collection}'"
            raise RetrievalError(msg) from exc

        chunks: list[RetrievedChunk] = []
        for point in response.points:
            payload: dict[str, Any] = dict(point.payload or {})
            text = payload.get(PAYLOAD_TEXT_KEY)
            source = payload.get(PAYLOAD_SOURCE_KEY)
            page = payload.get(PAYLOAD_PAGE_KEY)
            if not text or not source or page is None:
                logger.warning("Skipping point %s: incomplete payload", point.id)
                continue
            chunks.append(
                RetrievedChunk(
                    text=str(text),
                    source=str(source),
                    page=int(page),
                    chunk_index=int(payload.get(PAYLOAD_CHUNK_KEY, 0)),
                    dense_score=float(point.score),
                    rerank_score=0.0,
                )
            )
        return chunks

    # ------------------------------ Stage 2 -------------------------------- #
    async def _rerank(
        self, query: str, candidates: list[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]:
        """Score every (query, passage) pair and keep the best `top_n`."""
        pairs = [[query, chunk.text] for chunk in candidates]
        scores = await asyncio.to_thread(
            self._reranker.predict,
            pairs,
            batch_size=min(len(pairs), self._settings.embedding_batch_size),
            show_progress_bar=False,
        )
        rescored = [
            chunk.model_copy(update={"rerank_score": float(score)})
            for chunk, score in zip(candidates, scores, strict=True)
        ]
        rescored.sort(key=lambda chunk: chunk.rerank_score, reverse=True)
        return rescored[:top_n]

    # ------------------------------- Public -------------------------------- #
    async def retrieve(
        self,
        query: str,
        *,
        top_n: int | None = None,
        candidates: int | None = None,
    ) -> RetrievalResult:
        """Run dense recall then cross-encoder reranking for `query`."""
        limit = candidates or self._settings.retrieval_candidates
        keep = top_n or self._settings.rerank_top_n
        # Never rerank fewer candidates than we intend to return.
        limit = max(limit, keep)

        dense_started = time.perf_counter()
        vector = await self.embed_query(query)
        dense_candidates = await self._dense_search(vector, limit)
        dense_ms = (time.perf_counter() - dense_started) * 1000

        if not dense_candidates:
            logger.info("Dense retrieval returned no candidates for query=%r", query[:80])
            return RetrievalResult([], 0, dense_ms, 0.0)

        rerank_started = time.perf_counter()
        reranked = await self._rerank(query, dense_candidates, keep)
        rerank_ms = (time.perf_counter() - rerank_started) * 1000

        logger.debug(
            "Retrieved %d/%d chunks (dense %.1fms, rerank %.1fms)",
            len(reranked),
            len(dense_candidates),
            dense_ms,
            rerank_ms,
        )
        return RetrievalResult(reranked, len(dense_candidates), dense_ms, rerank_ms)

    # ------------------------------ Health --------------------------------- #
    async def collection_points(self) -> int | None:
        """Return the number of indexed points, or None when unavailable."""
        try:
            info = await self._client.get_collection(self._settings.qdrant_collection)
        except Exception as exc:  # noqa: BLE001 - a health probe must never raise
            logger.warning("Qdrant health probe failed: %s", exc)
            return None
        return int(info.points_count or 0)

    async def close(self) -> None:
        """Release the underlying HTTP connection pool."""
        await self._client.close()


def build_context_block(chunks: list[RetrievedChunk]) -> str:
    """Render retrieved chunks as an enumerated, citable context block.

    Each passage is fenced and labelled so the model can cite `[1]`, `[2]`…
    and so that untrusted document text is visually separated from the
    instruction channel (see the system prompt in `src.main`).
    """
    blocks = []
    for index, chunk in enumerate(chunks, start=1):
        blocks.append(
            f"[{index}] (source: {chunk.source}, page: {chunk.page})\n{chunk.text.strip()}"
        )
    return "\n\n".join(blocks)
