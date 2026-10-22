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

Failure behaviour is deterministic and bounded:

* Transient vector-store errors (transport failures, HTTP 429/5xx) are retried
  `QDRANT_MAX_RETRIES` times with a fixed linear backoff, all inside the
  request's retrieval deadline; anything else fails fast as `RetrievalError`.
* A malformed point payload is skipped, never turned into a 500.
* If the cross-encoder raises or returns unusable scores, the passages are
  ranked by dense similarity instead and the result says so
  (`ranking="dense_fallback"`) — degraded, still grounded, never silent.
* Ties are broken by (dense score, source, page, chunk index), so the same
  index and query always yield the same passages in the same order.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as rest
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from src.config import Settings
from src.contracts import (
    PairScorer,
    RetrievalResult,
    RetrieverComponents,
    TextEncoder,
    VectorStore,
)
from src.models import Ranking, RetrievedChunk

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps import time low
    from sentence_transformers import CrossEncoder, SentenceTransformer

logger = logging.getLogger(__name__)

PAYLOAD_TEXT_KEY = "text"
PAYLOAD_SOURCE_KEY = "source"
PAYLOAD_PAGE_KEY = "page"
PAYLOAD_CHUNK_KEY = "chunk_index"


class RetrievalError(RuntimeError):
    """Raised when the vector store cannot serve a query."""


def is_transient(exc: BaseException) -> bool:
    """True for failures a retry can fix: transport errors, throttling, 5xx."""
    if isinstance(exc, UnexpectedResponse):
        return exc.status_code is not None and (exc.status_code == 429 or exc.status_code >= 500)
    return isinstance(exc, ResponseHandlingException | ConnectionError | TimeoutError)


def _rerank_order(chunk: RetrievedChunk) -> tuple[float, float, str, int, int]:
    return (-chunk.rerank_score, -chunk.dense_score, chunk.source, chunk.page, chunk.chunk_index)


def _dense_order(chunk: RetrievedChunk) -> tuple[float, str, int, int]:
    return (-chunk.dense_score, chunk.source, chunk.page, chunk.chunk_index)


def chunk_from_point(point: rest.ScoredPoint) -> RetrievedChunk | None:
    """Map a Qdrant hit onto a `RetrievedChunk`, or None if its payload is unusable."""
    payload: dict[str, Any] = dict(point.payload or {})
    text = payload.get(PAYLOAD_TEXT_KEY)
    source = payload.get(PAYLOAD_SOURCE_KEY)
    page = payload.get(PAYLOAD_PAGE_KEY)
    if not text or not source or page is None:
        logger.warning("Skipping point %s: incomplete payload", point.id)
        return None
    try:
        return RetrievedChunk(
            text=str(text),
            source=str(source),
            page=int(page),
            chunk_index=int(payload.get(PAYLOAD_CHUNK_KEY, 0)),
            dense_score=float(point.score),
            rerank_score=0.0,
        )
    except (TypeError, ValueError) as exc:  # pydantic's ValidationError is a ValueError
        logger.warning("Skipping point %s: malformed payload (%s)", point.id, exc)
        return None


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
        client: VectorStore,
        encoder: TextEncoder,
        reranker: PairScorer,
        settings: Settings,
    ) -> None:
        # Fail at wiring time (lifespan), not on the first query, if a loader
        # handed back something that is not a usable model or store.
        components = RetrieverComponents(store=client, encoder=encoder, reranker=reranker)
        self._client = components.store
        self._encoder = components.encoder
        self._reranker = components.reranker
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

    async def _query(self, vector: list[float], limit: int) -> rest.QueryResponse:
        """`query_points` with bounded, deterministic retries on transient failures."""
        collection = self._settings.qdrant_collection
        attempts = self._settings.qdrant_max_retries + 1
        attempt = 1
        while True:
            try:
                return await self._client.query_points(
                    collection_name=collection,
                    query=vector,
                    limit=limit,
                    with_payload=True,
                    score_threshold=self._settings.score_threshold,
                )
            except Exception as exc:
                if attempt >= attempts or not is_transient(exc):
                    msg = (
                        f"Qdrant query failed for collection '{collection}' "
                        f"after {attempt} attempt(s)"
                    )
                    raise RetrievalError(msg) from exc
                delay = self._settings.qdrant_retry_backoff_seconds * attempt
                logger.warning(
                    "Transient Qdrant failure (%s), retry %d/%d in %.2fs",
                    type(exc).__name__,
                    attempt,
                    attempts - 1,
                    delay,
                )
                await asyncio.sleep(delay)
                attempt += 1

    async def _dense_search(self, vector: list[float], limit: int) -> list[RetrievedChunk]:
        """Stage 1: nearest chunks, minus unusable payloads and duplicate passages."""
        response = await self._query(vector, limit)
        chunks: list[RetrievedChunk] = []
        seen: set[tuple[str, int, int]] = set()
        # Qdrant returns hits by descending score, so the first copy kept is the best.
        for point in response.points:
            chunk = chunk_from_point(point)
            if chunk is None:
                continue
            key = (chunk.source, chunk.page, chunk.chunk_index)
            if key in seen:
                logger.debug("Dropping duplicate passage %s p.%d #%d", *key)
                continue
            seen.add(key)
            chunks.append(chunk)
        return chunks

    # ------------------------------ Stage 2 -------------------------------- #
    async def _score_pairs(self, query: str, candidates: Sequence[RetrievedChunk]) -> list[float]:
        """Cross-encoder scores, one finite float per candidate — or ValueError."""
        pairs = [[query, chunk.text] for chunk in candidates]
        raw = await asyncio.to_thread(
            self._reranker.predict,
            pairs,
            batch_size=min(len(pairs), self._settings.embedding_batch_size),
            show_progress_bar=False,
        )
        scores = [float(score) for score in raw]
        if len(scores) != len(candidates):
            msg = f"reranker returned {len(scores)} scores for {len(candidates)} pairs"
            raise ValueError(msg)
        if not all(math.isfinite(score) for score in scores):
            msg = "reranker returned NaN or infinite scores"
            raise ValueError(msg)
        return scores

    async def _rerank(
        self, query: str, candidates: list[RetrievedChunk], top_n: int
    ) -> tuple[list[RetrievedChunk], Ranking]:
        """Keep the best `top_n` by cross-encoder score, falling back to dense order."""
        try:
            scores = await self._score_pairs(query, candidates)
        except Exception:
            # A broken reranker must not take retrieval down with it: the dense
            # ranking is a weaker but still relevant, grounded ordering.
            logger.warning("Cross-encoder failed; ranking by dense similarity", exc_info=True)
            return sorted(candidates, key=_dense_order)[:top_n], "dense_fallback"
        rescored = [
            chunk.model_copy(update={"rerank_score": score})
            for chunk, score in zip(candidates, scores, strict=True)
        ]
        rescored.sort(key=_rerank_order)
        return rescored[:top_n], "cross_encoder"

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
            return RetrievalResult(
                chunks=(), candidates_considered=0, dense_ms=dense_ms, rerank_ms=0.0
            )

        rerank_started = time.perf_counter()
        reranked, ranking = await self._rerank(query, dense_candidates, keep)
        rerank_ms = (time.perf_counter() - rerank_started) * 1000

        logger.debug(
            "Retrieved %d/%d chunks via %s (dense %.1fms, rerank %.1fms)",
            len(reranked),
            len(dense_candidates),
            ranking,
            dense_ms,
            rerank_ms,
        )
        return RetrievalResult(
            chunks=tuple(reranked),
            candidates_considered=len(dense_candidates),
            dense_ms=dense_ms,
            rerank_ms=rerank_ms,
            ranking=ranking,
        )

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


def build_context_block(chunks: Sequence[RetrievedChunk]) -> str:
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
