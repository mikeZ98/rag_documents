"""Unit tests for the two-stage retriever."""

from __future__ import annotations

import asyncio
import math
from typing import Any, cast

import httpx
import numpy as np
import pytest
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from src.config import Settings
from src.models import RetrievedChunk
from src.retriever import (
    RetrievalError,
    TwoStageRetriever,
    build_context_block,
    is_transient,
    resolve_device,
)

from .conftest import CORPUS, FakeEncoder, FakeQdrantClient, FakeReranker

# `asyncio_mode = "auto"` (pyproject) runs every async test without a marker.


def _retriever(
    settings: Settings, qdrant: FakeQdrantClient | None = None
) -> tuple[TwoStageRetriever, FakeQdrantClient, FakeReranker]:
    client = qdrant or FakeQdrantClient()
    reranker = FakeReranker()
    retriever = TwoStageRetriever(
        client=client,
        encoder=FakeEncoder(),
        reranker=reranker,
        settings=settings,
    )
    return retriever, client, reranker


async def test_retrieve_pulls_candidates_then_keeps_top_n(settings: Settings) -> None:
    settings.retrieval_candidates = 4
    retriever, client, reranker = _retriever(settings)

    result = await retriever.retrieve("notice period termination", top_n=2)

    assert len(result.chunks) == 2
    assert result.candidates_considered == 4
    # Stage 2 scored every candidate, not just the returned ones.
    assert len(reranker.calls[0]) == 4
    # Stage 1 asked Qdrant for the candidate pool with payloads.
    assert client.calls[0]["limit"] == 4
    assert client.calls[0]["with_payload"] is True
    assert client.calls[0]["collection_name"] == settings.qdrant_collection


async def test_results_are_sorted_by_rerank_score(settings: Settings) -> None:
    retriever, _, _ = _retriever(settings)

    result = await retriever.retrieve("annual leave entitlement days", top_n=3)

    scores = [chunk.rerank_score for chunk in result.chunks]
    assert scores == sorted(scores, reverse=True)
    # The handbook passage about leave must outrank the contract passages,
    # even though the dense fake ranked the contract first.
    assert result.chunks[0].source == "handbook.pdf"
    assert result.chunks[0].page == 12


async def test_dense_scores_are_preserved_alongside_rerank_scores(settings: Settings) -> None:
    retriever, _, _ = _retriever(settings)

    result = await retriever.retrieve("payment invoices due", top_n=3)

    assert all(chunk.dense_score > 0 for chunk in result.chunks)
    assert result.dense_ms >= 0
    assert result.rerank_ms >= 0
    assert result.total_ms == pytest.approx(result.dense_ms + result.rerank_ms)


async def test_candidate_pool_is_never_smaller_than_requested_k(settings: Settings) -> None:
    settings.retrieval_candidates = 2
    retriever, client, _ = _retriever(settings)

    await retriever.retrieve("any question", top_n=5)

    assert client.calls[0]["limit"] == 5


async def test_empty_index_short_circuits_the_reranker(settings: Settings) -> None:
    retriever, _, reranker = _retriever(settings, FakeQdrantClient(corpus=[]))

    result = await retriever.retrieve("anything", top_n=3)

    assert result.chunks == ()
    assert result.candidates_considered == 0
    assert result.rerank_ms == 0.0
    assert reranker.calls == []


async def test_points_with_incomplete_payload_are_skipped(settings: Settings) -> None:
    corpus = [
        {"text": "orphan chunk without provenance"},  # no source/page -> dropped
        *CORPUS[:1],
    ]
    retriever, _, _ = _retriever(settings, FakeQdrantClient(corpus=corpus))

    result = await retriever.retrieve("notice period", top_n=3)

    assert result.candidates_considered == 1
    assert result.chunks[0].source == "contract.pdf"


async def test_vector_store_failure_becomes_retrieval_error(settings: Settings) -> None:
    retriever, _, _ = _retriever(settings, FakeQdrantClient(fail=ConnectionError("down")))

    with pytest.raises(RetrievalError, match="Qdrant query failed"):
        await retriever.retrieve("anything")


async def test_query_embedding_is_normalised_and_offloaded(settings: Settings) -> None:
    retriever, _, _ = _retriever(settings)
    # Reaching into the private attribute is deliberate: we assert the call contract.
    encoder = cast(FakeEncoder, retriever._encoder)

    vector = await retriever.embed_query("hello world")

    assert all(isinstance(value, float) for value in vector)
    assert encoder.calls == ["hello world"]


async def test_inference_never_blocks_the_event_loop(settings: Settings) -> None:
    """A slow, blocking `predict` must not stall concurrent coroutines."""

    class BlockingReranker:
        def predict(self, sentences: Any, **kwargs: Any) -> Any:
            import time

            time.sleep(0.2)  # blocking, CPU-bound work
            return [0.5] * len(sentences)

    retriever = TwoStageRetriever(
        client=FakeQdrantClient(),
        encoder=FakeEncoder(),
        reranker=BlockingReranker(),
        settings=settings,
    )
    ticks = 0

    async def heartbeat() -> None:
        nonlocal ticks
        for _ in range(10):
            await asyncio.sleep(0.02)
            ticks += 1

    await asyncio.gather(retriever.retrieve("question"), heartbeat())

    assert ticks == 10  # the loop kept scheduling while the model was busy


async def test_collection_points_returns_none_when_unavailable(settings: Settings) -> None:
    retriever, _, _ = _retriever(settings, FakeQdrantClient(points_count=None))

    assert await retriever.collection_points() is None


async def test_close_releases_the_client(settings: Settings) -> None:
    retriever, client, _ = _retriever(settings)

    await retriever.close()

    assert client.closed is True


def test_context_block_is_enumerated_and_attributed() -> None:
    chunks = [
        RetrievedChunk(
            text="First passage.",
            source="a.pdf",
            page=1,
            chunk_index=0,
            dense_score=0.9,
            rerank_score=8.0,
        ),
        RetrievedChunk(
            text="Second passage.",
            source="b.pdf",
            page=7,
            chunk_index=3,
            dense_score=0.8,
            rerank_score=5.0,
        ),
    ]

    block = build_context_block(chunks)

    assert block.startswith("[1] (source: a.pdf, page: 1)")
    assert "[2] (source: b.pdf, page: 7)" in block
    assert chunks[1].citation() == "b.pdf p.7"


def test_context_block_is_empty_for_no_chunks() -> None:
    assert build_context_block([]) == ""


def test_explicit_device_overrides_autodetection() -> None:
    assert resolve_device("mps") == "mps"
    assert resolve_device(None) in {"cpu", "cuda"}


# --------------------------------------------------------------------------- #
# Vector-store retry policy
# --------------------------------------------------------------------------- #
def _unexpected(status: int) -> UnexpectedResponse:
    return UnexpectedResponse(
        status_code=status, reason_phrase="error", content=b"", headers=httpx.Headers()
    )


@pytest.mark.parametrize(
    ("exc", "transient"),
    [
        (ResponseHandlingException(ConnectionError("reset")), True),
        (ConnectionError("refused"), True),
        (TimeoutError(), True),
        (_unexpected(503), True),
        (_unexpected(429), True),
        (_unexpected(404), False),
        (_unexpected(400), False),
        (ValueError("bad vector"), False),
    ],
)
def test_transient_failures_are_classified(exc: Exception, *, transient: bool) -> None:
    assert is_transient(exc) is transient


async def test_transient_failures_are_retried_until_the_store_answers(
    settings: Settings,
) -> None:
    qdrant = FakeQdrantClient(
        failures=[ResponseHandlingException(ConnectionError("reset")), _unexpected(503)]
    )
    retriever, _, _ = _retriever(settings, qdrant)

    result = await retriever.retrieve("notice period", top_n=2)

    assert len(qdrant.calls) == 3
    assert len(result.chunks) == 2


async def test_retries_are_bounded_and_then_fail_as_retrieval_error(settings: Settings) -> None:
    settings.qdrant_max_retries = 2
    qdrant = FakeQdrantClient(failures=[_unexpected(503)] * 5)
    retriever, _, _ = _retriever(settings, qdrant)

    with pytest.raises(RetrievalError, match="after 3 attempt"):
        await retriever.retrieve("notice period")

    assert len(qdrant.calls) == 3


async def test_permanent_failures_are_not_retried(settings: Settings) -> None:
    qdrant = FakeQdrantClient(failures=[_unexpected(404)])
    retriever, _, _ = _retriever(settings, qdrant)

    with pytest.raises(RetrievalError, match="after 1 attempt"):
        await retriever.retrieve("notice period")

    assert len(qdrant.calls) == 1


async def test_backoff_is_linear_and_deterministic(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings.qdrant_max_retries = 3
    settings.qdrant_retry_backoff_seconds = 0.25
    delays: list[float] = []

    async def record(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr("src.retriever.asyncio.sleep", record)
    retriever, _, _ = _retriever(settings, FakeQdrantClient(failures=[ConnectionError()] * 3))

    await retriever.retrieve("notice period")

    assert delays == [0.25, 0.5, 0.75]


# --------------------------------------------------------------------------- #
# Reranker fallback
# --------------------------------------------------------------------------- #
class _BrokenReranker:
    def __init__(self, result: Any) -> None:
        self.result = result

    def predict(self, sentences: Any, **kwargs: Any) -> Any:
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


@pytest.mark.parametrize(
    "broken",
    [
        RuntimeError("CUDA out of memory"),
        [0.1, 0.2],  # fewer scores than pairs
        [0.1, math.nan, 0.3, 0.4],
        [0.1, 0.2, math.inf, 0.4],
        None,  # not iterable
    ],
    ids=["raises", "short", "nan", "inf", "none"],
)
async def test_unusable_reranker_falls_back_to_dense_order(settings: Settings, broken: Any) -> None:
    retriever = TwoStageRetriever(
        client=FakeQdrantClient(),
        encoder=FakeEncoder(),
        reranker=_BrokenReranker(broken),
        settings=settings,
    )

    result = await retriever.retrieve("annual leave entitlement days", top_n=3)

    assert result.ranking == "dense_fallback"
    assert result.candidates_considered == 4
    dense = [chunk.dense_score for chunk in result.chunks]
    assert dense == sorted(dense, reverse=True)
    assert [chunk.source for chunk in result.chunks] == [item["source"] for item in CORPUS[:3]]
    assert all(chunk.rerank_score == 0.0 for chunk in result.chunks)


async def test_working_reranker_reports_cross_encoder_ranking(settings: Settings) -> None:
    retriever, _, _ = _retriever(settings)

    result = await retriever.retrieve("annual leave entitlement days", top_n=3)

    assert result.ranking == "cross_encoder"


# --------------------------------------------------------------------------- #
# Determinism and payload hygiene
# --------------------------------------------------------------------------- #
class _ConstantReranker:
    def predict(self, sentences: Any, **kwargs: Any) -> Any:
        return [1.0] * len(sentences)


async def test_ties_break_identically_whatever_order_the_store_returns(
    settings: Settings,
) -> None:
    async def ranked(corpus: list[dict[str, Any]]) -> list[tuple[str, int]]:
        retriever = TwoStageRetriever(
            client=FakeQdrantClient(corpus=corpus, scores=[0.5] * len(corpus)),
            encoder=FakeEncoder(),
            reranker=_ConstantReranker(),
            settings=settings,
        )
        result = await retriever.retrieve("anything", top_n=4)
        return [(chunk.source, chunk.page) for chunk in result.chunks]

    forward = await ranked(list(CORPUS))
    backward = await ranked(list(reversed(CORPUS)))

    assert forward == backward
    assert forward == sorted(forward)  # full tie -> provenance order


async def test_duplicate_passages_keep_only_the_best_hit(settings: Settings) -> None:
    corpus = [CORPUS[0], CORPUS[1], dict(CORPUS[0])]
    retriever, _, reranker = _retriever(settings, FakeQdrantClient(corpus=corpus))

    result = await retriever.retrieve("notice period", top_n=3)

    assert result.candidates_considered == 2
    assert len(reranker.calls[0]) == 2
    assert result.chunks[0].dense_score == pytest.approx(0.9)


@pytest.mark.parametrize(
    "bad",
    [
        {"text": "x" * 40, "source": "a.pdf", "page": "four"},
        {"text": "x" * 40, "source": "a.pdf", "page": 0},
        {"text": "x" * 40, "source": "a.pdf", "page": 2, "chunk_index": "first"},
        {"text": "x" * 40, "source": "a.pdf", "page": 2, "chunk_index": -1},
        {"text": "x" * 40, "source": "a.pdf", "page": [1]},
    ],
)
async def test_malformed_payloads_are_skipped_not_raised(
    settings: Settings, bad: dict[str, Any]
) -> None:
    retriever, _, _ = _retriever(settings, FakeQdrantClient(corpus=[bad, CORPUS[0]]))

    result = await retriever.retrieve("notice period", top_n=3)

    assert result.candidates_considered == 1
    assert result.chunks[0].source == "contract.pdf"


# --------------------------------------------------------------------------- #
# Mocked embedding model
# --------------------------------------------------------------------------- #
class _NumpyEncoder:
    """Mimics sentence-transformers: float32 ndarray output, keyword contract recorded."""

    def __init__(self) -> None:
        self.kwargs: dict[str, Any] = {}

    def encode(self, sentences: Any, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        return np.asarray([0.6, 0.8, 0.0, 0.0], dtype=np.float32)


async def test_numpy_embeddings_reach_qdrant_as_plain_floats(settings: Settings) -> None:
    encoder = _NumpyEncoder()
    qdrant = FakeQdrantClient()
    retriever = TwoStageRetriever(
        client=qdrant, encoder=encoder, reranker=FakeReranker(), settings=settings
    )

    await retriever.retrieve("notice period")

    vector = qdrant.calls[0]["query"]
    assert vector == pytest.approx([0.6, 0.8, 0.0, 0.0])
    assert all(type(value) is float for value in vector)
    assert encoder.kwargs["normalize_embeddings"] is True
    assert encoder.kwargs["convert_to_numpy"] is True


async def test_score_threshold_is_forwarded_to_the_store(settings: Settings) -> None:
    settings.score_threshold = 0.35
    retriever, qdrant, _ = _retriever(settings)

    await retriever.retrieve("notice period")

    assert qdrant.calls[0]["score_threshold"] == 0.35
