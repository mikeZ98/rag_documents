"""Unit tests for the two-stage retriever."""

from __future__ import annotations

import asyncio
from typing import Any, cast

import pytest
from qdrant_client import AsyncQdrantClient
from src.config import Settings
from src.models import RetrievedChunk
from src.retriever import (
    RetrievalError,
    TwoStageRetriever,
    build_context_block,
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
        client=cast(AsyncQdrantClient, client),
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

    assert result.chunks == []
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
        client=cast(AsyncQdrantClient, FakeQdrantClient()),
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
