"""Wire schemas and stage-transition contracts reject what the next stage cannot use."""

from __future__ import annotations

import inspect
import math
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError
from src import contracts, models
from src.config import Settings
from src.contracts import (
    Chunk,
    DocumentExtraction,
    IndexBatch,
    PageText,
    RetrievalResult,
    RetrieverComponents,
)
from src.ingest import embed_chunks
from src.models import QueryRequest, RetrievedChunk, TokenUsage

from .conftest import FakeEncoder, FakeQdrantClient, FakeReranker


def _schemas(module: Any) -> list[type[BaseModel]]:
    return [
        obj
        for _, obj in inspect.getmembers(module, inspect.isclass)
        if issubclass(obj, BaseModel)
        and obj.__module__ == module.__name__
        and obj not in {models.WireModel, contracts.StageContract}
    ]


ALL_SCHEMAS = _schemas(models) + _schemas(contracts)


def test_schema_discovery_found_every_module() -> None:
    names = {schema.__name__ for schema in ALL_SCHEMAS}
    assert {"QueryRequest", "SourcesEvent", "RetrievalResult", "IndexBatch"} <= names


@pytest.mark.parametrize("schema", ALL_SCHEMAS, ids=lambda schema: schema.__name__)
def test_every_schema_is_closed_frozen_and_strict(schema: type[BaseModel]) -> None:
    config = schema.model_config

    assert config.get("extra") == "forbid"
    assert config.get("frozen") is True
    assert config.get("strict") is True


def _chunk(score: float, *, index: int = 0) -> RetrievedChunk:
    return RetrievedChunk(
        text=f"passage {index}",
        source="a.pdf",
        page=1,
        chunk_index=index,
        dense_score=0.5,
        rerank_score=score,
    )


# --------------------------------------------------------------------------- #
# Wire models
# --------------------------------------------------------------------------- #
def test_wire_models_reject_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        TokenUsage.model_validate({"input_tokens": 1, "cost_usd": 0.1})


def test_wire_models_are_immutable() -> None:
    request = QueryRequest(query="notice period")

    with pytest.raises(ValidationError, match="frozen"):
        request.k = 5  # type: ignore[misc]


@pytest.mark.parametrize("k", ["3", True, 3.0])
def test_query_request_does_not_coerce_scalars(k: object) -> None:
    with pytest.raises(ValidationError):
        QueryRequest.model_validate({"query": "notice period", "k": k})


def test_query_request_accepts_valid_json() -> None:
    request = QueryRequest.model_validate_json('{"query": "  notice period ", "k": 2}')

    assert (request.query, request.k) == ("notice period", 2)


# --------------------------------------------------------------------------- #
# Retrieval → generation
# --------------------------------------------------------------------------- #
def test_retrieval_result_accepts_ranked_subset() -> None:
    result = RetrievalResult(
        chunks=(_chunk(9.0), _chunk(4.0, index=1)),
        candidates_considered=5,
        dense_ms=1.5,
        rerank_ms=2.5,
    )

    assert result.total_ms == 4.0


def test_retrieval_result_rejects_unranked_chunks() -> None:
    with pytest.raises(ValidationError, match="descending rerank score"):
        RetrievalResult(
            chunks=(_chunk(1.0), _chunk(5.0, index=1)),
            candidates_considered=2,
            dense_ms=0.0,
            rerank_ms=0.0,
        )


def test_retrieval_result_rejects_more_chunks_than_candidates() -> None:
    with pytest.raises(ValidationError, match="cannot come from"):
        RetrievalResult(chunks=(_chunk(1.0),), candidates_considered=0, dense_ms=0, rerank_ms=0)


def test_retrieval_result_rejects_negative_timings() -> None:
    with pytest.raises(ValidationError):
        RetrievalResult(chunks=(), candidates_considered=0, dense_ms=-1.0, rerank_ms=0.0)


# --------------------------------------------------------------------------- #
# Retriever wiring (InstanceOf protocols)
# --------------------------------------------------------------------------- #
def test_retriever_components_accept_structural_fakes() -> None:
    components = RetrieverComponents(
        store=FakeQdrantClient(), encoder=FakeEncoder(), reranker=FakeReranker()
    )

    assert isinstance(components.encoder, FakeEncoder)


@pytest.mark.parametrize(
    ("store", "encoder", "reranker"),
    [
        (FakeQdrantClient(), None, FakeReranker()),
        (FakeQdrantClient(), FakeEncoder(), FakeEncoder()),
        (FakeQdrantClient(), FakeReranker(), FakeReranker()),
        (object(), FakeEncoder(), FakeReranker()),
        ("http://localhost:6333", FakeEncoder(), FakeReranker()),
    ],
)
def test_retriever_components_reject_objects_without_the_expected_surface(
    store: object, encoder: object, reranker: object
) -> None:
    with pytest.raises(ValidationError, match="is_instance_of"):
        RetrieverComponents(store=store, encoder=encoder, reranker=reranker)  # type: ignore[arg-type]


def test_retriever_components_reject_a_model_name_in_place_of_the_model() -> None:
    with pytest.raises(ValidationError, match="expected a loaded model"):
        RetrieverComponents(
            store=FakeQdrantClient(), encoder="all-MiniLM-L6-v2", reranker=FakeReranker()
        )


# --------------------------------------------------------------------------- #
# Extraction → chunking
# --------------------------------------------------------------------------- #
def test_document_extraction_rejects_pages_from_another_document() -> None:
    with pytest.raises(ValidationError, match="must carry that source"):
        DocumentExtraction(source="a.pdf", pages=(PageText(source="b.pdf", page=1, text="x"),))


def test_document_extraction_rejects_out_of_order_pages() -> None:
    pages = (
        PageText(source="a.pdf", page=2, text="x"),
        PageText(source="a.pdf", page=1, text="y"),
    )

    with pytest.raises(ValidationError, match="unique and ascending"):
        DocumentExtraction(source="a.pdf", pages=pages)


@pytest.mark.parametrize("page", [0, -1])
def test_page_numbers_are_one_indexed(page: int) -> None:
    with pytest.raises(ValidationError):
        PageText(source="a.pdf", page=page, text="x")


# --------------------------------------------------------------------------- #
# Embedding → upsert
# --------------------------------------------------------------------------- #
def _chunks(count: int) -> tuple[Chunk, ...]:
    return tuple(
        Chunk(text=f"chunk {index}", source="a.pdf", page=1, chunk_index=index)
        for index in range(count)
    )


def test_index_batch_accepts_matching_vectors() -> None:
    batch = IndexBatch(chunks=_chunks(2), vectors=((0.1, 0.2), (0.3, 0.4)), dimension=2)

    assert len(batch.vectors) == 2


@pytest.mark.parametrize(
    ("vectors", "match"),
    [
        (((0.1, 0.2),), "2 chunks but 1 vectors"),
        (((0.1, 0.2), (0.3,)), "vector 1 has 1 dimensions"),
        (((0.1, math.nan), (0.3, 0.4)), "NaN or infinity"),
        (((0.1, 0.2), (math.inf, 0.4)), "NaN or infinity"),
    ],
)
def test_index_batch_rejects_vectors_qdrant_would_refuse(
    vectors: tuple[tuple[float, ...], ...], match: str
) -> None:
    with pytest.raises(ValidationError, match=match):
        IndexBatch(chunks=_chunks(2), vectors=vectors, dimension=2)


def test_index_batch_rejects_colliding_point_ids() -> None:
    chunk = _chunks(1)[0]

    with pytest.raises(ValidationError, match="duplicate"):
        IndexBatch(chunks=(chunk, chunk), vectors=((0.1,), (0.2,)), dimension=1)


def test_embed_chunks_rejects_a_model_of_the_wrong_width(settings: Settings) -> None:
    # FakeEncoder emits 4-dimensional vectors; the collection expects 384.
    with pytest.raises(ValidationError, match="the collection expects 384"):
        embed_chunks(FakeEncoder(), _chunks(2), settings)  # type: ignore[arg-type]
