"""Ingestion → index → retrieval against a real, in-process Qdrant engine.

qdrant-client's local mode (`path=...`) runs the actual collection, upsert and
query code with no server, so these tests prove the payload written by
`src.ingest` is exactly what `src.retriever` reads back — and that the real
`AsyncQdrantClient` satisfies the `VectorStore` protocol at runtime. Only the
neural models are replaced, by a deterministic hashing encoder.
"""

from __future__ import annotations

import math
import re
import shutil
import warnings
import zlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from qdrant_client import AsyncQdrantClient, QdrantClient
from src.config import PROJECT_ROOT, Settings
from src.ingest import ensure_collection, run_ingestion
from src.retriever import TwoStageRetriever

from .conftest import FakeReranker

SAMPLE_PDF = PROJECT_ROOT / "documents" / "sample_policy_handbook.pdf"
DIMENSION = 64
_TOKEN_RE = re.compile(r"[a-z0-9]+")


class HashingEncoder:
    """Bag-of-words feature hashing: similar wording → similar vector, no torch."""

    def __init__(self, dimension: int = DIMENSION) -> None:
        self.dimension = dimension

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for token in _TOKEN_RE.findall(text.lower()):
            vector[zlib.crc32(token.encode()) % self.dimension] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    def encode(self, sentences: str | list[str], **_kwargs: Any) -> Any:
        if isinstance(sentences, str):
            return self._vector(sentences)
        return [self._vector(text) for text in sentences]


@pytest.fixture
def corpus_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "documents"
    directory.mkdir()
    shutil.copy(SAMPLE_PDF, directory / SAMPLE_PDF.name)
    return directory


@pytest.fixture
def store_path(tmp_path: Path) -> Path:
    return tmp_path / "qdrant"


@pytest.fixture
def local_settings(settings: Settings, corpus_dir: Path) -> Settings:
    return settings.model_copy(
        update={
            "embedding_dimension": DIMENSION,
            "documents_dir": corpus_dir,
            "retrieval_candidates": 6,
        }
    )


@pytest.fixture
def opened_clients(
    monkeypatch: pytest.MonkeyPatch, store_path: Path
) -> Iterator[list[QdrantClient]]:
    """Route `src.ingest`'s Qdrant client and embedding model to local doubles."""
    opened: list[QdrantClient] = []

    def local_client(**_kwargs: Any) -> QdrantClient:
        client = QdrantClient(path=str(store_path))
        opened.append(client)
        return client

    monkeypatch.setattr("src.ingest.QdrantClient", local_client)
    monkeypatch.setattr("src.ingest.load_embedding_model", lambda _settings: HashingEncoder())
    with warnings.catch_warnings():
        # Local mode warns that payload indexes are a server-only feature.
        warnings.filterwarnings("ignore", message="Payload indexes have no effect")
        yield opened


def _count(store_path: Path, collection: str) -> int:
    client = QdrantClient(path=str(store_path))
    try:
        return client.count(collection, exact=True).count
    finally:
        client.close()


@pytest.mark.usefixtures("opened_clients")
async def test_ingested_corpus_is_retrievable_with_provenance(
    local_settings: Settings, store_path: Path
) -> None:
    written = run_ingestion(local_settings, recreate=True, dry_run=False)
    assert written > 0

    client = AsyncQdrantClient(path=str(store_path))
    retriever = TwoStageRetriever(
        client=client, encoder=HashingEncoder(), reranker=FakeReranker(), settings=local_settings
    )
    try:
        result = await retriever.retrieve("How many days of paid annual leave?", top_n=3)
        points = await retriever.collection_points()
    finally:
        await retriever.close()

    assert points == written
    assert result.ranking == "cross_encoder"
    assert 0 < len(result.chunks) <= 3
    assert result.candidates_considered <= local_settings.retrieval_candidates
    best = result.chunks[0]
    assert best.source == SAMPLE_PDF.name
    assert best.page == 4  # "6.1 Annual leave" is on page 4 of the sample
    assert "twenty-six (26)" in best.text


@pytest.mark.usefixtures("opened_clients")
def test_reingestion_upserts_in_place(local_settings: Settings, store_path: Path) -> None:
    first = run_ingestion(local_settings, recreate=True, dry_run=False)
    second = run_ingestion(local_settings, recreate=False, dry_run=False)

    assert first == second
    # Deterministic point ids: the second run overwrote, it did not duplicate.
    assert _count(store_path, local_settings.qdrant_collection) == first


def test_recreate_drops_existing_points(
    local_settings: Settings, store_path: Path, opened_clients: list[QdrantClient]
) -> None:
    run_ingestion(local_settings, recreate=True, dry_run=False)
    client = QdrantClient(path=str(store_path))
    try:
        ensure_collection(client, local_settings, recreate=True)
        assert client.count(local_settings.qdrant_collection, exact=True).count == 0
        ensure_collection(client, local_settings, recreate=False)  # reuse, no error
    finally:
        client.close()
    assert len(opened_clients) == 1


def test_wrong_vector_width_fails_before_qdrant_is_touched(
    local_settings: Settings, opened_clients: list[QdrantClient]
) -> None:
    mismatched = local_settings.model_copy(update={"embedding_dimension": DIMENSION // 2})

    with pytest.raises(ValidationError, match="the collection expects 32"):
        run_ingestion(mismatched, recreate=True, dry_run=False)

    assert opened_clients == []


def test_upsert_batches_cover_every_chunk(
    local_settings: Settings, store_path: Path, opened_clients: list[QdrantClient]
) -> None:
    tiny_batches = local_settings.model_copy(update={"ingest_upsert_batch_size": 2})

    written = run_ingestion(tiny_batches, recreate=True, dry_run=False)

    assert written > 2  # more than one batch was needed
    assert _count(store_path, tiny_batches.qdrant_collection) == written
