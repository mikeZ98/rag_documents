"""Offline ingestion pipeline: PDF → text → chunks → vectors → Qdrant.

Run it whenever the corpus changes:

    uv run python -m src.ingest --recreate

Design notes
------------
* Point ids are deterministic (`uuid5` of source+page+chunk index), so a
  re-run upserts in place instead of duplicating the corpus.
* Embedding happens in batches of `EMBEDDING_BATCH_SIZE` with L2-normalised
  output, matching the cosine distance configured on the collection.
* The collection is created with an explicit HNSW configuration rather than
  Qdrant's defaults, so index behaviour is reviewable in code.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import uuid
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from pypdf.errors import PdfReadError
from qdrant_client import QdrantClient, models
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from src.config import Settings, get_settings
from src.retriever import load_embedding_model

if TYPE_CHECKING:  # pragma: no cover - typing only
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger("src.ingest")
console = Console()

# uuid5 namespace for this project — keeps generated ids stable across runs.
POINT_NAMESPACE = uuid.UUID("6f1c2d3e-8a4b-5c6d-9e0f-1a2b3c4d5e6f")

# Ordered from strongest to weakest semantic boundary: markdown headings,
# paragraphs, lines, sentences, clauses, words, characters.
SPLIT_SEPARATORS: list[str] = [
    "\n# ",
    "\n## ",
    "\n### ",
    "\n#### ",
    "\n\n",
    "\n",
    ". ",
    "? ",
    "! ",
    "; ",
    ", ",
    " ",
    "",
]

_WHITESPACE_RE = re.compile(r"[ \t\x0b\f\r]+")
_BLANKLINE_RE = re.compile(r"\n{3,}")
_HYPHEN_BREAK_RE = re.compile(r"(\w)-\n(\w)")


@dataclass(frozen=True, slots=True)
class PageText:
    """Raw text of a single PDF page."""

    source: str
    page: int  # 1-indexed, as printed in the document
    text: str


@dataclass(frozen=True, slots=True)
class SkippedFile:
    """A directory entry that was deliberately not ingested."""

    name: str
    reason: str


@dataclass(frozen=True, slots=True)
class DocumentExtraction:
    """Everything one PDF contributed, including what went wrong inside it."""

    source: str
    pages: list[PageText]
    failed_pages: int = 0
    status: str = "ok"


@dataclass(frozen=True, slots=True)
class FileReport:
    """One row of the ingestion summary table."""

    name: str
    pages: int
    chunks: int
    failed_pages: int
    status: str


@dataclass(frozen=True, slots=True)
class Chunk:
    """A splitter-produced passage with full provenance."""

    text: str
    source: str
    page: int
    chunk_index: int

    @property
    def point_id(self) -> str:
        """Deterministic id so repeated ingestion updates rather than duplicates."""
        return str(uuid.uuid5(POINT_NAMESPACE, f"{self.source}:{self.page}:{self.chunk_index}"))


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #
def normalise_text(raw: str) -> str:
    """Clean up PDF extraction artefacts without destroying structure."""
    text = raw.replace("\x00", " ")
    text = _HYPHEN_BREAK_RE.sub(r"\1\2", text)  # re-join words split across lines
    text = _WHITESPACE_RE.sub(" ", text)
    text = _BLANKLINE_RE.sub("\n\n", text)
    return text.strip()


def extract_document(path: Path) -> DocumentExtraction:
    """Extract every page of `path`, degrading gracefully on damaged input.

    A corrupt file yields an empty result with a status, and a page that
    `pypdf` cannot decode is counted and skipped — one bad page (or one bad
    document) never aborts a corpus-wide ingestion run.
    """
    try:
        reader = PdfReader(str(path), strict=False)
        # Many "protected" PDFs open with an empty owner password; the rest we skip.
        if reader.is_encrypted and reader.decrypt("") == 0:
            logger.warning("Skipping encrypted PDF: %s", path.name)
            return DocumentExtraction(path.name, [], status="encrypted")
        page_count = len(reader.pages)
    except (PdfReadError, OSError, ValueError, KeyError) as exc:
        logger.error("Cannot read %s: %s", path.name, exc)
        return DocumentExtraction(path.name, [], status="unreadable")

    pages: list[PageText] = []
    failed = 0
    for page_number in range(1, page_count + 1):
        try:
            raw = reader.pages[page_number - 1].extract_text() or ""
        except Exception as exc:  # noqa: BLE001 - one unreadable page must not kill the run
            failed += 1
            logger.warning("Text extraction failed on %s p.%d: %s", path.name, page_number, exc)
            continue
        text = normalise_text(raw)
        if text:
            pages.append(PageText(source=path.name, page=page_number, text=text))

    if not pages:
        # Almost always a scanned/image-only PDF: real input, no text layer.
        logger.warning("No extractable text in %s (scanned or image-only?)", path.name)
        return DocumentExtraction(path.name, [], failed, "no text layer")
    status = "ok" if failed == 0 else f"{failed} page(s) unreadable"
    return DocumentExtraction(path.name, pages, failed, status)


def extract_pdf(path: Path) -> Iterator[PageText]:
    """Stream the pages of `path`; a thin view over :func:`extract_document`."""
    yield from extract_document(path).pages


def scan_documents(directory: Path) -> tuple[list[Path], list[SkippedFile]]:
    """Split `directory` into ingestible PDFs and explicitly skipped entries.

    Skipping is reported rather than silent: a user who drops `notes.docx` into
    `documents/` should be told why it never reached the index.
    """
    if not directory.is_dir():
        msg = f"Documents directory not found: {directory}"
        raise FileNotFoundError(msg)

    pdfs: list[Path] = []
    skipped: list[SkippedFile] = []
    for entry in sorted(directory.iterdir(), key=lambda item: item.name):
        if entry.name.startswith(".") or entry.name.lower() == "readme.md":
            continue  # housekeeping files, not corpus candidates
        if entry.is_dir():
            skipped.append(SkippedFile(entry.name, "directory (not scanned recursively)"))
        elif entry.suffix.lower() != ".pdf":
            extension = entry.suffix.lower() or "no extension"
            skipped.append(SkippedFile(entry.name, f"unsupported file type: {extension}"))
        elif entry.stat().st_size == 0:
            skipped.append(SkippedFile(entry.name, "empty file"))
        else:
            pdfs.append(entry)

    for item in skipped:
        logger.warning("Skipping %s — %s", item.name, item.reason)
    return pdfs, skipped


def discover_pdfs(directory: Path) -> list[Path]:
    """Return every ingestible `.pdf` in `directory`, sorted deterministically."""
    return scan_documents(directory)[0]


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #
def build_splitter(settings: Settings) -> RecursiveCharacterTextSplitter:
    """Recursive splitter that prefers markdown/paragraph/sentence boundaries."""
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=SPLIT_SEPARATORS,
        keep_separator=True,
        strip_whitespace=True,
        length_function=len,
    )


def chunk_pages(pages: Sequence[PageText], splitter: RecursiveCharacterTextSplitter) -> list[Chunk]:
    """Split each page independently so that page provenance is never lost."""
    chunks: list[Chunk] = []
    for page in pages:
        for index, piece in enumerate(splitter.split_text(page.text)):
            cleaned = piece.strip()
            if len(cleaned) < 32:  # drop headers/footers and stray page numbers
                continue
            chunks.append(
                Chunk(text=cleaned, source=page.source, page=page.page, chunk_index=index)
            )
    return chunks


# --------------------------------------------------------------------------- #
# Indexing
# --------------------------------------------------------------------------- #
def ensure_collection(client: QdrantClient, settings: Settings, *, recreate: bool) -> None:
    """Create (or recreate) the collection with cosine distance and HNSW."""
    exists = client.collection_exists(settings.qdrant_collection)
    if exists and recreate:
        logger.info("Dropping existing collection %s", settings.qdrant_collection)
        client.delete_collection(settings.qdrant_collection)
        exists = False
    if exists:
        logger.info("Reusing collection %s", settings.qdrant_collection)
        return

    logger.info(
        "Creating collection %s (dim=%d, cosine, HNSW m=16 ef_construct=100)",
        settings.qdrant_collection,
        settings.embedding_dimension,
    )
    client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config=models.VectorParams(
            size=settings.embedding_dimension,
            distance=models.Distance.COSINE,
            on_disk=False,
        ),
        hnsw_config=models.HnswConfigDiff(
            m=16,  # bi-directional links per node: recall/memory trade-off
            ef_construct=100,  # build-time candidate list: higher = better graph
            full_scan_threshold=10_000,  # below this, brute force is faster than ANN
        ),
        optimizers_config=models.OptimizersConfigDiff(default_segment_number=2),
    )
    # Payload index enables cheap metadata filtering (e.g. "search only in X.pdf").
    client.create_payload_index(
        collection_name=settings.qdrant_collection,
        field_name="source",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )


def embed_chunks(
    model: SentenceTransformer, chunks: Sequence[Chunk], settings: Settings
) -> list[list[float]]:
    """Batch-embed chunk texts into normalised vectors."""
    texts = [chunk.text for chunk in chunks]
    vectors = model.encode(
        texts,
        batch_size=settings.embedding_batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return [[float(value) for value in vector] for vector in vectors]


def upsert_chunks(
    client: QdrantClient,
    settings: Settings,
    chunks: Sequence[Chunk],
    vectors: Sequence[Sequence[float]],
) -> int:
    """Upsert points in batches; returns the number of points written."""
    batch_size = settings.ingest_upsert_batch_size
    written = 0
    for start in range(0, len(chunks), batch_size):
        window = chunks[start : start + batch_size]
        points = [
            models.PointStruct(
                id=chunk.point_id,
                vector=list(vector),
                payload={
                    "text": chunk.text,
                    "source": chunk.source,
                    "page": chunk.page,
                    "chunk_index": chunk.chunk_index,
                    "char_count": len(chunk.text),
                },
            )
            for chunk, vector in zip(window, vectors[start : start + batch_size], strict=True)
        ]
        client.upsert(collection_name=settings.qdrant_collection, points=points, wait=True)
        written += len(points)
    return written


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_ingestion(settings: Settings, *, recreate: bool, dry_run: bool) -> int:
    """Execute the full pipeline; returns the number of chunks written."""
    pdf_paths, skipped = scan_documents(settings.documents_dir)
    if skipped:
        _print_skipped_table(skipped)
    if not pdf_paths:
        console.print(
            f"[yellow]No PDF files found in {settings.documents_dir}. "
            "Drop your documents there and re-run.[/yellow]"
        )
        return 0

    console.print(
        f"[bold]Ingesting {len(pdf_paths)} PDF file(s)[/bold] from "
        f"[cyan]{settings.documents_dir}[/cyan]"
    )

    splitter = build_splitter(settings)
    reports: list[FileReport] = []
    all_chunks: list[Chunk] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Extracting & chunking", total=len(pdf_paths))
        for path in pdf_paths:
            extraction = extract_document(path)
            chunks = chunk_pages(extraction.pages, splitter)
            all_chunks.extend(chunks)
            # Text was extracted, but every fragment was too short to index.
            status = "no usable chunks" if extraction.pages and not chunks else extraction.status
            reports.append(
                FileReport(
                    name=extraction.source,
                    pages=len(extraction.pages),
                    chunks=len(chunks),
                    failed_pages=extraction.failed_pages,
                    status=status,
                )
            )
            progress.advance(task)

    _print_corpus_table(reports)

    if not all_chunks:
        console.print(
            "[red]No extractable text found in any document.[/red] "
            "Scanned PDFs need an OCR pass (e.g. `ocrmypdf`) before ingestion."
        )
        return 0

    if dry_run:
        console.print("[yellow]--dry-run: skipping embedding and upsert.[/yellow]")
        return len(all_chunks)

    model = load_embedding_model(settings)
    with console.status(f"Embedding {len(all_chunks)} chunks…", spinner="dots"):
        vectors = embed_chunks(model, all_chunks, settings)

    client = QdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        api_key=settings.secret(settings.qdrant_api_key),
        timeout=int(settings.qdrant_timeout_seconds),
    )
    try:
        ensure_collection(client, settings, recreate=recreate)
        with console.status("Upserting into Qdrant…", spinner="dots"):
            written = upsert_chunks(client, settings, all_chunks, vectors)
        total_points = client.count(settings.qdrant_collection, exact=True).count
    finally:
        client.close()

    _print_index_table(settings, written, total_points, recreate=recreate)
    return written


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _status_style(status: str) -> str:
    if status == "ok":
        return "[green]ok[/green]"
    if status in {"unreadable", "encrypted", "no text layer"}:
        return f"[red]{status}[/red]"
    return f"[yellow]{status}[/yellow]"


def _print_skipped_table(skipped: list[SkippedFile]) -> None:
    table = Table(title="Skipped files", header_style="bold", title_style="bold yellow")
    table.add_column("File")
    table.add_column("Reason")
    for item in skipped:
        table.add_row(item.name, f"[yellow]{item.reason}[/yellow]")
    console.print(table)


def _print_corpus_table(reports: list[FileReport]) -> None:
    table = Table(title="Corpus", header_style="bold")
    table.add_column("File", overflow="fold")
    table.add_column("Pages", justify="right")
    table.add_column("Chunks", justify="right")
    table.add_column("Status")
    for report in reports:
        table.add_row(
            report.name,
            str(report.pages),
            str(report.chunks),
            _status_style(report.status),
        )
    table.add_section()
    table.add_row(
        f"[bold]{len(reports)} file(s)[/bold]",
        f"[bold]{sum(r.pages for r in reports)}[/bold]",
        f"[bold]{sum(r.chunks for r in reports)}[/bold]",
        f"[bold]{sum(1 for r in reports if r.status == 'ok')} ok[/bold]",
    )
    console.print(table)


def _print_index_table(
    settings: Settings, written: int, total_points: int, *, recreate: bool
) -> None:
    table = Table(title="Index", header_style="bold", title_style="bold green")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("collection", settings.qdrant_collection)
    table.add_row("mode", "recreated" if recreate else "upsert (idempotent)")
    table.add_row("vector size / distance", f"{settings.embedding_dimension} / cosine")
    table.add_row("points upserted", f"[bold]{written}[/bold]")
    table.add_row("points in collection", f"[bold]{total_points}[/bold]")
    console.print(table)
    console.print("[green]✓[/green] Ingestion complete — the index is ready to query.")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="rag-ingest",
        description="Index the PDF corpus into Qdrant (PDF → chunks → vectors).",
    )
    parser.add_argument(
        "--documents-dir",
        type=Path,
        default=None,
        help="Override DOCUMENTS_DIR for this run.",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop and rebuild the collection before indexing.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Extract and chunk only; do not load models or touch Qdrant.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns a POSIX exit code."""
    args = _parse_args(argv)
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
    # The HuggingFace and Qdrant clients are chatty at INFO level.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    if args.documents_dir is not None:
        settings = settings.model_copy(update={"documents_dir": args.documents_dir.resolve()})

    try:
        indexed = run_ingestion(settings, recreate=args.recreate, dry_run=args.dry_run)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        return 2
    except Exception as exc:
        logger.exception("Ingestion failed")
        console.print(f"[red]Ingestion failed: {exc}[/red]")
        return 1
    return 0 if indexed else 3


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
