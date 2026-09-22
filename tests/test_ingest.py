"""Unit tests for the offline ingestion pipeline (no Qdrant, no models)."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import ClassVar

import pytest
from src.config import PROJECT_ROOT, Settings
from src.ingest import (
    Chunk,
    PageText,
    build_splitter,
    chunk_pages,
    discover_pdfs,
    extract_document,
    extract_pdf,
    normalise_text,
    run_ingestion,
    scan_documents,
)
from src.ingest import main as ingest_main

SAMPLE_PDF = PROJECT_ROOT / "documents" / "sample_policy_handbook.pdf"


def test_normalise_text_rejoins_hyphenated_line_breaks() -> None:
    assert normalise_text("termi-\nnation clause") == "termination clause"


def test_normalise_text_collapses_runs_of_whitespace() -> None:
    assert normalise_text("a   \t b\n\n\n\nc  ") == "a b\n\nc"


def test_normalise_text_strips_null_bytes() -> None:
    assert "\x00" not in normalise_text("bad\x00byte")


def test_chunking_respects_size_and_keeps_provenance(settings: Settings) -> None:
    page = PageText(source="doc.pdf", page=3, text="Clause text. " * 120)

    chunks = chunk_pages([page], build_splitter(settings))

    assert len(chunks) > 1
    assert all(chunk.source == "doc.pdf" and chunk.page == 3 for chunk in chunks)
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
    # The splitter must honour the configured budget (plus a small separator slack).
    assert max(len(chunk.text) for chunk in chunks) <= settings.chunk_size + 32


def test_chunking_drops_page_furniture(settings: Settings) -> None:
    page = PageText(source="doc.pdf", page=9, text="12")

    assert chunk_pages([page], build_splitter(settings)) == []


def test_chunk_ids_are_deterministic_and_unique() -> None:
    first = Chunk(text="a", source="doc.pdf", page=1, chunk_index=0)
    same = Chunk(text="different text", source="doc.pdf", page=1, chunk_index=0)
    other = Chunk(text="a", source="doc.pdf", page=1, chunk_index=1)

    # Identity is provenance-based, so re-ingesting updates instead of duplicating.
    assert first.point_id == same.point_id
    assert first.point_id != other.point_id


def test_discover_pdfs_is_sorted_and_filtered(tmp_path: Path) -> None:
    (tmp_path / "b.pdf").write_bytes(b"%PDF-1.4")
    (tmp_path / "a.pdf").write_bytes(b"%PDF-1.4")
    (tmp_path / "notes.txt").write_text("ignored")

    assert [path.name for path in discover_pdfs(tmp_path)] == ["a.pdf", "b.pdf"]


def test_discover_pdfs_rejects_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        discover_pdfs(tmp_path / "nope")


def test_unreadable_pdf_is_skipped_without_raising(tmp_path: Path) -> None:
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"not really a pdf")

    assert list(extract_pdf(broken)) == []


@pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="sample corpus not present")
def test_sample_corpus_extracts_pages_with_1_indexed_numbers() -> None:
    pages = list(extract_pdf(SAMPLE_PDF))

    assert len(pages) == 5
    assert [page.page for page in pages] == [1, 2, 3, 4, 5]
    assert all(page.source == "sample_policy_handbook.pdf" for page in pages)
    termination = next(page for page in pages if "terminate the agreement" in page.text)
    assert "thirty (30) days" in termination.text


@pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="sample corpus not present")
def test_end_to_end_chunking_of_the_sample_corpus(settings: Settings) -> None:
    chunks = chunk_pages(list(extract_pdf(SAMPLE_PDF)), build_splitter(settings))

    assert len(chunks) >= 10
    assert all(chunk.page >= 1 for chunk in chunks)
    assert any("thirty (30) days" in chunk.text for chunk in chunks)
    assert len({chunk.point_id for chunk in chunks}) == len(chunks)


# --------------------------------------------------------------------------- #
# Corpus scanning — arbitrary user files land in documents/
# --------------------------------------------------------------------------- #
def test_scan_separates_pdfs_from_everything_else(tmp_path: Path) -> None:
    (tmp_path / "policy.pdf").write_bytes(b"%PDF-1.4 content")
    (tmp_path / "contract.PDF").write_bytes(b"%PDF-1.4 content")  # case-insensitive
    (tmp_path / "notes.txt").write_text("ignored")
    (tmp_path / "deck.pptx").write_bytes(b"zip")
    (tmp_path / "LICENSE").write_text("no extension")
    (tmp_path / "empty.pdf").touch()
    (tmp_path / "archive").mkdir()
    (tmp_path / ".DS_Store").write_bytes(b"junk")
    (tmp_path / "README.md").write_text("housekeeping")

    pdfs, skipped = scan_documents(tmp_path)

    assert [path.name for path in pdfs] == ["contract.PDF", "policy.pdf"]
    reasons = {item.name: item.reason for item in skipped}
    assert reasons["notes.txt"] == "unsupported file type: .txt"
    assert reasons["deck.pptx"] == "unsupported file type: .pptx"
    assert reasons["LICENSE"] == "unsupported file type: no extension"
    assert reasons["empty.pdf"] == "empty file"
    assert reasons["archive"] == "directory (not scanned recursively)"
    # Hidden files and the corpus README are housekeeping, not skipped corpus.
    assert ".DS_Store" not in reasons
    assert "README.md" not in reasons


def test_skipped_files_are_logged(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    (tmp_path / "notes.txt").write_text("ignored")

    with caplog.at_level(logging.WARNING, logger="src.ingest"):
        scan_documents(tmp_path)

    assert "Skipping notes.txt" in caplog.text


# --------------------------------------------------------------------------- #
# Damaged input must not abort a run
# --------------------------------------------------------------------------- #
class _FakePage:
    def __init__(self, text: str | None, error: Exception | None = None) -> None:
        self._text = text
        self._error = error

    def extract_text(self) -> str | None:
        if self._error is not None:
            raise self._error
        return self._text


class _FakeReader:
    """Stand-in for `pypdf.PdfReader` with scriptable per-page behaviour."""

    pages_by_path: ClassVar[dict[str, list[_FakePage]]] = {}
    is_encrypted = False

    def __init__(self, path: str, strict: bool = True) -> None:
        self.pages = self.pages_by_path[Path(path).name]

    def decrypt(self, password: str) -> int:  # pragma: no cover - not used here
        return 1


def _install_reader(monkeypatch: pytest.MonkeyPatch, name: str, pages: list[_FakePage]) -> None:
    _FakeReader.pages_by_path = {name: pages}
    monkeypatch.setattr("src.ingest.PdfReader", _FakeReader)


def test_a_corrupt_page_is_counted_and_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "partly_broken.pdf"
    path.write_bytes(b"%PDF-1.4")
    _install_reader(
        monkeypatch,
        path.name,
        [
            _FakePage("Clause one: the notice period is thirty (30) days."),
            _FakePage(None, error=ValueError("corrupt content stream")),
            _FakePage("Clause three: invoices are due within fourteen (14) days."),
        ],
    )

    extraction = extract_document(path)

    # The run survives; the damage is reported, not swallowed.
    assert [page.page for page in extraction.pages] == [1, 3]
    assert extraction.failed_pages == 1
    assert extraction.status == "1 page(s) unreadable"


def test_scanned_pdf_without_a_text_layer_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "scanned.pdf"
    path.write_bytes(b"%PDF-1.4")
    _install_reader(monkeypatch, path.name, [_FakePage(""), _FakePage(None)])

    extraction = extract_document(path)

    assert extraction.pages == []
    assert extraction.status == "no text layer"


def test_unreadable_document_reports_status(tmp_path: Path) -> None:
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"not really a pdf")

    extraction = extract_document(broken)

    assert extraction.pages == []
    assert extraction.status == "unreadable"


# --------------------------------------------------------------------------- #
# End-to-end reporting (no models, no Qdrant)
# --------------------------------------------------------------------------- #
def test_dry_run_reports_corpus_and_skipped_files(
    tmp_path: Path, settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    shutil.copy(SAMPLE_PDF, tmp_path / "handbook.pdf")
    (tmp_path / "spreadsheet.xlsx").write_bytes(b"not a pdf")

    scoped = settings.model_copy(update={"documents_dir": tmp_path})
    chunks = run_ingestion(scoped, recreate=False, dry_run=True)

    assert chunks > 0
    out = capsys.readouterr().out
    assert "Skipped files" in out
    assert "spreadsheet.xlsx" in out
    assert "Corpus" in out
    assert "handbook.pdf" in out
    assert "dry-run" in out


def test_ingestion_of_an_empty_directory_is_not_an_error(
    tmp_path: Path, settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    scoped = settings.model_copy(update={"documents_dir": tmp_path})

    assert run_ingestion(scoped, recreate=False, dry_run=True) == 0
    assert "No PDF files found" in capsys.readouterr().out


def test_corpus_of_only_scanned_pdfs_reports_the_ocr_hint(
    tmp_path: Path,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "scanned.pdf"
    path.write_bytes(b"%PDF-1.4")
    _install_reader(monkeypatch, path.name, [_FakePage("")])
    scoped = settings.model_copy(update={"documents_dir": tmp_path})

    assert run_ingestion(scoped, recreate=False, dry_run=True) == 0
    assert "OCR" in capsys.readouterr().out


def test_cli_accepts_a_custom_documents_directory(tmp_path: Path) -> None:
    shutil.copy(SAMPLE_PDF, tmp_path / "handbook.pdf")

    exit_code = ingest_main(["--documents-dir", str(tmp_path), "--dry-run"])

    assert exit_code == 0
