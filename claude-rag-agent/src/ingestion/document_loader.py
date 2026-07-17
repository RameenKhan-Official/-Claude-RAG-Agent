"""
Document loading and chunking.

Supports plain text, Markdown, and PDF files. Chunking uses a simple
sliding-window splitter over characters, which is deterministic, fast,
and dependency-free (no tokenizer download required) -- a deliberate
tradeoff for a portfolio project that should run anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


@dataclass
class Chunk:
    """A single chunk of text extracted from a source document."""

    text: str
    source: str
    chunk_index: int

    def to_dict(self) -> dict:
        return {"text": self.text, "source": self.source, "chunk_index": self.chunk_index}


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_pdf_file(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover
        raise ImportError("pypdf is required to read PDF files. Install with `pip install pypdf`.") from exc

    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)


def load_document(path: str | Path) -> str:
    """Read a document from disk and return its raw text content."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No such file: {path}")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type '{suffix}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}")

    if suffix == ".pdf":
        return _read_pdf_file(path)
    return _read_text_file(path)


def chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 120, source: str = "unknown") -> list[Chunk]:
    """
    Split text into overlapping chunks.

    Args:
        text: Raw text to split.
        chunk_size: Target number of characters per chunk.
        chunk_overlap: Number of overlapping characters between consecutive chunks.
        source: Identifier (e.g. filename) stored on each chunk for citation purposes.

    Returns:
        A list of Chunk objects. Returns an empty list for blank input.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")

    cleaned = " ".join(text.split())
    if not cleaned:
        return []

    chunks: list[Chunk] = []
    start = 0
    index = 0
    step = chunk_size - chunk_overlap

    while start < len(cleaned):
        end = min(start + chunk_size, len(cleaned))
        piece = cleaned[start:end].strip()
        if piece:
            chunks.append(Chunk(text=piece, source=source, chunk_index=index))
            index += 1
        if end == len(cleaned):
            break
        start += step

    return chunks


def load_and_chunk(path: str | Path, chunk_size: int = 800, chunk_overlap: int = 120) -> list[Chunk]:
    """Convenience wrapper: load a document from disk and chunk it in one call."""
    path = Path(path)
    text = load_document(path)
    return chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap, source=path.name)
