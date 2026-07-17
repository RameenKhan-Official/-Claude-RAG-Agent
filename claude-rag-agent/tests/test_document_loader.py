import pytest

from src.ingestion.document_loader import chunk_text, load_and_chunk, load_document


def test_chunk_text_basic_split():
    text = "a" * 1000
    chunks = chunk_text(text, chunk_size=300, chunk_overlap=50)

    assert len(chunks) > 1
    assert all(len(c.text) <= 300 for c in chunks)
    # consecutive chunks should overlap by roughly chunk_overlap characters
    assert chunks[0].text[-50:] == chunks[1].text[:50]


def test_chunk_text_empty_input_returns_empty_list():
    assert chunk_text("   ", chunk_size=100, chunk_overlap=10) == []


def test_chunk_text_invalid_overlap_raises():
    with pytest.raises(ValueError):
        chunk_text("hello world", chunk_size=10, chunk_overlap=10)


def test_chunk_text_short_text_single_chunk():
    chunks = chunk_text("short text here", chunk_size=800, chunk_overlap=120, source="doc.txt")
    assert len(chunks) == 1
    assert chunks[0].source == "doc.txt"
    assert chunks[0].chunk_index == 0


def test_load_document_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_document("/nonexistent/path.txt")


def test_load_document_unsupported_extension_raises(tmp_path):
    bad_file = tmp_path / "notes.docx"
    bad_file.write_text("hello")
    with pytest.raises(ValueError):
        load_document(bad_file)


def test_load_and_chunk_txt_file(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("This is a sample document. " * 50)

    chunks = load_and_chunk(file_path, chunk_size=200, chunk_overlap=30)

    assert len(chunks) > 0
    assert all(c.source == "sample.txt" for c in chunks)
