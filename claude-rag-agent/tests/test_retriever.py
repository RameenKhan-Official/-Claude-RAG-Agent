from unittest.mock import MagicMock

import numpy as np

from src.ingestion.document_loader import Chunk
from src.retrieval.retriever import Retriever


def _mock_embedder(dim=4):
    embedder = MagicMock()
    embedder.embed.side_effect = lambda texts: np.random.RandomState(0).rand(len(texts), dim).astype("float32")
    embedder.embed_query.side_effect = lambda text: np.random.RandomState(0).rand(dim).astype("float32")
    return embedder


def test_index_chunks_and_retrieve_returns_results():
    retriever = Retriever(embedder=_mock_embedder())
    chunks = [Chunk(text="alpha", source="a.txt", chunk_index=0), Chunk(text="beta", source="a.txt", chunk_index=1)]

    retriever.index_chunks(chunks)
    results = retriever.retrieve("alpha", top_k=2)

    assert len(results) == 2
    assert {r["text"] for r in results} == {"alpha", "beta"}


def test_retrieve_before_indexing_returns_empty():
    retriever = Retriever(embedder=_mock_embedder())
    assert retriever.retrieve("anything") == []


def test_retrieve_as_context_formats_sources():
    retriever = Retriever(embedder=_mock_embedder())
    retriever.index_chunks([Chunk(text="alpha content", source="a.txt", chunk_index=0)])

    context = retriever.retrieve_as_context("query")
    assert "a.txt" in context
    assert "alpha content" in context


def test_retrieve_as_context_when_empty_says_so():
    retriever = Retriever(embedder=_mock_embedder())
    context = retriever.retrieve_as_context("query")
    assert "No relevant documents" in context


def test_index_chunks_with_empty_list_is_noop():
    retriever = Retriever(embedder=_mock_embedder())
    retriever.index_chunks([])
    assert retriever.retrieve("anything") == []
