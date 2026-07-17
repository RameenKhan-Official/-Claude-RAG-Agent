"""
Retriever: glues the embedder and vector store together into a single
"index documents, retrieve relevant chunks" interface.
"""

from __future__ import annotations

from pathlib import Path

from src.embeddings.embedder import Embedder
from src.ingestion.document_loader import Chunk, load_and_chunk
from src.vectorstore.faiss_store import FAISSVectorStore


class Retriever:
    def __init__(self, embedder: Embedder, vector_store: FAISSVectorStore | None = None):
        self.embedder = embedder
        self.vector_store = vector_store

    def index_chunks(self, chunks: list[Chunk]) -> None:
        """Embed and add a list of Chunk objects to the vector store."""
        if not chunks:
            return
        texts = [c.text for c in chunks]
        vectors = self.embedder.embed(texts)

        if self.vector_store is None:
            self.vector_store = FAISSVectorStore(dimension=vectors.shape[1])

        self.vector_store.add(vectors, [c.to_dict() for c in chunks])

    def index_files(self, paths: list[str | Path], chunk_size: int = 800, chunk_overlap: int = 120) -> int:
        """Load, chunk, and index a list of file paths. Returns total chunks indexed."""
        total = 0
        for path in paths:
            chunks = load_and_chunk(path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            self.index_chunks(chunks)
            total += len(chunks)
        return total

    def retrieve(self, query: str, top_k: int = 4) -> list[dict]:
        """Return the top_k most relevant chunks for a query, or [] if nothing is indexed."""
        if self.vector_store is None or self.vector_store.size == 0:
            return []
        query_vector = self.embedder.embed_query(query)
        return self.vector_store.search(query_vector, top_k=top_k)

    def retrieve_as_context(self, query: str, top_k: int = 4) -> str:
        """Return retrieved chunks pre-formatted as a single context string for prompting."""
        results = self.retrieve(query, top_k=top_k)
        if not results:
            return "No relevant documents have been indexed yet."

        blocks = []
        for i, r in enumerate(results, start=1):
            blocks.append(f"[{i}] (source: {r['source']})\n{r['text']}")
        return "\n\n".join(blocks)
