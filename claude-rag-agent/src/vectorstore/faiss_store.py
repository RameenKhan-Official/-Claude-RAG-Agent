"""
A thin, well-tested wrapper around FAISS for similarity search.

Stores chunk metadata (text + source) alongside the FAISS index so a
search returns ready-to-use context, not just raw vector ids.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np


class FAISSVectorStore:
    """Flat L2 similarity index with attached document metadata."""

    def __init__(self, dimension: int):
        import faiss  # imported lazily so unit tests can run without faiss installed for pure-python paths

        self.dimension = dimension
        self._faiss = faiss
        self.index = faiss.IndexFlatL2(dimension)
        self.metadata: list[dict] = []

    def add(self, vectors: np.ndarray, metadata: list[dict]) -> None:
        """Add a batch of vectors and their corresponding metadata dicts."""
        if len(vectors) != len(metadata):
            raise ValueError("vectors and metadata must have the same length")
        if len(vectors) == 0:
            return
        vectors = np.ascontiguousarray(vectors, dtype="float32")
        self.index.add(vectors)
        self.metadata.extend(metadata)

    def search(self, query_vector: np.ndarray, top_k: int = 4) -> list[dict]:
        """Return the top_k nearest metadata entries with their distances."""
        if self.index.ntotal == 0:
            return []
        query_vector = np.ascontiguousarray(query_vector, dtype="float32").reshape(1, -1)
        k = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(query_vector, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            entry = dict(self.metadata[idx])
            entry["distance"] = float(dist)
            results.append(entry)
        return results

    @property
    def size(self) -> int:
        return self.index.ntotal

    def save(self, directory: str | Path) -> None:
        """Persist the FAISS index and metadata to disk."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self._faiss.write_index(self.index, str(directory / "index.faiss"))
        with open(directory / "metadata.pkl", "wb") as f:
            pickle.dump(self.metadata, f)
        with open(directory / "config.json", "w") as f:
            json.dump({"dimension": self.dimension}, f)

    @classmethod
    def load(cls, directory: str | Path) -> "FAISSVectorStore":
        """Load a previously persisted vector store."""
        import faiss

        directory = Path(directory)
        with open(directory / "config.json") as f:
            config = json.load(f)

        store = cls(dimension=config["dimension"])
        store.index = faiss.read_index(str(directory / "index.faiss"))
        with open(directory / "metadata.pkl", "rb") as f:
            store.metadata = pickle.load(f)
        return store
