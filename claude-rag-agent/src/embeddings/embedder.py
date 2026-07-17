"""
Embedding generation.

Wraps `sentence-transformers` behind a small interface so the rest of
the codebase (and tests) never talk to the underlying model directly.
This makes it trivial to swap in a different embedding backend (e.g.
an API-based embedder) later without touching retrieval or vector
store code.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np


class Embedder:
    """Generates dense vector embeddings for a list of text strings."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None  # lazy-loaded so importing this module stays fast

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of texts. Returns an (n_texts, dim) float32 array."""
        if not texts:
            return np.empty((0, self.dimension), dtype="float32")
        vectors = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return np.asarray(vectors, dtype="float32")

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query string. Returns a 1D float32 array."""
        return self.embed([text])[0]

    @property
    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()


@lru_cache(maxsize=4)
def get_embedder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> Embedder:
    """Cached factory so the (potentially large) model is only loaded once per name."""
    return Embedder(model_name=model_name)
