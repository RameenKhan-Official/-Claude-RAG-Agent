from unittest.mock import MagicMock, patch

import numpy as np

from src.embeddings.embedder import Embedder, get_embedder


def _fake_model(dimension: int = 8):
    model = MagicMock()
    model.get_sentence_embedding_dimension.return_value = dimension
    model.encode.side_effect = lambda texts, **kwargs: np.random.RandomState(0).rand(len(texts), dimension).astype(
        "float32"
    )
    return model


def test_embed_returns_correct_shape():
    embedder = Embedder(model_name="fake-model")
    with patch.object(Embedder, "model", new=_fake_model()):
        vectors = embedder.embed(["hello", "world"])
    assert vectors.shape == (2, 8)
    assert vectors.dtype.name == "float32"


def test_embed_empty_list_returns_empty_array():
    embedder = Embedder(model_name="fake-model")
    with patch.object(Embedder, "model", new=_fake_model()):
        vectors = embedder.embed([])
    assert vectors.shape == (0, 8)


def test_embed_query_returns_1d_vector():
    embedder = Embedder(model_name="fake-model")
    with patch.object(Embedder, "model", new=_fake_model()):
        vector = embedder.embed_query("hello")
    assert vector.shape == (8,)


def test_get_embedder_is_cached():
    a = get_embedder("same-name")
    b = get_embedder("same-name")
    assert a is b
