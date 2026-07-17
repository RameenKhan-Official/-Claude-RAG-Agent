import numpy as np
import pytest

from src.vectorstore.faiss_store import FAISSVectorStore


def _sample_vectors(n=5, dim=4, seed=0):
    rng = np.random.RandomState(seed)
    return rng.rand(n, dim).astype("float32")


def test_add_and_search_returns_closest_match():
    store = FAISSVectorStore(dimension=4)
    vectors = _sample_vectors(n=5)
    metadata = [{"text": f"chunk-{i}", "source": "doc.txt", "chunk_index": i} for i in range(5)]

    store.add(vectors, metadata)
    results = store.search(vectors[2], top_k=1)

    assert len(results) == 1
    assert results[0]["text"] == "chunk-2"
    assert results[0]["distance"] == pytest.approx(0.0, abs=1e-4)


def test_search_on_empty_store_returns_empty_list():
    store = FAISSVectorStore(dimension=4)
    results = store.search(_sample_vectors(n=1)[0], top_k=3)
    assert results == []


def test_add_mismatched_lengths_raises():
    store = FAISSVectorStore(dimension=4)
    with pytest.raises(ValueError):
        store.add(_sample_vectors(n=3), [{"text": "only one"}])


def test_top_k_larger_than_store_size_is_clamped():
    store = FAISSVectorStore(dimension=4)
    vectors = _sample_vectors(n=2)
    store.add(vectors, [{"text": "a", "source": "x", "chunk_index": 0}, {"text": "b", "source": "x", "chunk_index": 1}])

    results = store.search(vectors[0], top_k=10)
    assert len(results) == 2


def test_save_and_load_round_trip(tmp_path):
    store = FAISSVectorStore(dimension=4)
    vectors = _sample_vectors(n=3)
    metadata = [{"text": f"chunk-{i}", "source": "doc.txt", "chunk_index": i} for i in range(3)]
    store.add(vectors, metadata)

    store.save(tmp_path / "index")
    loaded = FAISSVectorStore.load(tmp_path / "index")

    assert loaded.size == 3
    assert loaded.dimension == 4
    results = loaded.search(vectors[1], top_k=1)
    assert results[0]["text"] == "chunk-1"
