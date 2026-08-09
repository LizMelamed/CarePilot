import asyncio

import pytest

from src.db.faiss_vector_store import FaissVectorStore
from src.db.vector_store import get_clinical_vector_store


def test_faiss_upsert_query_filter_persist_delete(tmp_path):
    store = FaissVectorStore(tmp_path)

    asyncio.run(
        store.upsert(
            ["clinical", "insurance"],
            [[1.0, 0.0], [0.0, 1.0]],
            [
                {"text": "neutropenia infection", "topic_tags": ["clinical"], "title": "Clinical"},
                {"text": "appeal denied", "topic_tags": ["insurance"], "title": "Insurance"},
            ],
        )
    )

    assert asyncio.run(store.query([1.0, 0.0], top_k=1))[0].id == "clinical"
    assert asyncio.run(store.query([1.0, 0.0], top_k=1, filter={"topic_tags": "insurance"}))[0].id == "insurance"

    reloaded = FaissVectorStore(tmp_path)
    assert asyncio.run(reloaded.query([0.0, 1.0], top_k=1))[0].id == "insurance"

    asyncio.run(reloaded.delete(["insurance"]))
    remaining = asyncio.run(reloaded.query([0.0, 1.0], top_k=2))
    assert [match.id for match in remaining] == ["clinical"]


def test_faiss_validates_inputs(tmp_path):
    store = FaissVectorStore(tmp_path)

    with pytest.raises(ValueError, match="same length"):
        asyncio.run(store.upsert(["a"], [[1.0]], []))

    with pytest.raises(ValueError, match="unique"):
        asyncio.run(store.upsert(["a", "a"], [[1.0], [2.0]], [{}, {}]))

    with pytest.raises(ValueError, match="same dimension"):
        asyncio.run(store.upsert(["a", "b"], [[1.0], [1.0, 2.0]], [{}, {}]))


def test_vector_store_factory_rejects_unknown_backend(monkeypatch):
    monkeypatch.setenv("VECTOR_STORE_BACKEND", "unknown")

    with pytest.raises(ValueError, match="Unsupported VECTOR_STORE_BACKEND"):
        get_clinical_vector_store()
