import sys
from types import SimpleNamespace

import pytest

from src.db.pinecone_vector_store import PineconeVectorStore
from src.db.vector_store import get_clinical_vector_store


class FakePineconeClient:
    def __init__(self, api_key):
        self.api_key = api_key

    def Index(self, index_name):
        return SimpleNamespace(index_name=index_name)


def test_vector_store_factory_returns_pinecone(monkeypatch):
    monkeypatch.setenv("PINECONE_API_KEY", "test-key")
    monkeypatch.setenv("PINECONE_INDEX", "test-index")
    monkeypatch.setitem(sys.modules, "pinecone", SimpleNamespace(Pinecone=FakePineconeClient))

    store = get_clinical_vector_store()

    assert isinstance(store, PineconeVectorStore)
    assert store._index.index_name == "test-index"


def test_pinecone_vector_store_requires_api_key(monkeypatch):
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)
    monkeypatch.setenv("PINECONE_INDEX", "test-index")

    with pytest.raises(ValueError, match="PINECONE_API_KEY"):
        get_clinical_vector_store()


def test_pinecone_vector_store_uses_submission_index_default(monkeypatch):
    monkeypatch.setenv("PINECONE_API_KEY", "test-key")
    monkeypatch.delenv("PINECONE_INDEX", raising=False)
    monkeypatch.setitem(sys.modules, "pinecone", SimpleNamespace(Pinecone=FakePineconeClient))

    store = get_clinical_vector_store()

    assert store._index.index_name == "carepilot-clinical-rag"
