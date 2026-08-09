from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class VectorMatch:
    """One retrieved vector-store match."""

    id: str
    score: float
    metadata: dict[str, Any]


class VectorStore(ABC):
    """Small vector-store boundary used by RAG code."""

    @abstractmethod
    async def upsert(
        self,
        ids: list[str],
        vectors: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    async def query(
        self,
        vector: list[float],
        top_k: int,
        filter: dict[str, Any] | None = None,
    ) -> list[VectorMatch]:
        raise NotImplementedError

    @abstractmethod
    async def delete(self, ids: list[str]) -> None:
        raise NotImplementedError


def get_clinical_vector_store() -> VectorStore:
    """Return the clinical RAG vector store."""

    from src.db.pinecone_vector_store import PineconeVectorStore

    return PineconeVectorStore()
