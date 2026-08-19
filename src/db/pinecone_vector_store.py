from __future__ import annotations

from typing import Any

from src.db.vector_store import VectorMatch, VectorStore
from src.utils.my_env import MyEnv


class PineconeVectorStore(VectorStore):
    """Pinecone adapter for the clinical vector-store interface."""

    DEFAULT_INDEX = "carepilot-clinical-rag"

    def __init__(self):
        env = MyEnv()
        api_key = env.get("PINECONE_API_KEY")
        index_name = env.get("PINECONE_INDEX") or self.DEFAULT_INDEX
        if not api_key:
            raise ValueError("PINECONE_API_KEY is required for PineconeVectorStore")
        try:
            from pinecone import Pinecone
        except ImportError as exc:
            raise ImportError("pinecone is required for PineconeVectorStore") from exc

        self._client = Pinecone(api_key=api_key)
        self._index = self._client.Index(index_name)

    async def upsert(
        self,
        ids: list[str],
        vectors: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> None:
        if not (len(ids) == len(vectors) == len(metadatas)):
            raise ValueError("ids, vectors, and metadatas must have the same length")
        records = [
            {"id": item_id, "values": vector, "metadata": metadata}
            for item_id, vector, metadata in zip(ids, vectors, metadatas)
        ]
        if records:
            response = self._index.upsert(vectors=records)
            upserted_count = self._match_value(response, "upserted_count")
            if upserted_count is not None and int(upserted_count) != len(records):
                raise RuntimeError(f"Pinecone upserted {upserted_count}/{len(records)} records")

    async def query(
        self,
        vector: list[float],
        top_k: int,
        filter: dict[str, Any] | None = None,
    ) -> list[VectorMatch]:
        if top_k <= 0:
            return []

        response = self._index.query(
            vector=vector,
            top_k=top_k,
            include_metadata=True,
            filter=filter,
        )
        matches = response.get("matches", []) if isinstance(response, dict) else response.matches
        return [
            VectorMatch(
                id=self._match_value(match, "id"),
                score=float(self._match_value(match, "score")),
                metadata=self._match_value(match, "metadata") or {},
            )
            for match in matches
        ]

    async def delete(self, ids: list[str]) -> None:
        if ids:
            self._index.delete(ids=ids)

    @staticmethod
    def _match_value(match: Any, key: str) -> Any:
        if isinstance(match, dict):
            return match.get(key)
        return getattr(match, key)
