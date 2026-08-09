from __future__ import annotations

from src.agents.embedder import Embedder
from src.db.vector_store import VectorMatch, get_clinical_vector_store


async def query_clinical_rag(query: str, top_k: int = 5) -> list[VectorMatch]:
    """Retrieve clinical-corpus chunks for a natural-language query."""

    if top_k <= 0:
        return []
    query_vector = await Embedder().get().aembed_query(query)
    vector_store = get_clinical_vector_store()
    return await vector_store.query(query_vector, top_k=top_k)
