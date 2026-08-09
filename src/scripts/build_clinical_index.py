"""Build the clinical RAG vector index from the cleaned clinical corpus.

Run from the CarePilot repository root:
    python -m src.scripts.build_clinical_index
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.agents.embedder import Embedder
from src.db.chunking import (
    MEDICAL_CHUNK_OVERLAP,
    MEDICAL_CHUNK_SEPARATORS,
    MEDICAL_CHUNK_SIZE,
)
from src.db.vector_store import get_clinical_vector_store
from src.utils.utils import from_project_path

CLINICAL_CLEAN_DIR = from_project_path("data/clinical_corpus/cleaned")
EMBED_BATCH_SIZE = 10
UPSERT_BATCH_SIZE = 100


def _read_markdown_with_metadata(path: Path) -> tuple[dict[str, Any], str]:
    raw_text = path.read_text(encoding="utf-8")
    if not raw_text.startswith("---\n"):
        return {}, raw_text

    parts = raw_text.split("---\n", 2)
    if len(parts) != 3:
        return {}, raw_text

    metadata = json.loads(parts[1])
    return metadata, parts[2].strip()


def _collect_chunks() -> list[tuple[str, dict[str, Any]]]:
    if not CLINICAL_CLEAN_DIR.exists():
        raise FileNotFoundError(f"Cleaned corpus directory not found: {CLINICAL_CLEAN_DIR}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=MEDICAL_CHUNK_SIZE,
        chunk_overlap=MEDICAL_CHUNK_OVERLAP,
        separators=MEDICAL_CHUNK_SEPARATORS,
    )
    chunk_records: list[tuple[str, dict[str, Any]]] = []

    for path in sorted(CLINICAL_CLEAN_DIR.glob("*.md")):
        source_metadata, body = _read_markdown_with_metadata(path)
        chunks = splitter.split_text(body)
        for chunk_index, chunk_text in enumerate(chunks):
            chunk_id = f"{path.stem}:{chunk_index}"
            metadata = {
                "text": chunk_text,
                "source_url": source_metadata.get("source_url", ""),
                "title": source_metadata.get("title", path.stem),
                "topic_tags": source_metadata.get("topic_tags", []),
                "source_name": source_metadata.get("source_name", ""),
                "source_path": str(path.relative_to(from_project_path(""))),
                "chunk_index": chunk_index,
            }
            chunk_records.append((chunk_id, metadata))

    return chunk_records


async def _build_index() -> None:
    chunk_records = _collect_chunks()
    if not chunk_records:
        raise ValueError(f"No markdown files found in {CLINICAL_CLEAN_DIR}")

    vector_store = get_clinical_vector_store()
    clear = getattr(vector_store, "clear", None)
    if clear is not None:
        await clear()

    embedder = Embedder().get()
    ids = [chunk_id for chunk_id, _ in chunk_records]
    texts = [metadata["text"] for _, metadata in chunk_records]
    metadatas = [metadata for _, metadata in chunk_records]

    indexed = 0
    for start in range(0, len(ids), EMBED_BATCH_SIZE):
        end = start + EMBED_BATCH_SIZE
        vectors = await embedder.aembed_documents(
            texts[start:end],
            chunk_size=EMBED_BATCH_SIZE,
        )
        await vector_store.upsert(
            ids[start:end],
            vectors,
            metadatas[start:end],
        )
        indexed += len(vectors)
        if indexed % UPSERT_BATCH_SIZE == 0 or indexed == len(ids):
            print(f"Indexed {indexed}/{len(ids)} clinical chunks")


def main() -> None:
    asyncio.run(_build_index())


if __name__ == "__main__":
    main()
