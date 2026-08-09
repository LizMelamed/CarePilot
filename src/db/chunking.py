MEDICAL_CHUNK_SIZE = 3500
"""Chunk size for medical-document RAG ingestion."""

MEDICAL_CHUNK_OVERLAP = 600
"""Chunk overlap for medical-document RAG ingestion."""

MEDICAL_CHUNK_SEPARATORS = [
    "\n# ",
    "\n## ",
    "\n\n",
    "\n",
    ". ",
    " ",
]
"""Separators used for medical-document RAG ingestion."""
