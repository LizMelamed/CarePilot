"""Shared fixtures for integration tests that exercise the real LLM/embedder stack (Ollama).

These tests are slow and environment-dependent -- they skip (rather than fail) when the
configured LLM endpoint isn't reachable, so `pytest tests/` stays green on machines without
Ollama running. Run explicitly with: python -m pytest tests/integration
"""

from __future__ import annotations

import asyncio

import pytest
import requests

from src.utils.my_env import MyEnv


@pytest.fixture(scope="session")
def ollama_available() -> None:
    env = MyEnv()
    llm_url = env.get_llm_url() or "http://localhost:11434/v1"
    headers = {"Authorization": f"Bearer {env.get_llm_key()}"} if env.get_llm_key() else {}
    try:
        response = requests.get(f"{llm_url.rstrip('/')}/models", headers=headers, timeout=3)
        response.raise_for_status()
    except Exception as e:
        pytest.skip(f"LLM endpoint '{llm_url}' is not reachable ({e}); skipping integration test.")


@pytest.fixture(scope="session")
def clinical_index_ready(ollama_available) -> None:
    """Build the clinical FAISS index once (if it isn't already on disk) so query_clinical_rag has data."""
    from src.db.faiss_vector_store import INDEX_PATH
    from src.scripts.build_clinical_index import _build_index

    if not INDEX_PATH.exists():
        asyncio.run(_build_index())
