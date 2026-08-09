"""Shared fixtures for integration tests that exercise the real remote LLM/embedder stack.

These tests are slow and environment-dependent -- they skip (rather than fail) when the
configured LLM endpoint isn't reachable, so `pytest tests/` stays green on machines without
a configured remote stack. Run explicitly with: python -m pytest tests/integration
"""

from __future__ import annotations

import asyncio

import pytest
import requests

from src.utils.my_env import MyEnv


@pytest.fixture(scope="session")
def remote_llm_available() -> None:
    env = MyEnv()
    llm_url = env.get_llm_url()
    if not llm_url:
        pytest.skip("LLM_URL is not configured; skipping integration test.")
    headers = {"Authorization": f"Bearer {env.get_llm_key()}"} if env.get_llm_key() else {}
    try:
        response = requests.get(f"{llm_url.rstrip('/')}/models", headers=headers, timeout=3)
        response.raise_for_status()
    except Exception as e:
        pytest.skip(f"LLM endpoint '{llm_url}' is not reachable ({e}); skipping integration test.")


@pytest.fixture(scope="session")
def clinical_index_ready(remote_llm_available) -> None:
    """Build the remote clinical index so query_clinical_rag has data."""
    from src.scripts.build_clinical_index import _build_index

    asyncio.run(_build_index())
