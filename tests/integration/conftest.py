"""Shared fixtures for integration tests that exercise the real remote LLM/embedder stack.

These tests are slow and environment-dependent -- they skip (rather than fail) when the
configured LLM endpoint isn't reachable, so `pytest tests/` stays green on machines without
a configured remote stack. Run explicitly with: python -m pytest tests/integration
"""

from __future__ import annotations

import os

import pytest
import requests

from src.utils.my_env import MyEnv


@pytest.fixture(scope="session", autouse=True)
def isolated_integration_database():
    """Require a dedicated Supabase project before any live integration test can run."""
    env = MyEnv()
    integration_url = MyEnv.get("INTEGRATION_DB_URL")
    integration_token = MyEnv.get("INTEGRATION_DB_AUTH_TOKEN")
    if not integration_url or not integration_token:
        pytest.skip(
            "INTEGRATION_DB_URL and INTEGRATION_DB_AUTH_TOKEN are required; "
            "refusing to run integration tests against the application database."
        )
    if integration_url.rstrip("/") == (env.get_db_url() or "").rstrip("/"):
        pytest.skip("INTEGRATION_DB_URL must use a different Supabase project than DB_URL.")

    previous_url = os.environ.get("DB_URL")
    previous_token = os.environ.get("DB_AUTH_TOKEN")
    os.environ["DB_URL"] = integration_url
    os.environ["DB_AUTH_TOKEN"] = integration_token
    try:
        yield
    finally:
        if previous_url is None:
            os.environ.pop("DB_URL", None)
        else:
            os.environ["DB_URL"] = previous_url
        if previous_token is None:
            os.environ.pop("DB_AUTH_TOKEN", None)
        else:
            os.environ["DB_AUTH_TOKEN"] = previous_token


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
    """The clinical index already lives in Pinecone; query it as-is instead of rebuilding it."""
