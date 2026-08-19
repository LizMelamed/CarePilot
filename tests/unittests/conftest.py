from __future__ import annotations

import pytest

from src.agents.embedder import Embedder
from src.db.vector_store import get_clinical_vector_store
from src.utils.my_env import MyEnv
from src.utils.singleton import SingletonMeta


SERVICE_ENV_KEYS = (
    "LLM_URL",
    "LLM_KEY",
    "LLM_MODEL",
    "SAFETY_LLM_MODEL",
    "EMBEDDER_URL",
    "EMBEDDER_KEY",
    "EMBEDDER_MODEL",
    "DB_URL",
    "DB_AUTH_TOKEN",
    "PINECONE_API_KEY",
    "PINECONE_INDEX",
)


@pytest.fixture(autouse=True)
def isolate_unit_tests_from_local_secrets(monkeypatch):
    """Prevent offline unit tests from ever loading the developer's real .env."""
    monkeypatch.setattr(MyEnv, "ENV_PATH", "res/configs/.env.unit-tests-disabled")
    for key in SERVICE_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)

    SingletonMeta._instances.pop(MyEnv, None)
    SingletonMeta._instances.pop(Embedder, None)
    get_clinical_vector_store.cache_clear()
    yield
    SingletonMeta._instances.pop(MyEnv, None)
    SingletonMeta._instances.pop(Embedder, None)
    get_clinical_vector_store.cache_clear()
