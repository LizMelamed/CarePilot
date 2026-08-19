from src.utils.my_env import MyEnv


def test_llm_key_falls_back_to_shared_embedder_key(monkeypatch):
    # Keep this unit test independent of the developer's ignored local .env.
    monkeypatch.setattr(MyEnv, "ENV_PATH", "res/configs/.env.test-missing")
    monkeypatch.delenv("LLM_KEY", raising=False)
    monkeypatch.setenv("EMBEDDER_KEY", "shared-course-key")

    assert MyEnv().get_llm_key() == "shared-course-key"


def test_explicit_llm_key_takes_precedence(monkeypatch):
    monkeypatch.setenv("LLM_KEY", "text-key")
    monkeypatch.setenv("EMBEDDER_KEY", "embedding-key")

    assert MyEnv().get_llm_key() == "text-key"
