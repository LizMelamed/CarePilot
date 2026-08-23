from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import requests
from requests.exceptions import RequestException


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEPLOYMENT_URL = os.environ.get("CAREPILOT_VERCEL_URL", "https://care-pilot-dun.vercel.app").rstrip("/")
LIVE_TIMEOUT_SECONDS = 30
EXECUTE_TIMEOUT_SECONDS = 295


def _get_live(path: str, **kwargs):
    try:
        return requests.get(f"{DEPLOYMENT_URL}{path}", timeout=LIVE_TIMEOUT_SECONDS, **kwargs)
    except RequestException as exc:
        pytest.skip(f"Live Vercel deployment unreachable at {DEPLOYMENT_URL}{path}: {exc}")


def _post_live(path: str, **kwargs):
    try:
        return requests.post(f"{DEPLOYMENT_URL}{path}", **kwargs)
    except RequestException as exc:
        pytest.skip(f"Live Vercel deployment unreachable at {DEPLOYMENT_URL}{path}: {exc}")


def test_vercel_uses_root_fastapi_entrypoint_without_path_rewrite():
    config = json.loads((PROJECT_ROOT / "vercel.json").read_text(encoding="utf-8"))

    assert config["framework"] == "fastapi"
    assert config["fluid"] is True
    assert "rewrites" not in config
    assert config["functions"]["app.py"] == {
        "maxDuration": 295,
        "includeFiles": "static/**",
    }
    assert not (PROJECT_ROOT / "api" / "index.py").exists()


def test_vercel_entrypoint_exports_the_application():
    from app import app as vercel_app
    from src.api.main import app as application

    assert vercel_app is application


def test_vercel_python_version_is_supported_and_pinned():
    assert (PROJECT_ROOT / ".python-version").read_text(encoding="utf-8").strip() == "3.12"


def test_gui_distinguishes_llm_unavailable_and_offers_retry():
    html = (PROJECT_ROOT / "static" / "index.html").read_text(encoding="utf-8")

    assert "SERVICE_UNAVAILABLE_MARKER" in html
    assert "service-unavailable" in html
    assert "var(--warn-soft)" in html
    assert 'id="retry-button"' in html
    assert "retryButton.addEventListener('click', runAgent)" in html


def test_gui_includes_patient_scoped_document_viewer():
    html = (PROJECT_ROOT / "static" / "index.html").read_text(encoding="utf-8")

    assert 'id="document-modal"' in html
    assert "view-doc-btn" in html
    assert "encodeURIComponent(fileName)" in html
    assert "documentModalContent.textContent = data.content" in html


def test_live_root_gui_is_immediately_available_without_login():
    response = _get_live("/")

    assert response.status_code == 200
    assert "Run Agent" in response.text
    assert 'type="password"' not in response.text
    assert "Sign in" not in response.text


def test_live_team_info_matches_submission_contract():
    response = _get_live("/api/team_info")

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"group_batch_order_number", "team_name", "students"}
    assert isinstance(body["students"], list)
    for student in body["students"]:
        assert set(student) == {"name", "email"}


def test_live_agent_info_matches_submission_contract():
    response = _get_live("/api/agent_info")

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"description", "purpose", "prompt_template", "prompt_examples"}
    assert set(body["prompt_template"]) == {"template"}

    assert len(body["prompt_examples"]) > 0
    for example in body["prompt_examples"]:
        assert {"prompt", "full_response", "steps"} <= set(example)
        for step in example["steps"]:
            assert set(step) == {"module", "prompt", "response"}
            assert set(step["prompt"]) == {"System_prompt", "User_prompt"}


def test_live_model_architecture_returns_png():
    response = _get_live("/api/model_architecture")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/png")
    assert response.content.startswith(b"\x89PNG")


@pytest.mark.skipif(
    not os.environ.get("CAREPILOT_RUN_LIVE_EXECUTE_TEST"),
    reason=(
        "Hits the live LLM pipeline (real cost against the project's LLMod.ai budget) and can "
        "take up to 295s; set CAREPILOT_RUN_LIVE_EXECUTE_TEST=1 to opt in."
    ),
)
def test_live_execute_matches_submission_contract():
    response = _post_live(
        "/api/execute",
        json={"prompt": "Hi, what can you help me with?"},
        timeout=EXECUTE_TIMEOUT_SECONDS,
    )

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"status", "error", "response", "steps"}
    assert body["status"] in {"ok", "error"}
    if body["status"] == "ok":
        assert body["error"] is None
        assert isinstance(body["response"], str)
    else:
        assert body["response"] is None
        assert isinstance(body["error"], str)

    for step in body["steps"]:
        assert set(step) == {"module", "prompt", "response"}
        assert set(step["prompt"]) == {"System_prompt", "User_prompt"}
