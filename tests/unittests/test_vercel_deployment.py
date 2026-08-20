from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


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
