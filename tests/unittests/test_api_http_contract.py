from dataclasses import dataclass

import pytest

from fastapi.testclient import TestClient

from src.api import main as api_main


@dataclass(frozen=True)
class FakeResult:
    status: str = "success"
    error: str | None = None
    response: str = "Offline test response"
    steps: list[dict] | None = None
    execution_id: str | None = None


class FakeOrchestrator:
    async def execute(self, username: str, prompt: str):
        return FakeResult(
            steps=[
                {
                    "module": "PlanningLLM",
                    "prompt": {
                        "System_prompt": "offline system prompt",
                        "User_prompt": prompt,
                    },
                    "response": '{"direct_answer":"Offline test response"}',
                }
            ]
        )


def test_required_http_endpoints_match_submission_contract(monkeypatch):
    monkeypatch.setattr(api_main, "_orchestrator", lambda: FakeOrchestrator())
    client = TestClient(api_main.app)

    team = client.get("/api/team_info")
    assert team.status_code == 200
    assert set(team.json()) == {"group_batch_order_number", "team_name", "students"}

    agent = client.get("/api/agent_info")
    assert agent.status_code == 200
    assert set(agent.json()) == {"description", "purpose", "prompt_template", "prompt_examples"}

    architecture = client.get("/api/model_architecture")
    assert architecture.status_code == 200
    assert architecture.headers["content-type"].startswith("image/png")
    assert architecture.content.startswith(b"\x89PNG")

    execution = client.post("/api/execute", json={"prompt": "hello"})
    assert execution.status_code == 200
    body = execution.json()
    assert set(body) == {"status", "error", "response", "steps"}
    assert body["status"] == "ok"
    assert body["error"] is None
    assert body["response"] == "Offline test response"
    assert set(body["steps"][0]) == {"module", "prompt", "response"}
    assert set(body["steps"][0]["prompt"]) == {"System_prompt", "User_prompt"}


@pytest.mark.parametrize(
    ("content", "content_type"),
    [
        ('{"prompt":""}', "application/json"),
        ("{}", "application/json"),
        ('{"prompt":', "application/json"),
    ],
)
def test_execute_validation_errors_match_submission_contract(content, content_type):
    client = TestClient(api_main.app)

    response = client.post("/api/execute", content=content, headers={"content-type": content_type})

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"status", "error", "response", "steps"}
    assert body["status"] == "error"
    assert body["error"]
    assert body["response"] is None
    assert body["steps"] == []


def test_users_endpoint_hides_integration_test_identities(monkeypatch):
    class FakeDB:
        async def list_users(self):
            return ["patient_1", "integration_lab_deadbeef", "alex_t"]

    monkeypatch.setattr(api_main, "_db_handler", lambda: FakeDB())

    response = TestClient(api_main.app).get("/api/users")

    assert response.status_code == 200
    assert response.json() == {"users": ["alex_t", "patient_1"]}


def test_conversation_history_is_patient_scoped_and_returns_public_trace(monkeypatch):
    class FakeDB:
        calls = []

        async def get_execution_history(self, username, limit=10):
            self.calls.append((username, limit))
            return [
                {
                    "id": "exec_1",
                    "prompt": "What did my last test show?",
                    "final_response": "Your recorded result was stable.",
                    "status": "success",
                    "error": None,
                    "created_at": "2026-08-23T09:30:00+00:00",
                    "steps": [
                        {
                            "module": "PlanningLLM",
                            "system_prompt": "plan safely",
                            "user_prompt": "What did my last test show?",
                            "response": "{}",
                            "step_order": 1,
                        }
                    ],
                }
            ]

    db = FakeDB()
    monkeypatch.setattr(api_main, "_db_handler", lambda: db)

    response = TestClient(api_main.app).get(
        "/api/conversations?limit=7",
        headers={"X-CarePilot-Username": "patient_7"},
    )

    assert response.status_code == 200
    assert db.calls == [("patient_7", 7)]
    body = response.json()
    assert set(body) == {"conversations"}
    assert body["conversations"][0] == {
        "id": "exec_1",
        "prompt": "What did my last test show?",
        "response": "Your recorded result was stable.",
        "status": "success",
        "error": None,
        "created_at": "2026-08-23T09:30:00+00:00",
        "steps": [
            {
                "module": "PlanningLLM",
                "prompt": {
                    "System_prompt": "plan safely",
                    "User_prompt": "What did my last test show?",
                },
                "response": "{}",
            }
        ],
    }


def test_conversation_history_limit_is_bounded():
    response = TestClient(api_main.app).get("/api/conversations?limit=51")

    assert response.status_code == 422


def test_document_content_endpoint_is_patient_scoped_and_decodes_file_name(monkeypatch):
    class FakeDB:
        calls = []

        async def get_file(self, username, file_name):
            self.calls.append((username, file_name))
            return "Lab result: hemoglobin 11.0 g/dL"

    db = FakeDB()
    monkeypatch.setattr(api_main, "_db_handler", lambda: db)

    response = TestClient(api_main.app).get(
        "/api/documents/lab%20results.txt",
        headers={"X-CarePilot-Username": "patient_7"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "file_name": "lab results.txt",
        "content": "Lab result: hemoglobin 11.0 g/dL",
    }
    assert db.calls == [("patient_7", "lab results.txt")]


def test_document_content_endpoint_returns_404_when_file_is_not_owned(monkeypatch):
    class FakeDB:
        async def get_file(self, username, file_name):
            return None

    monkeypatch.setattr(api_main, "_db_handler", lambda: FakeDB())

    response = TestClient(api_main.app).get(
        "/api/documents/private.txt",
        headers={"X-CarePilot-Username": "other_patient"},
    )

    assert response.status_code == 404
    assert response.json() == {"detail": "File 'private.txt' not found."}


def test_root_gui_is_immediately_available_without_login():
    client = TestClient(api_main.app)
    response = client.get("/")

    assert response.status_code == 200
    assert "Run Agent" in response.text
    assert "Conversation history" in response.text
    assert "Ask a follow-up" in response.text
    assert "/api/conversations?limit=20" in response.text
    assert "type=\"password\"" not in response.text
    assert "Sign in" not in response.text
