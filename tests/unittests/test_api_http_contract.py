from dataclasses import dataclass

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


def test_root_gui_is_immediately_available_without_login():
    client = TestClient(api_main.app)
    response = client.get("/")

    assert response.status_code == 200
    assert "Run Agent" in response.text
    assert "type=\"password\"" not in response.text
    assert "Sign in" not in response.text
