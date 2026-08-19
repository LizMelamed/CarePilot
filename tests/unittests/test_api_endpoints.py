import asyncio
from dataclasses import dataclass

from src.api import main as api_main


@dataclass(frozen=True)
class FakeOrchestratorResult:
    status: str
    error: str | None
    response: str
    steps: list[dict]
    execution_id: str | None


class FakeOrchestrator:
    async def execute(self, username: str, prompt: str):
        return FakeOrchestratorResult(
            status="success",
            error=None,
            response=f"handled {username}: {prompt}",
            steps=[
                {
                    "module": "PlanningLLM",
                    "system_prompt": "s",
                    "user_prompt": prompt,
                    "response": "r",
                    "step_order": 1,
                },
                {
                    "module": "SingleTaskExecutorLLM",
                    "system_prompt": "s",
                    "user_prompt": prompt,
                    "response": "r",
                    "step_order": 2,
                },
                {
                    "module": "RePlanLLM",
                    "system_prompt": "s",
                    "user_prompt": prompt,
                    "response": "r",
                    "step_order": 3,
                },
                {
                    "module": "SafetyGuardLLM",
                    "system_prompt": "s",
                    "user_prompt": prompt,
                    "response": "r",
                    "step_order": 4,
                },
            ],
            execution_id="exec_1",
        )


def test_team_info_shape(monkeypatch):
    monkeypatch.setenv("CAREPILOT_GROUP_BATCH_ORDER_NUMBER", "1")
    monkeypatch.setenv("CAREPILOT_TEAM_NAME", "CarePilot")
    monkeypatch.setenv("CAREPILOT_STUDENTS_JSON", '[{"name":"A","email":"a@example.com"}]')

    body = asyncio.run(api_main.team_info())

    assert set(body) == {"group_batch_order_number", "team_name", "students"}
    assert set(body["students"][0]) == {"name", "email"}


def test_team_info_has_no_committed_identity_defaults(monkeypatch):
    monkeypatch.delenv("CAREPILOT_GROUP_BATCH_ORDER_NUMBER", raising=False)
    monkeypatch.delenv("CAREPILOT_TEAM_NAME", raising=False)
    monkeypatch.delenv("CAREPILOT_STUDENTS_JSON", raising=False)

    body = asyncio.run(api_main.team_info())

    assert body == {
        "group_batch_order_number": "",
        "team_name": "",
        "students": [],
    }


def test_agent_info_shape():
    body = asyncio.run(api_main.agent_info())

    assert set(body) == {"description", "purpose", "prompt_template", "prompt_examples"}
    assert set(body["prompt_template"]) == {"template"}
    assert {"prompt", "full_response", "steps"} <= set(body["prompt_examples"][0])


def test_model_architecture_png():
    api_main._architecture_png.cache_clear()
    response = asyncio.run(api_main.model_architecture())
    asyncio.run(api_main.model_architecture())

    assert response.media_type == "image/png"
    assert response.body.startswith(b"\x89PNG")
    assert api_main._architecture_png.cache_info().hits == 1


def test_execute_shape(monkeypatch):
    monkeypatch.setattr(api_main, "_orchestrator", lambda: FakeOrchestrator())

    body = asyncio.run(
        api_main.execute(
            api_main.ExecuteRequest(prompt="hello"),
            x_carepilot_username="patient_1",
        )
    )

    body_dict = body.model_dump()
    assert set(body_dict) == {"status", "error", "response", "steps"}
    assert body.status == "ok"
    assert body.error is None
    assert [step["module"] for step in body.steps] == [
        "PlanningLLM",
        "SingleTaskExecutorLLM",
        "RePlanLLM",
        "SafetyGuardLLM",
    ]


def test_execute_error_matches_required_contract(monkeypatch):
    class FailedOrchestrator:
        async def execute(self, username: str, prompt: str):
            return FakeOrchestratorResult(
                status="error",
                error="Human-readable failure",
                response="internal fallback must not leak into the API response",
                steps=[],
                execution_id=None,
            )

    monkeypatch.setattr(api_main, "_orchestrator", lambda: FailedOrchestrator())
    body = asyncio.run(
        api_main.execute(
            api_main.ExecuteRequest(prompt="hello"),
            x_carepilot_username="patient_1",
        )
    )

    assert body.model_dump() == {
        "status": "error",
        "error": "Human-readable failure",
        "response": None,
        "steps": [],
    }


def test_agent_step_shape_matches_assignment():
    body = asyncio.run(api_main.agent_info())

    for step in body["prompt_examples"][0]["steps"]:
        assert set(step) == {"module", "prompt", "response"}
        assert set(step["prompt"]) == {"System_prompt", "User_prompt"}
