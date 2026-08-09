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


def test_agent_info_shape():
    body = asyncio.run(api_main.agent_info())

    assert set(body) == {"description", "purpose", "prompt_template", "prompt_examples"}
    assert set(body["prompt_template"]) == {"template"}
    assert {"prompt", "full_response", "steps"} <= set(body["prompt_examples"][0])


def test_model_architecture_png():
    response = asyncio.run(api_main.model_architecture())

    assert response.media_type == "image/png"
    assert response.body.startswith(b"\x89PNG")


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
    assert body.status == "success"
    assert [step["module"] for step in body.steps] == [
        "PlanningLLM",
        "SingleTaskExecutorLLM",
        "RePlanLLM",
        "SafetyGuardLLM",
    ]
