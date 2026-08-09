import asyncio

from src.agents.executor_agent import ExecutorResult
from src.agents.planner_agent import PlannerResult
from src.agents.replanner_agent import ReplannerResult
from src.agents.workflow_types import AgentStep, PlannedTask, TaskResult
from src.carepilot.orchestrator import CarePilotOrchestrator


class FakeDB:
    def __init__(self):
        self.saved = None

    async def get_execution_history(self, username, limit=3):
        return [{"prompt": "previous"}]

    async def get_patient_data(self, username):
        return {"home_city": "Haifa"}

    async def save_execution(self, **kwargs):
        self.saved = kwargs
        return "exec_1"


class FakePlanner:
    def __init__(self, tasks=None, direct_answer=None):
        self._tasks = tasks if tasks is not None else [PlannedTask("task_1", "read docs", "patient_db")]
        self._direct_answer = direct_answer

    async def act(self, ctx):
        return PlannerResult(
            tasks=[] if self._direct_answer is not None else self._tasks,
            step=AgentStep("PlanningLLM", "s", "u", "r", 1),
            direct_answer=self._direct_answer,
        )


class FakeExecutor:
    def __init__(self, succeed=True):
        self._succeed = succeed

    async def act(self, ctx):
        return ExecutorResult(
            task_result=TaskResult(
                ctx.task.task_id, "executor result" if self._succeed else "", "get_patient_documents", self._succeed
            ),
            step=AgentStep("SingleTaskExecutorLLM", "s", "u", "r", ctx.step_order),
            source_documents=["source doc"] if self._succeed else [],
        )


class FakeReplanner:
    async def act(self, ctx):
        return ReplannerResult(
            done=True,
            final_answer_context="draft answer",
            new_tasks=[],
            step=AgentStep("RePlanLLM", "s", "u", "r", ctx.step_order),
        )


class FakeSafetyGuard:
    async def act(self, ctx):
        return {
            "is_grounded": True,
            "safety_violations_found": [],
            "action_taken": "PASS",
            "final_output": "safe final",
        }


class FakeLogger:
    def exception(self, message):
        self.message = message


def _orchestrator_with_fakes(db=None, planner=None, executor=None, replanner=None):
    orchestrator = CarePilotOrchestrator.__new__(CarePilotOrchestrator)
    orchestrator._logger = FakeLogger()
    orchestrator._db_handler = db or FakeDB()
    orchestrator._planner = planner or FakePlanner()
    orchestrator._executor = executor or FakeExecutor()
    orchestrator._replanner = replanner or FakeReplanner()
    orchestrator._safety_guard = FakeSafetyGuard()
    return orchestrator


def test_orchestrator_tier0_direct_answer_skips_executor_and_replanner():
    db = FakeDB()
    orchestrator = _orchestrator_with_fakes(db, planner=FakePlanner(direct_answer="no lookup needed"))
    result = asyncio.run(orchestrator.execute("patient_1", "hi"))

    assert result.status == "success"
    assert result.response == "safe final"
    assert [step["module"] for step in result.steps] == ["PlanningLLM", "SafetyGuardLLM"]
    assert db.saved["final_response"] == "safe final"


def test_orchestrator_tier1_single_task_skips_replanner():
    db = FakeDB()
    result = asyncio.run(_orchestrator_with_fakes(db).execute("patient_1", "prepare"))

    assert result.status == "success"
    assert result.response == "safe final"
    assert result.execution_id == "exec_1"
    assert [step["module"] for step in result.steps] == [
        "PlanningLLM",
        "SingleTaskExecutorLLM",
        "SafetyGuardLLM",
    ]
    assert db.saved["username_or_session"] == "patient_1"
    assert db.saved["final_response"] == "safe final"


def test_orchestrator_tier2_multi_task_uses_replanner():
    db = FakeDB()
    planner = FakePlanner(tasks=[
        PlannedTask("task_1", "read labs", "patient_db"),
        PlannedTask("task_2", "draft message", "message_drafting"),
    ])
    result = asyncio.run(_orchestrator_with_fakes(db, planner=planner).execute("patient_1", "prepare"))

    assert result.status == "success"
    assert [step["module"] for step in result.steps] == [
        "PlanningLLM",
        "SingleTaskExecutorLLM",
        "SingleTaskExecutorLLM",
        "RePlanLLM",
        "SafetyGuardLLM",
    ]


def test_orchestrator_escalates_to_replanner_when_single_task_fails():
    db = FakeDB()
    orchestrator = _orchestrator_with_fakes(db, executor=FakeExecutor(succeed=False))
    result = asyncio.run(orchestrator.execute("patient_1", "prepare"))

    assert result.status == "success"
    assert [step["module"] for step in result.steps] == [
        "PlanningLLM",
        "SingleTaskExecutorLLM",
        "RePlanLLM",
        "SafetyGuardLLM",
    ]


def test_orchestrator_error_path_still_persists():
    class FailingPlanner:
        async def act(self, ctx):
            return None

    db = FakeDB()
    orchestrator = _orchestrator_with_fakes(db, planner=FailingPlanner())
    result = asyncio.run(orchestrator.execute("patient_1", "prepare"))

    assert result.status == "error"
    assert result.error == "Planner failed."
    assert db.saved["status"] == "error"
