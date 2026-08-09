from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

from src.agents.executor_agent import ExecutorAgent, ExecutorContext
from src.agents.planner_agent import PlannerAgent, PlannerContext
from src.agents.replanner_agent import ReplannerAgent, ReplannerContext
from src.agents.safetyguard_agent import SafetyGuardAgent, SafetyGuardContext
from src.agents.workflow_types import AgentStep, PlannedTask, TaskResult
from src.carepilot.system import System
from src.utils.logger import Logger

MAX_REPLAN_ITERATIONS = 3
TIMEZONE = "Asia/Jerusalem"


@dataclass(frozen=True)
class OrchestratorResult:
    status: str
    error: str | None
    response: str
    steps: list[dict]
    execution_id: str | None


class CarePilotOrchestrator:
    """
    Planner, executor, replanner, safety-guard pipeline.

    Cost scales with how much the request actually needs:
    - Tier 0 (Planner + SafetyGuard): the planner can answer directly, no data lookup needed.
    - Tier 1 (Planner + Executor + SafetyGuard): the planner needs exactly one tool lookup, and it
      succeeds -- the Replanner is skipped since there is nothing left to plan.
    - Tier 2 (Planner + Executor(s) + Replanner + SafetyGuard): the planner needs multiple steps, or
      a Tier 1 attempt failed/came back empty and the Replanner is brought in as a safety net.
    """

    def __init__(self):
        self._logger = Logger()
        self._system = System()
        self._db_handler = self._system.get_db_handler()
        tools = self._system.get_tools()
        self._planner = PlannerAgent(tools=tools)
        self._executor = ExecutorAgent(tools=tools)
        self._replanner = ReplannerAgent(tools=tools)
        self._safety_guard = SafetyGuardAgent("SafetyGuardLLM", tools)

    async def execute(self, username: str, prompt: str) -> OrchestratorResult:
        steps: list[AgentStep] = []
        task_results: list[TaskResult] = []
        source_documents: list[str] = []
        final_response = ""
        status = "success"
        error = None

        try:
            history = await self._db_handler.get_execution_history(username, limit=3)
            patient_data = await self._db_handler.get_patient_data(username)
            home_city = None
            if isinstance(patient_data, dict):
                home_city = patient_data.get("home_city")

            planner_ctx = PlannerContext(
                prompts=[],
                prompt=prompt,
                username=username,
                home_city=home_city,
                current_datetime=datetime.now(ZoneInfo(TIMEZONE)).isoformat(),
                history=self._lean_history(history),
            )

            planner_result = await self._planner.act(planner_ctx)
            if planner_result is None:
                raise RuntimeError("Planner failed.")
            steps.append(planner_result.step)
            step_order = 2

            if planner_result.direct_answer is not None:
                # Tier 0: no patient data or tool needed at all.
                draft_response = planner_result.direct_answer
            else:
                tasks: list[PlannedTask] = planner_result.tasks
                step_order = await self._run_tasks(
                    username, tasks, step_order, steps, task_results, source_documents
                )

                if len(tasks) == 1 and self._task_succeeded(task_results[-1]):
                    # Tier 1: a single lookup answered it -- no Replanner call needed.
                    final_response = task_results[-1].result
                else:
                    # Tier 2, or an escalation safety-net for a failed/empty Tier 1 attempt.
                    final_response, step_order = await self._replan_until_done(
                        username, prompt, planner_result.tasks, task_results, source_documents, steps, step_order
                    )

                if not final_response:
                    final_response = self._fallback_final_context(task_results)
                draft_response = final_response

            safety_result = await self._safety_guard.act(
                SafetyGuardContext(
                    prompts=[],
                    query=prompt,
                    draft_response=draft_response,
                    documents=source_documents,
                )
            )
            if safety_result is None:
                raise RuntimeError("Safety guard failed.")

            final_response = str(safety_result.get("final_output") or "")
            steps.append(
                AgentStep(
                    module="SafetyGuardLLM",
                    system_prompt="See SafetyGuardAgent.SAFETY_PROMPT",
                    user_prompt=json.dumps(
                        {
                            "query": prompt,
                            "draft_response": draft_response,
                            "document_count": len(source_documents),
                        },
                        ensure_ascii=True,
                    ),
                    response=json.dumps(safety_result, ensure_ascii=True),
                    step_order=step_order,
                )
            )
        except Exception as e:
            self._logger.exception(f"CarePilot execution failed: {e}")
            status = "error"
            error = str(e)
            final_response = (
                "I'm sorry, but I could not complete this request safely. "
                "Please contact your care team for urgent or personal medical decisions."
            )

        step_dicts = [step.to_dict() for step in steps]
        execution_id = await self._safe_save_execution(
            username, prompt, final_response, status, step_dicts, error
        )

        return OrchestratorResult(status, error, final_response, step_dicts, execution_id)

    async def _run_tasks(
            self,
            username: str,
            tasks: list[PlannedTask],
            step_order: int,
            steps: list[AgentStep],
            task_results: list[TaskResult],
            source_documents: list[str],
    ) -> int:
        for task in tasks:
            executor_result = await self._executor.act(
                ExecutorContext(
                    prompts=[],
                    username=username,
                    task=task,
                    step_order=step_order,
                    prior_results=task_results,
                )
            )
            if executor_result is None:
                raise RuntimeError(f"Executor failed for task '{task.task_id}'.")
            steps.append(executor_result.step)
            task_results.append(executor_result.task_result)
            source_documents.extend(executor_result.source_documents)
            step_order += 1
        return step_order

    async def _replan_until_done(
            self,
            username: str,
            prompt: str,
            original_tasks: list[PlannedTask],
            task_results: list[TaskResult],
            source_documents: list[str],
            steps: list[AgentStep],
            step_order: int,
    ) -> tuple[str, int]:
        final_response = ""
        iteration = 0

        while True:
            replanner_result = await self._replanner.act(
                ReplannerContext(
                    prompts=[],
                    original_prompt=prompt,
                    original_tasks=original_tasks,
                    task_results=task_results,
                    step_order=step_order,
                    iteration=iteration,
                    max_iterations=MAX_REPLAN_ITERATIONS,
                )
            )
            if replanner_result is None:
                raise RuntimeError("Replanner failed.")
            steps.append(replanner_result.step)
            step_order += 1

            if replanner_result.done:
                final_response = replanner_result.final_answer_context
                break

            iteration += 1
            if iteration >= MAX_REPLAN_ITERATIONS:
                break

            step_order = await self._run_tasks(
                username, replanner_result.new_tasks, step_order, steps, task_results, source_documents
            )

        return final_response, step_order

    @staticmethod
    def _lean_history(history: list[dict]) -> list[dict]:
        """
        Strip DB execution-history rows down to just prompt/response pairs.

        The raw rows carry each past execution's full nested step trace (system_prompt,
        user_prompt, response per step) -- and since the planner's own step embeds this same
        history in its user_prompt, that step trace already contains a full copy of the prior
        turn's history. Passing the raw rows through would re-embed that on every turn, growing
        the payload exponentially (observed to reach tens of megabytes after a handful of turns).
        """
        return [
            {"prompt": entry.get("prompt"), "final_response": entry.get("final_response")}
            for entry in history
        ]

    _EMPTY_RESULT_MARKERS = {"[]", "{}", "null", "none"}
    _EMPTY_RESULT_STRIP_RE = re.compile(r"\[|\]|\{|\}|\(|\)|,|\s|null|none", re.IGNORECASE)

    @classmethod
    def _task_succeeded(cls, result: TaskResult) -> bool:
        if not result.success:
            return False
        text = result.result.strip()
        if not text or text.lower() in cls._EMPTY_RESULT_MARKERS:
            return False
        # Catches vacuous-but-technically-non-empty results like "[null, null, null]" or
        # "{}, {}" -- once brackets/commas/whitespace/null placeholders are stripped, nothing
        # meaningful is left, so it should escalate to the Replanner just like a true empty result.
        if not cls._EMPTY_RESULT_STRIP_RE.sub("", text):
            return False
        return True

    async def _safe_save_execution(
            self,
            username: str,
            prompt: str,
            final_response: str,
            status: str,
            step_dicts: list[dict],
            error: str | None,
    ) -> str | None:
        try:
            return await self._db_handler.save_execution(
                username_or_session=username,
                prompt=prompt,
                final_response=final_response,
                status=status,
                steps=step_dicts,
                error=error,
            )
        except Exception as e:
            self._logger.exception(f"Failed to persist execution history: {e}")
            return None

    @staticmethod
    def _fallback_final_context(task_results: list[TaskResult]) -> str:
        return "\n".join(
            result.result
            for result in task_results
            if result.result
        )
