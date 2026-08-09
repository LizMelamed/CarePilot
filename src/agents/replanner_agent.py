from __future__ import annotations

import json
from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.agent import AgentContext, BaseAgent
from src.agents.workflow_types import AgentStep, PlannedTask, TaskResult

REPLANNER_MODULE = "RePlanLLM"

REPLANNER_SYSTEM_PROMPT = """
You are the CarePilot re-planning agent.
Review the original prompt, original tasks, and task results.

If the results are enough to answer, return:
{"done":true,"final_answer_context":"<the real patient-facing answer, written out in full>"}

If the results are NOT enough (e.g. a lookup came back empty or didn't match what the patient needs) but one
more targeted task could plausibly find it, return:
{"done":false,"new_tasks":[{"task_id":"task_extra_1","description":"<a different, more specific lookup>","tool_hint":"patient_db"}]}

If the results are NOT enough and further tasks would not help (e.g. the information genuinely isn't in the
patient's records, or the request is missing a detail only the patient can supply, such as which appointment
or which document), return done:true, but final_answer_context must ask the patient a short, specific
clarifying question, or clearly state what wasn't found and what they can do next -- e.g. "I don't see any
upcoming appointments on file for you. Could you tell me the date or which specialist you're seeing?"

final_answer_context must always be the actual text to show the patient -- never a placeholder, ellipsis, or
empty string. Use the smallest number of new tasks. Do not request patient_rag.
Return only raw JSON.

EXAMPLE (empty lookup, nothing more to try):
task_results: [{"task_id":"task_1","result":"[]","tool_used":"get_patient_documents","success":true}]
-> {"done":true,"final_answer_context":"I couldn't find an upcoming appointment on file for you. Could you tell me the date or which doctor you're seeing, so I can help you prepare?"}
"""


@dataclass
class ReplannerContext(AgentContext):
    original_prompt: str
    original_tasks: list[PlannedTask]
    task_results: list[TaskResult]
    step_order: int
    iteration: int
    max_iterations: int = 3


@dataclass
class ReplannerResult:
    done: bool
    final_answer_context: str
    new_tasks: list[PlannedTask]
    step: AgentStep


class ReplannerAgent(BaseAgent):
    """Decide whether the executor results are sufficient."""

    def __init__(self, name: str = REPLANNER_MODULE, tools=None):
        super().__init__(name, tools or {})

    async def act(self, ctx: AgentContext) -> ReplannerResult | None:
        if not isinstance(ctx, ReplannerContext):
            self._logger.error("Given context is not an instance of ReplannerContext.")
            return None

        if ctx.iteration >= ctx.max_iterations:
            return self._fallback_done(ctx, "Iteration cap reached.")

        user_prompt = self._user_prompt(ctx)
        response = await self._model.ainvoke([
            SystemMessage(REPLANNER_SYSTEM_PROMPT),
            HumanMessage(user_prompt),
        ])
        output = str(getattr(response, "content", "") or "")
        result = self._parse_output(output, ctx)
        step = AgentStep(
            module=REPLANNER_MODULE,
            system_prompt=REPLANNER_SYSTEM_PROMPT.strip(),
            user_prompt=user_prompt,
            response=json.dumps(
                {
                    "done": result.done,
                    "final_answer_context": result.final_answer_context,
                    "new_tasks": [task.to_dict() for task in result.new_tasks],
                },
                ensure_ascii=True,
            ),
            step_order=ctx.step_order,
        )
        return ReplannerResult(
            done=result.done,
            final_answer_context=result.final_answer_context,
            new_tasks=result.new_tasks,
            step=step,
        )

    @staticmethod
    def _user_prompt(ctx: ReplannerContext) -> str:
        return json.dumps(
            {
                "original_prompt": ctx.original_prompt,
                "original_tasks": [task.to_dict() for task in ctx.original_tasks],
                "task_results": [result.to_dict() for result in ctx.task_results],
                "iteration": ctx.iteration,
                "max_iterations": ctx.max_iterations,
            },
            ensure_ascii=True,
        )

    def _parse_output(self, output: str, ctx: ReplannerContext) -> ReplannerResult:
        try:
            data = json.loads(output)
            done = bool(data.get("done"))
            if done:
                final_answer_context = str(data.get("final_answer_context") or "").strip()
                if not final_answer_context or self._is_placeholder(final_answer_context):
                    final_answer_context = self._joined_results(ctx) or self._CLARIFICATION_FALLBACK
                return ReplannerResult(
                    done=True,
                    final_answer_context=final_answer_context,
                    new_tasks=[],
                    step=self._empty_step(ctx),
                )

            raw_tasks = data.get("new_tasks") or []
            if not isinstance(raw_tasks, list):
                raw_tasks = []
            new_tasks = [
                PlannedTask.from_dict(task, f"task_extra_{index + 1}")
                for index, task in enumerate(raw_tasks)
                if isinstance(task, dict)
            ]
            if not new_tasks:
                return self._fallback_done(ctx, "No valid new tasks.")
            return ReplannerResult(False, "", new_tasks, self._empty_step(ctx))
        except Exception:
            return self._fallback_done(ctx, "Malformed replanner output.")

    def _fallback_done(self, ctx: ReplannerContext, reason: str) -> ReplannerResult:
        # `reason` (e.g. "Malformed replanner output.") is an internal diagnostic, not something a
        # patient should see -- the patient-facing text is either the real joined results, or a plain
        # clarification request if there's nothing usable to show.
        return ReplannerResult(
            done=True,
            final_answer_context=self._joined_results(ctx) or self._CLARIFICATION_FALLBACK,
            new_tasks=[],
            step=self._empty_step(ctx),
        )

    _CLARIFICATION_FALLBACK = (
        "I wasn't able to find enough information in your records to answer that. "
        "Could you give me a bit more detail (e.g. a date, a document name, or which appointment you mean)?"
    )

    _PLACEHOLDER_VALUES = {"...", "…", "tbd", "n/a", "na", "-"}

    @classmethod
    def _is_placeholder(cls, text: str) -> bool:
        stripped = text.strip(" .").lower()
        return not stripped or stripped in cls._PLACEHOLDER_VALUES

    @staticmethod
    def _joined_results(ctx: ReplannerContext) -> str:
        return "\n".join(
            f"{result.task_id}: {result.result}"
            for result in ctx.task_results
            if result.result and result.result.strip() not in ("[]", "{}", "null", "none")
        )

    @staticmethod
    def _empty_step(ctx: ReplannerContext) -> AgentStep:
        return AgentStep(REPLANNER_MODULE, "", "", "", ctx.step_order)
