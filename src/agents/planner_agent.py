from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.agent import AgentContext, BaseAgent
from src.agents.workflow_types import AgentStep, PlannedTask

PLANNING_MODULE = "PlanningLLM"

PLANNER_SYSTEM_PROMPT = """
You are the CarePilot planning agent.
CarePilot only helps with the patient's own healthcare: their records, appointments, medications, labs,
referrals, insurance, symptoms, and care logistics. Every patient of this app is dealing with an illness, so
words like "appointment", "visit", "checkup", or "meeting" almost always refer to a medical appointment (e.g.
with their doctor, care team, or coordinator) unless the request clearly states otherwise (e.g. explicitly
mentions work, a business call, or a personal/non-medical errand) -- treat these as in-scope by default, do
not ask the patient to clarify what kind of appointment they mean.
Only treat a request as unrelated to healthcare if it is clearly so (e.g. politics, celebrities, entertainment,
general trivia, coding help). For those, or for greetings/small talk/general questions you can already answer
without looking anything up, answer directly -- do not plan tasks for them.

recent_history contains prior conversation turns in chronological order (oldest first, newest last). Questions
about what the patient previously asked or what CarePilot previously answered must be answered directly from
recent_history. Conversation history is not a patient record: never use patient_db for these questions.

Decide which of these two response shapes applies:
1. DIRECT ANSWER: The request needs no patient data lookup and no tool at all -- greetings, small talk, or an
   off-topic request you're declining. NEVER use direct_answer for a clinical/medical fact question (e.g. "what
   is X", "what causes Y", anything asking for sources/citations) even if you already know the answer --
   answering those from unattributed model knowledge risks fabricated facts and fake sources; use clinical_rag
   instead so the answer is grounded in real, cited corpus text. Return:
   {"direct_answer": "the full patient-facing answer text"}
2. TASKS: The request needs one or more lookups or actions. Break it into the smallest number of necessary
   sub-tasks -- do not over-plan. If one tool lookup is enough, return exactly one task. Use only these
   tool_hint values:
   - clinical_rag: general clinical facts from the clinical corpus.
   - patient_db: patient profile, files, appointments, medications, labs, insurance, or referrals from the patient database.
   - summarization: summarize provided text or collected task results.
   - message_drafting: draft an outbound message ADDRESSED TO a specific recipient (e.g. their doctor, an
     insurance coordinator, a care team) that the patient will send. Only use this when the patient explicitly
     asks to draft, write, compose, or send a message/email/letter to someone. Never use it for guidance,
     checklists, or advice meant for the patient themselves -- e.g. "help me prepare for my appointment" is a
     patient_db lookup (find the appointment details) or a direct_answer with prep tips, NOT a message draft.
   Return:
   {"tasks":[{"task_id":"task_1","description":"...","tool_hint":"patient_db"}]}

EXAMPLES:
patient_prompt: "Hi, I need help preparing for my next meeting"
-> {"tasks":[{"task_id":"task_1","description":"Look up the patient's next appointment details","tool_hint":"patient_db"}]}
(WRONG: {"tasks":[{"task_id":"task_1","description":"draft message to care team about next meeting","tool_hint":"message_drafting"}]} -- the patient never asked to send anyone a message.)

patient_prompt: "Can you email my care coordinator that I need to reschedule?"
-> {"tasks":[{"task_id":"task_1","description":"Draft a message to the care coordinator requesting to reschedule","tool_hint":"message_drafting"}]}
(This one IS message_drafting, because the patient explicitly asked to send something to someone.)

patient_prompt: "Hi can you see my documents?"
-> {"tasks":[{"task_id":"task_1","description":"List and check the patient's uploaded documents/files on file","tool_hint":"patient_db"}]}
(WRONG: tool_hint "clinical_rag" -- clinical_rag is ONLY for general medical reference facts, e.g. "what is neutropenia?", never for the patient's own uploaded files or records. Any question about the patient's own documents, files, records, labs, referrals, appointments, medications, or insurance is patient_db, never clinical_rag.)

patient_prompt: "Based on my documents, what should I do next?"
-> {"tasks":[{"task_id":"task_1","description":"Read the patient's documents to find recommended next steps","tool_hint":"patient_db"}]}
(WRONG: tool_hint "summarization" -- summarization only condenses text you already have (e.g. prior_results from an earlier task); it cannot read the patient's files itself and has nothing to summarize on its own. Any request to read, check, or act on "my documents/records/files" needs a patient_db lookup FIRST, even if the patient's wording also sounds like they want a summary.)

patient_prompt: "What is melanoma? Provide sources"
-> {"tasks":[{"task_id":"task_1","description":"Look up clinical reference information about melanoma with sources","tool_hint":"clinical_rag"}]}
(WRONG: {"direct_answer": "Melanoma is a type of skin cancer..."} -- even though this is general medical knowledge you could phrase from memory, doing so is ungrounded and cannot honestly cite real sources; the patient explicitly asked for sources, so this MUST go through clinical_rag.)

Return only ONE raw JSON object matching exactly one of the two shapes above. Never return both keys.
"""


@dataclass
class PlannerContext(AgentContext):
    prompt: str
    username: str
    home_city: str | None = None
    current_datetime: str | None = None
    history: list[dict] = field(default_factory=list)


@dataclass
class PlannerResult:
    tasks: list[PlannedTask]
    step: AgentStep
    direct_answer: str | None = None


class PlannerAgent(BaseAgent):
    """Decide whether a request needs a direct answer or a task plan, at minimal cost."""

    def __init__(self, name: str = PLANNING_MODULE, tools=None):
        super().__init__(name, tools or {})

    async def act(self, ctx: AgentContext) -> PlannerResult | None:
        if not isinstance(ctx, PlannerContext):
            self._logger.error("Given context is not an instance of PlannerContext.")
            return None

        user_prompt = self._user_prompt(ctx)
        direct_answer = self._conversation_answer(ctx)
        if direct_answer is None:
            output = await self._invoke_json(PLANNER_SYSTEM_PROMPT, user_prompt)
            direct_answer, tasks = self._parse_output(output, ctx.prompt)
        else:
            tasks = []
        step = AgentStep(
            module=PLANNING_MODULE,
            system_prompt=PLANNER_SYSTEM_PROMPT.strip(),
            user_prompt=user_prompt,
            response=(
                json.dumps({"direct_answer": direct_answer}, ensure_ascii=True)
                if direct_answer is not None
                else json.dumps({"tasks": [task.to_dict() for task in tasks]}, ensure_ascii=True)
            ),
            step_order=1,
        )
        return PlannerResult(tasks=tasks, step=step, direct_answer=direct_answer)

    @staticmethod
    def _conversation_answer(ctx: PlannerContext) -> str | None:
        """Answer simple history-meta questions deterministically and without a needless tool call."""
        if not ctx.history:
            return None

        normalized = re.sub(r"[^a-z0-9\s]", " ", ctx.prompt.lower())
        refers_to_prior_turn = re.search(r"\b(previous|last|recent|earlier)\b", normalized)
        asks_about_question = re.search(r"\b(question|prompt|ask|asked|said)\b", normalized)
        asks_about_answer = re.search(r"\b(answer|response|reply|responded)\b", normalized)
        if not refers_to_prior_turn or not (asks_about_question or asks_about_answer):
            return None

        latest = ctx.history[-1]
        if asks_about_answer and not asks_about_question:
            prior_response = str(latest.get("final_response") or "").strip()
            return f'My previous response was: “{prior_response}”' if prior_response else None

        prior_prompt = str(latest.get("prompt") or "").strip()
        return f'Your previous question was: “{prior_prompt}”' if prior_prompt else None

    async def _invoke_json(self, system_prompt: str, user_prompt: str) -> str:
        response = await self._model.ainvoke([
            SystemMessage(system_prompt),
            HumanMessage(user_prompt),
        ])
        return str(getattr(response, "content", "") or "")

    @staticmethod
    def _user_prompt(ctx: PlannerContext) -> str:
        return json.dumps(
            {
                "username": ctx.username,
                "home_city": ctx.home_city,
                "current_datetime": ctx.current_datetime,
                "recent_history": ctx.history[-3:],
                "patient_prompt": ctx.prompt,
                "patient_storage_policy": "Use patient_db for patient-specific data. Do not plan patient_rag tasks.",
            },
            ensure_ascii=True,
        )

    @staticmethod
    def _parse_output(output: str, original_prompt: str) -> tuple[str | None, list[PlannedTask]]:
        try:
            data = json.loads(output)
            if not isinstance(data, dict):
                raise ValueError("planner output is not a JSON object")

            if "direct_answer" in data and not data.get("tasks"):
                direct_answer = str(data.get("direct_answer") or "").strip()
                if not direct_answer:
                    raise ValueError("empty direct_answer")
                return direct_answer, []

            raw_tasks = data.get("tasks")
            if not isinstance(raw_tasks, list) or not raw_tasks:
                raise ValueError("empty task list")
            tasks = [
                PlannedTask.from_dict(task, f"task_{index + 1}")
                for index, task in enumerate(raw_tasks)
                if isinstance(task, dict)
            ]
            if not tasks:
                raise ValueError("no valid task dictionaries")
            return None, tasks
        except Exception:
            return None, [
                PlannedTask(
                    task_id="task_1",
                    description=original_prompt,
                    tool_hint="clinical_rag",
                )
            ]
