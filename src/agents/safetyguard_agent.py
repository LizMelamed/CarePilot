import json
import re
from dataclasses import dataclass

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.agents.agent import BaseAgent, AgentContext
from src.utils.my_env import MyEnv

@dataclass
class SafetyGuardContext(AgentContext):
    query: str
    """The user query"""
    draft_response: str
    """The draft response the previous agent outputted"""
    documents: list[str]
    """The documents that the previous agent used as information to base its answer."""

class SafetyGuardAgent(BaseAgent):
    """
    An agent that is responsible for checking and sanitizing the output of the Plan&Execute Agent.
    """

    def __init__(self, name: str, tools):
        super().__init__(name, tools)

        # safety guard can use a stronger model than the rest of the pipeline,
        # since it needs to follow the safety-rules prompt reliably.
        safety_model_name = MyEnv.get("SAFETY_LLM_MODEL")
        if safety_model_name:
            self._model = ChatOpenAI(
                openai_api_base=MyEnv().get_llm_url(),
                model=safety_model_name,
                api_key=MyEnv().get_llm_key() or "d",
            )
            # bound reasoning: capped ("low"/"medium") instead of unlimited "high" thinking, for
            # reasoning-effort providers (e.g. OpenAI o-series/gpt-5).
            reasoning_effort = MyEnv.get("SAFETY_REASONING_EFFORT") or "low"
            self._model = self._model.bind(
                reasoning_effort=reasoning_effort,
                # Some OpenAI-compatible providers expose hybrid-thinking models that can leave raw
                # <think> traces in content, breaking strict JSON parsing. Providers that do not
                # recognize this ignore it.
                extra_body={"think": False},
            )


    async def act(self, ctx: AgentContext):
        """
        Returns sanitized output from the given result of the Plan&Execute agent.
        :param ctx: SafetyGuardContext. contains user query, draft_response of the previous agent,
        and source documents that the previous agent used as information to base its answer.
        :return: a dictionary result {is_grounded: bool, safety_violations_found: list, action_taken: str, final_output: str},
            or None if given context isn't of type SafetyGuardContext.
        """
        if not isinstance(ctx, SafetyGuardContext):
            self._logger.error("Given context is not an instance of SafetyGuardContext.")
            return None
        my_ctx: SafetyGuardContext = ctx
        medication_change_override = self._medication_change_override(my_ctx.query)
        if medication_change_override is not None:
            return medication_change_override

        given_input = HumanMessage(
            f"""
                Your ONLY task here is to review draft_response below against your safety rules and
                return the required JSON verdict. The "documents" field is reference evidence for that
                review only -- do NOT summarize, analyze, or answer questions about the documents
                themselves; that is not what is being asked of you.

                query: {my_ctx.query}
                draft_response: {my_ctx.draft_response}
                documents (supporting evidence only, not the subject of your task): {self._condensed_documents(my_ctx.documents)}

                Now return ONLY the JSON verdict object for draft_response, as specified in your instructions.
                """
        )
        instructions = SystemMessage(
            SAFETY_PROMPT
        )
        fallback_result = {
                "is_grounded": False,
                "safety_violations_found": ["deserialization failed."],
                "action_taken": ACTION_BLOCK,
                "final_output": FALLBACK_MESSAGE,
            }

        messages: list = [instructions, given_input]
        for attempt in range(self._MAX_ATTEMPTS):
            output_resp = await self._model.ainvoke(messages)

            if output_resp is None or output_resp.content is None:
                self._logger.error("Failed to retrieve output from SafetyGuard agent.")
                continue
            output = output_resp.content

            result = self._try_parse(output)
            if result is not None:
                return result

            # Small local models occasionally ignore the strict-JSON instruction entirely -- give it
            # one more chance with a sharper reminder before falling back to the safe default.
            messages = [
                instructions,
                given_input,
                AIMessage(str(output)),
                HumanMessage(
                    "Your previous reply was not a single valid JSON object matching the required "
                    "schema. Reply again with ONLY that raw JSON object -- no prose, no markdown, "
                    "no explanation before or after it."
                ),
            ]

        self._logger.error(f"SafetyGuard agent did not return valid JSON after {self._MAX_ATTEMPTS} attempts.")
        return fallback_result

    _MAX_ATTEMPTS = 1

    def _try_parse(self, output: str) -> dict | None:
        cleaned = self._strip_markdown_fence(self._strip_reasoning_trace(output))
        try:
            return self._normalize_result(json.loads(cleaned))
        except json.JSONDecodeError:
            pass

        # Some reasoning models (e.g. qwen3) can leave stray text around the JSON object even
        # after stripping <think> tags and code fences -- fall back to pulling out the first
        # balanced {...} block before giving up on this attempt.
        extracted = self._extract_json_object(cleaned)
        if extracted is not None:
            try:
                return self._normalize_result(json.loads(extracted))
            except json.JSONDecodeError:
                pass

        self._logger.exception(f"Failed to deserialize output from SafetyGuard agent: {cleaned!r}")
        return None

    @staticmethod
    def _strip_reasoning_trace(output: str) -> str:
        """Drop a leading chain-of-thought block some reasoning models (e.g. qwen3) emit even when
        a plain JSON response is requested. The opening <think> tag is sometimes swallowed by the
        chat template while the closing tag survives, so this splits on the LAST closing tag rather
        than requiring a matching open tag."""
        if "</think>" in output:
            return output.rsplit("</think>", 1)[1].strip()
        return output

    @staticmethod
    def _strip_markdown_fence(output: str) -> str:
        output = output.strip()
        if output.startswith("```"):
            lines = output.splitlines()
            if len(lines) >= 3 and lines[-1].strip() == "```":
                return "\n".join(lines[1:-1]).removeprefix("json").strip()
        return output

    _DOC_CHAR_CAP = 600
    _DOCS_TOTAL_CHAR_CAP = 2400

    @classmethod
    def _condensed_documents(cls, documents: list[str]) -> list[str]:
        """Cap each source document's length and the total documents payload before it goes into
        the safety prompt. Small local models have limited context; stuffing in several full
        multi-thousand-character chunks (as query_clinical_rag/get_patient_documents can produce)
        can overflow that budget, leaving no room for the model to produce a coherent -- let alone
        valid-JSON -- reply. The safety check only needs enough of each document to verify the
        draft's claims, not the full text."""
        condensed: list[str] = []
        total = 0
        for document in documents:
            text = str(document)
            if total >= cls._DOCS_TOTAL_CHAR_CAP:
                break
            remaining = cls._DOCS_TOTAL_CHAR_CAP - total
            snippet = text[:min(cls._DOC_CHAR_CAP, remaining)]
            if len(text) > len(snippet):
                snippet += "..."
            condensed.append(snippet)
            total += len(snippet)
        return condensed

    @staticmethod
    def _extract_json_object(text: str) -> str | None:
        """Return the first balanced top-level {...} substring in text, or None if there isn't one."""
        start = text.find("{")
        if start == -1:
            return None
        depth = 0
        for index in range(start, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    return text[start:index + 1]
        return None

    @staticmethod
    def _is_scope_violation(violations: list) -> bool:
        scope_keywords = ("scope", "off-topic", "off topic", "unrelated", "not related to")
        text = " ".join(str(v).lower() for v in violations)
        return any(keyword in text for keyword in scope_keywords)

    _MEDICATION_CHANGE_RE = re.compile(
        r"\b("
        r"stop|stopping|quit|discontinue|pause|skip|reduce|lower|change|hold|delay|"
        r"take less|not take"
        r")\b",
        re.IGNORECASE,
    )
    _MEDICATION_TERM_RE = re.compile(
        r"\b("
        r"medicine|medicines|medication|medications|meds|drug|drugs|pill|pills|"
        r"treatment|chemo|chemotherapy|capecitabine|ondansetron|acetaminophen"
        r")\b",
        re.IGNORECASE,
    )

    @classmethod
    def _medication_change_override(cls, query: str) -> dict | None:
        if not (cls._MEDICATION_CHANGE_RE.search(query) and cls._MEDICATION_TERM_RE.search(query)):
            return None
        return {
            "is_grounded": False,
            "safety_violations_found": ["medication change request requires clinician consultation."],
            "action_taken": ACTION_REWRITTEN,
            "final_output": MEDICATION_CHANGE_MESSAGE,
        }

    @staticmethod
    def _normalize_result(result):
        if not isinstance(result, dict):
            return {
                "is_grounded": False,
                "safety_violations_found": ["invalid safety result."],
                "action_taken": ACTION_BLOCK,
                "final_output": FALLBACK_MESSAGE,
            }

        action = result.get("action_taken")
        if action not in {ACTION_PASS, ACTION_REWRITTEN, ACTION_BLOCK}:
            action = ACTION_BLOCK

        violations = result.get("safety_violations_found") or []
        if isinstance(violations, str):
            violations = [violations]

        final_output = str(result.get("final_output") or "")
        if action == ACTION_BLOCK or not final_output.strip():
            final_output = OFF_TOPIC_MESSAGE if SafetyGuardAgent._is_scope_violation(violations) else FALLBACK_MESSAGE
            action = ACTION_BLOCK

        return {
            "is_grounded": bool(result.get("is_grounded")),
            "safety_violations_found": list(violations),
            "action_taken": action,
            "final_output": final_output,
        }

# actions available to the agent
ACTION_PASS = "PASS"
ACTION_REWRITTEN = "REWRITE"
ACTION_BLOCK = "BLOCK_AND_FALLBACK"

SAFETY_RULES = """
1. EMERGENCY CHECK: If the query or sources indicate potential life-threatening symptoms (e.g., severe chest pain, shortness of breath, sudden numbness), issue a clear recommendation to seek immediate emergency care.
2. LANGUAGE & TONE: Keep sentences short. Use clear, simple language (8th-grade reading level max). Be empathetic but firm on safety warnings.
3. ABSOLUTE GROUNDING: Remove any advice, medical claims, or home remedies that are not directly supported by the source documents.
4. NO DIAGNOSIS: Never say "You have X" or "This is caused by Y." Frame as "These symptoms can sometimes be related to..." and encourage consulting their healthcare provider.
5. MANDATORY FOOTER: Always append a disclaimer directing the user to contact their physician for personal medical decisions.
6. SCOPE: CarePilot only helps with the patient's own healthcare: their records, appointments, medications, labs, referrals, insurance, symptoms, and care logistics. Every patient of this app is dealing with an illness, so words like "appointment", "visit", "checkup", or "meeting" almost always refer to a medical appointment unless the query clearly states otherwise (e.g. explicitly mentions work, a business call, or a personal/non-medical errand) -- treat these as in-scope by default. Only if the query is CLEARLY unrelated to the patient's healthcare (e.g. politics, celebrities, opinions on public figures, entertainment, general trivia, coding help) should you set action_taken to "REWRITE" and replace final_output with a brief, polite redirect stating CarePilot can only help with health and care-related questions. Do not block or redirect a draft_response merely because it mentions scheduling, meetings, or appointments -- that is in scope.
7. INSUFFICIENT INFORMATION IS NOT UNSAFE: If the draft honestly says the needed information wasn't found and asks the patient a specific clarifying question (e.g. "I don't see an appointment on file -- could you tell me the date?"), that is a SAFE, desirable response -- it makes no unsupported claims. PASS it (or REWRITE only to fix tone/footer), never BLOCK_AND_FALLBACK it. Reserve BLOCK_AND_FALLBACK for content that is actually dangerous, fabricated, or gives unsupported medical claims -- not for a response that is short, uncertain, or asks for clarification.
"""

FALLBACK_MESSAGE = (
    "I'm sorry, but I cannot safely provide a personalized response to this query based on the available information. "
    "If you are experiencing severe or worsening symptoms, please contact a healthcare provider or seek immediate emergency medical care."
)

MEDICATION_CHANGE_MESSAGE = (
    "I cannot tell you whether to stop, pause, skip, or change a medicine. "
    "Please contact your healthcare provider before making any medication change. "
    "If you are having severe side effects or feel unsafe, seek urgent medical care."
)

OFF_TOPIC_MESSAGE = (
    "I'm CarePilot, and I can only help with your own healthcare: your records, appointments, medications, labs, "
    "referrals, insurance, symptoms, and care logistics. Try asking me something related to your care."
)

SAFETY_PROMPT = f"""
You are a Patient Safety and Health Communication Expert. You are reviewing a draft answer written for a patient, compared against verified source documents.

TASK:
Your task is to REWRITE the draft if it violates any of the rules in "SAFETY RULES TO ENFORCE"

CRITICAL INSTRUCTION ON EDITS:
- If the draft is accurate, grounded, and violates NO safety rules, DO NOT rewrite it. Set action_taken to "PASS" and return the original draft_response verbatim in final_output.
- Rewrite the response ONLY if any of the rules in "SAFETY RULES TO ENFORCE" are breached.
- If the draft is completely unsafe, severely ungrounded, or dangerous to answer, set action_taken to "BLOCK_AND_FALLBACK" and use the exact standard fallback message.

SAFETY RULES TO ENFORCE:
{SAFETY_RULES}

INPUT FORMAT:
- query: the original user query,
- draft_response: unchecked response to the query that needs to be checked and verified,
- documents: a list of source documents that the draft_response was based on,

STANDARD FALLBACK MESSAGE:
"{FALLBACK_MESSAGE}"

OUTPUT FORMAT:
Return ONLY a valid, raw JSON object matching the schema below. Do NOT wrap it in markdown block quotes (e.g. do not use ```json).

{{
  "is_grounded": bool,
  "safety_violations_found": ["list of issues"],
  "action_taken": "{ACTION_PASS}" | "{ACTION_REWRITTEN}" | "{ACTION_BLOCK}",
  "final_output": "The sanitized text to show the user..."
}}
"""
