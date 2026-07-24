import json
from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.agent import BaseAgent, AgentContext

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
        # this model uses high reasoning (bind is immutable)
        self._model = self._model.bind(reasoning_effort="high")


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
        given_input = HumanMessage(
            f"""
                query: {my_ctx.query}
                draft_response: {my_ctx.draft_response}
                documents: {my_ctx.documents}
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

        output_resp = await self._model.ainvoke([instructions, given_input])

        # get content of output
        if output_resp is None or output_resp.content is None:
            self._logger.error(f"Failed to retrieve output from SafetyGuard agent.")
            return fallback_result
        output = output_resp.content

        # deserialize json and extract fields
        try:
            deserialized_output = json.loads(output)
            return deserialized_output
        except json.JSONDecodeError as e:
            self._logger.exception(f"Failed to deserialize output from SafetyGuard agent: {e}")
            return fallback_result

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
"""

FALLBACK_MESSAGE = (
    "I'm sorry, but I cannot safely provide a personalized response to this query based on the available information. "
    "If you are experiencing severe or worsening symptoms, please contact a healthcare provider or seek immediate emergency medical care."
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