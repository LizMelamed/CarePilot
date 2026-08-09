from __future__ import annotations

from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.utils.my_env import MyEnv

SUMMARY_SYSTEM_PROMPT = """
You summarize patient documents in plain language.
Use only facts present in the source text.
Do not diagnose, predict prognosis, or add medical advice.
Use this structure:
Key findings:
Dates:
Next steps mentioned:
Uncertainties:
"""

DRAFTING_SYSTEM_PROMPT = """
You draft patient messages only.
Allowed: appointments, insurance coordination, referrals, medication logistics, record requests, and care-team questions.
Refuse deceptive, harassing, threatening, impersonation, or illegal requests.
Never send anything automatically. Produce a draft only.
Keep the message concise, factual, and polite.
"""

OUT_OF_SCOPE_TERMS = {
    "harass",
    "threaten",
    "blackmail",
    "impersonate",
    "fake diagnosis",
    "forge",
    "lie to",
    "deceive",
}


class SummarizationArgs(BaseModel):
    """
    Arguments for the summarization tool.
    """
    text: str = Field(description="Source document text to summarize.")


class DraftMessageArgs(BaseModel):
    """
    Arguments for the message drafting tool.
    """
    goal: str = Field(description="Patient's goal for the message.")
    key_points: list[str] = Field(description="Facts the draft should include.")
    recipient_type: str = Field(description="Recipient type, e.g. insurance coordinator or oncology nurse.")


async def summarization(text: str) -> str:
    """
    Summarize source text plainly without adding medical advice.
    """
    text = text.strip()
    if not text:
        return "I cannot summarize an empty document."

    try:
        model = _build_model()
        response = await model.ainvoke([
            SystemMessage(SUMMARY_SYSTEM_PROMPT),
            HumanMessage(text),
        ])
        output = str(getattr(response, "content", "") or "").strip()
        return output or _extractive_summary(text)
    except Exception:
        return _extractive_summary(text)


async def message_drafting(goal: str, key_points: list[str], recipient_type: str) -> str:
    """
    Draft an in-scope patient message; refuse unsafe or deceptive requests.
    """
    scope_issue = _scope_issue(goal=goal, key_points=key_points, recipient_type=recipient_type)
    if scope_issue:
        return scope_issue

    prompt = (
        f"Recipient type: {recipient_type}\n"
        f"Goal: {goal}\n"
        f"Key points:\n"
        + "\n".join(f"- {point}" for point in key_points)
        + "\nReturn only the draft message."
    )
    try:
        model = _build_model()
        response = await model.ainvoke([
            SystemMessage(DRAFTING_SYSTEM_PROMPT),
            HumanMessage(prompt),
        ])
        output = str(getattr(response, "content", "") or "").strip()
        return output or _template_draft(goal, key_points, recipient_type)
    except Exception:
        return _template_draft(goal, key_points, recipient_type)


def _build_model() -> ChatOpenAI:
    env_handler = MyEnv()
    base_url = env_handler.get_llm_url()
    model = env_handler.get_llm_model()
    if base_url is None:
        raise ValueError("No LLM URL provided.")
    if model is None:
        raise ValueError("No model provided.")
    return ChatOpenAI(
        openai_api_base=base_url,
        model=model,
        api_key=env_handler.get_llm_key() or "d",
    )


def _scope_issue(goal: str, key_points: list[str], recipient_type: str) -> str | None:
    text = " ".join([goal, recipient_type, *key_points]).lower()
    if any(term in text for term in OUT_OF_SCOPE_TERMS):
        return (
            "I cannot draft deceptive, harassing, threatening, impersonating, or illegal messages. "
            "I can help draft a factual message to your care team or insurer."
        )
    return None


def _extractive_summary(text: str) -> str:
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]
    status_lines = [
        line
        for line in lines
        if line.lower().startswith(("status:", "ground truth:", "next step", "patient action:", "requested service:", "insurer:"))
    ]
    selected = status_lines[:8] if status_lines else lines[:8]
    return (
        "Key findings:\n"
        + "\n".join(f"- {line}" for line in selected)
        + "\nDates:\n- See source text for exact dates mentioned.\n"
        + "Next steps mentioned:\n- Follow the document's listed next steps; contact the care team for personal medical decisions.\n"
        + "Uncertainties:\n- This summary is based only on the provided text."
    )


def _template_draft(goal: str, key_points: list[str], recipient_type: str) -> str:
    bullet_text = "\n".join(f"- {point}" for point in key_points)
    return f"""Hello,

I am writing about: {goal}

Relevant details:
{bullet_text}

Could you please review this and let me know the next step?

Thank you."""
