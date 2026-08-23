"""Integration tests for the SafetyGuardAgent's full act() pipeline: prompt construction,
document condensation, JSON parsing, and verdict normalization working together.

These are self-contained (a FakeModel stands in for the LLM boundary) so they run without a
live LLM or database, unlike the rest of tests/integration. Run explicitly with:
python -m pytest tests/integration/test_safety_guard_integration.py
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from src.agents.safetyguard_agent import (
    ACTION_BLOCK,
    ACTION_REWRITTEN,
    SafetyGuardAgent,
    SafetyGuardContext,
)


def _make_agent(verdict: dict) -> SafetyGuardAgent:
    class FakeModel:
        def __init__(self):
            self.received_prompt = None

        async def ainvoke(self, messages):
            self.received_prompt = messages[-1].content
            return SimpleNamespace(content=json.dumps(verdict))

    agent = SafetyGuardAgent.__new__(SafetyGuardAgent)
    agent._model = FakeModel()
    agent._logger = SimpleNamespace(error=lambda *a: None, exception=lambda *a: None)
    return agent


def test_doc_caps_pass_full_multi_value_lab_record_to_the_model():
    """A: raising the per-doc/total caps means a realistic multi-date lab panel reaches the
    safety model whole, instead of being cut off mid-record."""
    lab_document = (
        "Lab summary 2026-02-17: WBC 2.4 K/uL, ANC 1.1 K/uL, Hemoglobin 10.8 g/dL, Platelets 142 K/uL.\n"
        "Lab summary 2026-03-10: WBC 3.1 K/uL, ANC 1.6 K/uL, Hemoglobin 11.0 g/dL, Platelets 166 K/uL.\n"
        + ("Additional clinical narrative padding text. " * 40)
        + "END-OF-DOCUMENT-MARKER-7821"
    )
    assert len(lab_document) > 600, "fixture must exceed the old 600-char cap to be a meaningful test"

    agent = _make_agent({
        "is_grounded": True,
        "safety_violations_found": [],
        "action_taken": "PASS",
        "final_output": "Your labs on 2026-03-10 show WBC 3.1, ANC 1.6, Hemoglobin 11.0, Platelets 166.",
    })

    asyncio.run(agent.act(SafetyGuardContext(
        query="Summarize my most recent lab results in plain language.",
        draft_response="draft",
        documents=[lab_document],
    )))

    sent_prompt = agent._model.received_prompt
    assert "END-OF-DOCUMENT-MARKER-7821" in sent_prompt, (
        "The document was truncated before reaching the safety model; raise "
        "SafetyGuardAgent._DOC_CHAR_CAP / _DOCS_TOTAL_CHAR_CAP further."
    )
    assert "2026-02-17" in sent_prompt and "2026-03-10" in sent_prompt


def test_missing_disclaimer_alone_is_rewritten_not_blocked():
    """B: a draft whose only problem is a missing mandatory-footer disclaimer should come back
    as REWRITE with the footer appended, not BLOCK_AND_FALLBACK."""
    draft = "Your most recent lab results show Hemoglobin 11.0 g/dL and Platelets 166 K/uL."

    agent = _make_agent({
        "is_grounded": True,
        "safety_violations_found": [
            "No disclaimer directing the user to contact their physician for personal medical decisions."
        ],
        "action_taken": "BLOCK_AND_FALLBACK",
        "final_output": "",
    })

    result = asyncio.run(agent.act(SafetyGuardContext(
        query="Summarize my most recent lab results in plain language.",
        draft_response=draft,
        documents=["Hemoglobin 11.0 g/dL. Platelets 166 K/uL."],
    )))

    assert result["action_taken"] == ACTION_REWRITTEN
    assert "Hemoglobin 11.0" in result["final_output"], "original draft content must be preserved, not discarded"
    assert "healthcare provider" in result["final_output"].lower() or "physician" in result["final_output"].lower()


def test_ungrounded_values_still_block_even_with_missing_disclaimer():
    """Regression guard: when the draft has a real grounding problem (not just a missing
    footer), it must still be blocked -- the disclaimer fix must not soften genuine safety
    failures like the fabricated-lab-values scenario this fix was written for."""
    draft = (
        "Your most recent lab results are from March 10, 2026: WBC 3.1 K/uL, ANC 1.6 K/uL, "
        "Hemoglobin 11.0 g/dL, Platelets 166 K/uL."
    )

    agent = _make_agent({
        "is_grounded": False,
        "safety_violations_found": [
            "Unsupported lab date and values not found in the provided documents",
            "Potentially fabricated medical information",
            "No disclaimer directing the user to contact their physician for personal medical decisions",
        ],
        "action_taken": "BLOCK_AND_FALLBACK",
        "final_output": "",
    })

    result = asyncio.run(agent.act(SafetyGuardContext(
        query="Summarize my most recent lab results in plain language.",
        draft_response=draft,
        documents=["Unrelated note with no lab values."],
    )))

    assert result["action_taken"] == ACTION_BLOCK
