import asyncio

import pytest
from pydantic import BaseModel

from src.tools.text_tools import _extractive_summary, _scope_issue, _template_draft
from src.tools.tool import ToolRepository


class EchoArgs(BaseModel):
    value: str


def test_tool_repository_registers_and_rejects_duplicates():
    repo = ToolRepository()

    async def echo(value: str):
        return value

    tool = repo.register_func(
        name="echo",
        description="Echo value.",
        afunc=echo,
        args_schema=EchoArgs,
    )

    assert repo.get_tool("echo") is tool
    assert repo.list_tool_names() == ["echo"]
    assert asyncio.run(tool.ainvoke({"value": "x"})) == "x"
    with pytest.raises(ValueError, match="already registered"):
        repo.register(tool)
    with pytest.raises(KeyError, match="not found"):
        repo.get_tool("missing")


def test_text_tool_scope_check_blocks_bad_requests():
    assert _scope_issue("forge an insurance letter", [], "insurer") is not None
    assert _scope_issue("ask about preapproval", ["claim id 123"], "insurer") is None


def test_template_draft_includes_goal_points_and_recipient():
    draft = _template_draft("ask about referral", ["Referral is pending"], "clinic coordinator")

    assert "ask about referral" in draft
    assert "Referral is pending" in draft
    assert "send" not in draft.lower()


def test_extractive_summary_prefers_status_lines():
    summary = _extractive_summary(
        """
Noise line.
Status: pending.
Ground truth: oxaliplatin pre-approval is pending nurse review.
Patient action: call coordinator.
"""
    )

    assert "Status: pending." in summary
    assert "oxaliplatin" in summary
    assert "Noise line." not in summary
