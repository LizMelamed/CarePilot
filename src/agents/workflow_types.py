from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class PlannedTask:
    task_id: str
    description: str
    tool_hint: str

    @classmethod
    def from_dict(cls, data: dict[str, Any], fallback_id: str) -> "PlannedTask":
        return cls(
            task_id=str(data.get("task_id") or fallback_id),
            description=str(data.get("description") or ""),
            tool_hint=str(data.get("tool_hint") or "summarization"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TaskResult:
    task_id: str
    result: str
    tool_used: str | None
    success: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AgentStep:
    module: str
    system_prompt: str
    user_prompt: str
    response: str
    step_order: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
