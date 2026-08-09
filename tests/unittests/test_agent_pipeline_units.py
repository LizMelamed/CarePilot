import asyncio

from langchain_core.tools import InjectedToolArg

from src.db.vector_store import VectorMatch
from src.agents.executor_agent import ExecutorAgent
from src.agents.planner_agent import PlannerAgent
from src.agents.replanner_agent import ReplannerAgent, ReplannerContext
from src.agents.safetyguard_agent import ACTION_BLOCK, ACTION_REWRITTEN, FALLBACK_MESSAGE, SafetyGuardAgent
from src.agents.workflow_types import PlannedTask, TaskResult
from src.tools import db_tools
from src.tools.db_tools import GetFileArgs, GetPatientDocumentsArgs
from src.tools.text_tools import message_drafting, summarization


def test_planner_malformed_json_falls_back_to_one_task():
    direct_answer, tasks = PlannerAgent._parse_output("not json", "what does neutropenia mean")

    assert direct_answer is None
    assert len(tasks) == 1
    assert tasks[0].task_id == "task_1"
    assert tasks[0].description == "what does neutropenia mean"


def test_planner_parses_direct_answer():
    direct_answer, tasks = PlannerAgent._parse_output(
        '{"direct_answer": "I can only help with health-related questions."}',
        "what does neutropenia mean",
    )

    assert direct_answer == "I can only help with health-related questions."
    assert tasks == []


def test_planner_parses_tasks():
    direct_answer, tasks = PlannerAgent._parse_output(
        '{"tasks": [{"task_id": "task_1", "description": "get labs", "tool_hint": "patient_db"}]}',
        "get my labs",
    )

    assert direct_answer is None
    assert len(tasks) == 1
    assert tasks[0].tool_hint == "patient_db"


def test_replanner_iteration_cap_returns_done():
    ctx = ReplannerContext(
        prompts=[],
        original_prompt="help me prepare",
        original_tasks=[PlannedTask("task_1", "read labs", "patient_db")],
        task_results=[TaskResult("task_1", "labs found", "get_patient_documents", True)],
        step_order=3,
        iteration=3,
        max_iterations=3,
    )

    result = ReplannerAgent._fallback_done(ReplannerAgent.__new__(ReplannerAgent), ctx, "Iteration cap reached.")

    assert result.done is True
    # The internal reason is not patient-facing text; only the real task results should show through.
    assert "Iteration cap reached." not in result.final_answer_context
    assert "labs found" in result.final_answer_context


def test_replanner_fallback_with_no_usable_results_asks_for_clarification():
    ctx = ReplannerContext(
        prompts=[],
        original_prompt="what's my referral status",
        original_tasks=[PlannedTask("task_1", "read referral", "patient_db")],
        task_results=[TaskResult("task_1", "[]", "get_patient_documents", True)],
        step_order=3,
        iteration=3,
        max_iterations=3,
    )

    result = ReplannerAgent._fallback_done(ReplannerAgent.__new__(ReplannerAgent), ctx, "Iteration cap reached.")

    assert result.done is True
    assert result.final_answer_context == ReplannerAgent._CLARIFICATION_FALLBACK


def test_replanner_parse_output_rejects_placeholder_final_answer():
    agent = ReplannerAgent.__new__(ReplannerAgent)
    ctx = ReplannerContext(
        prompts=[],
        original_prompt="help me prepare for my meeting",
        original_tasks=[PlannedTask("task_1", "lookup", "patient_db")],
        task_results=[TaskResult("task_1", "[]", "get_patient_documents", True)],
        step_order=3,
        iteration=0,
        max_iterations=3,
    )

    result = agent._parse_output('{"done": true, "final_answer_context": "..."}', ctx)

    assert result.done is True
    assert result.final_answer_context == ReplannerAgent._CLARIFICATION_FALLBACK


def test_safety_guard_invalid_action_blocks():
    result = SafetyGuardAgent._normalize_result(
        {
            "is_grounded": True,
            "safety_violations_found": "bad action",
            "action_taken": "UNSAFE_PASS",
            "final_output": "",
        }
    )

    assert result["action_taken"] == ACTION_BLOCK
    assert result["final_output"] == FALLBACK_MESSAGE
    assert result["safety_violations_found"] == ["bad action"]


def test_safety_guard_medication_stop_question_is_rewritten():
    result = SafetyGuardAgent._medication_change_override("Can I stop taking my medicine?")

    assert result is not None
    assert result["action_taken"] == ACTION_REWRITTEN
    assert "cannot tell you whether to stop" in result["final_output"]
    assert "healthcare provider" in result["final_output"]


def test_patient_tool_username_is_injected_metadata():
    for schema in (GetFileArgs, GetPatientDocumentsArgs):
        field = schema.model_fields["username"]
        assert field.default.__class__.__name__ == "PydanticUndefinedType"
        assert any(isinstance(item, InjectedToolArg) for item in field.metadata)


def test_executor_marks_patient_db_tool_as_patient_scoped():
    assert ExecutorAgent._is_patient_tool("get_patient_documents") is True
    assert ExecutorAgent._is_patient_tool("query_clinical_rag") is False


def test_executor_fallback_tool_answer_hides_raw_document_json():
    raw = (
        '[{"file_name": "appointment.txt", "file_path": "patient1/appointment.txt", '
        '"content": "Document: Appointment Summary\\nCategory: appointment summary\\n'
        'Next steps: repeat labs before the next cycle.\\nSymptoms discussed: fatigue.", '
        '"metadata": {}}]'
    )

    result = ExecutorAgent._fallback_tool_answer(raw)

    assert "file_name" not in result
    assert "file_path" not in result
    assert "Next steps: repeat labs" in result


def test_clinical_rag_tool_returns_source_attribution(monkeypatch):
    async def fake_query(query, top_k):
        return [
            VectorMatch(
                id="chunk_1",
                score=0.9,
                metadata={
                    "text": "Neutropenia means low neutrophils.",
                    "source_url": "https://example.test/source",
                    "title": "Neutropenia",
                    "topic_tags": ["side_effects"],
                },
            )
        ]

    monkeypatch.setattr(db_tools, "query_clinical_rag", fake_query)

    result = asyncio.run(db_tools.query_clinical_rag_tool("neutropenia", 1))

    assert result[0]["source_url"] == "https://example.test/source"
    assert result[0]["title"] == "Neutropenia"
    assert "Neutropenia" in result[0]["text"]


def test_task_10_and_11_tools_register_with_system(monkeypatch):
    from src.carepilot import system as system_module

    class FakeDB:
        async def get_patient_data(self, username):
            return None

        async def get_file(self, username, file_name):
            return ""

        async def list_files(self, username):
            return []

        async def get_patient_documents(self, username, limit=10):
            return []

        async def query_file(self, username, query, top_k):
            return []

    names = system_module.System(db_handler=FakeDB())._tools.list_tool_names()

    assert "query_clinical_rag" in names
    assert "get_patient_documents" in names
    assert "summarization" in names
    assert "message_drafting" in names


def test_summarization_fallback_preserves_insurance_ground_truth(monkeypatch):
    def fail_model():
        raise ValueError("no model")

    monkeypatch.setattr("src.tools.text_tools._build_model", fail_model)
    text = """
Document: Insurance Pre-authorization Letter
Status: expired.
Ground truth: pembrolizumab authorization expired on 2026-03-15.
Patient action: keep this letter with oncology records.
"""

    result = asyncio.run(summarization(text))

    assert "expired" in result
    assert "2026-03-15" in result
    assert "Key findings:" in result


def test_message_drafting_refuses_deceptive_requests():
    result = asyncio.run(
        message_drafting(
            goal="Lie to the insurance coordinator about my diagnosis",
            key_points=["fake diagnosis"],
            recipient_type="insurance coordinator",
        )
    )

    assert "cannot draft" in result
    assert "deceptive" in result
