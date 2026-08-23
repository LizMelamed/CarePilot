from __future__ import annotations

import asyncio
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Header, HTTPException, Query, Request, UploadFile
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.agents.executor_agent import EXECUTOR_MODULE
from src.agents.planner_agent import PLANNING_MODULE, PLANNER_SYSTEM_PROMPT
from src.agents.replanner_agent import REPLANNER_MODULE
from src.agents.safetyguard_agent import SAFETY_MODULE, SAFETY_PROMPT
from src.carepilot.orchestrator import CarePilotOrchestrator

EXECUTE_TIMEOUT_SECONDS = 295
_ARCHITECTURE_PNG_PATH = Path(__file__).resolve().parents[2] / "static" / "model_architecture.png"
app = FastAPI(title="CarePilot API")


class ExecuteRequest(BaseModel):
    prompt: str = Field(min_length=1)


class ExecuteResponse(BaseModel):
    status: str
    error: str | None
    response: str | None
    steps: list[dict[str, Any]]


class DocumentMeta(BaseModel):
    file_name: str
    size_chars: int | None = None


class DocumentContent(BaseModel):
    file_name: str
    content: str


class UploadResponse(BaseModel):
    status: str
    file_name: str
    error: str | None = None


class ConversationTurn(BaseModel):
    id: str | None = None
    prompt: str
    response: str | None = None
    status: str
    error: str | None = None
    created_at: str | None = None
    steps: list[dict[str, Any]]


class ConversationHistoryResponse(BaseModel):
    conversations: list[ConversationTurn]


@app.exception_handler(RequestValidationError)
async def request_validation_error(request: Request, exc: RequestValidationError):
    """Keep validation failures for the required execute endpoint inside its API contract."""
    if request.url.path == "/api/execute":
        messages = [error.get("msg", "Invalid request") for error in exc.errors()]
        payload = ExecuteResponse(
            status="error",
            error="; ".join(messages) or "Invalid request",
            response=None,
            steps=[],
        )
        return JSONResponse(status_code=200, content=payload.model_dump())
    return await request_validation_exception_handler(request, exc)


class PatientProfile(BaseModel):
    username: str
    date_of_birth: str | None = None
    gender: str | None = None
    sex: str | None = None


@lru_cache(maxsize=1)
def _orchestrator() -> CarePilotOrchestrator:
    return CarePilotOrchestrator()


def _db_handler():
    return _orchestrator()._db_handler


def _resolve_username(x_carepilot_username: str | None) -> str:
    return x_carepilot_username or os.getenv("CAREPILOT_DEFAULT_USERNAME", "patient_1")


@app.get("/api/users")
async def list_users() -> dict[str, Any]:
    users = await _db_handler().list_users()
    visible_users = [username for username in users if not username.startswith("integration_")]
    return {"users": sorted(visible_users)}


@app.get("/api/patients/me", response_model=PatientProfile)
async def get_me(x_carepilot_username: str | None = Header(default=None)) -> PatientProfile:
    username = _resolve_username(x_carepilot_username)
    data = await _db_handler().get_patient_data(username)
    if data is None:
        raise HTTPException(status_code=404, detail=f"Unknown patient '{username}'.")
    date_of_birth, gender, sex = data
    return PatientProfile(username=username, date_of_birth=date_of_birth, gender=gender, sex=sex)


@app.get("/api/conversations", response_model=ConversationHistoryResponse)
async def conversation_history(
        limit: int = Query(default=20, ge=1, le=50),
        x_carepilot_username: str | None = Header(default=None),
) -> ConversationHistoryResponse:
    """Return recent patient-scoped turns without invoking the agent or an LLM."""
    username = _resolve_username(x_carepilot_username)
    rows = await _db_handler().get_execution_history(username, limit=limit)
    return ConversationHistoryResponse(
        conversations=[
            ConversationTurn(
                id=str(row["id"]) if row.get("id") is not None else None,
                prompt=row.get("prompt") or "",
                response=row.get("final_response"),
                status=row.get("status") or "error",
                error=row.get("error"),
                created_at=row.get("created_at"),
                steps=[_public_step(step) for step in row.get("steps") or []],
            )
            for row in rows
        ]
    )


@app.get("/api/documents")
async def list_documents(x_carepilot_username: str | None = Header(default=None)) -> dict[str, Any]:
    username = _resolve_username(x_carepilot_username)
    db = _db_handler()
    file_names = await db.list_files(username)
    documents = await db.get_patient_documents(username, limit=1000)
    by_name = {doc.get("file_name"): doc for doc in documents}
    results = []
    for file_name in file_names:
        doc = by_name.get(file_name, {})
        content = doc.get("content") or ""
        results.append(
            {
                "file_name": file_name,
                "size_chars": len(content) if content else None,
            }
        )
    return {"documents": results}


@app.post("/api/documents", response_model=UploadResponse)
async def upload_document(
        file: UploadFile = File(...),
        x_carepilot_username: str | None = Header(default=None),
) -> UploadResponse:
    username = _resolve_username(x_carepilot_username)
    if not file.filename:
        raise HTTPException(status_code=400, detail="File must have a name.")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    db = _db_handler()
    uploaded = await db.upload_file(username, file.filename, data)
    if not uploaded:
        return UploadResponse(status="error", file_name=file.filename, error="Upload failed.")

    try:
        await db.chunkify_file(username, file.filename)
    except Exception as e:
        return UploadResponse(status="uploaded", file_name=file.filename, error=f"Indexing failed: {e}")

    return UploadResponse(status="uploaded", file_name=file.filename)


@app.get("/api/documents/{file_name}", response_model=DocumentContent)
async def get_document(
        file_name: str,
        x_carepilot_username: str | None = Header(default=None),
) -> DocumentContent:
    username = _resolve_username(x_carepilot_username)
    content = await _db_handler().get_file(username, file_name)
    if content is None:
        raise HTTPException(status_code=404, detail=f"File '{file_name}' not found.")
    return DocumentContent(file_name=file_name, content=content)


@app.delete("/api/documents/{file_name}")
async def delete_document(
        file_name: str,
        x_carepilot_username: str | None = Header(default=None),
) -> dict[str, Any]:
    username = _resolve_username(x_carepilot_username)
    deleted = await _db_handler().delete_file(username, file_name)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"File '{file_name}' not found.")
    return {"status": "deleted", "file_name": file_name}


@app.get("/api/team_info")
async def team_info() -> dict[str, Any]:
    return {
        "group_batch_order_number": os.getenv("CAREPILOT_GROUP_BATCH_ORDER_NUMBER", ""),
        "team_name": os.getenv("CAREPILOT_TEAM_NAME", ""),
        "students": _students_from_env(),
    }


@app.get("/api/agent_info")
async def agent_info() -> dict[str, Any]:
    return {
        "description": "CarePilot helps cancer patients understand records, prepare questions, and draft care-logistics messages.",
        "purpose": "Provide grounded, patient-facing support using patient database records, clinical references, planning, execution, replanning, and final safety review.",
        "prompt_template": {
            "template": (
                "Authenticated patient prompt: {prompt}. "
                "Plan minimal tasks, execute one task at a time with injected identity, replan if needed, then safety-check before responding."
            )
        },
        "prompt_examples": [_example_prompt_response()],
    }


@app.get("/api/model_architecture")
async def model_architecture() -> Response:
    return Response(content=_ARCHITECTURE_PNG_PATH.read_bytes(), media_type="image/png")


@app.post("/api/execute", response_model=ExecuteResponse)
async def execute(
        request: ExecuteRequest,
        x_carepilot_username: str | None = Header(default=None),
) -> ExecuteResponse:
    username = _resolve_username(x_carepilot_username)
    try:
        result = await asyncio.wait_for(
            _orchestrator().execute(username=username, prompt=request.prompt),
            timeout=EXECUTE_TIMEOUT_SECONDS,
        )
        return ExecuteResponse(
            status="ok" if result.status == "success" else "error",
            error=result.error,
            response=result.response if result.status == "success" else None,
            steps=result.steps,
        )
    except asyncio.TimeoutError:
        return ExecuteResponse(
            status="error",
            error="Execution timed out.",
            response=None,
            steps=[],
        )
    except Exception as e:
        return ExecuteResponse(
            status="error",
            error=str(e),
            response=None,
            steps=[],
        )


def _students_from_env() -> list[dict[str, str]]:
    raw_students = os.getenv("CAREPILOT_STUDENTS_JSON")
    if raw_students:
        try:
            students = json.loads(raw_students)
            if isinstance(students, list):
                return [
                    {
                        "name": str(student.get("name", "")),
                        "email": str(student.get("email", "")),
                    }
                    for student in students
                    if isinstance(student, dict)
                ]
        except json.JSONDecodeError:
            pass
    return []


def _public_step(step: dict[str, Any]) -> dict[str, Any]:
    """Translate a persisted DB step back to the public trace shape used by the GUI."""
    prompt = step.get("prompt")
    if not isinstance(prompt, dict):
        prompt = {
            "System_prompt": step.get("system_prompt"),
            "User_prompt": step.get("user_prompt"),
        }
    return {
        "module": step.get("module"),
        "prompt": prompt,
        "response": step.get("response"),
    }


def _example_prompt_response() -> dict[str, Any]:
    prompt = "I have a checkup tomorrow, help me prepare."
    steps = [
        {
            "module": PLANNING_MODULE,
            "prompt": {
                "System_prompt": PLANNER_SYSTEM_PROMPT.strip(),
                "User_prompt": json.dumps({"patient_prompt": prompt}, ensure_ascii=True),
            },
            "response": json.dumps(
                {
                    "tasks": [
                        {
                            "task_id": "task_1",
                            "description": "Review recent patient documents for labs, appointments, insurance, referrals, and medications.",
                            "tool_hint": "patient_db",
                        },
                        {
                            "task_id": "task_2",
                            "description": "Summarize preparation items for the upcoming checkup.",
                            "tool_hint": "summarization",
                        },
                    ]
                },
                ensure_ascii=True,
            ),
        },
        {
            "module": EXECUTOR_MODULE,
            "prompt": {
                "System_prompt": "Executes one planned task with injected patient identity.",
                "User_prompt": json.dumps({"task_id": "task_1", "tool_hint": "patient_db"}, ensure_ascii=True),
            },
            "response": "Patient DB documents reviewed for appointment preparation.",
        },
        {
            "module": REPLANNER_MODULE,
            "prompt": {
                "System_prompt": "Checks whether more tasks are needed.",
                "User_prompt": json.dumps({"completed_tasks": ["task_1", "task_2"]}, ensure_ascii=True),
            },
            "response": json.dumps({"done": True, "final_answer_context": "Bring recent labs, medication list, referral status, and insurance letters."}, ensure_ascii=True),
        },
        {
            "module": SAFETY_MODULE,
            "prompt": {
                "System_prompt": SAFETY_PROMPT.strip(),
                "User_prompt": json.dumps({"query": prompt}, ensure_ascii=True),
            },
            "response": json.dumps({"action_taken": "PASS", "final_output": "Bring your recent lab results, medication list, referral status, and insurance letters. Ask your care team which symptoms should prompt urgent contact."}, ensure_ascii=True),
        },
    ]
    return {
        "prompt": prompt,
        "full_response": "Bring your recent lab results, medication list, referral status, and insurance letters. Ask your care team which symptoms should prompt urgent contact.",
        "steps": steps,
    }


_STATIC_DIR = Path(__file__).resolve().parents[2] / "static"
if _STATIC_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(_STATIC_DIR), html=True), name="static")
