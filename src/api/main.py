from __future__ import annotations

import asyncio
import io
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.agents.executor_agent import EXECUTOR_MODULE
from src.agents.planner_agent import PLANNING_MODULE, PLANNER_SYSTEM_PROMPT
from src.agents.replanner_agent import REPLANNER_MODULE
from src.agents.safetyguard_agent import SAFETY_PROMPT
from src.carepilot.orchestrator import CarePilotOrchestrator

SAFETY_MODULE = "SafetyGuardLLM"
EXECUTE_TIMEOUT_SECONDS = 295

app = FastAPI(title="CarePilot API")


class ExecuteRequest(BaseModel):
    prompt: str = Field(min_length=1)


class ExecuteResponse(BaseModel):
    status: str
    error: str | None
    response: str
    steps: list[dict[str, Any]]


class DocumentMeta(BaseModel):
    file_name: str
    size_chars: int | None = None


class UploadResponse(BaseModel):
    status: str
    file_name: str
    error: str | None = None


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
    return {"users": sorted(users)}


@app.get("/api/patients/me", response_model=PatientProfile)
async def get_me(x_carepilot_username: str | None = Header(default=None)) -> PatientProfile:
    username = _resolve_username(x_carepilot_username)
    data = await _db_handler().get_patient_data(username)
    if data is None:
        raise HTTPException(status_code=404, detail=f"Unknown patient '{username}'.")
    date_of_birth, gender, sex = data
    return PatientProfile(username=username, date_of_birth=date_of_birth, gender=gender, sex=sex)


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
    return Response(content=_architecture_png(), media_type="image/png")


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
            status=result.status,
            error=result.error,
            response=result.response,
            steps=result.steps,
        )
    except asyncio.TimeoutError:
        return ExecuteResponse(
            status="error",
            error="Execution timed out.",
            response="I'm sorry, but I could not complete this request within the allowed time.",
            steps=[],
        )
    except Exception as e:
        return ExecuteResponse(
            status="error",
            error=str(e),
            response="I'm sorry, but I could not complete this request due to a server error.",
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


def _example_prompt_response() -> dict[str, Any]:
    prompt = "I have a checkup tomorrow, help me prepare."
    steps = [
        {
            "module": PLANNING_MODULE,
            "system_prompt": PLANNER_SYSTEM_PROMPT.strip(),
            "user_prompt": json.dumps({"patient_prompt": prompt}, ensure_ascii=True),
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
            "step_order": 1,
        },
        {
            "module": EXECUTOR_MODULE,
            "system_prompt": "Executes one planned task with injected patient identity.",
            "user_prompt": json.dumps({"task_id": "task_1", "tool_hint": "patient_db"}, ensure_ascii=True),
            "response": "Patient DB documents reviewed for appointment preparation.",
            "step_order": 2,
        },
        {
            "module": REPLANNER_MODULE,
            "system_prompt": "Checks whether more tasks are needed.",
            "user_prompt": json.dumps({"completed_tasks": ["task_1", "task_2"]}, ensure_ascii=True),
            "response": json.dumps({"done": True, "final_answer_context": "Bring recent labs, medication list, referral status, and insurance letters."}, ensure_ascii=True),
            "step_order": 3,
        },
        {
            "module": SAFETY_MODULE,
            "system_prompt": SAFETY_PROMPT.strip(),
            "user_prompt": json.dumps({"query": prompt}, ensure_ascii=True),
            "response": json.dumps({"action_taken": "PASS", "final_output": "Bring your recent lab results, medication list, referral status, and insurance letters. Ask your care team which symptoms should prompt urgent contact."}, ensure_ascii=True),
            "step_order": 4,
        },
    ]
    return {
        "prompt": prompt,
        "full_response": "Bring your recent lab results, medication list, referral status, and insurance letters. Ask your care team which symptoms should prompt urgent contact.",
        "steps": steps,
    }


def _architecture_png() -> bytes:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return _minimal_png()

    width, height = 1400, 520
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    boxes = [
        ("Patient", 40, 200, 160, 270),
        (PLANNING_MODULE, 230, 200, 410, 270),
        ("Task List", 480, 200, 620, 270),
        (EXECUTOR_MODULE, 690, 180, 930, 290),
        ("Tools", 760, 340, 860, 410),
        (REPLANNER_MODULE, 1000, 200, 1160, 270),
        (SAFETY_MODULE, 1220, 200, 1380, 270),
    ]
    for label, x1, y1, x2, y2 in boxes:
        draw.rectangle((x1, y1, x2, y2), outline="black", width=3)
        draw.text((x1 + 12, y1 + 24), label, fill="black", font=font)

    arrows = [
        ((160, 235), (230, 235)),
        ((410, 235), (480, 235)),
        ((620, 235), (690, 235)),
        ((930, 235), (1000, 235)),
        ((1160, 235), (1220, 235)),
        ((1220, 270), (160, 310)),
        ((810, 290), (810, 340)),
        ((860, 340), (900, 290)),
    ]
    for start, end in arrows:
        _draw_arrow(draw, start, end)

    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _draw_arrow(draw, start: tuple[int, int], end: tuple[int, int]) -> None:
    draw.line((start, end), fill="black", width=2)
    x1, y1 = start
    x2, y2 = end
    if x2 >= x1:
        head = [(x2, y2), (x2 - 10, y2 - 6), (x2 - 10, y2 + 6)]
    else:
        head = [(x2, y2), (x2 + 10, y2 - 6), (x2 + 10, y2 + 6)]
    draw.polygon(head, fill="black")


def _minimal_png() -> bytes:
    return bytes.fromhex(
        "89504e470d0a1a0a0000000d4948445200000001000000010802000000907753de"
        "0000000c49444154789c63606060000000040001f61738550000000049454e44ae426082"
    )


_STATIC_DIR = Path(__file__).resolve().parents[2] / "static"
if _STATIC_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(_STATIC_DIR), html=True), name="static")
