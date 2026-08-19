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
    response: str | None
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


@lru_cache(maxsize=1)
def _architecture_png() -> bytes:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return _minimal_png()

    width, height = 1600, 820
    image = Image.new("RGB", (width, height), "#F4F7FB")
    draw = ImageDraw.Draw(image)
    title_font = _architecture_font(ImageFont, 40, bold=True)
    subtitle_font = _architecture_font(ImageFont, 22)
    box_font = _architecture_font(ImageFont, 22, bold=True)
    detail_font = _architecture_font(ImageFont, 17)

    draw.text((70, 55), "CarePilot Agent Architecture", fill="#17233A", font=title_font)
    draw.text(
        (70, 112),
        "Minimal plan-execute-replan pipeline with grounded retrieval and final safety review",
        fill="#53627A",
        font=subtitle_font,
    )

    boxes = [
        ("Patient request", 55, 270, 235, 385, "#E8F0FE", "#2F6FED"),
        (PLANNING_MODULE, 300, 255, 525, 400, "#FFFFFF", "#2F6FED"),
        (EXECUTOR_MODULE, 605, 255, 915, 400, "#FFFFFF", "#2F6FED"),
        (REPLANNER_MODULE, 995, 255, 1215, 400, "#FFFFFF", "#2F6FED"),
        (SAFETY_MODULE, 1285, 255, 1515, 400, "#E7F7EF", "#157A4F"),
    ]
    for label, x1, y1, x2, y2, fill, outline in boxes:
        draw.rounded_rectangle((x1, y1, x2, y2), radius=18, fill=fill, outline=outline, width=4)
        _draw_centered_text(draw, label, (x1, y1, x2, y2), box_font, "#17233A")

    for start, end in [
        ((235, 327), (300, 327)),
        ((525, 327), (605, 327)),
        ((915, 327), (995, 327)),
        ((1215, 327), (1285, 327)),
    ]:
        _draw_arrow(draw, start, end, fill="#53627A", width=4)

    draw.text((336, 420), "creates the smallest useful task list", fill="#53627A", font=detail_font)
    draw.text((664, 420), "executes one task at a time", fill="#53627A", font=detail_font)
    draw.text((1017, 420), "only when more work is needed", fill="#53627A", font=detail_font)

    tool_boxes = [
        ("Patient records\nSupabase", 555, 570, 790, 700),
        ("Clinical RAG\nPinecone", 815, 570, 1050, 700),
        ("Text tools\nDeterministic", 1075, 570, 1310, 700),
    ]
    draw.text((555, 505), "Executor tools", fill="#17233A", font=box_font)
    for label, x1, y1, x2, y2 in tool_boxes:
        draw.rounded_rectangle((x1, y1, x2, y2), radius=16, fill="#FFFFFF", outline="#9AA8BC", width=3)
        _draw_centered_text(draw, label, (x1, y1, x2, y2), detail_font, "#17233A")
        _draw_arrow(draw, ((x1 + x2) // 2, y1), (760, 400), fill="#7B8798", width=3)

    # Replanning feeds a revised task back to the executor without crossing the
    # main request/response path.
    draw.line((1105, 255, 1105, 205, 760, 205, 760, 255), fill="#8A5B00", width=4)
    _draw_arrow(draw, (800, 205), (760, 255), fill="#8A5B00", width=4)
    draw.text((864, 170), "revised task", fill="#8A5B00", font=detail_font)

    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _architecture_font(image_font, size: int, bold: bool = False):
    try:
        family = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
        return image_font.truetype(family, size=size)
    except OSError:
        return image_font.load_default()


def _draw_centered_text(draw, text: str, box: tuple[int, int, int, int], font, fill: str) -> None:
    x1, y1, x2, y2 = box
    left, top, right, bottom = draw.multiline_textbbox((0, 0), text, font=font, align="center", spacing=7)
    width = right - left
    height = bottom - top
    draw.multiline_text(
        (x1 + (x2 - x1 - width) / 2, y1 + (y2 - y1 - height) / 2),
        text,
        fill=fill,
        font=font,
        align="center",
        spacing=7,
    )


def _draw_arrow(draw, start: tuple[int, int], end: tuple[int, int], fill: str = "black", width: int = 2) -> None:
    draw.line((start, end), fill=fill, width=width)
    x1, y1 = start
    x2, y2 = end
    if x2 >= x1:
        head = [(x2, y2), (x2 - 10, y2 - 6), (x2 - 10, y2 + 6)]
    else:
        head = [(x2, y2), (x2 + 10, y2 - 6), (x2 + 10, y2 + 6)]
    draw.polygon(head, fill=fill)


def _minimal_png() -> bytes:
    return bytes.fromhex(
        "89504e470d0a1a0a0000000d4948445200000001000000010802000000907753de"
        "0000000c49444154789c63606060000000040001f61738550000000049454e44ae426082"
    )


_STATIC_DIR = Path(__file__).resolve().parents[2] / "static"
if _STATIC_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(_STATIC_DIR), html=True), name="static")
