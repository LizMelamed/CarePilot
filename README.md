# CarePilot

CarePilot is a web UI and FastAPI backend for grounded patient-record and clinical-reference Q&A.

## Run the Web UI

Install dependencies into the project venv:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

From the `CarePilot` folder:

```bash
source .venv/bin/activate
uvicorn api.index:app --host 127.0.0.1 --port 8000
```

Open this in a browser:

```text
http://127.0.0.1:8000/
```

## Required Remote Services

Before starting the web UI, `res/configs/.env` must point to existing working remote services:

```text
LOCAL_MODE=false
REMOTE_LLM_URL=https://api.llmod.ai/v1
REMOTE_LLM_MODEL=MB5R2CF-azure/gpt-5.4-mini
REMOTE_EMBEDDER_URL=https://api.llmod.ai/v1
REMOTE_EMBEDDER_MODEL=MB5R2CF-azure/text-embedding-3-small
REMOTE_DB_BACKEND=supabase
REMOTE_VECTOR_STORE_BACKEND=pinecone
PINECONE_INDEX=carepilot-clinical-rag
```

Do not commit real keys.

## Use the Web UI

1. Start `uvicorn`.
2. Open `http://127.0.0.1:8000/`.
3. Use `Ask CarePilot` to run the agent.
4. Use `My Documents` to upload patient files.
5. The default patient is `patient_1` unless `CAREPILOT_DEFAULT_USERNAME` is set.

## Test

Run all tests:

```bash
pytest tests -rs
```
