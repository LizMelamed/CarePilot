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

## Use the Web UI

1. Start `uvicorn`.
2. Open `http://127.0.0.1:8000/`.
3. Use `Ask CarePilot` to run the agent.
4. Use `My Documents` to upload patient files.
5. The public submission has no login gate and opens with synthetic `patient_1`.
   `CAREPILOT_DEFAULT_USERNAME` can select another synthetic demo identity.

## Test

Run all tests:

```bash
pytest tests -rs
```

Unit tests are offline. Integration tests contact the configured LLMod,
Supabase, and Pinecone services and therefore consume remote resources; run them
only after confirming that live-service testing is intended.

## Required Submission API

- `GET /api/team_info`
- `GET /api/agent_info`
- `GET /api/model_architecture` (PNG)
- `POST /api/execute`

Successful execution responses use `status: "ok"`. Errors use
`status: "error"` with `response: null`. Every returned trace item contains a
module name, a nested system/user prompt object, and the corresponding response.
