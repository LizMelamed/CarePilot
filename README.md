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

## Configuration

Copy the tracked template, then fill the ignored local file with the team's real
values. Never commit the resulting `.env` file.

```bash
mkdir -p res/configs
cp .env.example res/configs/.env
```

The required remote-service fields are:

```text
LLM_URL=https://api.llmod.ai/v1
LLM_KEY=...
LLM_MODEL=MB5R2CF-azure/gpt-5.4-mini
EMBEDDER_URL=https://api.llmod.ai/v1
EMBEDDER_KEY=...
EMBEDDER_MODEL=MB5R2CF-azure/text-embedding-3-small
DB_URL=https://<project>.supabase.co
DB_AUTH_TOKEN=...
PINECONE_API_KEY=...
PINECONE_INDEX=carepilot-clinical-rag
```

The submission metadata fields are also required:

```text
CAREPILOT_GROUP_BATCH_ORDER_NUMBER=<batch>_<presentation-order>
CAREPILOT_TEAM_NAME=<team-name>
CAREPILOT_STUDENTS_JSON=[{"name":"...","email":"..."}]
CAREPILOT_DEFAULT_USERNAME=patient_1
```

On Vercel, configure these as project environment variables instead of uploading
the local `.env` file. Team metadata is intentionally not stored in Git, so all
three `CAREPILOT_*` metadata variables must also be configured in Vercel.

Never commit `res/configs/.env`: it contains service credentials. Give the
professor only the deployed Vercel URL and GitHub repository URL, as required by
the assignment. The professor can retrieve the team details from
`GET /api/team_info` on the deployed application.

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
