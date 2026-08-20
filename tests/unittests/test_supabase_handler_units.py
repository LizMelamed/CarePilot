import asyncio

import pytest

from src.db.supabase_handler import SupabaseHandler


class FakeResponse:
    def __init__(self, data):
        self.data = data


class FakeQuery:
    def __init__(self, table, response_data):
        self.table = table
        self.response_data = response_data
        self.operations = []

    def select(self, value):
        self.operations.append(("select", value))
        return self

    def eq(self, key, value):
        self.operations.append(("eq", key, value))
        return self

    def limit(self, value):
        self.operations.append(("limit", value))
        return self

    def order(self, key, desc=False):
        self.operations.append(("order", key, desc))
        return self

    def in_(self, key, value):
        self.operations.append(("in", key, value))
        return self

    def upsert(self, payload, on_conflict=None):
        self.operations.append(("upsert", payload, on_conflict))
        return self

    def insert(self, payload):
        self.operations.append(("insert", payload))
        return self

    async def execute(self):
        return FakeResponse(self.response_data)


class FakeClient:
    def __init__(self):
        self.queries = []
        self.responses = {
            "users": [{"id": "user_1", "username": "patient_1"}],
            "files": [{"id": "file_1"}],
            "executions": [{"id": "exec_1", "prompt": "p"}],
            "execution_steps": [{"execution_id": "exec_1", "module": "PlanningLLM", "step_order": 1}],
        }

    def table(self, name):
        query = FakeQuery(name, self.responses.get(name, []))
        self.queries.append(query)
        return query


def _handler(client):
    handler = SupabaseHandler.__new__(SupabaseHandler)
    handler._db = client
    handler._logger = type("Logger", (), {"error": lambda *args, **kwargs: None, "exception": lambda *args, **kwargs: None})()
    return handler


def test_execution_step_payload_defaults_order_and_requires_module():
    payload = SupabaseHandler._SupabaseHandler__execution_step_payload(
        "exec_1",
        {"module": "PlanningLLM", "response": "ok"},
        3,
    )

    assert payload["execution_id"] == "exec_1"
    assert payload["step_order"] == 3
    assert payload["module"] == "PlanningLLM"


def test_upsert_patient_document_scopes_by_username():
    client = FakeClient()
    handler = _handler(client)

    ok = asyncio.run(
        handler.upsert_patient_document(
            username="patient_1",
            file_name="labs.txt",
            content="content",
            metadata={"synthetic": True},
        )
    )

    assert ok is True
    user_query, file_query = client.queries
    assert ("eq", "username", "patient_1") in user_query.operations
    assert file_query.operations[-1][0] == "upsert"
    assert file_query.operations[-1][1]["file_path"] == "patient_1/labs.txt"


def test_file_record_upsert_retries_without_optional_content():
    class ContentFallbackClient(FakeClient):
        def table(self, name):
            query = super().table(name)
            original_execute = query.execute

            async def execute():
                payload = next(op[1] for op in query.operations if op[0] == "upsert")
                if "content" in payload:
                    raise RuntimeError("content column does not exist")
                return await original_execute()

            query.execute = execute
            return query

    client = ContentFallbackClient()
    handler = _handler(client)

    ok = asyncio.run(
        handler._upsert_file_record(
            {"file_path": "patient_1/labs.txt", "file_name": "labs.txt", "content": "labs"}
        )
    )

    assert ok is True
    assert len(client.queries) == 2
    fallback_payload = next(op[1] for op in client.queries[1].operations if op[0] == "upsert")
    assert "content" not in fallback_payload


def test_file_record_upsert_reraises_unrelated_failure():
    class BrokenQuery(FakeQuery):
        async def execute(self):
            raise RuntimeError("network unavailable")

    class BrokenClient(FakeClient):
        def table(self, name):
            query = BrokenQuery(name, [])
            self.queries.append(query)
            return query

    handler = _handler(BrokenClient())

    with pytest.raises(RuntimeError, match="network unavailable"):
        asyncio.run(
            handler._upsert_file_record(
                {"file_path": "patient_1/labs.txt", "file_name": "labs.txt", "content": "labs"}
            )
        )


def test_get_execution_history_attaches_steps():
    client = FakeClient()
    handler = _handler(client)

    history = asyncio.run(handler.get_execution_history("patient_1"))

    assert history[0]["id"] == "exec_1"
    assert history[0]["steps"][0]["module"] == "PlanningLLM"
