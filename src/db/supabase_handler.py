import io
import os
from itertools import batched
from typing import Any, List

try:
    from storage3.utils import StorageException
except ImportError:
    StorageException = Exception

try:
    from supabase import AsyncClient
except ImportError:
    AsyncClient = object

from src.agents.embedder import Embedder
from src.db.db_handler import DBHandler

FILES_BUCKET = "carepilot_files"
CHUNK_BATCH_SIZE = 10
"""size of batch to send embedder to convert to vectors"""
SUPABASE_BATCH_SIZE = 100
"""size of a batch to send to supabase"""


class SupabaseHandler(DBHandler):
    """
    A handler for a supabase database
    """

    def _generate_db_object(self, url: str, auth_token: str) -> AsyncClient:
        try:
            from supabase import AsyncClient as SupabaseAsyncClient
        except ImportError as exc:
            raise ImportError("supabase is required to instantiate SupabaseHandler") from exc

        conn: AsyncClient = SupabaseAsyncClient(url, auth_token)
        return conn

    async def get_patient_data(self, username: str):
        client: AsyncClient = self._db
        resp = (
            await client.table("users")
            .select("date_of_birth, gender, sex")
            .eq("username", username)
            .execute()
        )
        rows = resp.data

        # no matching username, return None
        if not rows:
            return None

        if len(rows) > 1:
            self._logger.warning(
                f"Multiple user records found for username: '{username}', (found: {len(rows)}). Using the first row.",
            )

        first_row = rows[0]
        return (
            first_row.get("date_of_birth"),
            first_row.get("gender"),
            first_row.get("sex"),
        )

    async def upload_file(self, username: str, file_name: str, data: bytes):
        client: AsyncClient = self._db
        file_path = self.__file_path(username, file_name)

        try:

            # get user_id, creating the patient record if it doesn't exist yet
            response = await client.table("users").select("id").eq("username", username).execute()
            if not response or not response.data:
                self._logger.warning(f"User '{username}' not found; creating it.")
                user_id = await self.upsert_patient_profile(username, {})
                if user_id is None:
                    self._logger.error(f"Failed to create user '{username}'.")
                    return False
            else:
                rows = response.data
                user_id = rows[0].get("id")

            self._logger.info(f"Uploading file: '{file_path}'.")

            # overwrite file
            response = await client.storage.from_(FILES_BUCKET).upload(
                path=file_path,
                file=data,
                file_options={"upsert": "true"}  # Allows overwriting existing files
            )

            # sanity check: file upload must succeed to continue
            is_uploaded = (
                    response and (
                    (isinstance(response, dict) and "path" in response)
                    or hasattr(response, "path")
            )
            )
            if not is_uploaded:
                self._logger.error(f"Failed to upload '{file_path}' to storage.")
                return False

            self._logger.info(f"Uploaded file '{file_path}' to storage. Saving DB record...")

            content = self._extract_text(data, file_name)

            # update files table
            payload = {
                "user_id": user_id,
                "file_path": file_path,
                "file_name": file_name,
                "content": content,
            }
            db_response = await self._upsert_file_record(payload)

            if db_response:
                self._logger.info(f"Successfully recorded '{file_path}' in 'files' table.")
                return True
            self._logger.error(f"File uploaded, but failed to insert record into database.")
            return False

        except StorageException as e:
            self._logger.exception(f"Storage API Error: {e}")
            return False
        except Exception as e:
            self._logger.exception(f"Unexpected Error during upload: {e}")
            return False

    async def _upsert_file_record(self, payload: dict[str, Any]) -> bool:
        try:
            db_response = (
                await self._db.table("files")
                .upsert(
                    payload,
                    on_conflict="file_path"  # Replaces existing record if file_path is already registered
                )
                .execute()
            )
            return bool(db_response.data)
        except Exception as e:
            if "content" not in payload or "content" not in str(e).lower():
                raise
            fallback_payload = {key: value for key, value in payload.items() if key != "content"}
            db_response = (
                await self._db.table("files")
                .upsert(fallback_payload, on_conflict="file_path")
                .execute()
            )
            return bool(db_response.data)

    async def _download_file_text(self, file_path: str, file_name: str) -> str | None:
        _, file_ext = os.path.splitext(file_name)
        try:
            file_bytes: bytes = await self._db.storage.from_(FILES_BUCKET).download(file_path)
            stream = io.BytesIO(file_bytes)
            result = self._md.convert(stream, file_extension=file_ext)
            return str(result)
        except Exception as e:
            self._logger.exception(f"Unexpected Error during storage file read: {e}")
            return None

    async def _select_patient_document_rows(self, username: str, limit: int):
        try:
            return await (
                self._db.table("files")
                .select("file_name, file_path, content, metadata, users!inner(username)")
                .eq("users.username", username)
                .limit(limit)
                .execute()
            )
        except Exception as e:
            if "content" not in str(e).lower():
                raise
            try:
                return await (
                    self._db.table("files")
                    .select("file_name, file_path, metadata, users!inner(username)")
                    .eq("users.username", username)
                    .limit(limit)
                    .execute()
                )
            except Exception as metadata_error:
                if "metadata" not in str(metadata_error).lower():
                    raise
                return await (
                    self._db.table("files")
                    .select("file_name, file_path, users!inner(username)")
                    .eq("users.username", username)
                    .limit(limit)
                    .execute()
                )

    async def delete_file(self, username: str, file_name: str) -> bool:
        client: AsyncClient = self._db
        file_path = self.__file_path(username, file_name)

        try:
            # get user_id
            response = await client.table("users").select("id").eq("username", username).execute()
            if not response or not response.data:
                self._logger.error(f"User '{username}' not found.")
                return False
            user_id = response.data[0].get("id")

            # delete file from Supabase storage
            self._logger.info(f"Deleting file: '{file_path}' from storage.")
            del_resp = await client.storage.from_(FILES_BUCKET).remove([file_path])
            # sanity check: deletion must succeed
            is_removed = False
            if isinstance(del_resp, dict):
                # `deleted` key contains a list of successfully deleted paths
                is_removed = bool(del_resp.get("deleted"))
            elif del_resp:  # otherwise, treated as deleted
                is_removed = True

            if not is_removed:
                self._logger.error(f"Failed to delete '{file_path}' from storage.")
                return False

            self._logger.info(
                f"Deleted file '{file_path}' from storage. Removing DB record..."
            )
            db_response = (
                await client.table("files")
                .delete()
                .eq("user_id", user_id)
                .eq("file_name", file_name)
                .execute()
            )
            # We treat any non‑empty dict as success.
            if db_response and (db_response.data is not None):
                self._logger.info(
                    f"Successfully removed '{file_path}' record from 'files' table."
                )
                return True
            else:
                self._logger.error(
                    "File deleted from storage, but failed to remove record from database."
                )
                return False

        except StorageException as e:
            self._logger.exception(f"Storage API Error: {e}")
            return False
        except Exception as e:
            self._logger.exception(f"Unexpected Error during delete: {e}")
            return False

    async def get_file(self, username: str, file_name: str) -> str | None:
        client: AsyncClient = self._db

        _, file_ext = os.path.splitext(file_name)

        try:
            self._logger.info(f"Getting {username}'s file: '{file_name}' from storage...")
            response = await (
                client.table("files")
                .select("file_path, users!inner()")
                .eq("users.username", username)
                .eq("file_name", file_name)
                .execute()
            )
            if not response or not response.data:
                self._logger.error(f"File '{file_name}' of user '{username}' not found.")
                return None
            file_path = response.data[0].get("file_path")
            self._logger.info(f"Found path: '{file_path}'.")

            self._logger.info(f"Retrieving file '{file_path}' from storage...")
            file_bytes: bytes = await client.storage.from_(FILES_BUCKET).download(file_path)
            stream = io.BytesIO(file_bytes)
            result = self._md.convert(stream, file_extension=file_ext)

            return str(result)

        except StorageException as e:
            self._logger.exception(f"Storage API Error: {e}")
            return None
        except Exception as e:
            self._logger.exception(f"Unexpected Error during delete: {e}")
            return None

    async def chunkify_file(self, username: str, file_name: str) -> bool:
        client: AsyncClient = self._db
        embedder: Embedder = Embedder()

        self._logger.info("Getting file id for user...")
        response = await (
            client.table("files")
            .select("id, users!inner()")
            .eq("users.username", username)
            .eq("file_name", file_name)
            .execute()
        )
        if not response or not response.data:
            self._logger.error(f"File '{file_name}' of user '{username}' not found.")
            return False
        file_id = response.data[0].get("id")

        file_text = await self.get_file(username, file_name)
        if file_text is None:
            return False

        self._logger.info("Embedding chunks...")
        chunks = self._medical_splitter.split_text(file_text)
        vectors = await embedder.get().aembed_documents(chunks, chunk_size=CHUNK_BATCH_SIZE)

        self._logger.info("Uploading chunks...")
        try:
            await self._db.table("chunks").delete().eq("file_id", file_id).execute()
        except Exception as e:
            self._logger.exception(f"Deleting old chunks failed: {e}")
            return False
        rows = [
            {
                "file_id": file_id,
                "chunk_index": idx,
                "content": content,
                "embedding": vector
            }
            for idx, (content, vector) in enumerate(zip(chunks, vectors))
        ]
        # whether all batches were uploaded successfully
        all_success = True
        for batch_rows in batched(rows, SUPABASE_BATCH_SIZE):
            try:
                resp = await self._db.table("chunks").upsert(list(batch_rows)).execute()
            except Exception as e:
                self._logger.exception(f"Uploading chunks failed: {e}")
                all_success = False

        return all_success

    async def list_files(self, username: str) -> List[str]:
        client: AsyncClient = self._db

        response = await (
            client.table("files")
            .select("file_name, users!inner()")
            .eq("users.username", username)
            .execute()
        )
        return [row["file_name"] for row in response.data]

    async def get_patient_documents(
            self,
            username: str,
            limit: int = 10,
    ) -> list[dict[str, Any]]:
        client: AsyncClient = self._db

        try:
            response = await self._select_patient_document_rows(username, limit)
            documents = []
            for row in response.data:
                file_name = row.get("file_name")
                file_path = row.get("file_path")
                content = row.get("content")
                if content is None and file_path and file_name:
                    content = await self._download_file_text(file_path, file_name)
                documents.append(
                    {
                        "file_name": file_name,
                        "file_path": file_path,
                        "content": content,
                        "metadata": row.get("metadata"),
                    }
                )
            return documents
        except Exception as e:
            self._logger.exception(f"Unexpected Error during patient document read: {e}")
            return []

    async def query_file(self, username: str, query: str, top_k: int) -> List[tuple[str, str, str]]:
        client: AsyncClient = self._db
        embedder: Embedder = Embedder()

        self._logger.info(f"Querying user '{username}' and query: '{query}'...")

        self._logger.info("Embedding query...")
        query_vector = await embedder.get().aembed_query(query)
        self._logger.info("Querying file...")
        try:
            response = await client.rpc(
                "query_file",
                {
                    "p_username": username,
                    "p_query_embedding": query_vector,
                    "p_top_k": top_k,
                }
            ).execute()

            selected_chunks_tuples = [
                (item["file_name"], item["chunk_index"], item["content"])
                for item in response.data
            ]
            return selected_chunks_tuples
        except Exception as e:
            self._logger.exception(f"Query API Error: {e}")
            return []

    async def file_exists(self, username: str, file_name: str) -> bool:
        client: AsyncClient = self._db
        file_path = self.__file_path(username, file_name)

        # looks for file_id, if found -> file exists
        response = await (
            client.table("files")
            .select("id, users!inner()")
            .eq("users.username", username)
            .eq("file_name", file_name)
            .execute()
        )
        # either response is None, response.data is None or [] ([] evaluates to False)
        if not response or not response.data:
            return False

        # check if file exists in storage
        response = await (client.storage.from_(FILES_BUCKET).exists(file_path))
        return response

    async def list_users(self) -> list[str]:
        client: AsyncClient = self._db

        response = await (
            client.table("users").select("username").execute())
        if not response or not response.data:
            self._logger.error(f"Failed to get response for users table.")
            return []
        return [row["username"] for row in response.data]

    async def upsert_patient_profile(
            self,
            username: str,
            profile: dict[str, Any],
    ) -> str | None:
        client: AsyncClient = self._db

        payload = {
            "username": username,
            "first_name": profile.get("first_name") or profile.get("display_name") or username,
            "last_name": profile.get("last_name") or "Patient",
            "date_of_birth": profile.get("date_of_birth") or "1970-01-01",
            "age": profile.get("age"),
            "gender": profile.get("gender") or "unknown",
            "sex": profile.get("sex") or "unknown",
            "display_name": profile.get("display_name"),
            "home_city": profile.get("home_city"),
            "diagnosis": profile.get("diagnosis"),
            "cancer_stage": profile.get("stage"),
            "treatment_plan": profile.get("treatment_plan"),
        }
        payload = {
            key: value
            for key, value in payload.items()
            if value is not None
        }

        try:
            response = await (
                client.table("users")
                .upsert(payload, on_conflict="username")
                .execute()
            )
            if not response or not response.data:
                self._logger.error(f"Failed to upsert user '{username}'.")
                return None
            return response.data[0].get("id")
        except Exception as e:
            self._logger.exception(f"Unexpected Error during patient profile upsert: {e}")
            return None

    async def upsert_patient_document(
            self,
            username: str,
            file_name: str,
            content: str,
            metadata: dict[str, Any] | None = None,
    ) -> bool:
        client: AsyncClient = self._db

        try:
            user_response = await (
                client.table("users")
                .select("id")
                .eq("username", username)
                .execute()
            )
            if not user_response or not user_response.data:
                self._logger.warning(f"User '{username}' not found; creating it.")
                user_id = await self.upsert_patient_profile(username, {})
                if user_id is None:
                    self._logger.error(f"Failed to create user '{username}'.")
                    return False
            else:
                user_id = user_response.data[0].get("id")

            payload = {
                "user_id": user_id,
                "file_path": self.__file_path(username, file_name),
                "file_name": file_name,
                "content": content,
                "metadata": metadata or {},
            }
            return await self._upsert_file_record(payload)
        except Exception as e:
            self._logger.exception(f"Unexpected Error during patient document upsert: {e}")
            return False

    async def save_execution(
            self,
            username_or_session: str,
            prompt: str,
            final_response: str | None,
            status: str,
            steps: list[dict[str, Any]],
            error: str | None = None,
    ) -> str | None:
        client: AsyncClient = self._db

        try:
            execution_payload: dict[str, Any] = {
                "username_or_session": username_or_session,
                "prompt": prompt,
                "final_response": final_response,
                "status": status,
                "error": error,
            }

            user_response = await (
                client.table("users")
                .select("id")
                .eq("username", username_or_session)
                .execute()
            )
            if user_response and user_response.data:
                execution_payload["user_id"] = user_response.data[0].get("id")

            execution_response = await (
                client.table("executions")
                .insert(execution_payload)
                .execute()
            )
            if not execution_response or not execution_response.data:
                self._logger.error("Failed to save execution.")
                return None

            execution_id = execution_response.data[0].get("id")
            step_rows = [
                self.__execution_step_payload(execution_id, step, index + 1)
                for index, step in enumerate(steps)
            ]

            if step_rows:
                steps_response = await (
                    client.table("execution_steps")
                    .insert(step_rows)
                    .execute()
                )
                if not steps_response or steps_response.data is None:
                    self._logger.error(f"Failed to save steps for execution '{execution_id}'.")
                    return None

            return execution_id
        except Exception as e:
            self._logger.exception(f"Unexpected Error during execution save: {e}")
            return None

    async def get_execution_history(
            self,
            username_or_session: str,
            limit: int = 10,
    ) -> list[dict[str, Any]]:
        client: AsyncClient = self._db

        try:
            executions_response = await (
                client.table("executions")
                .select("*")
                .eq("username_or_session", username_or_session)
                .order("created_at", desc=True)
                .limit(limit)
                .execute()
            )
            if not executions_response or not executions_response.data:
                return []

            executions = executions_response.data
            execution_ids = [execution["id"] for execution in executions]
            steps_response = await (
                client.table("execution_steps")
                .select("*")
                .in_("execution_id", execution_ids)
                .order("step_order")
                .execute()
            )

            steps_by_execution_id: dict[str, list[dict[str, Any]]] = {
                execution_id: []
                for execution_id in execution_ids
            }
            for step in steps_response.data if steps_response else []:
                steps_by_execution_id.setdefault(step["execution_id"], []).append(step)

            return [
                {
                    **execution,
                    "steps": steps_by_execution_id.get(execution["id"], []),
                }
                for execution in executions
            ]
        except Exception as e:
            self._logger.exception(f"Unexpected Error during execution history read: {e}")
            return []


    # utils
    def _extract_text(self, data: bytes, file_name: str) -> str | None:
        _, file_ext = os.path.splitext(file_name)
        try:
            result = self._md.convert(io.BytesIO(data), file_extension=file_ext)
            return str(result)
        except Exception as e:
            self._logger.exception(f"Unexpected Error during content extraction: {e}")
            return None

    @staticmethod
    def __file_path(username: str, file_name: str) -> str:
        """
        generate the file path from username and file_name
        :param username:
        :param file_name:
        :return:
        """
        return f"{username}/{file_name}"

    @staticmethod
    def __execution_step_payload(
            execution_id: str,
            step: dict[str, Any],
            default_step_order: int,
    ) -> dict[str, Any]:
        module = step.get("module")
        if not module:
            raise ValueError("Execution step is missing required field: module")

        prompt = step.get("prompt")
        if not isinstance(prompt, dict):
            prompt = {}

        return {
            "execution_id": execution_id,
            "module": module,
            # Accept the current public schema and the legacy flat schema so old
            # callers and stored fixtures remain compatible during migration.
            "system_prompt": prompt.get("System_prompt", step.get("system_prompt")),
            "user_prompt": prompt.get("User_prompt", step.get("user_prompt")),
            "response": step.get("response"),
            "step_order": step.get("step_order", default_step_order),
        }
