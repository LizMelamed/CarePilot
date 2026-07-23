import io
import os
from itertools import batched
from typing import List

from storage3.utils import StorageException
from supabase import AsyncClient

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
        conn: AsyncClient = AsyncClient(url, auth_token)
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

            # get user_id
            response = await client.table("users").select("id").eq("username", username).execute()
            if not response:
                self._logger.error(f"User '{username}' not found.")
                return False
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

            # update files table
            db_response = (
                await client.table("files")
                .upsert(
                    {
                        "user_id": user_id,
                        "file_path": file_path,
                        "file_name": file_name,
                    },
                    on_conflict="file_path"  # Replaces existing record if file_path is already registered
                )
                .execute()
            )

            if db_response.data:
                self._logger.info(f"Successfully recorded '{file_path}' in 'files' table.")
                return True
            else:
                self._logger.error(f"File uploaded, but failed to insert record into database.")
                return False

        except StorageException as e:
            self._logger.exception(f"Storage API Error: {e}")
            return False
        except Exception as e:
            self._logger.exception(f"Unexpected Error during upload: {e}")
            return False

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


    # utils
    @staticmethod
    def __file_path(username: str, file_name: str) -> str:
        """
        generate the file path from username and file_name
        :param username:
        :param file_name:
        :return:
        """
        return f"{username}/{file_name}"
