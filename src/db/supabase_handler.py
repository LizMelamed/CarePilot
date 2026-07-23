import io
import os
from typing import List, Optional, Dict, Any, Union

from jedi.api import file_name
from storage3.utils import StorageException
from supabase import create_client, Client

from src.db.db_handler import DBHandler

FILES_BUCKET = "carepilot_files"


class SupabaseHandler(DBHandler):
    """
    A handler for a supabase database
    """

    def _generate_db_object(self, url: str, auth_token: str) -> Client:
        conn: Client = create_client(url, auth_token)
        return conn

    def get_patient_data(self, username: str):
        client: Client = self._db
        resp = (
            client.table("users")
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

    def upload_file(self, username: str, file_name: str, data: bytes):
        client: Client = self._db
        file_path = self.__file_path(username, file_name)

        try:

            # get user_id
            response = client.table("users").select("id").eq("username", username).execute()
            if not response:
                self._logger.error(f"User '{username}' not found.")
                return False
            rows = response.data
            user_id = rows[0].get("id")

            self._logger.info(f"Uploading file: '{file_path}'.")

            # overwrite file
            response = client.storage.from_(FILES_BUCKET).upload(
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
                client.table("files")
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

    def delete_file(self, username: str, file_name: str) -> bool:
        client: Client = self._db
        file_path = self.__file_path(username, file_name)

        try:
            # get user_id
            response = client.table("users").select("id").eq("username", username).execute()
            if not response or not response.data:
                self._logger.error(f"User '{username}' not found.")
                return False
            user_id = response.data[0].get("id")

            # delete file from Supabase storage
            self._logger.info(f"Deleting file: '{file_path}' from storage.")
            del_resp = client.storage.from_(FILES_BUCKET).remove([file_path])
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
                client.table("files")
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

    def get_file(self, username: str, file_name: str) -> str|None:
        client: Client = self._db

        _, file_ext = os.path.splitext(file_name)

        try:
            # get user_id
            response = client.table("users").select("id").eq("username", username).execute()
            if not response or not response.data:
                self._logger.error(f"User '{username}' not found.")
                return
            user_id = response.data[0].get("id")

            self._logger.info(f"Getting {username}'s file: '{file_name}' from storage...")
            # get file_path in storage
            response = (client.table("files").select("file_path")
                        .eq("user_id", user_id)
                        .eq("file_name", file_name)
                        .execute())
            if not response or not response.data:
                self._logger.error(f"User '{username}' not found.")
                return
            file_path = response.data[0].get("file_path")
            self._logger.info(f"Found path: '{file_path}'.")

            file_bytes: bytes = client.storage.from_(FILES_BUCKET).download(file_path)
            stream = io.BytesIO(file_bytes)
            result = self._md.convert(stream, file_extension=file_ext)

            return str(result)

        except StorageException as e:
            self._logger.exception(f"Storage API Error: {e}")
            return
        except Exception as e:
            self._logger.exception(f"Unexpected Error during delete: {e}")
            return


    def chunkify_file(self, username: str, file_name: str) -> bool:
        pass

    def list_files(self, username: str) -> List[str]:
        client: Client = self._db
        user_path = self.__file_path(username, "")

        files = client.storage.from_(FILES_BUCKET).list(user_path)
        return [f["name"] for f in files if f.get("id")]

    def query_file(self, username: str, query: str, top_k: int) -> List[tuple[str, str]]:
        pass

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
