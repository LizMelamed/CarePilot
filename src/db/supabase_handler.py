from typing import List, Optional, Dict, Any, Union

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
        file_path = f"{username}/{file_name}"

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

    def delete_file(self, username: str, file_name: str) -> None:
        pass

    def get_file(self, username: str, file_name: str) -> str:
        #TODO: implement, use pypdf to convert pdf to text
        pass

    def chunkify_file(self, username: str, file_name: str) -> bool:
        pass

    def list_files(self, username: str) -> List[str]:
        pass

    def query_file(self, username: str, query: str, top_k: int) -> List[tuple[str, str]]:
        pass
