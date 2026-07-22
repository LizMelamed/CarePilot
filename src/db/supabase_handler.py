from typing import List, Optional, Dict, Any, Union

from supabase import create_client, Client

from src.db.db_handler import DBHandler


class SupabaseHandler(DBHandler):
    """
    A singleton handler for any libsql database
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

    def create_file(self, username: str, file_name: str, data: str):
        pass
