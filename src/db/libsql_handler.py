from typing import List, Optional, Dict, Any, Union

from libsql import libsql

from src.db.db_handler import DBHandler


class LibSqlHandler(DBHandler):
    """
    A singleton handler for any libsql database
    """

    def _generate_db_object(self, url: str, auth_token: str):
        conn = libsql.connect(
            database=url,
            auth_token=auth_token
        )
        return conn

    def execute_all(
            self,
            queries: List[str],
            params: Optional[List[Union[Dict[str, Any], tuple, list]]] = None
    ) -> List[Any]:
        results: List[Any] = []

        with self._db:
            cursor = self._db.cursor()

            for idx, query in enumerate(queries):
                query_params = params[idx] if (params and idx < len(params)) else ()
                cursor.execute(query, query_params)

                if cursor.description:
                    results.append(cursor.fetchall())
                else:
                    results.append(None)

        return results