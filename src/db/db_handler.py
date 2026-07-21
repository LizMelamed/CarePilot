from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from src.utils.my_env import MyEnv
from src.utils.logger import Logger
from src.utils.singleton import SingletonMeta


class DBHandler(ABC):
    """
    A singleton abstract handler for the database
    """

    def __init__(self):
        self._logger: Logger = Logger()
        self._logger.info("Initializing DBHandler...")
        env: MyEnv = MyEnv()
        url = env.get_db_url()
        auth_token = env.get_db_token()

        assert url is not None, "URL of database not set"
        assert auth_token is not None, "Authorization token of database not set"

        self._logger.info("Initializing database object...")
        self._db = self._generate_db_object(url=url, auth_token=auth_token)
        self._logger.info("Database object initialized.")

        self._logger.info("DBHandler initialized.")

    @abstractmethod
    def _generate_db_object(self, url: str, auth_token: str):
        """
        Build the database object used to interact with the database.
        the object depends on what database is being used (supabase, turso, etc...).
        the object is referenced with self._db.
        :param url: the database url
        :param auth_token: the authentication token.
        :return: the database object.
        """
        raise NotImplementedError

    @abstractmethod
    def execute_all(
            self,
            queries: List[str],
            params: Optional[List[Union[Dict[str, Any], tuple, list]]] = None,
    ) -> List[Any]:
        """
        Execute one or multiple SQL queries against the underlying database.

        :param queries: The SQL statements to run, in the order they should be executed.
        :param params: A list of parameter mappings that match each query. If ``None`` the
            queries are executed without parameters.
            params may be a list of lists/tuples/dicts.
        :return:
            The result of each query.  The concrete implementation decides
            the exact type of the returned objects.
        """
        raise NotImplementedError

    def execute(
            self,
            query: str,
            params: Optional[Union[Dict[str, Any], tuple, list]] = None,
    ) -> Any:
        """
        Executes a single SQL query against the underlying database.
        :param query: the SQL query to execute.
        :param params: the arguments to pass to the SQL query.
        :return: the result of the SQL query: a list of rows
        """
        new_params = [params] if params is not None else []
        return self.execute_all(queries=[query], params=new_params)[0]