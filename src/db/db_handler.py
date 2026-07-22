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
    def get_patient_data(self, username: str):
        """
        Get the patient's (LLM visible) data from the database.
        :param username: username identifying the patient
        :return: a tuple of the data, otherwise None if not found based on username.
        """
        raise NotImplementedError

    @abstractmethod
    def create_file(self, username: str, file_name: str, data: str):
        """
        Create a new file in the database.
        :param username: username identifying the patient that owns the file
        :param file_name: name of the file
        :param data: the textual data stored in the file
        :return:
        """
        raise NotImplementedError