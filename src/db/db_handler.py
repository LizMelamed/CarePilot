from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from langchain_text_splitters import RecursiveCharacterTextSplitter
from markitdown import MarkItDown

from src.utils.my_env import MyEnv
from src.utils.logger import Logger
from src.utils.singleton import SingletonMeta


class DBHandler(ABC):
    """
    An abstract handler for the database
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
        self._md = MarkItDown()

        self._medical_splitter = RecursiveCharacterTextSplitter(
            chunk_size=3500,  # ~875 tokens
            chunk_overlap=600,  # ~150 tokens (~17% margin)
            separators=[
                "\n# ",
                "\n## ",
                "\n\n",
                "\n",
                ". ",
                " "
            ])
        """The splitter used for chunkifying medical documents."""
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
    async def get_patient_data(self, username: str):
        """
        Get the patient's (LLM visible) data from the database.
        :param username: username identifying the patient
        :return: a tuple of the data, otherwise None if not found based on username.
        """
        raise NotImplementedError

    @abstractmethod
    async def upload_file(self, username: str, file_name: str, data: bytes) -> bool:
        """
        Create or replace an existing file in the database.
        ONLY TEXT FILES ARE ALLOWED.
        :param username: username identifying the patient that owns the file
        :param file_name: name of the file
        :param data: the data stored in the file
        :return: True if the file was successfully uploaded, False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    async def delete_file(self, username: str, file_name: str) -> bool:
        """
        Delete a file in the database.
        :param username: identifying name of the patient that owns the file
        :param file_name: the name of the file to be deleted
        :return: True if the file was successfully deleted, False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    async def get_file(self, username: str, file_name: str) -> str|None:
        """
        Get a file from the database.
        Uses markitdown to convert the file to a string in Markdown format.
        :param username: identifying name of the patient that owns the file.
        :param file_name: the name of the file to be retrieved.
        :return: The textual content of the retrieved file, otherwise None if not found based on username and file_name.
        """
        raise NotImplementedError

    @abstractmethod
    async def chunkify_file(self, username: str, file_name: str) -> bool:
        """
        Looks for a file, builds and stores a list of chunks
        :param username: identifying name of the patient that owns the file.
        :param file_name: the name of the file to be chunked.
        :return: True if the file was successfully chunked, False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    async def list_files(self, username: str) -> List[str]:
        """
        List all files in the database that are owned by the user.
        :param username: identifying name of the patient that owns the files.
        :return: a list of file names.
        """
        raise NotImplementedError

    @abstractmethod
    async def query_file(self, username: str, query: str, top_k: int) -> List[tuple[str, str]]:
        """
        Query for files in the database that are owned by the user.
        returns the top-k chunks that match the query.
        :param username: identifying name of the patient that owns the files.
        :param query: the query to be queried
        :param top_k: the number of files to return
        :return: a list of tuples (file_name, chunk).
        """
        raise NotImplementedError