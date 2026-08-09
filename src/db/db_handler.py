from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from langchain_text_splitters import RecursiveCharacterTextSplitter
try:
    from markitdown import MarkItDown
except ImportError:
    MarkItDown = None

from src.db.chunking import (
    MEDICAL_CHUNK_OVERLAP,
    MEDICAL_CHUNK_SEPARATORS,
    MEDICAL_CHUNK_SIZE,
)
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
        if MarkItDown is None:
            raise ImportError("markitdown is required to instantiate DBHandler")
        self._md = MarkItDown()

        self._medical_splitter = RecursiveCharacterTextSplitter(
            chunk_size=MEDICAL_CHUNK_SIZE,  # ~875 tokens
            chunk_overlap=MEDICAL_CHUNK_OVERLAP,  # ~150 tokens (~17% margin)
            separators=MEDICAL_CHUNK_SEPARATORS)
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
    async def get_patient_documents(
            self,
            username: str,
            limit: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Read patient documents from DB rows without vector retrieval.
        :param username: identifying name of the patient that owns the files.
        :param limit: maximum number of documents to return.
        :return: document records.
        """
        raise NotImplementedError

    @abstractmethod
    async def query_file(self, username: str, query: str, top_k: int) -> List[tuple[str, str, str]]:
        """
        Query for files in the database that are owned by the user.
        The query is written in natural language, and information is retrieved as RAG chunks.
        returns the top-k chunks that match the query.
        :param username: identifying name of the patient that owns the files.
        :param query: the query to be queried
        :param top_k: the number of files to return
        :return: a list of tuples (file_name, chunk_index, chunk_text).
        """
        raise NotImplementedError

    @abstractmethod
    async def file_exists(self, username: str, file_name: str) -> bool:
        """
        Determines if the file exists in the database and is owned by the user.
        also verifies that the same file exists in storage.
        :param username: identifying name of the patient that owns the file.
        :param file_name: name of the file to be checked
        :return: True if the file exists, False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    async def list_users(self) -> list[str]:
        """
        List all users in the database. THIS INFORMATION IS NOT FOR THE AGENT.
        :return: a list of usernames.
        """
        raise NotImplementedError

    @abstractmethod
    async def save_execution(
            self,
            username_or_session: str,
            prompt: str,
            final_response: str | None,
            status: str,
            steps: list[dict[str, Any]],
            error: str | None = None,
    ) -> str | None:
        """
        Save one API execution and its step trace.
        :param username_or_session: authenticated username or anonymous session id.
        :param prompt: raw user prompt.
        :param final_response: final answer returned to the user, if any.
        :param status: execution status.
        :param steps: ordered step trace dictionaries.
        :param error: error text, if execution failed.
        :return: saved execution id, otherwise None.
        """
        raise NotImplementedError

    @abstractmethod
    async def get_execution_history(
            self,
            username_or_session: str,
            limit: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Read recent executions and their steps for one username or session.
        :param username_or_session: authenticated username or anonymous session id.
        :param limit: maximum number of executions to return.
        :return: execution records newest-first, each with a steps list.
        """
        raise NotImplementedError


def get_db_handler() -> "DBHandler":
    """Return the remote DB handler."""

    from src.db.supabase_handler import SupabaseHandler

    return SupabaseHandler()
