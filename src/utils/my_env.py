import os

from src.utils.logger import Logger
from src.utils.singleton import SingletonMeta
from typing import Final

from src.utils.utils import from_project_path

class MyEnv(metaclass=SingletonMeta):
    """
    Singleton class to manage environment variables.
    """

    ENV_PATH: Final = "res/configs/.env"
    # field name of the url of the service
    LLM_URL_FIELD: Final = "LLM_URL"
    # field name of the API KEY sent to the service
    LLM_KEY_FIELD: Final = "LLM_KEY"
    # field name of the model-name (e.g: 'o1-preview') used by the service
    LLM_MODEL_FIELD: Final = "LLM_MODEL"
    # field name of the url of the database
    DB_URL_FIELD: Final = "DB_URL"
    # field name of the authentication token used to access the database
    DB_AUTH_TOKEN_FIELD: Final = "DB_AUTH_TOKEN"
    # field name of the url of the embedder service
    EMBEDDER_URL_FIELD: Final = "EMBEDDER_URL"
    # field name of the API KEY sent to the embedder service
    EMBEDDER_KEY_FIELD: Final = "EMBEDDER_KEY"
    # field name of the embedder model-name (e.g: 'o1-preview') used by the service
    EMBEDDER_MODEL_FIELD: Final = "EMBEDDER_MODEL"

    def __init__(self):
        self._logger = Logger()
        self._logger.info("Initializing MyEnv...")

        env_path = from_project_path(self.ENV_PATH)

        # sanity check: .env file must exist
        if not env_path.exists() or not env_path.is_file():
            self._logger.warning(f"File '{env_path}' not found. Assuming variables are loaded. Skipping...")
        else:
            with open(env_path, "r") as f:
                for line in f:
                    line = line.strip()
                    # Skip empty lines and comments
                    if not line or line.startswith("#"):
                        continue
                    if "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip()

                    # Skip keys without an assigned value
                    if value == "":
                        self._logger.warning(f"Given variable '{key}' is empty. Skipping...")
                        continue

                    # skip if key is already assigned
                    if key not in os.environ:
                        os.environ[key] = value
                    else:
                        self._logger.warning(f"Environment variable '{key}' already set; leaving it unchanged.")
                    self._logger.info(f"Variable '{key}' loaded.")

        self._logger.info("MyEnv initialized.")

    @staticmethod
    def get(key: str):
        """
        Get an environment variable.
        :param key: environment variable name.
        :return: The environment variable, otherwise None.
        """
        if key not in os.environ:
            return None
        return os.environ[key]

    def get_llm_url(self):
        """
        Get the URL of the LLM service.
        :return: The URL of the LLM service, otherwise None.
        """
        return self.get(self.LLM_URL_FIELD)

    def get_llm_key(self):
        """
        Get the API key of the LLM service.
        :return: The API key of the LLM service, otherwise None.
        """
        return self.get(self.LLM_KEY_FIELD)

    def get_llm_model(self):
        """
        Get the model-name of the LLM service.
        :return: The model-name of the LLM service, otherwise None.
        """
        return self.get(self.LLM_MODEL_FIELD)

    def get_db_url(self):
        """
        Get the URL of the DB service.
        :return: The URL of the DB service, otherwise None.
        """
        return self.get(self.DB_URL_FIELD)
    def get_db_token(self):
        """
        Get the auth token of the DB service.
        :return: The auth token of the DB service, otherwise None.
        """
        return self.get(self.DB_AUTH_TOKEN_FIELD)

    def get_embedder_url(self):
        """
        Get the URL of the embedder service.
        :return: the URL of the embedder service, otherwise None.
        """
        return self.get(self.EMBEDDER_URL_FIELD)

    def get_embedder_key(self):
        """
        Get the API key of the embedder service.
        :return: The API key of the embedder service, otherwise None.
        """
        return self.get(self.EMBEDDER_KEY_FIELD)

    def get_embedder_model(self):
        """
        Get the model-name of the embedder service.
        :return: The model-name of the embedder service, otherwise None.
        """
        return self.get(self.EMBEDDER_MODEL_FIELD)
