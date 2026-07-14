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
    LLM_URL_FIELD: Final = "LLM_URL"
    LLM_KEY_FIELD: Final = "LLM_KEY"

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

    def get_url(self):
        """
        Get the URL of the LLM service.
        :return: The URL of the LLM service, otherwise None.
        """
        return self.get(self.LLM_URL_FIELD)

    def get_key(self):
        """
        Get the API key of the LLM service.
        :return: The API key of the LLM service, otherwise None.
        """
        return self.get(self.LLM_KEY_FIELD)
