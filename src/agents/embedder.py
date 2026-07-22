from types import CoroutineType
from typing import Any

from langchain_openai import OpenAIEmbeddings

from src.utils.logger import Logger
from src.utils.my_env import MyEnv
from src.utils.singleton import SingletonMeta


class Embedder(metaclass=SingletonMeta):

    def __init__(self):
        self._logger: Logger = Logger()
        """logger reference"""

        self._logger.info(f"Initializing embedder...")

        env_handler: MyEnv = MyEnv()

        url = env_handler.get_embedder_url()
        api_key = env_handler.get_embedder_key()
        model = env_handler.get_embedder_model()

        assert url is not None, "Embedder URL not provided."
        assert api_key is not None, "Embedder API key not provided."
        assert model is not None, "Embedder model not provided."

        self._embeddings = OpenAIEmbeddings(
            api_key=api_key,
            base_url=url,
            model=model
        )
        """The Embedder object used to generate vector embeddings."""
        self._logger.info(f"Embedder initialized.")

    async def generate_embedding_async(self, text: str):
        """
        Generate embedding for given text.
        :param text: Given text to generate embedding for.
        :return: The embedding for given text.
        """
        return self._embeddings.aembed_query(text)

    def generate_embedding(self, text: str) -> list[float]:
        """
        Generate embedding for given text.
        :param text: Given text to generate embedding for.
        :return: The embedding for given text.
        """
        return self._embeddings.embed_query(text)