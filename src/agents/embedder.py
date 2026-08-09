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
            model=model,
            # Ollama's OpenAI-compatible embeddings endpoint only accepts raw text input.
            # Without this, langchain pre-tokenizes text into integer token arrays via
            # tiktoken (valid for real OpenAI), which Ollama rejects as "invalid input type".
            check_embedding_ctx_length=False,
        )
        """The Embedder object used to generate vector embeddings."""
        self._logger.info(f"Embedder initialized.")

    def get(self) -> OpenAIEmbeddings:
        """
        Get the embedder object used to generate vector embeddings.
        :return:
        """
        return self._embeddings