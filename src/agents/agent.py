from abc import abstractmethod, ABC
from typing import Dict

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

from src.utils.logger import Logger
from src.utils.my_env import MyEnv


class BaseAgent(ABC):
    """
    Represents an AI agent that can receives
    a request and act as it was instructed
    """

    def __init__(self, name: str, tools: Dict[str, BaseTool]):

        self._logger: Logger = Logger()
        """logger reference"""
        self._name = name
        """Agent name, for debugging purposes."""
        self._tools = tools
        """List of tools that the agent can use."""

        self._logger.info(f"Initializing agent '{self._name}'...")

        env_handler: MyEnv = MyEnv()
        params = {}

        self._logger.info("Initializing model...")
        if env_handler.get_url() is None:
            raise ValueError("No LLM URL provided.")
        params["openai_api_base"] = env_handler.get_url()

        if env_handler.get_model() is None:
            raise ValueError("No model provided.")
        params["model"] = env_handler.get_model()


        if env_handler.get_key() is None:
            self._logger.warning("No key provided, using dummy key instead.")
            key = "d"
        else:
            key = env_handler.get_key()
        params["api_key"] = key
        key = None

        self._model = ChatOpenAI(**params)
        """The LLM used by the agent. Use self._model.bind() to modify its properties."""
        self._logger.info(f"Agent '{self._name}' initialized.")

    @abstractmethod
    async def act(self, prompts: list[HumanMessage|AIMessage]):
        """
        Performs an action given a list of prompts.
        :param prompts: one or more prompts that contain the information
        required to perform the action.
        :return: The result of the action.
        """
        raise NotImplementedError