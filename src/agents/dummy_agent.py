from langchain_core.messages import HumanMessage, AIMessage

from src.agents.agent import BaseAgent

class DummyAgent(BaseAgent):
    """
    Example agent that sends a request to the LLM and receives a response.
    """

    async def act(self, prompts: list[HumanMessage|AIMessage]):
        response = self._model.invoke(
            prompts
        )

        if isinstance(response, dict):
            return response.get("content", "")

        if hasattr(response, "content"):
            return getattr(response, "content")

        return str(response)