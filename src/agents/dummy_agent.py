from src.agents.agent import BaseAgent, AgentContext


class DummyAgent(BaseAgent):
    """
    Example agent that sends a request to the LLM and receives a response.
    """

    async def act(self, ctx: AgentContext):
        response = self._model.invoke(
            ctx.prompts
        )

        if isinstance(response, dict):
            return response.get("content", "")

        if hasattr(response, "content"):
            return getattr(response, "content")

        return str(response)