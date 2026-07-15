from typing import Dict, Callable, Type, List

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel


class ToolRepository:
    """
    A central repository to register, manage, and retrieve LangChain tools.
    """

    def __init__(self):
        # name -> tool
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """
        Registers a tool to the repository.
        :param tool: the tool to register.
        :return:
        """
        if tool.name in self._tools:
            raise ValueError(f"Tool with name '{tool.name}' is already registered.")
        self._tools[tool.name] = tool

    def register_func(
            self,
            func: Callable,
            name: str,
            description: str,
            args_schema: Type[BaseModel] = None
    ) -> BaseTool:
        """
        Registers a function to the repository.
        :param func: The function to register.
        :param name: The name of the tool.
        :param description: The description of the tool.
        :param args_schema: The schema of the arguments of the tool.
        :return: The registered tool.
        """
        tool = StructuredTool.from_function(
            func=func,
            name=name,
            description=description,
            args_schema=args_schema
        )
        self.register(tool)
        return tool

    def get_tool(self, name: str) -> BaseTool:
        """
        Retrieves a tool from the repository.
        :param name: The name of the tool.
        :return: The tool.
        """
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not found in repository.")
        return self._tools[name]

    def get_tools(self, names: List[str]) -> Dict[str, BaseTool]:
        """
        Retrieves a mapping of names to their tools from the repository.
        :param names: A list of tool names, to filter on.
        :return: The filtered dict of tools.
        """
        return {name: self.get_tool(name) for name in names}

    def get_all_tools(self) -> Dict[str, BaseTool]:
        """
        Retrieves all tools from the repository.
        :return: A list of all tools.
        """
        return self._tools.copy()

    def list_tool_names(self) -> List[str]:
        """
        Lists the names of all registered tools.
        :return: The names of all tools.
        """
        return list(self._tools.keys())