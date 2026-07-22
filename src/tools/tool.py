from functools import partial
from typing import Dict, Callable, Type, List, Any

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel

from src.db.db_handler import DBHandler


class HiddenContext:
    """
    Arguments context passed to tools. All agents are completely blind to these arguments.
    """
    def __init__(self):

        self.username: str|None = None

    def to_dict(self) -> dict[str, Any]:
        """
        Converts the object to a dictionary of public variables. skips any fields that their values are None.
        :return:
        """
        return {
            key: value
            for key, value in vars(self).items()
            if not key.startswith('_') and value is not None
        }


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

    def get_tool(self, name: str, context: HiddenContext|None = None) -> BaseTool:
        """
        Retrieves a tool from the repository.
        :param name: The name of the tool.
        :param context: hidden context arguments to initialize the tool.
            The LLM is blind to these arguments (e.g: username).
        :return: The tool.
        """
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not found in repository.")

        pristine_tool = self._tools[name]
        if not context:
            return pristine_tool

        bound_tool = pristine_tool.model_copy()
        if isinstance(bound_tool, StructuredTool) and bound_tool.func is not None:
            bound_tool.func = partial(bound_tool.func, **context.to_dict())

        return bound_tool

    def get_tools(self, names: List[str], context: HiddenContext|None = None) -> Dict[str, BaseTool]:
        """
        Retrieves a mapping of names to their tools from the repository.
        :param names: A list of tool names, to filter on.
        :param context: hidden context arguments to initialize the tools.
            The LLM is blind to these arguments (e.g: username).
        :return: The filtered dict of tools.
        """
        return {name: self.get_tool(name, context=context) for name in names}

    def get_all_tools(self, context: HiddenContext|None = None) -> Dict[str, BaseTool]:
        """
        Retrieves all tools from the repository.
        :param context: hidden context arguments to initialize the tools.
            The LLM is blind to these arguments (e.g: username).
        :return: A list of all tools.
        """
        return {name: self.get_tool(name, context=context) for name in self._tools.keys()}

    def list_tool_names(self) -> List[str]:
        """
        Lists the names of all registered tools.
        :return: The names of all tools.
        """
        return list(self._tools.keys())