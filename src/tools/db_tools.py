from typing import Annotated

from langchain_core.tools import InjectedToolArg
from pydantic import BaseModel, Field


class GetPatientDataArgs(BaseModel):
    """
    Arguments for the GetPatientData tool.
    """
    username: str = Annotated[str, InjectedToolArg]

class GetFileArgs(BaseModel):
    """
    Arguments for the GetFile tool.
    """
    username: str = Annotated[str, InjectedToolArg]
    file_name: str = Field(description="The name of the file to get.")

class ListFilesArgs(BaseModel):
    """
    Arguments for the ListFiles tool.
    """
    username: str = Annotated[str, InjectedToolArg]

class QueryFileArgs(BaseModel):
    """
    Arguments for the QueryFile tool.
    """
    username: str = Annotated[str, InjectedToolArg]
    query: str = Field(description="Natural language query to search for files based on their content.")