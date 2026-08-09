from typing import Annotated

from langchain_core.tools import InjectedToolArg
from pydantic import BaseModel, Field

from src.db.clinical_rag import query_clinical_rag


class GetPatientDataArgs(BaseModel):
    """
    Arguments for the GetPatientData tool.
    """
    username: Annotated[str, InjectedToolArg()]

class GetFileArgs(BaseModel):
    """
    Arguments for the GetFile tool.
    """
    username: Annotated[str, InjectedToolArg()]
    file_name: str = Field(description="The name of the file to get.")

class ListFilesArgs(BaseModel):
    """
    Arguments for the ListFiles tool.
    """
    username: Annotated[str, InjectedToolArg()]

class GetPatientDocumentsArgs(BaseModel):
    """
    Arguments for the GetPatientDocuments tool.
    """
    username: Annotated[str, InjectedToolArg()]
    limit: int = Field(default=10, description="Maximum number of documents to return.")

class QueryFileArgs(BaseModel):
    """
    Arguments for the QueryFile tool.
    """
    username: Annotated[str, InjectedToolArg()]
    query: str = Field(description="Natural language query to search for files based on their content.")

class QueryClinicalRagArgs(BaseModel):
    """
    Arguments for the QueryClinicalRag tool.
    """
    query: str = Field(description="Natural language clinical question.")
    top_k: int = Field(default=5, description="Maximum number of clinical chunks to return.")


async def query_clinical_rag_tool(query: str, top_k: int = 5) -> list[dict]:
    """
    Source-attributed clinical retrieval tool wrapper.
    """
    matches = await query_clinical_rag(query=query, top_k=top_k)
    return [
        {
            "id": match.id,
            "score": match.score,
            "text": match.metadata.get("text"),
            "source_url": match.metadata.get("source_url"),
            "title": match.metadata.get("title"),
            "topic_tags": match.metadata.get("topic_tags", []),
        }
        for match in matches
    ]
