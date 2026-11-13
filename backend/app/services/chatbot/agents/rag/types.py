from dataclasses import dataclass
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from app.services.chatbot.agents.rag.prompts import SEARCH_QUERY_DESCRIPTION


class RagState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


@dataclass(frozen=True)
class KnowledgeBasePassage:
    file_id: str
    filename: str
    score: float
    text: str


class SearchKnowledgeBaseArgs(BaseModel):
    query: str = Field(min_length=1, description=SEARCH_QUERY_DESCRIPTION)
