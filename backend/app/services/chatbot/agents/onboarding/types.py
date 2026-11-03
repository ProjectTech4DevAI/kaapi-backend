from typing import Annotated, Any, NotRequired, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class OnboardingState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    context_messages: NotRequired[list[BaseMessage]]
    collected_fields: dict[str, Any]
    missing_fields: NotRequired[list[str]]
    is_complete: NotRequired[bool]


class ExtractedField(BaseModel):
    name: str = Field(description="snake_case field name.")
    value: str = Field(description="Value the user clearly provided; never guessed.")


class FieldExtraction(BaseModel):
    """Fields the admin's flow prompt asks for, as answered so far by the user."""

    collected_fields: list[ExtractedField] = Field(
        default_factory=list,
        description="Every field the user has clearly provided, one entry per field.",
    )
    missing_fields: list[str] = Field(
        default_factory=list,
        description="snake_case names of required fields the user has not answered yet.",
    )
    is_complete: bool = Field(
        description="True only when no required field is missing.",
    )
