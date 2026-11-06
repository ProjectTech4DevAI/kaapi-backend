from dataclasses import dataclass, field
from typing import Any, NamedTuple, TypedDict
from uuid import UUID

from openai import OpenAI

from app.models.chatbot import (
    ChatbotCompletionStatusEnum,
    ChatbotFlowNodeTypeEnum,
    ChatbotSessionMessage,
)
from app.services.chatbot.utils import ChatbotModelSettings

ChatbotMessageDict = dict[str, str]


class ChatbotSource(TypedDict):
    file_id: str
    filename: str


@dataclass(frozen=True)
class ChatbotTurnResult:
    reply: str
    collected_fields: dict[str, Any]
    is_complete: bool
    sources: list[ChatbotSource] = field(default_factory=list)

    @property
    def completion_status(self) -> ChatbotCompletionStatusEnum:
        if self.is_complete:
            return ChatbotCompletionStatusEnum.COMPLETE
        return ChatbotCompletionStatusEnum.IN_PROGRESS


class TurnNode(NamedTuple):
    """The runnable agent node a session's turn runs on, read from its pinned flow."""

    node_id: str
    node_type: ChatbotFlowNodeTypeEnum
    prompt: str
    config_id: UUID
    config_version: int | None


class TurnContext(NamedTuple):
    """Everything a turn needs, loaded in one DB session before the LLM call."""

    chatbot_id: UUID
    session_status: str
    collected_fields: dict[str, Any]
    stored_messages: list[ChatbotSessionMessage]
    turn_node: TurnNode
    model_settings: ChatbotModelSettings
    openai_client: OpenAI | None
