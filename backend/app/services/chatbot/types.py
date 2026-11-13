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
    # Complete and no optional field left to ask; only this hands a sub-flow back to its coordinator.
    is_done: bool = False

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


# Lives here, not in the coordinator graph, so TurnContext can reference it without an import cycle.
class CoordinatorRoute(NamedTuple):
    intent: str
    description: str | None
    target_node_id: str


class TurnContext(NamedTuple):
    """Everything a turn needs, loaded in one DB session before the first LLM call."""

    chatbot_id: UUID
    session_status: str
    collected_fields: dict[str, Any]
    stored_messages: list[ChatbotSessionMessage]
    turn_node: TurnNode
    model_settings: ChatbotModelSettings
    # Knowledge-base search client: set only when the active node is rag.
    openai_client: OpenAI | None
    # Coordinator only: its routes and each target's node; the chosen target's settings load after routing.
    routes: list[CoordinatorRoute]
    route_target_nodes: dict[str, TurnNode]
    # Set when the active node is an agent behind the coordinator: the sub-flow the coordinator defaults to.
    in_progress_node_id: str | None
