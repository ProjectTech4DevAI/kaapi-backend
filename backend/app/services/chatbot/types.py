from dataclasses import dataclass
from typing import Any, NamedTuple
from uuid import UUID

from app.models.chatbot import (
    ChatbotCompletionStatusEnum,
    ChatbotFlowNodeTypeEnum,
    ChatbotSessionMessage,
)
from app.services.chatbot.utils import ChatbotModelSettings

ChatbotMessageDict = dict[str, str]


@dataclass(frozen=True)
class ChatbotTurnResult:
    reply: str
    collected_fields: dict[str, Any]
    is_complete: bool

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
    """Everything a turn needs, loaded in one DB session before the LLM call."""

    chatbot_id: UUID
    session_status: str
    collected_fields: dict[str, Any]
    stored_messages: list[ChatbotSessionMessage]
    turn_node: TurnNode
    model_settings: ChatbotModelSettings
    # Coordinator only: its routes and each target's node + settings, resolved up front.
    routes: list[CoordinatorRoute]
    route_targets: dict[str, tuple[TurnNode, ChatbotModelSettings]]
    # Agent behind a coordinator: where the session returns once the agent completes.
    return_node_id: str | None
