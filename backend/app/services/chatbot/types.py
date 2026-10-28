from dataclasses import dataclass, field
from typing import Any, TypedDict

from app.models.chatbot import ChatbotCompletionStatusEnum

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
