from dataclasses import dataclass
from typing import Any

from app.models.chatbot import ChatbotCompletionStatusEnum

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
