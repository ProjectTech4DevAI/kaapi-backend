from enum import StrEnum
from uuid import UUID

from sqlmodel import Field, SQLModel


class ChatbotMessageRoleEnum(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ChatbotMessage(SQLModel):
    role: ChatbotMessageRoleEnum
    content: str


class ChatbotTurnRequest(SQLModel):
    config_id: UUID = Field(
        description="Saved Kaapi config (Anthropic text completion) that supplies the model and reasoning settings.",
    )
    config_version: int | None = Field(
        default=None,
        ge=1,
        description="Version of `config_id` to use; omit to use its latest version.",
    )
    max_tokens: int | None = Field(
        default=None,
        gt=0,
        description="Cap on output tokens per model call; omit for the server default.",
    )
    history_token_limit: int | None = Field(
        default=None,
        gt=0,
        description="Token size of the history above which older turns are summarized; omit for the server default.",
    )
    # Required every call in v0: there is no server-side storage of the flow prompt yet.
    system_prompt: str = Field(
        min_length=1,
        description="Admin-authored flow prompt (goal, fields to collect, rules).",
    )
    history: list[ChatbotMessage] = Field(
        default_factory=list,
        description="Every message before this turn, as returned by the previous call; empty on the first call.",
    )
    message: str | None = Field(
        default=None,
        min_length=1,
        description="The new user message; omit only to get the opening greeting on the first call.",
    )


class ChatbotTurnResponse(SQLModel):
    messages: list[ChatbotMessage] = Field(
        description="Full updated transcript (system prompt + every turn, including the new reply); resend it as `history` next call.",
    )
