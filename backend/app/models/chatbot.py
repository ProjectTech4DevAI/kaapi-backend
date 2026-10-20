from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import JSON, Column
from sqlalchemy.dialects.postgresql import ENUM
from sqlmodel import Field, SQLModel

from app.core.util import now


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


class ChatbotSessionStatusEnum(StrEnum):
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ABANDONED = "abandoned"


# values_callable stores the lowercase member values the migration created the
# Postgres enum types with; SQLAlchemy's default would store member names.
_SESSION_STATUS_PG_ENUM = ENUM(
    ChatbotSessionStatusEnum,
    name="chatbotsessionstatusenum",
    values_callable=lambda enum_cls: [member.value for member in enum_cls],
    create_type=False,
)
_MESSAGE_ROLE_PG_ENUM = ENUM(
    ChatbotMessageRoleEnum,
    name="chatbotmessageroleenum",
    values_callable=lambda enum_cls: [member.value for member in enum_cls],
    create_type=False,
)

CONTACT_REF_MAX_LENGTH = 500


class ChatbotSession(SQLModel, table=True):
    __tablename__ = "chatbot_session"

    id: UUID = Field(
        default_factory=uuid4,
        primary_key=True,
        sa_column_kwargs={"comment": "Session identifier returned to the caller"},
    )
    contact_ref: str = Field(
        max_length=CONTACT_REF_MAX_LENGTH,
        sa_column_kwargs={
            "comment": "Caller-side identifier of the end user (e.g. a Glific contact id); opaque to Kaapi"
        },
    )
    config_id: UUID = Field(
        foreign_key="config.id",
        nullable=False,
        index=True,
        ondelete="RESTRICT",
        sa_column_kwargs={
            "comment": "Saved Anthropic text config that supplies the chat model"
        },
    )
    config_version: int | None = Field(
        default=None,
        ge=1,
        sa_column_kwargs={
            "comment": "Pinned config version; NULL resolves to the latest version on every turn"
        },
    )
    system_prompt: str = Field(
        sa_column_kwargs={
            "comment": "Admin-authored flow prompt captured at session creation"
        },
    )
    status: ChatbotSessionStatusEnum = Field(
        default=ChatbotSessionStatusEnum.IN_PROGRESS,
        sa_column=Column(
            _SESSION_STATUS_PG_ENUM,
            nullable=False,
            comment="Conversation lifecycle: in_progress, completed, or abandoned",
        ),
    )
    collected_fields: dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(
            JSON,
            nullable=False,
            comment="Flat map of field name -> value gathered from the user so far",
        ),
    )
    organization_id: int = Field(
        foreign_key="organization.id",
        nullable=False,
        index=True,
        ondelete="CASCADE",
        sa_column_kwargs={"comment": "Reference to the organization"},
    )
    project_id: int = Field(
        foreign_key="project.id",
        nullable=False,
        index=True,
        ondelete="CASCADE",
        sa_column_kwargs={"comment": "Reference to the project"},
    )
    inserted_at: datetime = Field(
        default_factory=now,
        nullable=False,
        sa_column_kwargs={"comment": "Timestamp when the session was created"},
    )
    updated_at: datetime = Field(
        default_factory=now,
        nullable=False,
        sa_column_kwargs={
            "comment": "Timestamp when the session was last updated",
            "onupdate": now,
        },
    )


class ChatbotSessionMessage(SQLModel, table=True):
    __tablename__ = "chatbot_session_message"

    id: UUID = Field(
        default_factory=uuid4,
        primary_key=True,
        sa_column_kwargs={"comment": "Unique identifier"},
    )
    session_id: UUID = Field(
        foreign_key="chatbot_session.id",
        nullable=False,
        index=True,
        ondelete="CASCADE",
        sa_column_kwargs={"comment": "Reference to the owning chatbot session"},
    )
    role: ChatbotMessageRoleEnum = Field(
        sa_column=Column(
            _MESSAGE_ROLE_PG_ENUM,
            nullable=False,
            comment="Speaker: user or assistant (system is reserved; the prompt lives on the session)",
        ),
    )
    content: str = Field(
        sa_column_kwargs={"comment": "Message text"},
    )
    inserted_at: datetime = Field(
        default_factory=now,
        nullable=False,
        sa_column_kwargs={
            "comment": "Timestamp when the message was stored; defines transcript order"
        },
    )


class CreateChatbotSessionRequest(SQLModel):
    config_id: UUID = Field(
        description="Saved Kaapi config (Anthropic text completion) that supplies the model and reasoning settings.",
    )
    config_version: int | None = Field(
        default=None,
        ge=1,
        description="Version of `config_id` to pin; omit to use the latest version on every turn.",
    )
    contact_ref: str = Field(
        min_length=1,
        max_length=CONTACT_REF_MAX_LENGTH,
        description="Caller-side identifier of the end user this conversation is with.",
    )
    system_prompt: str = Field(
        min_length=1,
        description="Admin-authored flow prompt (goal, fields to collect, rules); stored for the session's lifetime.",
    )


class CreateChatbotSessionResponse(SQLModel):
    session_id: UUID


class SendChatbotSessionMessageRequest(SQLModel):
    message: str | None = Field(
        default=None,
        min_length=1,
        description="The new user message; omit only on the first call to get the opening greeting.",
    )
    webhook_url: str = Field(
        min_length=1,
        description="HTTPS URL the bot reply is POSTed to once the turn completes.",
    )


class SendChatbotSessionMessageResponse(SQLModel):
    status: str = "queued"
