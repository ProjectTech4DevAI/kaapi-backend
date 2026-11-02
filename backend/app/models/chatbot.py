from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator
from pydantic import Field as PydanticField
from sqlalchemy import JSON, Column, Index, UniqueConstraint, text
from sqlalchemy.dialects.postgresql import ENUM, JSONB
from sqlmodel import Field, SQLModel

from app.core.util import now


class ChatbotMessageRoleEnum(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ChatbotSessionStatusEnum(StrEnum):
    ACTIVE = "active"
    INACTIVE = "inactive"


class ChatbotCompletionStatusEnum(StrEnum):
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"
    ABANDONED = "abandoned"


class ChatbotFlowNodeTypeEnum(StrEnum):
    INPUT = "input"
    OUTPUT = "output"
    COORDINATOR = "coordinator"
    ONBOARDING = "onboarding"
    RAG = "rag"


AGENT_NODE_TYPES: frozenset[ChatbotFlowNodeTypeEnum] = frozenset(
    {
        ChatbotFlowNodeTypeEnum.COORDINATOR,
        ChatbotFlowNodeTypeEnum.ONBOARDING,
        ChatbotFlowNodeTypeEnum.RAG,
    }
)

_SESSION_STATUS_PG_ENUM = ENUM(
    ChatbotSessionStatusEnum,
    name="chatbotsessionstatusenum",
    values_callable=lambda enum_cls: [member.value for member in enum_cls],
    create_type=False,
)
_COMPLETION_STATUS_PG_ENUM = ENUM(
    ChatbotCompletionStatusEnum,
    name="chatbotcompletionstatusenum",
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
CHATBOT_NAME_MAX_LENGTH = 255
FLOW_NODE_ID_MAX_LENGTH = 255
ACTIVE_CHATBOT_SESSION_UNIQUE_INDEX = "uq_chatbot_session_active_chatbot_contact"
LIVE_CHATBOT_FLOW_VERSION_UNIQUE_INDEX = "uq_chatbot_flow_version_live_chatbot"
CHATBOT_FLOW_VERSION_UNIQUE_CONSTRAINT = "uq_chatbot_flow_version_chatbot_version"


class ChatbotFlowPosition(BaseModel):
    model_config = ConfigDict(extra="allow")

    x: float
    y: float


class ChatbotAgentNodeData(BaseModel):
    """`data` of a coordinator/onboarding/rag node; UI-only keys (labels) pass through."""

    model_config = ConfigDict(extra="allow")

    config_id: UUID | None = None
    config_version: int | None = PydanticField(default=None, ge=1)
    prompt: str | None = None


class ChatbotFlowNode(BaseModel):
    """React Flow node, stored as the UI sent it (extra keys such as `measured` round-trip)."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    id: str = PydanticField(min_length=1, max_length=FLOW_NODE_ID_MAX_LENGTH)
    type: ChatbotFlowNodeTypeEnum
    position: ChatbotFlowPosition
    data: dict[str, Any] = PydanticField(default_factory=dict)

    def agent_data(self) -> ChatbotAgentNodeData:
        return ChatbotAgentNodeData.model_validate(self.data)

    @model_validator(mode="after")
    def _validate_agent_data(self) -> "ChatbotFlowNode":
        if self.type not in AGENT_NODE_TYPES:
            return self
        try:
            self.agent_data()
        except ValidationError as e:
            raise ValueError(f"Invalid data for node '{self.id}': {e}") from e
        return self


class ChatbotFlowEdge(BaseModel):
    """React Flow edge; `data.intent` / `data.description` are reserved for coordinator routing."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    id: str = PydanticField(min_length=1, max_length=FLOW_NODE_ID_MAX_LENGTH)
    source: str = PydanticField(min_length=1, max_length=FLOW_NODE_ID_MAX_LENGTH)
    target: str = PydanticField(min_length=1, max_length=FLOW_NODE_ID_MAX_LENGTH)
    source_handle: str | None = PydanticField(default=None, alias="sourceHandle")
    target_handle: str | None = PydanticField(default=None, alias="targetHandle")
    data: dict[str, Any] = PydanticField(default_factory=dict)


class ChatbotBase(SQLModel):
    name: str = Field(
        min_length=1,
        max_length=CHATBOT_NAME_MAX_LENGTH,
        sa_column_kwargs={"comment": "Display name of the chatbot shown in the UI"},
    )


class ChatbotCreate(ChatbotBase):
    pass


class ChatbotUpdate(SQLModel):
    name: str | None = Field(
        default=None, min_length=1, max_length=CHATBOT_NAME_MAX_LENGTH
    )


class Chatbot(ChatbotBase, table=True):
    __tablename__ = "chatbot"

    id: UUID = Field(
        default_factory=uuid4,
        primary_key=True,
        sa_column_kwargs={"comment": "Chatbot identifier used in API paths"},
    )
    organization_id: int = Field(
        foreign_key="organization.id",
        nullable=False,
        index=True,
        ondelete="CASCADE",
        sa_column_kwargs={"comment": "Organization that owns the chatbot"},
    )
    project_id: int = Field(
        foreign_key="project.id",
        nullable=False,
        index=True,
        ondelete="CASCADE",
        sa_column_kwargs={"comment": "Project that owns the chatbot"},
    )
    inserted_at: datetime = Field(
        default_factory=now,
        nullable=False,
        sa_column_kwargs={"comment": "Timestamp when the chatbot was created"},
    )
    updated_at: datetime = Field(
        default_factory=now,
        nullable=False,
        sa_column_kwargs={
            "comment": "Timestamp when the chatbot was last updated",
            "onupdate": now,
        },
    )
    deleted_at: datetime | None = Field(
        default=None,
        nullable=True,
        sa_column_kwargs={
            "comment": "Soft-delete timestamp; the row is kept because sessions reference it"
        },
    )


class ChatbotPublic(ChatbotBase):
    id: UUID
    organization_id: int
    project_id: int
    inserted_at: datetime
    updated_at: datetime


class ChatbotFlowVersion(SQLModel, table=True):
    __tablename__ = "chatbot_flow_version"
    __table_args__ = (
        UniqueConstraint(
            "chatbot_id", "version", name=CHATBOT_FLOW_VERSION_UNIQUE_CONSTRAINT
        ),
        Index(
            LIVE_CHATBOT_FLOW_VERSION_UNIQUE_INDEX,
            "chatbot_id",
            unique=True,
            postgresql_where=text("is_live"),
        ),
    )

    id: UUID = Field(
        default_factory=uuid4,
        primary_key=True,
        sa_column_kwargs={"comment": "Flow version identifier pinned by sessions"},
    )
    chatbot_id: UUID = Field(
        foreign_key="chatbot.id",
        nullable=False,
        index=True,
        ondelete="CASCADE",
        sa_column_kwargs={"comment": "Chatbot this flow version belongs to"},
    )
    version: int = Field(
        ge=1,
        nullable=False,
        sa_column_kwargs={
            "comment": "Sequential version number per chatbot, starting at 1"
        },
    )
    nodes: list[dict[str, Any]] = Field(
        sa_column=Column(
            JSONB,
            nullable=False,
            comment="React Flow nodes as saved by the UI: [{id, type (input/output/coordinator/onboarding/rag), position {x, y}, data}]; agent node data carries config_id, config_version, prompt",
        ),
    )
    edges: list[dict[str, Any]] = Field(
        default_factory=list,
        sa_column=Column(
            JSONB,
            nullable=False,
            server_default=text("'[]'::jsonb"),
            comment="React Flow edges as saved by the UI: [{id, source, target, sourceHandle, targetHandle, data}]",
        ),
    )
    is_live: bool = Field(
        default=False,
        nullable=False,
        sa_column_kwargs={
            "server_default": text("false"),
            "comment": "True for the single published version new sessions run on; live versions are immutable, the highest non-live version is the draft",
        },
    )
    inserted_at: datetime = Field(
        default_factory=now,
        nullable=False,
        sa_column_kwargs={"comment": "Timestamp when the version was created"},
    )
    updated_at: datetime = Field(
        default_factory=now,
        nullable=False,
        sa_column_kwargs={
            "comment": "Timestamp when the draft was last saved or the version was published/unpublished",
            "onupdate": now,
        },
    )


class ChatbotFlowUpdate(SQLModel):
    nodes: list[ChatbotFlowNode] = Field(
        description="React Flow nodes; only their shape is checked on save."
    )
    edges: list[ChatbotFlowEdge] = Field(
        default_factory=list, description="React Flow edges between `nodes`."
    )


class ChatbotFlowVersionPublic(SQLModel):
    id: UUID
    chatbot_id: UUID
    version: int
    is_live: bool
    nodes: list[ChatbotFlowNode]
    edges: list[ChatbotFlowEdge]
    inserted_at: datetime
    updated_at: datetime


class ChatbotSession(SQLModel, table=True):
    __tablename__ = "chatbot_session"
    __table_args__ = (
        # Also serves FK lookups on chatbot_id via its prefix, so chatbot_id has no own index.
        Index("ix_chatbot_session_chatbot_contact", "chatbot_id", "contact_ref"),
        Index(
            ACTIVE_CHATBOT_SESSION_UNIQUE_INDEX,
            "chatbot_id",
            "contact_ref",
            unique=True,
            postgresql_where=text("status = 'active'"),
        ),
    )

    id: UUID = Field(
        default_factory=uuid4,
        primary_key=True,
        sa_column_kwargs={"comment": "Session identifier returned to the caller"},
    )
    chatbot_id: UUID = Field(
        foreign_key="chatbot.id",
        nullable=False,
        ondelete="RESTRICT",
        sa_column_kwargs={
            "comment": "Chatbot this session runs; tenant scope comes from the chatbot"
        },
    )
    flow_version_id: UUID = Field(
        foreign_key="chatbot_flow_version.id",
        nullable=False,
        index=True,
        ondelete="RESTRICT",
        sa_column_kwargs={
            "comment": "Live flow version pinned at creation, so publishing never changes a running conversation"
        },
    )
    contact_ref: str = Field(
        max_length=CONTACT_REF_MAX_LENGTH,
        index=True,
        sa_column_kwargs={
            "comment": "Caller-side identifier of the end user (e.g. a Glific contact id); opaque to Kaapi"
        },
    )
    active_node_id: str = Field(
        max_length=FLOW_NODE_ID_MAX_LENGTH,
        nullable=False,
        sa_column_kwargs={
            "comment": "React Flow node id inside the pinned flow version that handles the next turn"
        },
    )
    status: ChatbotSessionStatusEnum = Field(
        default=ChatbotSessionStatusEnum.ACTIVE,
        sa_column=Column(
            _SESSION_STATUS_PG_ENUM,
            nullable=False,
            comment="Session in use: active, or inactive once superseded by a newer session for the same chatbot + contact_ref or the chatbot was deleted",
        ),
    )
    completion_status: ChatbotCompletionStatusEnum = Field(
        default=ChatbotCompletionStatusEnum.IN_PROGRESS,
        sa_column=Column(
            _COMPLETION_STATUS_PG_ENUM,
            nullable=False,
            server_default=ChatbotCompletionStatusEnum.IN_PROGRESS.value,
            comment="Field collection progress: in_progress, complete once every required field is gathered, or abandoned when deactivated while in_progress",
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
            comment="Speaker: user or assistant (system is reserved; the prompt lives on the flow node)",
        ),
    )
    content: str = Field(
        sa_column_kwargs={"comment": "Message text"},
    )
    node_id: str | None = Field(
        default=None,
        max_length=FLOW_NODE_ID_MAX_LENGTH,
        nullable=True,
        sa_column_kwargs={
            "comment": "React Flow node id of the pinned flow that was active for this turn"
        },
    )
    inserted_at: datetime = Field(
        default_factory=now,
        nullable=False,
        sa_column_kwargs={
            "comment": "Timestamp when the message was stored; defines transcript order"
        },
    )


class CreateChatbotSessionRequest(SQLModel):
    contact_ref: str = Field(
        min_length=1,
        max_length=CONTACT_REF_MAX_LENGTH,
        description="Caller-side identifier of the end user this conversation is with.",
    )


class CreateChatbotSessionResponse(SQLModel):
    session_id: UUID
    chatbot_id: UUID
    flow_version: int
    active_node_id: str


class ChatbotSessionSummary(SQLModel):
    session_id: UUID
    chatbot_id: UUID
    chatbot_name: str
    flow_version: int
    active_node_id: str
    prompt: str | None
    contact_ref: str
    status: ChatbotSessionStatusEnum
    completion_status: ChatbotCompletionStatusEnum
    collected_fields: dict[str, Any]
    inserted_at: datetime
    updated_at: datetime


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
