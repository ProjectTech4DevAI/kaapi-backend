from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import Column, Index
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, SQLModel

from app.core.util import now


class WebhookResourceTypeEnum(StrEnum):
    LLM_CALL = "LLM_CALL"
    LLM_CHAIN = "LLM_CHAIN"
    GUARDRAILS = "GUARDRAILS"
    RESPONSE = "RESPONSE"
    PROMPT_IMPROVEMENT = "PROMPT_IMPROVEMENT"
    DOC_TRANSFORM = "DOC_TRANSFORM"
    COLLECTION = "COLLECTION"
    EVALUATION = "EVALUATION"
    EVALUATION_ITERATION = "EVALUATION_ITERATION"
    ASSESSMENT = "ASSESSMENT"
    DUPLICATE_DETECTION = "DUPLICATE_DETECTION"
    CHATBOT_SESSION = "CHATBOT_SESSION"


class WebhookDeliveryStatusEnum(StrEnum):
    DELIVERED = "DELIVERED"
    FAILED = "FAILED"
    REJECTED = "REJECTED"


class WebhookLog(SQLModel, table=True):
    __tablename__ = "webhook_log"
    __table_args__ = (
        Index("ix_webhook_log_resource", "resource_type", "resource_id"),
        # Serves the project-scoped list (newest first) and the project FK cascade.
        Index("ix_webhook_log_project_id_inserted_at", "project_id", "inserted_at"),
    )

    id: UUID = Field(
        default_factory=uuid4,
        primary_key=True,
        sa_column_kwargs={"comment": "Unique identifier"},
    )
    project_id: int | None = Field(
        default=None,
        foreign_key="project.id",
        nullable=True,
        ondelete="CASCADE",
        sa_column_kwargs={
            "comment": "Project whose job triggered the callback; null for unattributed callers"
        },
    )
    organization_id: int | None = Field(
        default=None,
        foreign_key="organization.id",
        nullable=True,
        index=True,
        ondelete="CASCADE",
        sa_column_kwargs={
            "comment": "Organization whose job triggered the callback; null for unattributed callers"
        },
    )
    resource_type: WebhookResourceTypeEnum | None = Field(
        default=None,
        nullable=True,
        sa_column_kwargs={
            "comment": "Kind of resource the callback reports on: LLM_CALL, LLM_CHAIN, GUARDRAILS, RESPONSE, PROMPT_IMPROVEMENT, DOC_TRANSFORM, COLLECTION, EVALUATION, EVALUATION_ITERATION, ASSESSMENT, DUPLICATE_DETECTION, CHATBOT_SESSION",
        },
    )
    resource_id: str | None = Field(
        default=None,
        nullable=True,
        sa_column_kwargs={
            "comment": "Id of the resource as a string; job UUID, eval int id, assessment or session UUID depending on resource_type"
        },
    )
    url: str = Field(
        nullable=False,
        sa_column_kwargs={"comment": "Callback URL the payload was sent to"},
    )
    status: WebhookDeliveryStatusEnum = Field(
        nullable=False,
        sa_column_kwargs={
            "comment": "Outcome: DELIVERED (2xx), FAILED (retries exhausted), or REJECTED (URL failed SSRF validation, no request sent)",
        },
    )
    status_code: int | None = Field(
        default=None,
        nullable=True,
        sa_column_kwargs={
            "comment": "HTTP status of the last attempt; null on connection errors or rejection"
        },
    )
    attempt_count: int = Field(
        nullable=False,
        sa_column_kwargs={"comment": "HTTP attempts made; 0 when rejected"},
    )
    duration_ms: float | None = Field(
        default=None,
        nullable=True,
        sa_column_kwargs={
            "comment": "Total wall time in milliseconds across all attempts including retry backoff"
        },
    )
    request_body: dict[str, Any] = Field(
        sa_column=Column(
            JSONB,
            nullable=False,
            comment="JSON payload exactly as sent to the receiver",
        ),
    )
    request_bytes: int = Field(
        nullable=False,
        sa_column_kwargs={"comment": "Size of the serialized request body in bytes"},
    )
    response_body: str | None = Field(
        default=None,
        nullable=True,
        sa_column_kwargs={
            "comment": "Receiver's response text from the last attempt, truncated"
        },
    )
    error: str | None = Field(
        default=None,
        nullable=True,
        sa_column_kwargs={
            "comment": "Failure or rejection reason; null when delivered"
        },
    )
    signed: bool = Field(
        nullable=False,
        sa_column_kwargs={
            "comment": "Whether the request carried HMAC X-Webhook-Signature headers"
        },
    )
    trace_id: str | None = Field(
        default=None,
        nullable=True,
        sa_column_kwargs={
            "comment": "Request correlation id (X-Request-ID) active when the callback was sent"
        },
    )
    inserted_at: datetime = Field(
        default_factory=now,
        nullable=False,
        sa_column_kwargs={"comment": "Timestamp when the delivery finished"},
    )


class WebhookLogPublic(SQLModel):
    id: UUID
    project_id: int | None
    organization_id: int | None
    resource_type: WebhookResourceTypeEnum | None
    resource_id: str | None
    url: str
    status: WebhookDeliveryStatusEnum
    status_code: int | None
    attempt_count: int
    duration_ms: float | None
    request_body: dict[str, Any]
    request_bytes: int
    response_body: str | None
    error: str | None
    signed: bool
    trace_id: str | None
    inserted_at: datetime
