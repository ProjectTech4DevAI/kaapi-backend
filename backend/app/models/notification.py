from datetime import datetime
from enum import Enum
from typing import Any

from sqlalchemy import Column, Index
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, SQLModel

from app.core.util import now


class NotificationStatus(str, Enum):
    PENDING = "pending"
    SENT = "sent"
    FAILED = "failed"
    SKIPPED = "skipped"


class NotificationProvider(str, Enum):
    EMAIL = "email"
    PUSH = "push"
    SMS = "sms"
    WEBHOOK = "webhook"


class NotificationType(str, Enum):
    EVAL_COMPLETED = "eval_completed"
    EVAL_FAILED = "eval_failed"
    MAGIC_LINK_LOGIN = "magic_link_login"


class NotificationEntityType(str, Enum):
    EVAL_RUN = "eval_run"
    PROJECT = "project"
    USER = "user"


class Notification(SQLModel, table=True):
    """Generic outbound notification record.

    One row per (recipient, channel, event). Existence of rows for a given
    (entity_type, entity_id, notification_type) is used as the idempotency
    guard preventing duplicate sends for the same event.
    """

    __tablename__ = "notification"
    __table_args__ = (
        Index(
            "idx_notification_entity",
            "entity_type",
            "entity_id",
            "notification_type",
        ),
        Index("idx_notification_recipient", "recipient_user_id"),
        Index("idx_notification_status", "status"),
    )

    id: int = Field(
        default=None,
        primary_key=True,
        sa_column_kwargs={"comment": "Unique identifier for the notification record"},
    )
    notification_type: str = Field(
        max_length=64,
        nullable=False,
        sa_column_kwargs={
            "comment": (
                "Event triggering the notification, e.g. eval_completed, "
                "eval_failed, magic_link_login"
            )
        },
    )
    provider: str = Field(
        max_length=32,
        nullable=False,
        sa_column_kwargs={"comment": "Delivery channel: email, push, sms, webhook"},
    )
    recipient_user_id: int = Field(
        foreign_key="user.id",
        nullable=False,
        ondelete="CASCADE",
        sa_column_kwargs={
            "comment": "Reference to the user receiving the notification"
        },
    )
    entity_type: str = Field(
        max_length=64,
        nullable=False,
        sa_column_kwargs={"comment": "Polymorphic entity type, e.g. eval_run, project"},
    )
    entity_id: int = Field(
        nullable=False,
        sa_column_kwargs={
            "comment": "Polymorphic entity id (not a hard FK across types)"
        },
    )
    project_id: int | None = Field(
        default=None,
        foreign_key="project.id",
        nullable=True,
        ondelete="CASCADE",
        sa_column_kwargs={
            "comment": (
                "Reference to the project the notification belongs to. "
                "Null for project-agnostic notifications (e.g. magic_link_login)."
            )
        },
    )
    subject: str | None = Field(
        default=None,
        nullable=True,
        sa_column_kwargs={"comment": "Subject line (for email-like providers)"},
    )
    body_template: str | None = Field(
        default=None,
        max_length=128,
        nullable=True,
        sa_column_kwargs={
            "comment": (
                "Template identifier used to render the body, "
                "e.g. eval_completion_v1"
            )
        },
    )
    payload: dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(
            JSONB,
            nullable=False,
            comment=(
                "Snapshot of dynamic values used to render the notification "
                "(eval name, status, timestamp, result link, etc.)"
            ),
        ),
    )
    status: str = Field(
        default=NotificationStatus.PENDING.value,
        max_length=16,
        nullable=False,
        sa_column_kwargs={"comment": "Delivery status: pending, sent, failed, skipped"},
    )
    sent_at: datetime | None = Field(
        default=None,
        nullable=True,
        sa_column_kwargs={"comment": "Timestamp when delivery succeeded"},
    )
    failed_reason: str | None = Field(
        default=None,
        nullable=True,
        sa_column_kwargs={"comment": "Error message if delivery failed"},
    )
    inserted_at: datetime = Field(
        default_factory=now,
        nullable=False,
        sa_column_kwargs={"comment": "Timestamp when the record was created"},
    )
    updated_at: datetime = Field(
        default_factory=now,
        nullable=False,
        sa_column_kwargs={"comment": "Timestamp when the record was last updated"},
    )
