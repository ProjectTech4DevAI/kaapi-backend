"""add chatbot session and message tables

Revision ID: 088
Revises: 087
Create Date: 2026-10-20 00:00:00.000000
"""

import sqlalchemy as sa
import sqlmodel.sql.sqltypes
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "088"
down_revision = "087"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "chatbot_session",
        sa.Column(
            "id",
            sa.Uuid(),
            nullable=False,
            comment="Session identifier returned to the caller",
        ),
        sa.Column(
            "contact_ref",
            sqlmodel.sql.sqltypes.AutoString(length=500),
            nullable=False,
            comment="Caller-side identifier of the end user (e.g. a contact id); opaque to Kaapi",
        ),
        sa.Column(
            "config_id",
            sa.Uuid(),
            nullable=False,
            comment="Saved Anthropic text config that supplies the chat model",
        ),
        sa.Column(
            "config_version",
            sa.Integer(),
            nullable=True,
            comment="Pinned config version; NULL resolves to the latest version on every turn",
        ),
        sa.Column(
            "system_prompt",
            sqlmodel.sql.sqltypes.AutoString(),
            nullable=False,
            comment="Admin-authored flow prompt captured at session creation",
        ),
        sa.Column(
            "status",
            sa.Enum(
                "active",
                "inactive",
                name="chatbotsessionstatusenum",
            ),
            nullable=False,
            comment="Conversation lifecycle: active or inactive",
        ),
        sa.Column(
            "collected_fields",
            postgresql.JSON(astext_type=sa.Text()),
            nullable=False,
            comment="Flat map of field name -> value gathered from the user so far",
        ),
        sa.Column(
            "organization_id",
            sa.Integer(),
            nullable=False,
            comment="Reference to the organization",
        ),
        sa.Column(
            "project_id",
            sa.Integer(),
            nullable=False,
            comment="Reference to the project",
        ),
        sa.Column(
            "inserted_at",
            sa.DateTime(),
            nullable=False,
            comment="Timestamp when the session was created",
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(),
            nullable=False,
            comment="Timestamp when the session was last updated",
        ),
        sa.ForeignKeyConstraint(["config_id"], ["config.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(
            ["organization_id"], ["organization.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(["project_id"], ["project.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_chatbot_session_config_id"),
        "chatbot_session",
        ["config_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_chatbot_session_organization_id"),
        "chatbot_session",
        ["organization_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_chatbot_session_project_id"),
        "chatbot_session",
        ["project_id"],
        unique=False,
    )

    op.create_table(
        "chatbot_session_message",
        sa.Column("id", sa.Uuid(), nullable=False, comment="Unique identifier"),
        sa.Column(
            "session_id",
            sa.Uuid(),
            nullable=False,
            comment="Reference to the owning chatbot session",
        ),
        sa.Column(
            "role",
            sa.Enum("system", "user", "assistant", name="chatbotmessageroleenum"),
            nullable=False,
            comment="Speaker: user or assistant (system is reserved; the prompt lives on the session)",
        ),
        sa.Column(
            "content",
            sqlmodel.sql.sqltypes.AutoString(),
            nullable=False,
            comment="Message text",
        ),
        sa.Column(
            "inserted_at",
            sa.DateTime(),
            nullable=False,
            comment="Timestamp when the message was stored; defines transcript order",
        ),
        sa.ForeignKeyConstraint(
            ["session_id"], ["chatbot_session.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_chatbot_session_message_session_id"),
        "chatbot_session_message",
        ["session_id"],
        unique=False,
    )


def downgrade():
    raise NotImplementedError(
        "not reversible: dropping chatbot_session tables discards stored conversations"
    )
