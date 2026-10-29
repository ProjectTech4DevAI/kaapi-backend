"""add chatbot, flow version, session and message tables

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
        "chatbot",
        sa.Column(
            "id",
            sa.Uuid(),
            nullable=False,
            comment="Chatbot identifier used in API paths",
        ),
        sa.Column(
            "name",
            sqlmodel.sql.sqltypes.AutoString(length=255),
            nullable=False,
            comment="Display name of the chatbot shown in the UI",
        ),
        sa.Column(
            "organization_id",
            sa.Integer(),
            nullable=False,
            comment="Organization that owns the chatbot",
        ),
        sa.Column(
            "project_id",
            sa.Integer(),
            nullable=False,
            comment="Project that owns the chatbot",
        ),
        sa.Column(
            "inserted_at",
            sa.DateTime(),
            nullable=False,
            comment="Timestamp when the chatbot was created",
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(),
            nullable=False,
            comment="Timestamp when the chatbot was last updated",
        ),
        sa.Column(
            "deleted_at",
            sa.DateTime(),
            nullable=True,
            comment="Soft-delete timestamp; the row is kept because sessions reference it",
        ),
        sa.ForeignKeyConstraint(
            ["organization_id"], ["organization.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(["project_id"], ["project.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_chatbot_organization_id"),
        "chatbot",
        ["organization_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_chatbot_project_id"), "chatbot", ["project_id"], unique=False
    )

    op.create_table(
        "chatbot_flow_version",
        sa.Column(
            "id",
            sa.Uuid(),
            nullable=False,
            comment="Flow version identifier pinned by sessions",
        ),
        sa.Column(
            "chatbot_id",
            sa.Uuid(),
            nullable=False,
            comment="Chatbot this flow version belongs to",
        ),
        sa.Column(
            "version",
            sa.Integer(),
            nullable=False,
            comment="Sequential version number per chatbot, starting at 1",
        ),
        sa.Column(
            "nodes",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            comment="React Flow nodes as saved by the UI: [{id, type (input/output/coordinator/onboarding/rag), position {x, y}, data}]; agent node data carries config_id, config_version, prompt",
        ),
        sa.Column(
            "edges",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
            comment="React Flow edges as saved by the UI: [{id, source, target, sourceHandle, targetHandle, data}]",
        ),
        sa.Column(
            "is_live",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
            comment="True for the single published version new sessions run on; live versions are immutable, the highest non-live version is the draft",
        ),
        sa.Column(
            "inserted_at",
            sa.DateTime(),
            nullable=False,
            comment="Timestamp when the version was created",
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(),
            nullable=False,
            comment="Timestamp when the draft was last saved or the version was published/unpublished",
        ),
        sa.ForeignKeyConstraint(["chatbot_id"], ["chatbot.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "chatbot_id",
            "version",
            name="uq_chatbot_flow_version_chatbot_version",
        ),
    )
    op.create_index(
        op.f("ix_chatbot_flow_version_chatbot_id"),
        "chatbot_flow_version",
        ["chatbot_id"],
        unique=False,
    )
    op.create_index(
        "uq_chatbot_flow_version_live_chatbot",
        "chatbot_flow_version",
        ["chatbot_id"],
        unique=True,
        postgresql_where=sa.text("is_live"),
    )

    op.create_table(
        "chatbot_session",
        sa.Column(
            "id",
            sa.Uuid(),
            nullable=False,
            comment="Session identifier returned to the caller",
        ),
        sa.Column(
            "chatbot_id",
            sa.Uuid(),
            nullable=False,
            comment="Chatbot this session runs; tenant scope comes from the chatbot",
        ),
        sa.Column(
            "flow_version_id",
            sa.Uuid(),
            nullable=False,
            comment="Live flow version pinned at creation, so publishing never changes a running conversation",
        ),
        sa.Column(
            "contact_ref",
            sqlmodel.sql.sqltypes.AutoString(length=500),
            nullable=False,
            comment="Caller-side identifier of the end user (e.g. a Glific contact id); opaque to Kaapi",
        ),
        sa.Column(
            "active_node_id",
            sqlmodel.sql.sqltypes.AutoString(length=255),
            nullable=False,
            comment="React Flow node id inside the pinned flow version that handles the next turn",
        ),
        sa.Column(
            "status",
            sa.Enum(
                "active",
                "inactive",
                name="chatbotsessionstatusenum",
            ),
            nullable=False,
            comment="Session in use: active, or inactive once superseded by a newer session for the same chatbot + contact_ref or the chatbot was deleted",
        ),
        sa.Column(
            "completion_status",
            sa.Enum(
                "in_progress",
                "complete",
                "abandoned",
                name="chatbotcompletionstatusenum",
            ),
            nullable=False,
            server_default="in_progress",
            comment="Field collection progress: in_progress, complete once every required field is gathered, or abandoned when deactivated while in_progress",
        ),
        sa.Column(
            "collected_fields",
            postgresql.JSON(astext_type=sa.Text()),
            nullable=False,
            comment="Flat map of field name -> value gathered from the user so far",
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
        sa.ForeignKeyConstraint(["chatbot_id"], ["chatbot.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(
            ["flow_version_id"], ["chatbot_flow_version.id"], ondelete="RESTRICT"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_chatbot_session_flow_version_id"),
        "chatbot_session",
        ["flow_version_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_chatbot_session_contact_ref"),
        "chatbot_session",
        ["contact_ref"],
        unique=False,
    )
    op.create_index(
        "ix_chatbot_session_chatbot_contact",
        "chatbot_session",
        ["chatbot_id", "contact_ref"],
        unique=False,
    )
    op.create_index(
        "uq_chatbot_session_active_chatbot_contact",
        "chatbot_session",
        ["chatbot_id", "contact_ref"],
        unique=True,
        postgresql_where=sa.text("status = 'active'"),
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
            comment="Speaker: user or assistant (system is reserved; the prompt lives on the flow node)",
        ),
        sa.Column(
            "content",
            sqlmodel.sql.sqltypes.AutoString(),
            nullable=False,
            comment="Message text",
        ),
        sa.Column(
            "node_id",
            sqlmodel.sql.sqltypes.AutoString(length=255),
            nullable=True,
            comment="React Flow node id of the pinned flow that was active for this turn",
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
        "not reversible: dropping chatbot tables discards chatbot flows and stored conversations"
    )
