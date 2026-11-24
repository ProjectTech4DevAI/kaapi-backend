"""add webhook_log table

Revision ID: 090
Revises: 089
Create Date: 2026-11-24 00:00:00.000000

One row per send_callback call so callback deliveries can be audited and
debugged per resource. Project/org FKs are nullable because some callers send
callbacks without tenant attribution.
"""

import sqlalchemy as sa
import sqlmodel.sql.sqltypes
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "090"
down_revision = "089"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "webhook_log",
        sa.Column("id", sa.Uuid(), nullable=False, comment="Unique identifier"),
        sa.Column(
            "project_id",
            sa.Integer(),
            nullable=True,
            comment="Project whose job triggered the callback; null for unattributed callers",
        ),
        sa.Column(
            "organization_id",
            sa.Integer(),
            nullable=True,
            comment="Organization whose job triggered the callback; null for unattributed callers",
        ),
        sa.Column(
            "resource_type",
            sa.Enum(
                "LLM_CALL",
                "LLM_CHAIN",
                "GUARDRAILS",
                "RESPONSE",
                "PROMPT_IMPROVEMENT",
                "DOC_TRANSFORM",
                "COLLECTION",
                "EVALUATION",
                "EVALUATION_ITERATION",
                "ASSESSMENT",
                "DUPLICATE_DETECTION",
                "CHATBOT_SESSION",
                name="webhookresourcetypeenum",
            ),
            nullable=True,
            comment="Kind of resource the callback reports on: LLM_CALL, LLM_CHAIN, GUARDRAILS, RESPONSE, PROMPT_IMPROVEMENT, DOC_TRANSFORM, COLLECTION, EVALUATION, EVALUATION_ITERATION, ASSESSMENT, DUPLICATE_DETECTION, CHATBOT_SESSION",
        ),
        sa.Column(
            "resource_id",
            sqlmodel.sql.sqltypes.AutoString(),
            nullable=True,
            comment="Id of the resource as a string; job UUID, eval int id, assessment or session UUID depending on resource_type",
        ),
        sa.Column(
            "url",
            sqlmodel.sql.sqltypes.AutoString(),
            nullable=False,
            comment="Callback URL the payload was sent to",
        ),
        sa.Column(
            "status",
            sa.Enum(
                "DELIVERED",
                "FAILED",
                "REJECTED",
                name="webhookdeliverystatusenum",
            ),
            nullable=False,
            comment="Outcome: DELIVERED (2xx), FAILED (retries exhausted), or REJECTED (URL failed SSRF validation, no request sent)",
        ),
        sa.Column(
            "status_code",
            sa.Integer(),
            nullable=True,
            comment="HTTP status of the last attempt; null on connection errors or rejection",
        ),
        sa.Column(
            "attempt_count",
            sa.Integer(),
            nullable=False,
            comment="HTTP attempts made; 0 when rejected",
        ),
        sa.Column(
            "duration_ms",
            sa.Float(),
            nullable=True,
            comment="Total wall time in milliseconds across all attempts including retry backoff",
        ),
        sa.Column(
            "request_body",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            comment="JSON payload exactly as sent to the receiver",
        ),
        sa.Column(
            "request_bytes",
            sa.Integer(),
            nullable=False,
            comment="Size of the serialized request body in bytes",
        ),
        sa.Column(
            "response_body",
            sqlmodel.sql.sqltypes.AutoString(),
            nullable=True,
            comment="Receiver's response text from the last attempt, truncated",
        ),
        sa.Column(
            "error",
            sqlmodel.sql.sqltypes.AutoString(),
            nullable=True,
            comment="Failure or rejection reason; null when delivered",
        ),
        sa.Column(
            "signed",
            sa.Boolean(),
            nullable=False,
            comment="Whether the request carried HMAC X-Webhook-Signature headers",
        ),
        sa.Column(
            "trace_id",
            sqlmodel.sql.sqltypes.AutoString(),
            nullable=True,
            comment="Request correlation id (X-Request-ID) active when the callback was sent",
        ),
        sa.Column(
            "inserted_at",
            sa.DateTime(),
            nullable=False,
            comment="Timestamp when the delivery finished",
        ),
        sa.ForeignKeyConstraint(
            ["organization_id"], ["organization.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(["project_id"], ["project.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_webhook_log_organization_id"),
        "webhook_log",
        ["organization_id"],
        unique=False,
    )
    op.create_index(
        "ix_webhook_log_project_id_inserted_at",
        "webhook_log",
        ["project_id", "inserted_at"],
        unique=False,
    )
    op.create_index(
        "ix_webhook_log_resource",
        "webhook_log",
        ["resource_type", "resource_id"],
        unique=False,
    )


def downgrade():
    op.drop_table("webhook_log")
    sa.Enum(name="webhookdeliverystatusenum").drop(op.get_bind())
    sa.Enum(name="webhookresourcetypeenum").drop(op.get_bind())
