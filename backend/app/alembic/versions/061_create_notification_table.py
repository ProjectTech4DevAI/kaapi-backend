"""Create generic notification table

Revision ID: 061
Revises: 060
Create Date: 2026-05-14 12:00:00.000000

Adds a generic `notification` table used to record outbound notifications
across providers (email, push, sms, webhook) and entity types (eval_run,
project, ...). Each row captures a single delivery attempt to a single
recipient with a JSON snapshot of the template payload, the delivery
status, and any failure reason. Existence of rows for
(entity_type, entity_id, notification_type) acts as the idempotency
guard preventing duplicate sends for the same event.
"""

import sqlalchemy as sa
import sqlmodel.sql.sqltypes
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "061"
down_revision = "060"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "notification",
        sa.Column(
            "id",
            sa.Integer(),
            nullable=False,
            comment="Unique identifier for the notification record",
        ),
        sa.Column(
            "notification_type",
            sa.String(length=64),
            nullable=False,
            comment=(
                "Event triggering the notification, e.g. eval_completed, "
                "eval_failed, magic_link_login"
            ),
        ),
        sa.Column(
            "provider",
            sa.String(length=32),
            nullable=False,
            comment="Delivery channel: email, push, sms, webhook",
        ),
        sa.Column(
            "recipient_user_id",
            sa.Integer(),
            nullable=False,
            comment="Reference to the user receiving the notification",
        ),
        sa.Column(
            "entity_type",
            sa.String(length=64),
            nullable=False,
            comment="Polymorphic entity type, e.g. eval_run, project",
        ),
        sa.Column(
            "entity_id",
            sa.Integer(),
            nullable=False,
            comment="Polymorphic entity id (not a hard FK across types)",
        ),
        sa.Column(
            "project_id",
            sa.Integer(),
            nullable=True,
            comment=(
                "Reference to the project the notification belongs to. "
                "Null for project-agnostic notifications (e.g. magic_link_login)."
            ),
        ),
        sa.Column(
            "subject",
            sqlmodel.sql.sqltypes.AutoString(),
            nullable=True,
            comment="Subject line (for email-like providers)",
        ),
        sa.Column(
            "body_template",
            sa.String(length=128),
            nullable=True,
            comment="Template identifier used to render the body, e.g. eval_completion_v1",
        ),
        sa.Column(
            "payload",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
            comment=(
                "Snapshot of dynamic values used to render the notification "
                "(eval name, status, timestamp, result link, etc.)"
            ),
        ),
        sa.Column(
            "status",
            sa.String(length=16),
            nullable=False,
            server_default="pending",
            comment="Delivery status: pending, sent, failed, skipped",
        ),
        sa.Column(
            "sent_at",
            sa.DateTime(),
            nullable=True,
            comment="Timestamp when delivery succeeded",
        ),
        sa.Column(
            "failed_reason",
            sa.Text(),
            nullable=True,
            comment="Error message if delivery failed",
        ),
        sa.Column(
            "inserted_at",
            sa.DateTime(),
            nullable=False,
            comment="Timestamp when the record was created",
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(),
            nullable=False,
            comment="Timestamp when the record was last updated",
        ),
        sa.ForeignKeyConstraint(
            ["recipient_user_id"],
            ["user.id"],
            name="fk_notification_recipient_user_id",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["project_id"],
            ["project.id"],
            name="fk_notification_project_id",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "idx_notification_entity",
        "notification",
        ["entity_type", "entity_id", "notification_type"],
        unique=False,
    )
    op.create_index(
        "idx_notification_recipient",
        "notification",
        ["recipient_user_id"],
        unique=False,
    )
    op.create_index(
        "idx_notification_status",
        "notification",
        ["status"],
        unique=False,
    )


def downgrade():
    op.drop_index("idx_notification_status", table_name="notification")
    op.drop_index("idx_notification_recipient", table_name="notification")
    op.drop_index("idx_notification_entity", table_name="notification")
    op.drop_table("notification")
