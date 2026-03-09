"""add tts evaluation tables

Revision ID: 049
Revises: 048
Create Date: 2026-02-14 12:00:00.000000

"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "049"
down_revision = "048"
branch_labels = None
depends_on = None


def upgrade():
    # Create tts_result table
    op.create_table(
        "tts_result",
        sa.Column(
            "id",
            sa.Integer(),
            nullable=False,
            comment="Unique identifier for the TTS result",
        ),
        sa.Column(
            "sample_text",
            sa.Text(),
            nullable=False,
            comment="Input text that will be synthesized to speech",
        ),
        sa.Column(
            "object_store_url",
            sa.String(),
            nullable=True,
            comment="S3 URL of the generated WAV audio file",
        ),
        sa.Column(
            "metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
            comment="Audio metadata: {duration_seconds, size_bytes}",
        ),
        sa.Column(
            "provider",
            sa.String(length=100),
            nullable=False,
            comment="TTS provider used (e.g., gemini-2.5-pro-preview-tts)",
        ),
        sa.Column(
            "status",
            sa.String(length=20),
            nullable=False,
            server_default="PENDING",
            comment="Result status: PENDING, SUCCESS, FAILED",
        ),
        sa.Column(
            "score",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
            comment="Extensible evaluation metrics",
        ),
        sa.Column(
            "is_correct",
            sa.Boolean(),
            nullable=True,
            comment="Human feedback flag on audio quality correctness",
        ),
        sa.Column(
            "comment",
            sa.Text(),
            nullable=True,
            comment="Human feedback comment on audio quality",
        ),
        sa.Column(
            "error_message",
            sa.Text(),
            nullable=True,
            comment="Error message if synthesis failed",
        ),
        sa.Column(
            "evaluation_run_id",
            sa.Integer(),
            nullable=False,
            comment="Reference to the evaluation run",
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
            comment="Timestamp when the result was created",
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(),
            nullable=False,
            comment="Timestamp when the result was last updated",
        ),
        sa.ForeignKeyConstraint(
            ["evaluation_run_id"],
            ["evaluation_run.id"],
            name="fk_tts_result_run_id",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["organization_id"],
            ["organization.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["project_id"],
            ["project.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_tts_result_run_id",
        "tts_result",
        ["evaluation_run_id"],
        unique=False,
    )
    op.create_index(
        "idx_tts_result_feedback",
        "tts_result",
        ["evaluation_run_id", "is_correct"],
        unique=False,
    )
    op.create_index(
        "idx_tts_result_status",
        "tts_result",
        ["evaluation_run_id", "status"],
        unique=False,
    )


def downgrade():
    op.drop_index("idx_tts_result_status", table_name="tts_result")
    op.drop_index("idx_tts_result_feedback", table_name="tts_result")
    op.drop_index("ix_tts_result_run_id", table_name="tts_result")
    op.drop_table("tts_result")
