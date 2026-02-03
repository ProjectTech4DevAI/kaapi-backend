"""add stt evaluation tables

Revision ID: 043
Revises: 042
Create Date: 2026-01-28 12:00:00.000000

"""

import sqlalchemy as sa
import sqlmodel.sql.sqltypes
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "043"
down_revision = "042"
branch_labels = None
depends_on = None


def upgrade():
    # Add type and language columns to evaluation_dataset table
    op.add_column(
        "evaluation_dataset",
        sa.Column(
            "type",
            sa.String(length=20),
            nullable=False,
            server_default="text",
            comment="Evaluation type: text, stt, or tts",
        ),
    )
    op.add_column(
        "evaluation_dataset",
        sa.Column(
            "language",
            sa.String(length=10),
            nullable=True,
            comment="ISO 639-1 language code (e.g., en, hi)",
        ),
    )

    # Add type, language, providers, and processed_samples columns to evaluation_run table
    op.add_column(
        "evaluation_run",
        sa.Column(
            "type",
            sa.String(length=20),
            nullable=False,
            server_default="text",
            comment="Evaluation type: text, stt, or tts",
        ),
    )
    op.add_column(
        "evaluation_run",
        sa.Column(
            "language",
            sa.String(length=10),
            nullable=True,
            comment="ISO 639-1 language code",
        ),
    )
    op.add_column(
        "evaluation_run",
        sa.Column(
            "providers",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
            comment="List of STT/TTS providers used (e.g., ['gemini-2.5-pro'])",
        ),
    )
    op.add_column(
        "evaluation_run",
        sa.Column(
            "processed_samples",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
            comment="Number of samples processed so far",
        ),
    )

    # Create stt_sample table
    op.create_table(
        "stt_sample",
        sa.Column(
            "id",
            sa.Integer(),
            nullable=False,
            comment="Unique identifier for the STT sample",
        ),
        sa.Column(
            "object_store_url",
            sqlmodel.sql.sqltypes.AutoString(),
            nullable=False,
            comment="S3 URL of the audio file",
        ),
        sa.Column(
            "language",
            sa.String(length=10),
            nullable=True,
            comment="ISO 639-1 language code for this sample",
        ),
        sa.Column(
            "ground_truth",
            sa.Text(),
            nullable=True,
            comment="Reference transcription for comparison (optional)",
        ),
        sa.Column(
            "sample_metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
            server_default=sa.text("'{}'::jsonb"),
            comment="Additional metadata (format, bitrate, original filename, etc.)",
        ),
        sa.Column(
            "dataset_id",
            sa.Integer(),
            nullable=False,
            comment="Reference to the parent evaluation dataset",
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
            comment="Timestamp when the sample was created",
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(),
            nullable=False,
            comment="Timestamp when the sample was last updated",
        ),
        sa.ForeignKeyConstraint(
            ["dataset_id"],
            ["evaluation_dataset.id"],
            name="fk_stt_sample_dataset_id",
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
        "ix_stt_sample_dataset_id",
        "stt_sample",
        ["dataset_id"],
        unique=False,
    )
    op.create_index(
        "idx_stt_sample_org_project",
        "stt_sample",
        ["organization_id", "project_id"],
        unique=False,
    )

    # Create stt_result table
    op.create_table(
        "stt_result",
        sa.Column(
            "id",
            sa.Integer(),
            nullable=False,
            comment="Unique identifier for the STT result",
        ),
        sa.Column(
            "transcription",
            sa.Text(),
            nullable=True,
            comment="Generated transcription from STT provider",
        ),
        sa.Column(
            "provider",
            sa.String(length=50),
            nullable=False,
            comment="STT provider used (e.g., gemini-2.5-pro)",
        ),
        sa.Column(
            "status",
            sa.String(length=20),
            nullable=False,
            server_default="pending",
            comment="Result status: pending, completed, failed",
        ),
        sa.Column(
            "score",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
            comment="Evaluation metrics (e.g., wer, cer, mer, wil) - extensible for future metrics",
        ),
        sa.Column(
            "is_correct",
            sa.Boolean(),
            nullable=True,
            comment="Human feedback: transcription correctness (null=not reviewed)",
        ),
        sa.Column(
            "comment",
            sa.Text(),
            nullable=True,
            comment="Human feedback comment",
        ),
        sa.Column(
            "provider_metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
            server_default=sa.text("'{}'::jsonb"),
            comment="Provider-specific response metadata (tokens, latency, etc.)",
        ),
        sa.Column(
            "error_message",
            sa.Text(),
            nullable=True,
            comment="Error message if transcription failed",
        ),
        sa.Column(
            "stt_sample_id",
            sa.Integer(),
            nullable=False,
            comment="Reference to the STT sample",
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
            ["stt_sample_id"],
            ["stt_sample.id"],
            name="fk_stt_result_sample_id",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["evaluation_run_id"],
            ["evaluation_run.id"],
            name="fk_stt_result_run_id",
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
        "ix_stt_result_sample_id",
        "stt_result",
        ["stt_sample_id"],
        unique=False,
    )
    op.create_index(
        "ix_stt_result_run_id",
        "stt_result",
        ["evaluation_run_id"],
        unique=False,
    )
    op.create_index(
        "idx_stt_result_feedback",
        "stt_result",
        ["evaluation_run_id", "is_correct"],
        unique=False,
    )
    op.create_index(
        "idx_stt_result_status",
        "stt_result",
        ["evaluation_run_id", "status"],
        unique=False,
    )


def downgrade():
    # Drop stt_result table
    op.drop_index("idx_stt_result_status", table_name="stt_result")
    op.drop_index("idx_stt_result_feedback", table_name="stt_result")
    op.drop_index("ix_stt_result_run_id", table_name="stt_result")
    op.drop_index("ix_stt_result_sample_id", table_name="stt_result")
    op.drop_table("stt_result")

    # Drop stt_sample table
    op.drop_index("idx_stt_sample_org_project", table_name="stt_sample")
    op.drop_index("ix_stt_sample_dataset_id", table_name="stt_sample")
    op.drop_table("stt_sample")

    # Remove columns from evaluation_run table
    op.drop_column("evaluation_run", "processed_samples")
    op.drop_column("evaluation_run", "providers")
    op.drop_column("evaluation_run", "language")
    op.drop_column("evaluation_run", "type")

    # Remove columns from evaluation_dataset table
    op.drop_column("evaluation_dataset", "language")
    op.drop_column("evaluation_dataset", "type")
