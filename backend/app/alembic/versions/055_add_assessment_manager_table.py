"""add assessment and assessment_run tables

Revision ID: 055
Revises: 054
Create Date: 2026-03-26 23:30:00.000000

"""

import sqlalchemy as sa
import sqlmodel.sql.sqltypes
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "055"
down_revision = "054"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "assessment",
        sa.Column(
            "id",
            sa.Integer(),
            nullable=False,
            comment="Unique identifier for the assessment",
        ),
        sa.Column(
            "experiment_name",
            sqlmodel.sql.sqltypes.AutoString(),
            nullable=False,
            comment="Name of the experiment grouping its config runs",
        ),
        sa.Column(
            "dataset_id",
            sa.Integer(),
            nullable=False,
            comment="Reference to the evaluation dataset",
        ),
        sa.Column(
            "status",
            sqlmodel.sql.sqltypes.AutoString(),
            nullable=False,
            server_default="pending",
            comment=(
                "Aggregate status: pending, processing, completed, "
                "completed_with_errors, failed"
            ),
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
            comment="Timestamp when the assessment was created",
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(),
            nullable=False,
            comment="Timestamp when the assessment was last updated",
        ),
        sa.ForeignKeyConstraint(
            ["dataset_id"],
            ["evaluation_dataset.id"],
            name="fk_assessment_dataset_id",
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
        op.f("ix_assessment_experiment_name"),
        "assessment",
        ["experiment_name"],
        unique=False,
    )
    op.create_index(
        "idx_assessment_org_project",
        "assessment",
        ["organization_id", "project_id", "inserted_at"],
        unique=False,
    )
    op.create_index(
        "idx_assessment_status",
        "assessment",
        ["status"],
        unique=False,
    )

    op.create_table(
        "assessment_run",
        sa.Column(
            "id",
            sa.Integer(),
            nullable=False,
            comment="Unique identifier for the assessment run",
        ),
        sa.Column(
            "assessment_id",
            sa.Integer(),
            nullable=False,
            comment="Reference to the parent assessment",
        ),
        sa.Column(
            "config_id",
            sa.Uuid(),
            nullable=False,
            comment="Reference to the stored config used",
        ),
        sa.Column(
            "config_version",
            sa.Integer(),
            nullable=False,
            comment="Version of the config used",
        ),
        sa.Column(
            "status",
            sqlmodel.sql.sqltypes.AutoString(),
            nullable=False,
            server_default="pending",
            comment="Run status: pending, processing, completed, failed",
        ),
        sa.Column(
            "batch_job_id",
            sa.Integer(),
            nullable=True,
            comment="Reference to the batch job processing this run",
        ),
        sa.Column(
            "total_items",
            sa.Integer(),
            nullable=False,
            server_default="0",
            comment="Total number of dataset items in this run",
        ),
        sa.Column(
            "input",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            comment=(
                "Assessment input: prompt_template, text_columns, attachments, "
                "output_schema"
            ),
        ),
        sa.Column(
            "object_store_url",
            sqlmodel.sql.sqltypes.AutoString(),
            nullable=True,
            comment="S3 URL of processed batch results",
        ),
        sa.Column(
            "error_message",
            sa.Text(),
            nullable=True,
            comment="Error message if the run failed",
        ),
        sa.Column(
            "inserted_at",
            sa.DateTime(),
            nullable=False,
            comment="Timestamp when the run was created",
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(),
            nullable=False,
            comment="Timestamp when the run was last updated",
        ),
        sa.ForeignKeyConstraint(
            ["assessment_id"],
            ["assessment.id"],
            name="fk_assessment_run_assessment_id",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["config_id"],
            ["config.id"],
            name="fk_assessment_run_config_id",
        ),
        sa.ForeignKeyConstraint(
            ["batch_job_id"],
            ["batch_job.id"],
            name="fk_assessment_run_batch_job_id",
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "idx_assessment_run_assessment_id",
        "assessment_run",
        ["assessment_id"],
        unique=False,
    )


def downgrade():
    op.drop_index("idx_assessment_run_assessment_id", table_name="assessment_run")
    op.drop_table("assessment_run")
    op.drop_index("idx_assessment_status", table_name="assessment")
    op.drop_index("idx_assessment_org_project", table_name="assessment")
    op.drop_index(op.f("ix_assessment_experiment_name"), table_name="assessment")
    op.drop_table("assessment")
