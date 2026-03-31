"""add assessment manager table

Revision ID: 050
Revises: 049
Create Date: 2026-03-26 23:30:00.000000

"""

import sqlalchemy as sa
import sqlmodel.sql.sqltypes
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "050"
down_revision = "049"
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
            comment="Experiment name shared by child config runs",
        ),
        sa.Column(
            "dataset_id",
            sa.Integer(),
            nullable=False,
            comment="Reference to the evaluation dataset",
        ),
        sa.Column(
            "dataset_name",
            sqlmodel.sql.sqltypes.AutoString(),
            nullable=False,
            comment="Name of the dataset used by this assessment",
        ),
        sa.Column(
            "status",
            sqlmodel.sql.sqltypes.AutoString(),
            nullable=False,
            server_default="pending",
            comment="Overall assessment status across all child evaluation runs",
        ),
        sa.Column(
            "total_runs",
            sa.Integer(),
            nullable=False,
            server_default="0",
            comment="Total number of child evaluation runs",
        ),
        sa.Column(
            "pending_runs",
            sa.Integer(),
            nullable=False,
            server_default="0",
            comment="Number of child runs in pending state",
        ),
        sa.Column(
            "processing_runs",
            sa.Integer(),
            nullable=False,
            server_default="0",
            comment="Number of child runs in processing state",
        ),
        sa.Column(
            "completed_runs",
            sa.Integer(),
            nullable=False,
            server_default="0",
            comment="Number of child runs in completed state",
        ),
        sa.Column(
            "failed_runs",
            sa.Integer(),
            nullable=False,
            server_default="0",
            comment="Number of child runs in failed state",
        ),
        sa.Column(
            "run_stats",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
            comment="Cached status snapshot for child evaluation runs",
        ),
        sa.Column(
            "error_message",
            sa.Text(),
            nullable=True,
            comment="Aggregated error message for child run failures",
        ),
        sa.Column(
            "callback_url",
            sqlmodel.sql.sqltypes.AutoString(),
            nullable=True,
            comment="Optional frontend callback URL for status updates",
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
        "idx_assessment_status_org",
        "assessment",
        ["status", "organization_id"],
        unique=False,
    )
    op.create_index(
        "idx_assessment_status_project",
        "assessment",
        ["status", "project_id"],
        unique=False,
    )

    op.add_column(
        "evaluation_run",
        sa.Column(
            "assessment_id",
            sa.Integer(),
            nullable=True,
            comment="Reference to parent assessment manager row, if applicable",
        ),
    )
    op.create_foreign_key(
        "fk_evaluation_run_assessment_id",
        "evaluation_run",
        "assessment",
        ["assessment_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index(
        "idx_eval_run_assessment_id",
        "evaluation_run",
        ["assessment_id"],
        unique=False,
    )


def downgrade():
    op.drop_index("idx_eval_run_assessment_id", table_name="evaluation_run")
    op.drop_constraint(
        "fk_evaluation_run_assessment_id",
        "evaluation_run",
        type_="foreignkey",
    )
    op.drop_column("evaluation_run", "assessment_id")

    op.drop_index("idx_assessment_status_project", table_name="assessment")
    op.drop_index("idx_assessment_status_org", table_name="assessment")
    op.drop_index(op.f("ix_assessment_experiment_name"), table_name="assessment")
    op.drop_table("assessment")
