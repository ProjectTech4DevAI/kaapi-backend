"""Add evaluation_iteration_run table

Revision ID: 082
Revises: 081
Create Date: 2026-08-02 00:00:00.000000

Thin tracking row for the eval-iterate-improve LangGraph loop (see
docs/srd-ai-prompt-improvement.md follow-on: the Evaluation Iteration Loop). The
round-by-round trajectory itself lives in the LangGraph checkpoint (owned by
`langgraph-checkpoint-postgres`, set up separately via `checkpointer.setup()`,
not this migration) — this table only tracks enough to create/look up a loop,
scope it by org/project, and let the cron tick find loops still in flight.
"""

import sqlalchemy as sa
import sqlmodel.sql.sqltypes
from alembic import op

revision = "082"
down_revision = "081"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "evaluation_iteration_run",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("dataset_id", sa.Integer(), nullable=False),
        sa.Column(
            "experiment_name",
            sqlmodel.sql.sqltypes.AutoString(length=255),
            nullable=False,
        ),
        sa.Column("config_id", sa.Uuid(), nullable=False),
        sa.Column("initial_config_version", sa.Integer(), nullable=False),
        sa.Column(
            "status",
            sa.Enum(
                "processing",
                "completed",
                "failed",
                name="evaluationiterationstatusenum",
            ),
            nullable=False,
            comment="Loop bookkeeping status: processing, completed, or failed",
        ),
        sa.Column(
            "stop_reason",
            sqlmodel.sql.sqltypes.AutoString(),
            nullable=True,
            comment="Copied from the final graph state once terminal: ceiling_reached, max_rounds_reached, or round_failed",
        ),
        sa.Column(
            "callback_url",
            sqlmodel.sql.sqltypes.AutoString(),
            nullable=False,
            comment="HTTPS webhook validated via validate_callback_url before create",
        ),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("organization_id", sa.Integer(), nullable=False),
        sa.Column("project_id", sa.Integer(), nullable=False),
        sa.Column("inserted_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["dataset_id"],
            ["evaluation_dataset.id"],
            name="fk_evaluation_iteration_run_dataset_id",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["config_id"],
            ["config.id"],
            name="fk_evaluation_iteration_run_config_id",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["organization_id"], ["organization.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(["project_id"], ["project.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_evaluation_iteration_run_dataset_id"),
        "evaluation_iteration_run",
        ["dataset_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_evaluation_iteration_run_config_id"),
        "evaluation_iteration_run",
        ["config_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_evaluation_iteration_run_organization_id"),
        "evaluation_iteration_run",
        ["organization_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_evaluation_iteration_run_project_id"),
        "evaluation_iteration_run",
        ["project_id"],
        unique=False,
    )


def downgrade():
    op.drop_index(
        op.f("ix_evaluation_iteration_run_project_id"),
        table_name="evaluation_iteration_run",
    )
    op.drop_index(
        op.f("ix_evaluation_iteration_run_organization_id"),
        table_name="evaluation_iteration_run",
    )
    op.drop_index(
        op.f("ix_evaluation_iteration_run_config_id"),
        table_name="evaluation_iteration_run",
    )
    op.drop_index(
        op.f("ix_evaluation_iteration_run_dataset_id"),
        table_name="evaluation_iteration_run",
    )
    op.drop_table("evaluation_iteration_run")
    sa.Enum(name="evaluationiterationstatusenum").drop(op.get_bind(), checkfirst=True)
