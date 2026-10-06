"""Fold assessment_run into assessment

Revision ID: 086
Revises: 085
Create Date: 2026-10-06 00:00:00.000000

The child was 1:1 with its parent everywhere, so its config + runtime columns move
onto `assessment` and the table is dropped. `error_message` lands in
`execution.error`; keys the code no longer reads are stripped. `input` goes: nothing
writes it since the console pipeline was retired. Downgrade recreates `assessment_run`
empty: take a `pg_dump -t assessment_run` before upgrading, the rows come back from it.
An assessment with no run fails the `SET NOT NULL` on purpose rather than migrating half.
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "086"
down_revision = "085"
branch_labels = None
depends_on = None

CONFIG_INDEX = "idx_assessment_config"
CONFIG_FK = "fk_assessment_config_id_config"
RUN_TABLE = "assessment_run"
# Derivable from config_version or already in the stage dumps.
DROPPED_EXECUTION_KEYS = (
    "stage_output_urls",
    "provider",
    "model",
    "input_schema",
    "verdicts",
    "gate_passed",
)


def upgrade() -> None:
    op.add_column(
        "assessment",
        sa.Column("config_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.add_column(
        "assessment", sa.Column("config_version", sa.Integer(), nullable=True)
    )
    op.add_column(
        "assessment",
        sa.Column(
            "total_items",
            sa.Integer(),
            nullable=True,
            comment="Submitted row count; sizes the result without reading the rows",
        ),
    )
    op.add_column(
        "assessment",
        sa.Column(
            "execution",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
            comment="Staged-batch runtime (AssessmentExecution); NULL for RESPONSE",
        ),
    )

    strip = " ".join(f"- '{key}'" for key in DROPPED_EXECUTION_KEYS)
    op.execute(
        f"""
        UPDATE assessment AS a
        SET config_id = r.config_id,
            config_version = r.config_version,
            total_items = r.total_items,
            execution = CASE
                WHEN r.error_message IS NULL THEN r.execution {strip}
                ELSE (COALESCE(r.execution, '{{}}'::jsonb) {strip})
                     || jsonb_build_object('error', r.error_message)
            END
        FROM {RUN_TABLE} AS r
        WHERE r.assessment_id = a.id
          AND a.config_id IS NULL
        """
    )

    op.alter_column("assessment", "config_id", nullable=False)
    op.alter_column("assessment", "config_version", nullable=False)
    op.alter_column("assessment", "total_items", nullable=False)
    op.create_foreign_key(CONFIG_FK, "assessment", "config", ["config_id"], ["id"])
    op.create_index(CONFIG_INDEX, "assessment", ["config_id", "config_version"])

    op.drop_column("assessment", "input")
    op.drop_table(RUN_TABLE)


def downgrade() -> None:
    op.create_table(
        RUN_TABLE,
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "assessment_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey(
                "assessment.id",
                name="fk_assessment_run_assessment_id",
                ondelete="CASCADE",
            ),
            nullable=False,
        ),
        sa.Column(
            "config_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("config.id", name="fk_assessment_run_config_id"),
            nullable=False,
        ),
        sa.Column("config_version", sa.Integer(), nullable=False),
        sa.Column(
            "status",
            postgresql.ENUM(name="assessment_status", create_type=False),
            server_default="PENDING",
            nullable=False,
        ),
        sa.Column(
            "batch_job_id",
            sa.Integer(),
            sa.ForeignKey(
                "batch_job.id",
                name="fk_assessment_run_batch_job_id",
                ondelete="SET NULL",
            ),
            nullable=True,
        ),
        sa.Column("total_items", sa.Integer(), server_default="0", nullable=False),
        sa.Column("execution", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "post_processing_config",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("inserted_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )
    op.create_index("idx_assessment_run_assessment_id", RUN_TABLE, ["assessment_id"])
    op.create_index(
        "idx_assessment_run_config", RUN_TABLE, ["config_id", "config_version"]
    )
    op.create_index("idx_assessment_run_status", RUN_TABLE, ["status"])
    op.add_column(
        "assessment",
        sa.Column("input", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )
    op.drop_index(CONFIG_INDEX, table_name="assessment")
    op.drop_constraint(CONFIG_FK, "assessment", type_="foreignkey")
    op.drop_column("assessment", "execution")
    op.drop_column("assessment", "total_items")
    op.drop_column("assessment", "config_version")
    op.drop_column("assessment", "config_id")
