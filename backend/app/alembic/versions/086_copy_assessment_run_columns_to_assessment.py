"""Copy assessment_run's per-config columns onto assessment (expand step)

Revision ID: 086
Revises: 085
Create Date: 2026-10-06 00:00:00.000000

`assessment_run` was designed as a 1:N child so one submission could fan out to up to
four configs. The console dropped multi-config submission and the API-client path
always creates exactly one execution, so in every environment the two tables are 1:1
and the child is redundant. This is the additive half of folding it into the parent:
the seven run-only columns are added to `assessment` and backfilled from
`assessment_run`, with `assessment_run` left untouched so the deployed code keeps
working. The code switch and the `assessment_run` drop ship in a later migration once
the application reads from these columns.

Every new column is nullable with no server default so the ADD COLUMN is a catalog-only
change (no table rewrite, no long lock on a live table). RESPONSE assessments have no
run and stay NULL in all seven. The backfill is a single statement guarded by
`config_id IS NULL`, so re-running it after a partial failure copies only what is
missing. Downgrade drops the columns; nothing is lost because `assessment_run` still
holds the source rows.
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "086"
down_revision = "085"
branch_labels = None
depends_on = None

CONFIG_INDEX = "idx_assessment_config"
BATCH_JOB_INDEX = "idx_assessment_batch_job"
CONFIG_FK = "fk_assessment_config_id_config"
BATCH_JOB_FK = "fk_assessment_batch_job_id_batch_job"


def upgrade() -> None:
    op.add_column(
        "assessment",
        sa.Column(
            "config_id",
            postgresql.UUID(as_uuid=True),
            nullable=True,
            comment="ASSESSMENT-tagged config this assessment runs; NULL for RESPONSE",
        ),
    )
    op.add_column(
        "assessment",
        sa.Column("config_version", sa.Integer(), nullable=True),
    )
    op.add_column(
        "assessment",
        sa.Column(
            "total_items",
            sa.Integer(),
            nullable=True,
            comment="Submitted row count; drives result sizing without reading rows",
        ),
    )
    op.add_column(
        "assessment",
        sa.Column(
            "batch_job_id",
            sa.Integer(),
            nullable=True,
            comment="Provider batch job; staged runs track per-stage jobs in execution",
        ),
    )
    op.add_column(
        "assessment",
        sa.Column(
            "execution",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
            comment="Staged-batch runtime bag (BatchRunState for BATCH, RunExecution for RUN)",
        ),
    )
    op.add_column(
        "assessment",
        sa.Column(
            "post_processing_config",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
    )
    op.add_column(
        "assessment",
        sa.Column("error_message", sa.Text(), nullable=True),
    )

    op.create_foreign_key(CONFIG_FK, "assessment", "config", ["config_id"], ["id"])
    op.create_foreign_key(
        BATCH_JOB_FK,
        "assessment",
        "batch_job",
        ["batch_job_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index(
        CONFIG_INDEX, "assessment", ["config_id", "config_version"], unique=False
    )
    op.create_index(
        BATCH_JOB_INDEX,
        "assessment",
        ["batch_job_id"],
        unique=False,
        postgresql_where=sa.text("batch_job_id IS NOT NULL"),
    )

    op.execute(
        """
        UPDATE assessment AS a
        SET config_id = r.config_id,
            config_version = r.config_version,
            total_items = r.total_items,
            batch_job_id = r.batch_job_id,
            execution = r.execution,
            post_processing_config = r.post_processing_config,
            error_message = r.error_message
        FROM assessment_run AS r
        WHERE r.assessment_id = a.id
          AND a.config_id IS NULL
        """
    )


def downgrade() -> None:
    op.drop_index(BATCH_JOB_INDEX, table_name="assessment")
    op.drop_index(CONFIG_INDEX, table_name="assessment")
    op.drop_constraint(BATCH_JOB_FK, "assessment", type_="foreignkey")
    op.drop_constraint(CONFIG_FK, "assessment", type_="foreignkey")
    op.drop_column("assessment", "error_message")
    op.drop_column("assessment", "post_processing_config")
    op.drop_column("assessment", "execution")
    op.drop_column("assessment", "batch_job_id")
    op.drop_column("assessment", "total_items")
    op.drop_column("assessment", "config_version")
    op.drop_column("assessment", "config_id")
