"""refactor assessment tables for method-based pipeline

Revision ID: 077
Revises: 076
Create Date: 2026-08-01 00:00:00.000000
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "077"
down_revision = "076"
branch_labels = None
depends_on = None

ASSESSMENT_METHOD_ENUM = "assessment_method"
ASSESSMENT_STATUS_ENUM = "assessment_status"
ASSESSMENT_METHOD_VALUES = ("RESPONSE", "BATCH", "RUN")
ASSESSMENT_STATUS_VALUES = (
    "PENDING",
    "PROCESSING",
    "COMPLETED",
    "COMPLETED_WITH_ERRORS",
    "FAILED",
)
# folded into `execution`, then dropped
LEGACY_RUN_COLUMNS = (
    "input",
    "pipeline",
    "stage",
    "stage_status",
    "stage_batches",
    "object_store_url",
    "prefilter_object_store_url",
    "prefilter_total_rows",
    "prefilter_total_passed",
    "prefilter_total_rejected",
)


def _status_to_enum(table: str) -> None:
    op.execute(f"ALTER TABLE {table} ALTER COLUMN status DROP DEFAULT")
    op.execute(
        f"ALTER TABLE {table} ALTER COLUMN status TYPE {ASSESSMENT_STATUS_ENUM} "
        f"USING upper(status)::{ASSESSMENT_STATUS_ENUM}"
    )
    op.execute(
        f"ALTER TABLE {table} ALTER COLUMN status "
        f"SET DEFAULT 'PENDING'::{ASSESSMENT_STATUS_ENUM}"
    )


def _status_to_varchar(table: str) -> None:
    op.execute(f"ALTER TABLE {table} ALTER COLUMN status DROP DEFAULT")
    op.execute(
        f"ALTER TABLE {table} ALTER COLUMN status TYPE VARCHAR "
        f"USING lower(status::text)"
    )
    op.execute(f"ALTER TABLE {table} ALTER COLUMN status SET DEFAULT 'pending'")


def upgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute("ALTER TYPE jobtype ADD VALUE IF NOT EXISTS 'ASSESSMENT'")

    # assessment data is disposable — drop it for a clean schema swap
    op.execute("TRUNCATE TABLE assessment_run, assessment RESTART IDENTITY CASCADE")

    assessment_method = postgresql.ENUM(
        *ASSESSMENT_METHOD_VALUES, name=ASSESSMENT_METHOD_ENUM, create_type=False
    )
    assessment_status = postgresql.ENUM(
        *ASSESSMENT_STATUS_VALUES, name=ASSESSMENT_STATUS_ENUM, create_type=False
    )
    assessment_method.create(op.get_bind(), checkfirst=True)
    assessment_status.create(op.get_bind(), checkfirst=True)

    op.add_column(
        "assessment",
        sa.Column("method", assessment_method, nullable=False),
    )
    op.add_column(
        "assessment",
        sa.Column(
            "job_id",
            sa.Uuid(),
            nullable=True,
            comment="RESPONSE execution job; resolve call/chain via job_id",
        ),
    )
    op.create_foreign_key(
        "fk_assessment_job_id",
        "assessment",
        "job",
        ["job_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.add_column(
        "assessment",
        sa.Column(
            "input",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
            comment="Method-shaped: RequestInput (RESPONSE) / BatchInput (BATCH) / InputBinding (RUN)",
        ),
    )
    op.add_column(
        "assessment_run",
        sa.Column(
            "execution",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
            comment="RUN-only runtime (RunExecution)",
        ),
    )
    op.add_column(
        "assessment_run",
        sa.Column(
            "post_processing_config",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
    )

    _status_to_enum("assessment")
    _status_to_enum("assessment_run")

    op.alter_column(
        "assessment",
        "experiment_name",
        existing_type=sa.String(),
        nullable=True,
        existing_comment="Name of the experiment grouping its config runs",
        comment="Experiment name; required for BATCH/RUN",
    )
    op.drop_constraint("fk_assessment_dataset_id", "assessment", type_="foreignkey")
    op.alter_column(
        "assessment",
        "dataset_id",
        existing_type=sa.Integer(),
        nullable=True,
        existing_comment="Reference to the evaluation dataset",
        comment="External dataset (RUN required; BATCH optional by-ref)",
    )
    op.create_foreign_key(
        "fk_assessment_dataset_id",
        "assessment",
        "evaluation_dataset",
        ["dataset_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index("idx_assessment_method", "assessment", ["method"], unique=False)
    op.create_index(
        "idx_assessment_job",
        "assessment",
        ["job_id"],
        unique=False,
        postgresql_where=sa.text("job_id IS NOT NULL"),
    )

    for column in LEGACY_RUN_COLUMNS:
        op.drop_column("assessment_run", column)
    op.create_index(
        "idx_assessment_run_config",
        "assessment_run",
        ["config_id", "config_version"],
        unique=False,
    )
    op.create_index(
        "idx_assessment_run_status", "assessment_run", ["status"], unique=False
    )

    op.drop_constraint(
        "fk_assessment_run_assessment_id", "assessment_run", type_="foreignkey"
    )
    op.drop_index("idx_assessment_run_assessment_id", table_name="assessment_run")
    op.drop_constraint("assessment_pkey", "assessment", type_="primary")
    op.drop_column("assessment", "id")
    op.add_column(
        "assessment",
        sa.Column(
            "id",
            sa.Uuid(),
            nullable=False,
            comment="Unique identifier for the assessment",
        ),
    )
    op.create_primary_key("assessment_pkey", "assessment", ["id"])
    op.drop_column("assessment_run", "assessment_id")
    op.add_column(
        "assessment_run",
        sa.Column(
            "assessment_id",
            sa.Uuid(),
            nullable=False,
            comment="Reference to the parent assessment",
        ),
    )
    op.create_index(
        "idx_assessment_run_assessment_id",
        "assessment_run",
        ["assessment_id"],
        unique=False,
    )
    op.create_foreign_key(
        "fk_assessment_run_assessment_id",
        "assessment_run",
        "assessment",
        ["assessment_id"],
        ["id"],
        ondelete="CASCADE",
    )


def downgrade() -> None:
    op.execute("TRUNCATE TABLE assessment_run, assessment RESTART IDENTITY CASCADE")
    op.drop_constraint(
        "fk_assessment_run_assessment_id", "assessment_run", type_="foreignkey"
    )
    op.drop_index("idx_assessment_run_assessment_id", table_name="assessment_run")
    op.drop_constraint("assessment_pkey", "assessment", type_="primary")
    op.drop_column("assessment", "id")
    op.add_column(
        "assessment",
        sa.Column(
            "id",
            sa.Integer(),
            sa.Identity(always=False),
            nullable=False,
            comment="Unique identifier for the assessment",
        ),
    )
    op.create_primary_key("assessment_pkey", "assessment", ["id"])
    op.drop_column("assessment_run", "assessment_id")
    op.add_column(
        "assessment_run",
        sa.Column(
            "assessment_id",
            sa.Integer(),
            nullable=False,
            comment="Reference to the parent assessment",
        ),
    )
    op.create_index(
        "idx_assessment_run_assessment_id",
        "assessment_run",
        ["assessment_id"],
        unique=False,
    )
    op.create_foreign_key(
        "fk_assessment_run_assessment_id",
        "assessment_run",
        "assessment",
        ["assessment_id"],
        ["id"],
        ondelete="CASCADE",
    )

    op.drop_index("idx_assessment_run_status", table_name="assessment_run")
    op.drop_index("idx_assessment_run_config", table_name="assessment_run")
    _status_to_varchar("assessment_run")

    # recreated NULLable; original data is lost
    op.add_column(
        "assessment_run",
        sa.Column(
            "input",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
            comment="Assessment input: prompt_template, text_columns, attachments, output_schema",
        ),
    )
    op.add_column(
        "assessment_run",
        sa.Column("object_store_url", sa.String(), nullable=True),
    )
    op.add_column("assessment_run", sa.Column("stage", sa.String(), nullable=True))
    op.add_column(
        "assessment_run", sa.Column("stage_status", sa.String(), nullable=True)
    )
    op.add_column(
        "assessment_run",
        sa.Column("pipeline", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )
    op.add_column(
        "assessment_run",
        sa.Column(
            "stage_batches", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
    )
    op.add_column(
        "assessment_run",
        sa.Column("prefilter_object_store_url", sa.String(), nullable=True),
    )
    op.add_column(
        "assessment_run",
        sa.Column("prefilter_total_rows", sa.Integer(), nullable=True),
    )
    op.add_column(
        "assessment_run",
        sa.Column("prefilter_total_passed", sa.Integer(), nullable=True),
    )
    op.add_column(
        "assessment_run",
        sa.Column("prefilter_total_rejected", sa.Integer(), nullable=True),
    )
    op.drop_column("assessment_run", "post_processing_config")
    op.drop_column("assessment_run", "execution")

    op.drop_index("idx_assessment_job", table_name="assessment")
    op.drop_index("idx_assessment_method", table_name="assessment")
    _status_to_varchar("assessment")

    op.drop_constraint("fk_assessment_dataset_id", "assessment", type_="foreignkey")
    op.alter_column(
        "assessment",
        "dataset_id",
        existing_type=sa.Integer(),
        nullable=False,
        existing_comment="External dataset (RUN required; BATCH optional by-ref)",
        comment="Reference to the evaluation dataset",
    )
    op.create_foreign_key(
        "fk_assessment_dataset_id",
        "assessment",
        "evaluation_dataset",
        ["dataset_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.alter_column(
        "assessment",
        "experiment_name",
        existing_type=sa.String(),
        nullable=False,
        existing_comment="Experiment name; required for BATCH/RUN",
        comment="Name of the experiment grouping its config runs",
    )

    op.drop_constraint("fk_assessment_job_id", "assessment", type_="foreignkey")
    op.drop_column("assessment", "job_id")
    op.drop_column("assessment", "input")
    op.drop_column("assessment", "method")

    sa.Enum(name=ASSESSMENT_STATUS_ENUM).drop(op.get_bind(), checkfirst=True)
    sa.Enum(name=ASSESSMENT_METHOD_ENUM).drop(op.get_bind(), checkfirst=True)
