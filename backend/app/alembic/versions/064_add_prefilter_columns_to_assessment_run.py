"""Add prefilter columns and pipeline stage-machine columns to assessment_run

Revision ID: 064
Revises: 063
Create Date: 2026-05-27 00:00:00.000000

"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "064"
down_revision = "063"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "assessment_run",
        sa.Column(
            "prefilter_object_store_url",
            sa.String(),
            nullable=True,
            comment="S3 URL of prefilter results JSON",
        ),
    )
    op.add_column(
        "assessment_run",
        sa.Column(
            "prefilter_total_rows",
            sa.Integer(),
            nullable=True,
            comment="Total rows fed into the prefilter stages",
        ),
    )
    op.add_column(
        "assessment_run",
        sa.Column(
            "prefilter_total_passed",
            sa.Integer(),
            nullable=True,
            comment="Rows that passed the go/no-go gates and went to L2",
        ),
    )
    op.add_column(
        "assessment_run",
        sa.Column(
            "prefilter_total_rejected",
            sa.Integer(),
            nullable=True,
            comment="Rows rejected by a go/no-go gate",
        ),
    )
    op.add_column(
        "assessment_run",
        sa.Column(
            "stage",
            sa.String(),
            nullable=True,
            comment=(
                "Current pipeline stage: PRE_FILTER_TOPIC_RELEVANCE, "
                "PRE_FILTER_DUPLICATE_DETECTION, L2_ASSESSMENT, COMPLETED, FAILED"
            ),
        ),
    )
    op.add_column(
        "assessment_run",
        sa.Column(
            "stage_status",
            sa.String(),
            nullable=True,
            comment="Status of stage: PENDING, PROCESSING, COMPLETED, FAILED",
        ),
    )
    op.add_column(
        "assessment_run",
        sa.Column(
            "pipeline",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
            comment="Ordered stage config driving execution: {'stages': [...]}",
        ),
    )
    op.add_column(
        "assessment_run",
        sa.Column(
            "stage_batches",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
            comment="Map of stage name -> batch_job id, for per-stage result lookup",
        ),
    )


def downgrade() -> None:
    op.drop_column("assessment_run", "stage_batches")
    op.drop_column("assessment_run", "pipeline")
    op.drop_column("assessment_run", "stage_status")
    op.drop_column("assessment_run", "stage")
    op.drop_column("assessment_run", "prefilter_total_rejected")
    op.drop_column("assessment_run", "prefilter_total_passed")
    op.drop_column("assessment_run", "prefilter_total_rows")
    op.drop_column("assessment_run", "prefilter_object_store_url")
