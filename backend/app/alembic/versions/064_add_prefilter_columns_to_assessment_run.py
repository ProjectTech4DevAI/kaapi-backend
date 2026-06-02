"""Add prefilter pipeline columns to assessment_run

Revision ID: 064
Revises: 063
Create Date: 2026-05-27 00:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

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
            comment="S3 URL of stored prefilter filter results JSON",
        ),
    )
    op.add_column(
        "assessment_run",
        sa.Column(
            "prefilter_total_rows",
            sa.Integer(),
            nullable=True,
            comment="Total rows fed into prefilter pipeline",
        ),
    )
    op.add_column(
        "assessment_run",
        sa.Column(
            "prefilter_total_passed",
            sa.Integer(),
            nullable=True,
            comment="Rows that passed topic relevance and went to L2",
        ),
    )
    op.add_column(
        "assessment_run",
        sa.Column(
            "prefilter_total_rejected",
            sa.Integer(),
            nullable=True,
            comment="Rows rejected by topic relevance, stopped at prefilter",
        ),
    )


def downgrade() -> None:
    op.drop_column("assessment_run", "prefilter_total_rejected")
    op.drop_column("assessment_run", "prefilter_total_passed")
    op.drop_column("assessment_run", "prefilter_total_rows")
    op.drop_column("assessment_run", "prefilter_object_store_url")
