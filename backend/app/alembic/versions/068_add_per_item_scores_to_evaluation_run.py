"""add per_item_scores and unscoreable columns to evaluation_run

Revision ID: 068
Revises: 067
Create Date: 2026-06-17 00:00:00.000000

"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "068"
down_revision = "067"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "evaluation_run",
        sa.Column(
            "per_item_scores",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
            comment=(
                "Durable {trace_id: cosine_similarity} map of computed pair "
                "scores; source of truth used to backfill Langfuse on resync"
            ),
        ),
    )
    op.add_column(
        "evaluation_run",
        sa.Column(
            "unscoreable",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
            comment=(
                "{trace_id: reason} for items that cannot be scored "
                "(empty_output / empty_ground_truth / embedding_failed)"
            ),
        ),
    )


def downgrade():
    op.drop_column("evaluation_run", "unscoreable")
    op.drop_column("evaluation_run", "per_item_scores")
