"""Add is_judge_run to evaluation_run

Revision ID: 075
Revises: 074
Create Date: 2026-07-16 00:00:00.000000

The chunked aggregate task only knows an eval_run_id, so the judge intent has to be
durable on the run row for it to read at judge time.
"""

import sqlalchemy as sa
from alembic import op

revision = "075"
down_revision = "074"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "evaluation_run",
        sa.Column(
            "is_judge_run",
            sa.Boolean(),
            nullable=True,
            comment=(
                "True for v2 runs that run the native LLM-as-judge (and skip the "
                "Langfuse score sync). NULL/False = v1 run, cosine-only, Langfuse-synced"
            ),
        ),
    )


def downgrade():
    op.drop_column("evaluation_run", "is_judge_run")
