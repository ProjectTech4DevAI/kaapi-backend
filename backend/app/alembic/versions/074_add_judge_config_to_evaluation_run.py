"""add judge_config column to evaluation_run

Revision ID: 074
Revises: 073
Create Date: 2026-07-09 00:00:00.000000

The chunked fast-eval pipeline enqueues the aggregate task from the cron barrier
with only eval_run_id, so a per-run judge LLMCallConfig can't survive as a Celery
arg. Persist it on the run row instead. Nullable with no backfill: NULL is the
"zero-config default judge" sentinel, which is exactly how pre-feature runs behave.

"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "074"
down_revision = "073"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "evaluation_run",
        sa.Column(
            "judge_config",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
            comment=(
                "Per-run LLMCallConfig payload tailoring the correctness judge "
                "(saved id+version ref or ad-hoc blob); NULL = zero-config default "
                "judge. Persisted here because the cron barrier enqueues the "
                "aggregate with only eval_run_id, so it can't ride a Celery arg"
            ),
        ),
    )


def downgrade():
    op.drop_column("evaluation_run", "judge_config")
