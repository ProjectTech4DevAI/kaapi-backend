"""Add native LLM-as-judge columns to evaluation_run

Revision ID: 074
Revises: 073
Create Date: 2026-07-16 00:00:00.000000

The v2 judged fast-eval run stores its judge state on the existing evaluation_run
row rather than a new table. The chunked aggregate task only knows an eval_run_id,
so the judge intent (is_judge_run) must be durable on the row for the aggregate to
read at judge time. per_item_ground_truth mirrors per_item_scores as Kaapi's native
per-row store (v2 never syncs to Langfuse). Judging is system-config only — always
the fallback model + built-in prompt — so no per-run judge config is persisted.

Both columns are nullable with default NULL: v1 and pre-feature runs carry no judge
data, so no backfill is needed.
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "074"
down_revision = "073"
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
    op.add_column(
        "evaluation_run",
        sa.Column(
            "per_item_ground_truth",
            JSONB(),
            nullable=True,
            comment=(
                "Durable {ref: score} map of the Adherence to Ground Truth judge "
                "scores (ref = trace_id when traced, else item_id); Kaapi's own store"
            ),
        ),
    )


def downgrade():
    op.drop_column("evaluation_run", "per_item_ground_truth")
    op.drop_column("evaluation_run", "is_judge_run")
