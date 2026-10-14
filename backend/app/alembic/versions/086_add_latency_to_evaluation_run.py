"""Add latency to evaluation_run

Revision ID: 086
Revises: 085
Create Date: 2026-10-14 00:00:00.000000

A sibling of `cost` rather than nested inside it, so a pricing failure and a
latency failure stay independent (one swallowed exception can't drop the other)
and analytics that SUM `cost->>'total_cost_usd'` are unaffected.

Nullable because the table is shared with STT/TTS runs and batch runs have no
per-call timing. No backfill: rows that predate this migration legitimately
read as NULL.
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "086"
down_revision = "085"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "evaluation_run",
        sa.Column(
            "latency",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
            comment="Avg OpenAI call latency per stage (fast runs only; NULL for batch)",
        ),
    )


def downgrade():
    op.drop_column("evaluation_run", "latency")
