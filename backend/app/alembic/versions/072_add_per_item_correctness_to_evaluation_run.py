"""add per_item_correctness column to evaluation_run

Revision ID: 072
Revises: 071
Create Date: 2026-07-06 00:00:00.000000

Native LLM-as-a-judge correctness rides the existing score/cost columns; this
adds the one durable map it needs. Nullable with no backfill: pre-feature runs
carry no correctness data, so NULL is the correct "never judged" sentinel and
distinguishes them from a run judged into an empty map.

"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "072"
down_revision = "071"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "evaluation_run",
        sa.Column(
            "per_item_correctness",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
            comment=("Durable {trace_id: correctness} map of LLM-as-a-judge scores; "),
        ),
    )


def downgrade():
    op.drop_column("evaluation_run", "per_item_correctness")
