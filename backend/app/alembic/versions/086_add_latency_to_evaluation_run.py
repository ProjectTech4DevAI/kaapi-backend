"""Add latency to evaluation_run

Revision ID: 086
Revises: 085
Create Date: 2026-10-14 00:00:00.000000
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
