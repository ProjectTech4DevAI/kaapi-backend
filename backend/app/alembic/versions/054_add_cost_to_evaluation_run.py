"""add cost tracking to evaluation_run

Revision ID: 054
Revises: 053
Create Date: 2026-04-09 12:00:00.000000

"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "054"
down_revision = "053"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "evaluation_run",
        sa.Column(
            "cost",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
            comment="Cost tracking (response/embedding tokens and USD)",
        ),
    )


def downgrade():
    op.drop_column("evaluation_run", "cost")
