"""add run_mode to evaluation_run

Revision ID: 055
Revises: 054
Create Date: 2026-04-28 12:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "055"
down_revision = "054"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "evaluation_run",
        sa.Column(
            "run_mode",
            sa.String(length=10),
            nullable=False,
            server_default="batch",
            comment="Execution mode: batch | live",
        ),
    )


def downgrade():
    op.drop_column("evaluation_run", "run_mode")
