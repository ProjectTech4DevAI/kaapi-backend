"""make llm_call input_type, provider, model nullable

Revision ID: 058
Revises: 057
Create Date: 2026-05-11 00:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "058"
down_revision = "057"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column("llm_call", "input_type", nullable=True)
    op.alter_column("llm_call", "provider", nullable=True)
    op.alter_column("llm_call", "model", nullable=True)


def downgrade() -> None:
    op.alter_column("llm_call", "model", nullable=False)
    op.alter_column("llm_call", "provider", nullable=False)
    op.alter_column("llm_call", "input_type", nullable=False)
