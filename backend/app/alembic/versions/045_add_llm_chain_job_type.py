"""add LLM_CHAIN job type

Revision ID: 045
Revises: 044
Create Date: 2026-02-04 00:35:43.891644

"""
from alembic import op

# revision identifiers, used by Alembic.
revision = "045"
down_revision = "044"
branch_labels = None
depends_on = None


def upgrade():
    op.execute("ALTER TYPE jobtype ADD VALUE IF NOT EXISTS 'LLM_CHAIN'")


def downgrade():
    pass
