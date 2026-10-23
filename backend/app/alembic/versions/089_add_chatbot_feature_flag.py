"""add CHATBOT feature flag key

Revision ID: 089
Revises: 088
Create Date: 2026-10-23 00:00:00.000000
"""

from alembic import op

revision = "089"
down_revision = "088"
branch_labels = None
depends_on = None


def upgrade():
    with op.get_context().autocommit_block():
        op.execute("ALTER TYPE featureflagkey ADD VALUE IF NOT EXISTS 'CHATBOT'")


def downgrade():
    pass
