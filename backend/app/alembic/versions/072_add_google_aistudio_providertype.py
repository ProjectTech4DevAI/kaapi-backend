"""add google_aistudio value to the providertype enum

Revision ID: 072
Revises: 071
Create Date: 2026-07-07 00:00:00.000000

Stored value is the member NAME 'google_aistudio'
"""

from alembic import op

revision = "072"
down_revision = "071"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute("ALTER TYPE providertype ADD VALUE IF NOT EXISTS 'google_aistudio'")


def downgrade() -> None:
    # Postgres cannot drop a value from an enum type; downgrade is a no-op.
    pass
