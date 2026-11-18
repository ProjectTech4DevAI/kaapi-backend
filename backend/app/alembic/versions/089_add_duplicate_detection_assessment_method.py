"""add DUPLICATE_DETECTION assessment_method enum value

Revision ID: 089
Revises: 088
Create Date: 2026-11-14 00:00:00.000000

Duplicate-detection jobs are stored in `assessment` under their own method so
they can be told apart from RESPONSE rows. ALTER TYPE ... ADD VALUE must run
outside a transaction, hence the autocommit block.
"""

from alembic import op

revision = "089"
down_revision = "088"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute(
            "ALTER TYPE assessment_method ADD VALUE IF NOT EXISTS 'DUPLICATE_DETECTION'"
        )


def downgrade() -> None:
    # Postgres cannot drop a value from an enum type; leaving it is harmless.
    pass
