"""add PROMPT_IMPROVEMENT jobtype enum value

Revision ID: 072
Revises: 071
Create Date: 2026-07-09 00:00:00.000000

Prompt improvement moved from a synchronous request to a Celery job, so its runs
are tracked in the `job` table under this new jobtype. ALTER TYPE ... ADD VALUE
must run outside a transaction, hence the autocommit block.
"""

from alembic import op

revision = "072"
down_revision = "071"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute("ALTER TYPE jobtype ADD VALUE IF NOT EXISTS 'PROMPT_IMPROVEMENT'")


def downgrade() -> None:
    # Postgres cannot drop a value from an enum type; leaving it is harmless.
    pass
