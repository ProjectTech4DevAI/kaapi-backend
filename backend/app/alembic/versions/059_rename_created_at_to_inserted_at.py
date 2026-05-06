"""rename created_at to inserted_at on job and llm_call

Revision ID: 059
Revises: 058
Create Date: 2026-05-06 14:00:00.000000

Aligns `job` and `llm_call` with the rest of the schema (51 other tables
use `inserted_at`). Pure rename — no type or default change.
"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "059"
down_revision = "058"
branch_labels = None
depends_on = None


def upgrade():
    op.alter_column("job", "created_at", new_column_name="inserted_at")
    op.alter_column("llm_call", "created_at", new_column_name="inserted_at")


def downgrade():
    op.alter_column("llm_call", "inserted_at", new_column_name="created_at")
    op.alter_column("job", "inserted_at", new_column_name="created_at")
