"""add composite index for conversation query optimization

Revision ID: 044
Revises: 043
Create Date: 2026-02-04 15:24:00.000000

"""
from alembic import op


# revision identifiers, used by Alembic.
revision = "044"
down_revision = "043"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create composite index to optimize the get_conversation_by_ancestor_id query
    # This query filters by: ancestor_response_id, project_id, is_deleted
    # and orders by: inserted_at DESC
    op.create_index(
        "ix_openai_conversation_ancestor_project_active_time",
        "openai_conversation",
        ["ancestor_response_id", "project_id", "is_deleted", "inserted_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_openai_conversation_ancestor_project_active_time",
        table_name="openai_conversation",
    )
