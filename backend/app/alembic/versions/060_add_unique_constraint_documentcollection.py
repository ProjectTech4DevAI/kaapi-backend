"""add unique constraint to documentcollection

Revision ID: 060
Revises: 059
Create Date: 2026-05-07 12:00:00.000000

The `documentcollection` junction table never had a uniqueness constraint
on (document_id, collection_id), so the same document could be linked to
the same collection multiple times. This migration:

  1. Removes any existing duplicate rows, keeping the row with the lowest
     `id` for each (document_id, collection_id) pair.
  2. Adds the unique constraint going forward.
"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "060"
down_revision = "059"
branch_labels = None
depends_on = None


def upgrade():
    op.execute(
        """
        DELETE FROM documentcollection
        WHERE id NOT IN (
            SELECT MIN(id)
            FROM documentcollection
            GROUP BY document_id, collection_id
        )
        """
    )
    op.create_unique_constraint(
        "uq_document_collection",
        "documentcollection",
        ["document_id", "collection_id"],
    )


def downgrade():
    op.drop_constraint("uq_document_collection", "documentcollection", type_="unique")
