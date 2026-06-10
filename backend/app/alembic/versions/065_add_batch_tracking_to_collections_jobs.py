"""add batch tracking to collection_jobs

Revision ID: 065
Revises: 064
Create Date: 2026-04-13

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "065"
down_revision = "064"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "collection_jobs",
        sa.Column(
            "total_batches",
            sa.Integer(),
            nullable=True,
            comment="Total number of batches the documents are split into",
        ),
    )
    op.add_column(
        "collection_jobs",
        sa.Column(
            "current_batch_number",
            sa.Integer(),
            nullable=True,
            comment="Which batch is currently being processed (1-indexed)",
        ),
    )
    op.add_column(
        "collection_jobs",
        sa.Column(
            "documents_uploaded",
            sa.JSON(),
            nullable=True,
            comment="List of document IDs successfully uploaded so far",
        ),
    )
    op.add_column(
        "document",
        sa.Column(
            "openai_file_id",
            sa.String(),
            nullable=True,
            comment="File ID assigned by the LLM provider (e.g. OpenAI file ID) to avoid re-uploading",
        ),
    )


def downgrade():
    op.drop_column("collection_jobs", "total_batches")
    op.drop_column("collection_jobs", "current_batch_number")
    op.drop_column("collection_jobs", "documents_uploaded")
    op.drop_column("document", "openai_file_id")
