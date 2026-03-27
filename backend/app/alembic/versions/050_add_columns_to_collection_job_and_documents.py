"""add columns to collection job and documents table

Revision ID: 050
Revises: 049
Create Date: 2026-03-25 10:09:47.318575

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "050"
down_revision = "049"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "collection_jobs",
        sa.Column(
            "docs_num",
            sa.Integer(),
            nullable=True,
            comment="Total number of documents to be processed in this job",
        ),
    )
    op.add_column(
        "collection_jobs",
        sa.Column(
            "total_size_mb",
            sa.Float(),
            nullable=True,
            comment="Total size of documents being uploaded to collection",
        ),
    )
    op.add_column(
        "document",
        sa.Column(
            "file_size_kb",
            sa.Float(),
            nullable=True,
            comment="Size of the document in bytes",
        ),
    )
    op.add_column(
        "collection_jobs",
        sa.Column(
            "documents",
            sa.JSON(),
            nullable=True,
            comment="JSON array of document UUIDs included in this job",
        ),
    )


def downgrade():
    op.drop_column("document", "file_size_kb")
    op.drop_column("collection_jobs", "total_size_mb")
    op.drop_column("collection_jobs", "docs_num")
    op.drop_column("collection_jobs", "documents")
