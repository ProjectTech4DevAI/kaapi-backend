"""add columns to collection job and documents table

Revision ID: 051
Revises: 050
Create Date: 2026-03-25 10:09:47.318575

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "051"
down_revision = "050"
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
            comment="Total size of documents being uploaded to collection in MB",
        ),
    )
    op.add_column(
        "collection_jobs",
        sa.Column(
            "documents",
            sa.JSON(),
            nullable=True,
            comment="List of documents given to make collection",
        ),
    )
    op.add_column(
        "document",
        sa.Column(
            "file_size_kb",
            sa.Float(),
            nullable=True,
            comment="Size of the document in kilobytes (KB)",
        ),
    )


def downgrade():
    op.drop_column("document", "file_size_kb")
    op.drop_column("collection_jobs", "total_size_mb")
    op.drop_column("collection_jobs", "docs_num")
    op.drop_column("collection_jobs", "documents")
