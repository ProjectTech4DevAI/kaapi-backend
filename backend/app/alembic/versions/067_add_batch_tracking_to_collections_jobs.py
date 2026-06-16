"""add batch tracking to collection_jobs and rename collection knowledge base columns

Revision ID: 067
Revises: 066
Create Date: 2026-04-13

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = "067"
down_revision = "066"
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
        "collection_jobs",
        sa.Column(
            "knowledge_base_id",
            sa.String(),
            nullable=True,
            comment="Provider knowledge base ID (e.g. OpenAI vector store ID) created during setup; batch tasks attach files to it",
        ),
    )
    op.add_column(
        "document",
        sa.Column(
            "file_id",
            postgresql.JSONB(),
            nullable=True,
            comment='Provider-keyed file IDs, e.g. {"openai": "file-abc"}, to avoid re-uploading',
        ),
    )
    op.alter_column(
        "collection",
        "llm_service_id",
        new_column_name="knowledge_base_id",
        existing_type=sa.String(),
        existing_nullable=False,
        comment="Provider knowledge base ID (e.g. OpenAI vector store ID)",
        existing_comment="External LLM service identifier (e.g., OpenAI vector store ID)",
    )
    op.alter_column(
        "collection",
        "llm_service_name",
        new_column_name="knowledge_base_provider",
        existing_type=sa.String(),
        existing_nullable=False,
        comment="Name of the knowledge base provider service",
        existing_comment="Name of the LLM service provider",
    )


def downgrade():
    op.alter_column(
        "collection",
        "knowledge_base_provider",
        new_column_name="llm_service_name",
        existing_type=sa.String(),
        existing_nullable=False,
        comment="Name of the LLM service provider",
        existing_comment="Name of the knowledge base provider service",
    )
    op.alter_column(
        "collection",
        "knowledge_base_id",
        new_column_name="llm_service_id",
        existing_type=sa.String(),
        existing_nullable=False,
        comment="External LLM service identifier (e.g., OpenAI vector store ID)",
        existing_comment="Provider knowledge base ID (e.g. OpenAI vector store ID)",
    )
    op.drop_column("document", "file_id")
    op.drop_column("collection_jobs", "knowledge_base_id")
    op.drop_column("collection_jobs", "total_batches")
    op.drop_column("collection_jobs", "current_batch_number")
    op.drop_column("collection_jobs", "documents_uploaded")
