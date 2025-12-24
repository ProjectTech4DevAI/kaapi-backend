"""adding blob column in collection table

Revision ID: 041
Revises: 040
Create Date: 2025-12-24 11:03:44.620424

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "041"
down_revision = "040"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "collection",
        sa.Column(
            "collection_blob",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
            comment="Provider-specific knowledge base creation parameters (name, description, chunking params etc.)",
        ),
    )
    op.alter_column(
        "collection",
        "llm_service_name",
        existing_type=sa.VARCHAR(),
        comment="Name of the LLM service",
        existing_comment="Name of the LLM service provider",
        existing_nullable=False,
    )


def downgrade():
    op.alter_column(
        "collection",
        "llm_service_name",
        existing_type=sa.VARCHAR(),
        comment="Name of the LLM service provider",
        existing_comment="Name of the LLM service",
        existing_nullable=False,
    )
    op.drop_column("collection", "collection_blob")
