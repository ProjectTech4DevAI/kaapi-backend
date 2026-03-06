"""Create llm_chain table

Revision ID: 048
Revises: 047
Create Date: 2026-02-20 00:00:00.000000

"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "048"
down_revision = "047"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. Create llm_chain table
    op.create_table(
        "llm_chain",
        sa.Column(
            "id",
            sa.Uuid(),
            nullable=False,
            comment="Unique identifier for the LLM chain record",
        ),
        sa.Column(
            "job_id",
            sa.Uuid(),
            nullable=False,
            comment="Reference to the parent job (status tracked in job table)",
        ),
        sa.Column(
            "project_id",
            sa.Integer(),
            nullable=False,
            comment="Reference to the project this LLM call belongs to",
        ),
        sa.Column(
            "organization_id",
            sa.Integer(),
            nullable=False,
            comment="Reference to the organization this LLM call belongs to",
        ),
        sa.Column(
            "status",
            sa.String(),
            nullable=False,
            server_default="pending",
            comment="Chain execution status (pending, running, failed, completed)",
        ),
        sa.Column(
            "error",
            sa.Text(),
            nullable=True,
            comment="Error message if the chain execution failed",
        ),
        sa.Column(
            "block_sequences",
            JSONB(),
            nullable=True,
            comment="Ordered list of llm_call UUIDs as blocks complete",
        ),
        sa.Column(
            "total_blocks",
            sa.Integer(),
            nullable=False,
            comment="Total number of blocks to execute",
        ),
        sa.Column(
            "number_of_blocks_processed",
            sa.Integer(),
            nullable=False,
            server_default="0",
            comment="Number of blocks processed so far (used for tracking progress)",
        ),
        sa.Column(
            "input",
            sa.String(),
            nullable=False,
            comment="First block user's input - text string, binary data, or file path for multimodal",
        ),
        sa.Column(
            "output",
            JSONB(),
            nullable=True,
            comment="Last block's final output (set on chain completion)",
        ),
        sa.Column(
            "configs",
            JSONB(),
            nullable=True,
            comment="Ordered list of block configs as submitted in the request",
        ),
        sa.Column(
            "total_usage",
            JSONB(),
            nullable=True,
            comment="Aggregated token usage: {input_tokens, output_tokens, total_tokens}",
        ),
        sa.Column(
            "metadata",
            JSONB(),
            nullable=True,
            comment="Future-proof extensibility catch-all",
        ),
        sa.Column(
            "inserted_at",
            sa.DateTime(),
            nullable=False,
            comment="Timestamp when the chain record was created",
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(),
            nullable=False,
            comment="Timestamp when the chain record was last updated",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["job_id"], ["job.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["project_id"], ["project.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["organization_id"], ["organization.id"], ondelete="CASCADE"
        ),
    )

    op.create_index(
        "idx_llm_chain_job_id",
        "llm_chain",
        ["job_id"],
    )

    # 2. Add chain_id FK column to llm_call table
    op.add_column(
        "llm_call",
        sa.Column(
            "chain_id",
            sa.Uuid(),
            nullable=True,
            comment="Reference to the parent chain (NULL for standalone /llm/call requests)",
        ),
    )
    op.create_foreign_key(
        "fk_llm_call_chain_id",
        "llm_call",
        "llm_chain",
        ["chain_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index(
        "idx_llm_call_chain_id",
        "llm_call",
        ["chain_id"],
        postgresql_where=sa.text("chain_id IS NOT NULL"),
    )

    op.execute("ALTER TYPE jobtype ADD VALUE IF NOT EXISTS 'LLM_CHAIN'")


def downgrade() -> None:
    op.drop_index("idx_llm_call_chain_id", table_name="llm_call")
    op.drop_constraint("fk_llm_call_chain_id", "llm_call", type_="foreignkey")
    op.drop_column("llm_call", "chain_id")

    op.drop_index("idx_llm_chain_job_id", table_name="llm_chain")
    op.drop_table("llm_chain")
