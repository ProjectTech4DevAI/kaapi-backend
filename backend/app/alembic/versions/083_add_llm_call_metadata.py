"""Add llm_call.metadata (generic extensibility catch-all)

Revision ID: 083
Revises: 082
Create Date: 2026-09-08 00:00:00.000000

Re-adds a `metadata` JSONB column on `llm_call`, previously dropped by 079
(as collateral of an unrelated feature revert, not because the column was a
bad idea). This time it's a generic catch-all, mirroring `llm_chain.metadata`
(added via `metadata_` in the model to dodge SQLAlchemy's reserved
`Base.metadata` attribute) — first use case: persisting input/output
guardrail results so /llm/call polling (GET /llm/call/{job_id}) can surface
them, matching what's already sent on the callback payload's `metadata`
field.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "083"
down_revision = "082"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "llm_call",
        sa.Column(
            "metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
            comment="Future-proof extensibility catch-all (e.g. guardrail results)",
        ),
    )


def downgrade():
    op.drop_column("llm_call", "metadata")
