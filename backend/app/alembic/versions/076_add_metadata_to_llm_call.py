"""Add metadata to llm_call

Revision ID: 076
Revises: 075
Create Date: 2026-08-10 00:00:00.000000

`llm_call.provider` records what the client asked for; platform routing
(GEMINI_DEFAULT_INFERENCE_ROUTE) can send "google" to a different backend, so
monitoring needs the resolved backend. Left nullable with no backfill: rows
predating routing metadata have no recoverable effective provider and are
grouped as unknown/legacy.

The model attribute is `call_metadata` because `metadata` is reserved on
SQLModel declarative classes; the column itself is plain `metadata`.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "076"
down_revision = "075"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "llm_call",
        sa.Column(
            "metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
            comment=(
                "Platform metadata: {effective_provider} - backend actually used after "
                "routing (NULL for rows predating routing metadata)"
            ),
        ),
    )


def downgrade():
    op.drop_column("llm_call", "metadata")
