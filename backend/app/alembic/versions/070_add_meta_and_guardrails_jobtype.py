"""add job.meta JSONB column and LLM_GUARDRAILS jobtype enum value

Revision ID: 070
Revises: 069
Create Date: 2026-06-15 00:00:00.000000

"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "070"
down_revision = "069"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute("ALTER TYPE jobtype ADD VALUE IF NOT EXISTS 'LLM_GUARDRAILS'")

    op.add_column(
        "job",
        sa.Column(
            "meta",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
            comment=(
                "Per-job-type tracking payload. For LLM_GUARDRAILS this stores "
                "{'request': {...}, 'response': {...}} capturing the inbound "
                "guardrails request and the upstream guardrails service response."
            ),
        ),
    )


def downgrade() -> None:
    op.drop_column("job", "meta")
