"""add job.meta JSONB column and LLM_GUARDRAILS jobtype enum value

Revision ID: 067
Revises: 066
Create Date: 2026-06-15 00:00:00.000000

"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "067"
down_revision = "066"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add new enum value for guardrails-only jobs. ALTER TYPE ... ADD VALUE
    # cannot run inside a transaction block, hence the autocommit guard.
    with op.get_context().autocommit_block():
        op.execute("ALTER TYPE jobtype ADD VALUE IF NOT EXISTS 'LLM_GUARDRAILS'")

    # Add nullable meta JSONB column on job for per-job-type tracking payloads
    # (e.g. guardrails request/response). Nullable + no default so it imposes
    # zero cost on existing rows and on job types that do not write to it.
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
    # NOTE: Postgres has no clean way to remove a single enum value. Leaving
    # 'LLM_GUARDRAILS' on the type on downgrade is intentional and harmless.
