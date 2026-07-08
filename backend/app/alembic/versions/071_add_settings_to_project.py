"""add settings jsonb column to project

Revision ID: 071
Revises: 070
Create Date: 2026-06-17

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = "071"
down_revision = "070"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "project",
        sa.Column(
            "settings",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
            comment=(
                "Project-level settings (JSONB). Keys: 'tracing' (bool) — "
                "Langfuse tracing opt-in, off by default to conserve Langfuse "
                "rate-limit/credit budget. Gates tracing for both the response "
                "path and evaluations (which fall back to cosine-only scoring)."
            ),
        ),
    )


def downgrade():
    op.drop_column("project", "settings")
