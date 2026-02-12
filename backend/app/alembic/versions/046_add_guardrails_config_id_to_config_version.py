"""Add guardrails_config_id to config_version

Revision ID: 046
Revises: 045
Create Date: 2026-02-12 12:30:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "046"
down_revision = "045"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "config_version",
        sa.Column(
            "guardrails_config_id",
            sa.Uuid(),
            nullable=True,
            comment="Reference to the kaapi_guardrails validator configuration",
        ),
    )


def downgrade():
    op.drop_column("config_version", "guardrails_config_id")
