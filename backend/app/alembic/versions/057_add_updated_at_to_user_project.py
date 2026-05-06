"""add updated_at to user_project

Revision ID: 057
Revises: 056
Create Date: 2026-05-06 12:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "057"
down_revision = "056"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "user_project",
        sa.Column(
            "updated_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.text("NOW()"),
            comment="Timestamp when the mapping was last updated",
        ),
    )
    op.alter_column("user_project", "updated_at", server_default=None)


def downgrade():
    op.drop_column("user_project", "updated_at")
