"""add config in evals run table

Revision ID: 041
Revises: 040
Create Date: 2025-12-15 14:03:22.082746

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "041"
down_revision = "040"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "evaluation_run",
        sa.Column(
            "config_id",
            sa.Uuid(),
            nullable=True,
            comment="Reference to the stored config used",
        ),
    )
    op.add_column(
        "evaluation_run",
        sa.Column(
            "config_version",
            sa.Integer(),
            nullable=True,
            comment="Version of the config used",
        ),
    )
    op.create_foreign_key(
        "fk_evaluation_run_config_id", "evaluation_run", "config", ["config_id"], ["id"]
    )
    op.drop_column("evaluation_run", "config")


def downgrade():
    op.add_column(
        "evaluation_run",
        sa.Column(
            "config",
            postgresql.JSONB(astext_type=sa.Text()),
            autoincrement=False,
            nullable=False,
            comment="Evaluation configuration (model, instructions, etc.)",
        ),
    )
    op.drop_constraint(
        "fk_evaluation_run_config_id", "evaluation_run", type_="foreignkey"
    )
    op.drop_column("evaluation_run", "config_version")
    op.drop_column("evaluation_run", "config_id")
