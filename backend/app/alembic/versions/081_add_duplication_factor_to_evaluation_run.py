"""Add duplication_factor override to evaluation_run

Revision ID: 081
Revises: 080
Create Date: 2026-08-17 00:00:00.000000

A v2 run may override the dataset's stored duplication_factor for that run only.
The factor is read in three places, two of which (the chunk re-load and the
ai_summary math) only know an eval_run_id, so the override has to be durable on
the run row for the fan-out sizing and the chunk re-load to agree. NULL keeps the
existing behavior (dataset's stored factor, forced to 1 for v1/Langfuse datasets).
"""

import sqlalchemy as sa
from alembic import op

revision = "081"
down_revision = "080"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "evaluation_run",
        sa.Column(
            "duplication_factor",
            sa.Integer(),
            nullable=True,
            comment=(
                "Per-run override of the dataset's stored duplication_factor for "
                "runtime-duplicated (v2) datasets. Read by fan-out sizing, the "
                "chunk re-load, and the ai_summary math so all three agree. NULL = "
                "use the dataset's stored factor (v1/Langfuse forced to 1)"
            ),
        ),
    )


def downgrade() -> None:
    op.drop_column("evaluation_run", "duplication_factor")
