"""Add callback_url to evaluation_run

Revision ID: 076
Revises: 075
Create Date: 2026-08-06 00:00:00.000000

v2 runs may register an HTTPS webhook that is POSTed the run's result once it
reaches a terminal state. The completion hook lives in the CRUD layer and only
knows an eval_run_id, so the target URL has to be durable on the run row.
"""

import sqlalchemy as sa
from alembic import op

revision = "076"
down_revision = "075"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "evaluation_run",
        sa.Column(
            "callback_url",
            sa.Text(),
            nullable=True,
            comment=(
                "Optional HTTPS webhook (v2 runs only) POSTed a slim run snapshot "
                "(id/run_name/dataset_name/status/run_mode/timestamps) once the run "
                "reaches a terminal state (completed/failed). NULL when no callback "
                "was requested"
            ),
        ),
    )


def downgrade():
    op.drop_column("evaluation_run", "callback_url")
