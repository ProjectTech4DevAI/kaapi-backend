"""Add last_dispatched_at to evaluation_iteration_run

Revision ID: 083
Revises: 082
Create Date: 2026-09-08 00:00:00.000000

The cron tick used to fan a `resume=True` graph step out to every PROCESSING
loop unconditionally, so a step slower than the tick interval got a second one
dispatched on top of it — two workers against the same LangGraph checkpoint
thread, and a duplicate eval run or improvement job charged for.

Nullable with no backfill on purpose: NULL means "never dispatched", which is
exactly what a freshly created loop needs so its first tick isn't skipped. Rows
that predate this migration read as NULL and get one immediate resume, which is
the behaviour they had anyway.

No index — the PROCESSING set is small and already indexed on `status`; the
cooldown comparison happens in Python, mirroring the fast-eval barrier.
"""

import sqlalchemy as sa
from alembic import op

revision = "083"
down_revision = "082"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "evaluation_iteration_run",
        sa.Column(
            "last_dispatched_at",
            sa.DateTime(),
            nullable=True,
            comment="When the cron last dispatched a resume; NULL until the first tick",
        ),
    )


def downgrade():
    op.drop_column("evaluation_iteration_run", "last_dispatched_at")
