"""clear stale 'Checking failed' error on completed fast evaluation runs

The batch poller used to pick up in-flight fast evaluation runs and mark them
with "Checking failed: ... has no batch_job_id". The run then completed
successfully but kept the stale error_message, so it displayed as COMPLETED
with a red error. Clear that message on completed fast runs.

Revision ID: 068
Revises: 067
Create Date: 2026-06-16

"""
from alembic import op


# revision identifiers, used by Alembic.
revision = "068"
down_revision = "067"
branch_labels = None
depends_on = None


def upgrade():
    op.execute(
        """
        UPDATE evaluation_run
        SET error_message = NULL
        WHERE run_mode = 'fast'
          AND status = 'completed'
          AND error_message LIKE 'Checking failed:%'
        """
    )


def downgrade():
    # Data-only cleanup; the cleared messages are not restorable.
    pass
