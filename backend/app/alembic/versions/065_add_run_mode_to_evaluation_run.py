"""add run_mode column and unique run-name constraint to evaluation_run

Revision ID: 065
Revises: 064
Create Date: 2026-05-20 00:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "065"
down_revision = "064"
branch_labels = None
depends_on = None

disable_per_migration_transaction = True

_UNIQUE_INDEX = "uq_evaluation_run_org_project_run_name"
_UNIQUE_CONSTRAINT = "uq_evaluation_run_org_project_run_name"


def upgrade():
    # 1. Add run_mode as nullable first so existing rows are backfilled by the
    #    server default, then tighten to NOT NULL.  The server default is left
    #    in place as a safety net.
    with op.get_context().autocommit_block():
        op.add_column(
            "evaluation_run",
            sa.Column(
                "run_mode",
                sa.String(length=10),
                nullable=True,
                server_default=sa.text("'batch'"),
                comment="Execution mode: batch or fast",
            ),
        )
        op.execute("ALTER TABLE evaluation_run ALTER COLUMN run_mode SET NOT NULL")

    # 2. Resolve duplicate (organization_id, project_id, run_name) tuples
    #    non-destructively before adding the unique constraint. Keep the
    #    lowest-id row's run_name untouched and rename the rest by appending a
    #    unique "__dup_<id>" suffix so no historical run (and its scores, result
    #    URLs, or batch_job links) is lost.
    with op.get_context().autocommit_block():
        op.execute(
            """
            UPDATE evaluation_run e
            SET run_name = e.run_name || '__dup_' || e.id
            WHERE e.id <> (
                SELECT MIN(x.id)
                FROM evaluation_run x
                WHERE x.organization_id = e.organization_id
                  AND x.project_id = e.project_id
                  AND x.run_name = e.run_name
            )
            """
        )

    # 3. Build the unique index CONCURRENTLY so the scan does not take an
    #    AccessExclusiveLock, then attach it as a named constraint via
    #    ADD CONSTRAINT ... USING INDEX (brief catalog-only lock).
    with op.get_context().autocommit_block():
        op.execute(
            f"CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS "
            f'"{_UNIQUE_INDEX}" '
            f"ON evaluation_run (organization_id, project_id, run_name)"
        )
    op.execute(
        f"ALTER TABLE evaluation_run "
        f'ADD CONSTRAINT "{_UNIQUE_CONSTRAINT}" '
        f'UNIQUE USING INDEX "{_UNIQUE_INDEX}"'
    )


def downgrade():
    # Reverse in opposite order to upgrade().
    op.execute(
        f"ALTER TABLE evaluation_run "
        f'DROP CONSTRAINT IF EXISTS "{_UNIQUE_CONSTRAINT}"'
    )
    with op.get_context().autocommit_block():
        op.execute(f'DROP INDEX CONCURRENTLY IF EXISTS "{_UNIQUE_INDEX}"')
    op.drop_column("evaluation_run", "run_mode")
