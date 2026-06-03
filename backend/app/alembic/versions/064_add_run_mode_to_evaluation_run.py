"""add run_mode column and unique run-name constraint to evaluation_run

Revision ID: 064
Revises: 063
Create Date: 2026-05-20 00:00:00.000000

Two schema changes are required to support the fast-evaluation feature:

1. run_mode VARCHAR(10) NOT NULL DEFAULT 'batch'
   Every evaluation run must now carry an explicit execution mode ('batch'
   or 'fast').  Because evaluation_run may already contain rows the column
   is added as nullable with a server_default of 'batch' so existing rows
   are backfilled atomically, then SET NOT NULL is applied.  The server
   default is kept on the column to act as a safety net for any insert path
   that does not explicitly set the field; the SQLModel default="batch" will
   also cover application inserts.

2. UNIQUE constraint uq_evaluation_run_org_project_run_name
   on (organization_id, project_id, run_name)
   Guards against double-click races on POST /evaluations where two
   concurrent requests could create duplicate runs with the same name inside
   the same project.  Lower environments may already contain duplicate
   (organization_id, project_id, run_name) tuples, so we dedupe first
   (keeping the lowest-id survivor), then build the unique index
   CONCURRENTLY (no AccessExclusiveLock on the table during the build) and
   attach it as a named constraint via ADD CONSTRAINT ... USING INDEX.
   Because CONCURRENTLY cannot execute inside a transaction the whole
   migration is marked disable_per_migration_transaction = True and each
   CONCURRENTLY step is wrapped in autocommit_block().  The dedupe DELETE and
   the non-concurrent DDL steps run inside autocommit mode too, which is
   safe — they are each individually atomic at the statement level.
"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "064"
down_revision = "063"
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

    # 2. Dedupe existing rows before adding the unique constraint.
    #    Keep the lowest-id row for each (organization_id, project_id,
    #    run_name) tuple and remove the rest.
    with op.get_context().autocommit_block():
        op.execute(
            """
            DELETE FROM evaluation_run
            WHERE id IN (
                SELECT id
                FROM (
                    SELECT id,
                           ROW_NUMBER() OVER (
                               PARTITION BY organization_id, project_id, run_name
                               ORDER BY id ASC
                           ) AS rn
                    FROM evaluation_run
                ) sub
                WHERE rn > 1
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
