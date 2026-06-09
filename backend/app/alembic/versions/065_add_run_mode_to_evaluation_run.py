"""add run_mode column and unique run-name constraint to evaluation_run

Revision ID: 065
Revises: 064
Create Date: 2026-05-20 00:00:00.000000

"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "065"
down_revision = "064"
branch_labels = None
depends_on = None

disable_per_migration_transaction = True

_UNIQUE_INDEX = "uq_evaluation_run_org_project_run_name"
_UNIQUE_CONSTRAINT = "uq_evaluation_run_org_project_run_name"
_RUN_MODE_ENUM_NAME = "run_mode_enum"
_RUN_MODE_VALUES = ("batch", "fast")


def upgrade():
    run_mode_enum = postgresql.ENUM(
        *_RUN_MODE_VALUES,
        name=_RUN_MODE_ENUM_NAME,
        create_type=False,
    )
    run_mode_enum.create(op.get_bind(), checkfirst=True)

    with op.get_context().autocommit_block():
        op.add_column(
            "evaluation_run",
            sa.Column(
                "run_mode",
                run_mode_enum,
                nullable=False,
                server_default=sa.text(f"'batch'::{_RUN_MODE_ENUM_NAME}"),
                comment="Execution mode: batch or fast",
            ),
        )

    # Rename duplicate (org, project, run_name) rows (keeping the lowest id) so
    # the unique constraint can be added without dropping any history.
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

    # Build the index CONCURRENTLY (avoids an AccessExclusiveLock), then attach
    # it as the named unique constraint.
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
    op.execute(
        f"ALTER TABLE evaluation_run "
        f'DROP CONSTRAINT IF EXISTS "{_UNIQUE_CONSTRAINT}"'
    )
    with op.get_context().autocommit_block():
        op.execute(f'DROP INDEX CONCURRENTLY IF EXISTS "{_UNIQUE_INDEX}"')
    op.drop_column("evaluation_run", "run_mode")
    sa.Enum(name=_RUN_MODE_ENUM_NAME).drop(op.get_bind(), checkfirst=True)
