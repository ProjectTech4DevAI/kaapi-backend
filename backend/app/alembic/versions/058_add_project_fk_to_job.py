"""add project_id foreign key to job table

Revision ID: 058
Revises: 057
Create Date: 2026-05-06 13:00:00.000000

Migration 051 added job.project_id as a plain Integer with no foreign key
constraint, leaving the column without referential integrity. This migration:

  1. Backfills orphan rows: any job.project_id that doesn't match a real
     project.id is set to NULL (the column is nullable). This preserves
     historical job records whose project was deleted before the FK existed.
     Switch the cleanup to a DELETE if you'd rather discard orphans
     retroactively under CASCADE semantics.

  2. Adds the foreign key constraint with ON DELETE CASCADE, matching the
     pattern used by every other project_id FK in the schema.

The supporting index (ix_job_project_id) is created by migration 055.
"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "058"
down_revision = "057"
branch_labels = None
depends_on = None


def upgrade():
    op.execute(
        """
        UPDATE job
        SET project_id = NULL
        WHERE project_id IS NOT NULL
          AND project_id NOT IN (SELECT id FROM project)
        """
    )
    op.create_foreign_key(
        "job_project_id_fkey",
        "job",
        "project",
        ["project_id"],
        ["id"],
        ondelete="CASCADE",
    )


def downgrade():
    op.drop_constraint("job_project_id_fkey", "job", type_="foreignkey")
