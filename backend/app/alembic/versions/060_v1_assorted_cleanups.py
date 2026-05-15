"""v1.0 assorted schema cleanups

Revision ID: 060
Revises: 059
Create Date: 2026-05-07 14:00:00.000000

Bundles five small, mutually independent v1.0 cleanups in source order:

  1. user_project + user: add timestamp columns missing from the model.
     `user_project` gets `updated_at`; `user` gets both `inserted_at`
     and `updated_at`. Each new column is added with NOW() server
     default so existing rows are backfilled, then the default is
     dropped so future inserts use the model's `default_factory=now`.

  2. job: backfill orphan project_id rows to NULL, then add the missing
     foreign key constraint with ON DELETE CASCADE. The supporting
     ix_job_project_id index is created by migration 058.

  3. job + llm_call: rename `created_at` → `inserted_at` to align with
     the rest of the schema (every other table uses `inserted_at`).
     Also creates the partial index ix_llm_call_job_inserted_at_active
     here (rather than in migration 058) so the index name reflects the
     post-rename column.

  4. documentcollection: dedupe any existing duplicate (document_id,
     collection_id) pairs (keeps the lowest id), then add the missing
     unique constraint. The unique index is built CONCURRENTLY and then
     attached via ADD CONSTRAINT ... USING INDEX so the index build does
     not take AccessExclusiveLock on the table.

  5. global.languages: align id column type to INTEGER. Migration 043
     originally created it as BIGINT, but every FK column referencing it
     is INTEGER. The IDENTITY sequence stays BIGINT (PG doesn't change
     it on ALTER COLUMN TYPE) — harmless at this scale.
"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "060"
down_revision = "059"
branch_labels = None
depends_on = None


def upgrade():
    # 1. Backfill missing timestamp columns. Each column is created with
    #    a NOW() server default so existing rows are populated atomically;
    #    the default is then dropped so future inserts get their value
    #    from the model's `default_factory=now`.
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

    op.add_column(
        "user",
        sa.Column(
            "inserted_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.text("NOW()"),
            comment="Timestamp when the user was created",
        ),
    )
    op.alter_column("user", "inserted_at", server_default=None)

    op.add_column(
        "user",
        sa.Column(
            "updated_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.text("NOW()"),
            comment="Timestamp when the user was last updated",
        ),
    )
    op.alter_column("user", "updated_at", server_default=None)

    # 2. job.project_id foreign key (with orphan backfill)
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

    # 3. Rename created_at → inserted_at on job and llm_call, then create
    #    the llm_call hot-path index using the new column name. Index
    #    creation is CONCURRENTLY and must run outside a transaction.
    op.alter_column("job", "created_at", new_column_name="inserted_at")
    op.alter_column("llm_call", "created_at", new_column_name="inserted_at")
    with op.get_context().autocommit_block():
        op.execute(
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS "
            '"ix_llm_call_job_inserted_at_active" '
            'ON "llm_call" ("job_id", "inserted_at" DESC) '
            'WHERE "deleted_at" IS NULL'
        )

    # 4. documentcollection unique constraint (with dedupe).
    #    Build the underlying unique index CONCURRENTLY (no AccessExclusive
    #    on the table during the scan/build), then attach it as a constraint
    #    via ADD CONSTRAINT ... USING INDEX (catalog-only, brief lock).
    op.execute(
        """
        DELETE FROM documentcollection
        WHERE id NOT IN (
            SELECT MIN(id)
            FROM documentcollection
            GROUP BY document_id, collection_id
        )
        """
    )
    with op.get_context().autocommit_block():
        op.execute(
            "CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS "
            '"uq_document_collection" '
            'ON "documentcollection" ("document_id", "collection_id")'
        )
    op.execute(
        'ALTER TABLE "documentcollection" '
        'ADD CONSTRAINT "uq_document_collection" '
        'UNIQUE USING INDEX "uq_document_collection"'
    )

    # 5. Align global.languages.id to INTEGER
    op.execute("ALTER TABLE global.languages ALTER COLUMN id SET DATA TYPE INTEGER")


def downgrade():
    # Reverse order of upgrade()
    op.execute("ALTER TABLE global.languages ALTER COLUMN id SET DATA TYPE BIGINT")
    op.drop_constraint("uq_document_collection", "documentcollection", type_="unique")
    with op.get_context().autocommit_block():
        op.execute(
            'DROP INDEX CONCURRENTLY IF EXISTS "ix_llm_call_job_inserted_at_active"'
        )
    op.alter_column("llm_call", "inserted_at", new_column_name="created_at")
    op.alter_column("job", "inserted_at", new_column_name="created_at")
    op.drop_constraint("job_project_id_fkey", "job", type_="foreignkey")
    op.drop_column("user", "updated_at")
    op.drop_column("user", "inserted_at")
    op.drop_column("user_project", "updated_at")
