"""cascade document foreign keys on delete

Revision ID: 093
Revises: 092
Create Date: 2026-11-26 00:00:00.000000

Permanent document delete now removes the document row, so rows that point at it
(transformed documents, transformation jobs, fine-tuning jobs, model evaluations)
go with it instead of blocking the delete. These FKs were created unnamed, so they
carry Postgres's default <table>_<column>_fkey names.

document.source_document_id was the only cascading FK without an index; without
it every document delete seq-scans the document table to find children.
"""

from alembic import op

revision = "093"
down_revision = "092"
branch_labels = None
depends_on = None

DOCUMENT_FKS = [
    ("document", "source_document_id"),
    ("doc_transformation_job", "source_document_id"),
    ("doc_transformation_job", "transformed_document_id"),
    ("fine_tuning", "document_id"),
    ("model_evaluation", "document_id"),
]
SOURCE_DOCUMENT_INDEX = "ix_document_source_document_id"


def _recreate_fks(ondelete: str | None) -> None:
    for table, column in DOCUMENT_FKS:
        name = f"{table}_{column}_fkey"
        op.drop_constraint(name, table, type_="foreignkey")
        op.create_foreign_key(
            name, table, "document", [column], ["id"], ondelete=ondelete
        )


def upgrade():
    with op.get_context().autocommit_block():
        op.execute(
            f'CREATE INDEX CONCURRENTLY IF NOT EXISTS "{SOURCE_DOCUMENT_INDEX}" '
            'ON "document" ("source_document_id")'
        )
    _recreate_fks("CASCADE")


def downgrade():
    _recreate_fks(None)
    with op.get_context().autocommit_block():
        op.execute(f'DROP INDEX CONCURRENTLY IF EXISTS "{SOURCE_DOCUMENT_INDEX}"')
