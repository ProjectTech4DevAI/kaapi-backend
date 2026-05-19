"""add pending job monitoring indexes

Revision ID: 061
Revises: 060
Create Date: 2026-05-13 00:00:00.000000

"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "062"
down_revision = "061"
branch_labels = None
depends_on = None

disable_per_migration_transaction = True


INDEXES = [
    ("ix_job_status_inserted_at", "job", ["status", "inserted_at"]),
    (
        "ix_job_status_job_type_inserted_at",
        "job",
        ["status", "job_type", "inserted_at"],
    ),
    (
        "ix_collection_jobs_status_inserted_at",
        "collection_jobs",
        ["status", "inserted_at"],
    ),
    (
        "ix_collection_jobs_status_action_type_inserted_at",
        "collection_jobs",
        ["status", "action_type", "inserted_at"],
    ),
    (
        "ix_doc_transformation_job_status_inserted_at",
        "doc_transformation_job",
        ["status", "inserted_at"],
    ),
]


def upgrade():
    with op.get_context().autocommit_block():
        for name, table, cols in INDEXES:
            op.create_index(
                name,
                table,
                cols,
                unique=False,
                postgresql_concurrently=True,
                if_not_exists=True,
            )


def downgrade():
    with op.get_context().autocommit_block():
        for name, table, _cols in reversed(INDEXES):
            op.drop_index(
                name,
                table_name=table,
                postgresql_concurrently=True,
                if_exists=True,
            )
