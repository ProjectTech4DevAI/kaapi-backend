"""Add name to collection_jobs with in-flight unique index

Revision ID: 087
Revises: 086
Create Date: 2026-10-28 00:00:00.000000
"""

import sqlalchemy as sa
from alembic import op

revision = "087"
down_revision = "086"
branch_labels = None
depends_on = None

INDEX_NAME = "uq_collection_jobs_project_id_name_in_flight"


def upgrade():
    op.add_column(
        "collection_jobs",
        sa.Column(
            "name",
            sa.String(),
            nullable=True,
            comment="Requested collection name; keeps in-flight CREATE jobs unique per project",
        ),
    )
    op.create_index(
        INDEX_NAME,
        "collection_jobs",
        ["project_id", "name"],
        unique=True,
        postgresql_where=sa.text(
            "action_type = 'CREATE' AND status IN ('PENDING', 'PROCESSING')"
        ),
    )


def downgrade():
    op.drop_index(INDEX_NAME, table_name="collection_jobs")
    op.drop_column("collection_jobs", "name")
