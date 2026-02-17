"""backfill legacy config completion type

Revision ID: 047
Revises: 046
Create Date: 2026-02-17 00:00:00.000000

"""
from alembic import op

revision = "047"
down_revision = "046"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        UPDATE config_version
        SET config_blob = jsonb_set(config_blob, '{completion,type}', '"text"')
        WHERE config_blob->'completion' IS NOT NULL
          AND config_blob->'completion'->>'type' IS NULL
        """
    )


def downgrade() -> None:
    # No-op: removing type='text' would drop a required field and break validation; NULL safely defaults to 'text'.
    pass
