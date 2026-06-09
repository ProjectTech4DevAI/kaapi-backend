"""completion_type scalar to array

Revision ID: 065
Revises: 064
Create Date: 2026-05-26 00:00:00.000000

"""

from alembic import op

revision = "065"
down_revision = "064"
branch_labels = None
depends_on = None


def upgrade():
    # Drop scalar index before altering column
    op.execute("DROP INDEX IF EXISTS global.ix_model_config_provider_type_active")

    # Convert scalar enum to array, wrapping existing values
    op.execute(
        """
        ALTER TABLE global.model_config
            ALTER COLUMN completion_type TYPE global.completion_type_enum[]
                USING ARRAY[completion_type]
        """
    )

    # GIN index for efficient array containment queries
    op.execute(
        "CREATE INDEX ix_model_config_completion_type ON global.model_config USING gin (completion_type)"
    )


def downgrade():
    op.execute("DROP INDEX IF EXISTS global.ix_model_config_completion_type")

    op.execute(
        """
        ALTER TABLE global.model_config
            ALTER COLUMN completion_type TYPE global.completion_type_enum
                USING completion_type[1]
        """
    )

    op.execute(
        "CREATE INDEX ix_model_config_provider_type_active ON global.model_config (provider, completion_type, is_active)"
    )
