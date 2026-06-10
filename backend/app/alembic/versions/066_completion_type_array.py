"""completion_type scalar to array; add providers to provider_enum

Revision ID: 066
Revises: 065
Create Date: 2026-05-26 00:00:00.000000

"""

from alembic import op

revision = "066"
down_revision = "065"
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

    with op.get_context().autocommit_block():
        op.execute(
            "ALTER TYPE global.provider_enum ADD VALUE IF NOT EXISTS 'anthropic'"
        )
        op.execute(
            "ALTER TYPE global.provider_enum ADD VALUE IF NOT EXISTS 'google-aistudio'"
        )
        op.execute("ALTER TYPE global.provider_enum ADD VALUE IF NOT EXISTS 'proxy'")
        op.execute(
            "ALTER TYPE global.provider_enum ADD VALUE IF NOT EXISTS 'google-aistudio'"
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
