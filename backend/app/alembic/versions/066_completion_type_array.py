"""completion_type scalar to array; add providers to provider_enum;
backfill credential providers for the google/google-vertex semantics flip

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

    # Convert scalar enum to array, wrapping existing values. Guarded so a
    # rerun after a partially-applied earlier attempt does not double-wrap.
    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_schema = 'global'
                  AND table_name = 'model_config'
                  AND column_name = 'completion_type'
                  AND data_type <> 'ARRAY'
            ) THEN
                ALTER TABLE global.model_config
                    ALTER COLUMN completion_type TYPE global.completion_type_enum[]
                        USING ARRAY[completion_type];
            END IF;
        END$$;
        """
    )

    # GIN index for efficient array containment queries
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_model_config_completion_type ON global.model_config USING gin (completion_type)"
    )

    with op.get_context().autocommit_block():
        op.execute(
            "ALTER TYPE global.provider_enum ADD VALUE IF NOT EXISTS 'anthropic'"
        )
        op.execute("ALTER TYPE global.provider_enum ADD VALUE IF NOT EXISTS 'proxy'")
        op.execute(
            "ALTER TYPE global.provider_enum ADD VALUE IF NOT EXISTS 'google-aistudio'"
        )

    op.execute(
        "UPDATE credential SET provider = 'google-aistudio' WHERE provider = 'google'"
    )
    op.execute(
        "UPDATE credential SET provider = 'google' WHERE provider = 'google-vertex'"
    )

    op.execute(
        """
        INSERT INTO global.model_config
            (provider, model_name, completion_type, config, input_modalities,
             output_modalities, pricing, is_active, inserted_at, updated_at)
        SELECT 'google-aistudio', model_name, completion_type, config, input_modalities,
               output_modalities, pricing, is_active, NOW(), NOW()
        FROM global.model_config
        WHERE provider = 'google'
        ON CONFLICT (provider, model_name) DO NOTHING
        """
    )

    op.execute(
        """
        UPDATE config_version
        SET config_blob = jsonb_set(config_blob, '{completion,provider}', '"google-aistudio"')
        WHERE config_blob->'completion'->>'provider' = 'google'
        """
    )
    op.execute(
        """
        UPDATE config_version
        SET config_blob = jsonb_set(config_blob, '{completion,provider}', '"google-aistudio-native"')
        WHERE config_blob->'completion'->>'provider' = 'google-native'
        """
    )

    op.execute(
        "UPDATE batch_job SET provider = 'google-aistudio' WHERE provider = 'google'"
    )


def downgrade():
    op.execute(
        "UPDATE batch_job SET provider = 'google' WHERE provider = 'google-aistudio'"
    )
    op.execute(
        """
        UPDATE config_version
        SET config_blob = jsonb_set(config_blob, '{completion,provider}', '"google-native"')
        WHERE config_blob->'completion'->>'provider' = 'google-aistudio-native'
        """
    )
    op.execute(
        """
        UPDATE config_version
        SET config_blob = jsonb_set(config_blob, '{completion,provider}', '"google"')
        WHERE config_blob->'completion'->>'provider' = 'google-aistudio'
        """
    )
    op.execute("DELETE FROM global.model_config WHERE provider = 'google-aistudio'")
    op.execute(
        "UPDATE credential SET provider = 'google-vertex' WHERE provider = 'google'"
    )
    op.execute(
        "UPDATE credential SET provider = 'google' WHERE provider = 'google-aistudio'"
    )

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
