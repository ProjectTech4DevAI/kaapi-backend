"""seed stt/tts model_config rows for google, sarvamai, elevenlabs

Revision ID: 063
Revises: 062
Create Date: 2026-05-19 00:00:00.000000

"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "063"
down_revision = "062"
branch_labels = None
depends_on = None


SEEDED_MODELS = [
    ("google", "gemini-2.5-pro"),
    ("google", "gemini-3.1-pro-preview"),
    ("google", "gemini-3-flash-preview"),
    ("google", "gemini-2.5-flash"),
    ("google", "gemini-2.5-flash-preview-tts"),
    ("google", "gemini-2.5-pro-preview-tts"),
    ("sarvamai", "saaras:v3"),
    ("sarvamai", "bulbul:v3"),
    ("elevenlabs", "scribe_v2"),
    ("elevenlabs", "eleven_v3"),
]


def upgrade():
    # 1. Create enum types
    op.execute(
        "CREATE TYPE global.provider_enum AS ENUM ('openai', 'google', 'sarvamai', 'elevenlabs')"
    )
    op.execute("CREATE TYPE global.completion_type_enum AS ENUM ('text', 'stt', 'tts')")

    # 2. Alter provider column to use enum; add completion_type column
    op.execute(
        """
        ALTER TABLE global.model_config
            ALTER COLUMN provider TYPE global.provider_enum
                USING provider::global.provider_enum,
            ADD COLUMN completion_type global.completion_type_enum
        """
    )

    # 3. Backfill completion_type for pre-existing rows (openai models seeded before this migration)
    op.execute(
        """
        UPDATE global.model_config SET completion_type =
            CASE
                WHEN 'AUDIO' = ANY(input_modalities::text[]) AND NOT ('AUDIO' = ANY(output_modalities::text[])) THEN 'stt'::global.completion_type_enum
                WHEN 'AUDIO' = ANY(output_modalities::text[]) AND NOT ('AUDIO' = ANY(input_modalities::text[])) THEN 'tts'::global.completion_type_enum
                ELSE 'text'::global.completion_type_enum
            END
        WHERE completion_type IS NULL
        """
    )

    # 4. Set NOT NULL now that all rows are backfilled
    op.execute(
        "ALTER TABLE global.model_config ALTER COLUMN completion_type SET NOT NULL"
    )

    # 5. Add indexes
    op.execute(
        "CREATE INDEX ix_model_config_provider_active ON global.model_config (provider, is_active)"
    )
    op.execute(
        "CREATE INDEX ix_model_config_provider_type_active ON global.model_config (provider, completion_type, is_active)"
    )
    op.execute(
        "CREATE INDEX ix_model_config_input_modalities ON global.model_config USING gin (input_modalities)"
    )
    op.execute(
        "CREATE INDEX ix_model_config_output_modalities ON global.model_config USING gin (output_modalities)"
    )

    op.execute(
        "SELECT setval(pg_get_serial_sequence('global.model_config', 'id'), MAX(id)) FROM global.model_config"
    )

    op.execute(
        """
        INSERT INTO global.model_config
            (provider, model_name, completion_type, config, input_modalities, output_modalities, pricing, is_active, inserted_at, updated_at)
        VALUES
            ('google', 'gemini-2.5-pro', 'stt',
                '{"temperature": {"type": "float", "default": 1.0, "min": 0.0, "max": 2.0, "description": "Controls randomness. Lower = more deterministic."}}',
                '{AUDIO}', '{TEXT}',
                '{"response": {"input_token_cost": 1.25, "output_token_cost": 10.0}, "batch": {"input_token_cost": 0.625, "output_token_cost": 5.0}, "audio": {"input_token_cost": 3.5, "output_token_cost": 10.0}}',
                true, NOW(), NOW()),
            ('google', 'gemini-3.1-pro-preview', 'stt',
                '{"thinking_level": {"type": "enum", "default": "high", "options": ["low", "medium", "high"], "description": "Max reasoning depth before output. high = best quality, low = faster/cheaper."}}',
                '{AUDIO}', '{TEXT}',
                '{"response": {"input_token_cost": 2.0, "output_token_cost": 12.0}, "batch": {"input_token_cost": 1.0, "output_token_cost": 6.0}, "audio": {"input_token_cost": 3.5, "output_token_cost": 12.0}}',
                true, NOW(), NOW()),
            ('google', 'gemini-3-flash-preview', 'stt',
                '{"thinking_level": {"type": "enum", "default": "high", "options": ["minimal", "low", "medium", "high"], "description": "Max reasoning depth before output."}}',
                '{AUDIO}', '{TEXT}',
                '{"response": {"input_token_cost": 0.5, "output_token_cost": 3.0}, "batch": {"input_token_cost": 0.25, "output_token_cost": 1.5}, "audio": {"input_token_cost": 1.0, "output_token_cost": 3.0}}',
                true, NOW(), NOW()),
            ('google', 'gemini-2.5-flash', 'stt',
                '{"temperature": {"type": "float", "default": 1.0, "min": 0.0, "max": 2.0, "description": "Controls randomness. Lower = more deterministic."}}',
                '{AUDIO}', '{TEXT}',
                '{"response": {"input_token_cost": 0.3, "output_token_cost": 2.5}, "batch": {"input_token_cost": 0.15, "output_token_cost": 1.25}, "audio": {"input_token_cost": 1.0, "output_token_cost": 2.5}}',
                true, NOW(), NOW()),
            ('google', 'gemini-2.5-flash-preview-tts', 'tts',
                '{"voice": {"type": "enum", "default": "Kore", "options": ["Kore", "Orus", "Leda", "Charon"], "description": "TTS voice."}}',
                '{TEXT}', '{AUDIO}',
                '{"response": {"input_token_cost": 0.5, "output_token_cost": 10.0}, "batch": {"input_token_cost": 0.25, "output_token_cost": 5.0}, "audio": {"input_token_cost": 0.5, "output_token_cost": 10.0}}',
                true, NOW(), NOW()),
            ('google', 'gemini-2.5-pro-preview-tts', 'tts',
                '{"voice": {"type": "enum", "default": "Kore", "options": ["Kore", "Orus", "Leda", "Charon"], "description": "TTS voice."}}',
                '{TEXT}', '{AUDIO}',
                '{"response": {"input_token_cost": 1.0, "output_token_cost": 20.0}, "batch": {"input_token_cost": 0.5, "output_token_cost": 10.0}, "audio": {"input_token_cost": 1.0, "output_token_cost": 20.0}}',
                true, NOW(), NOW()),
            ('sarvamai', 'saaras:v3', 'stt',
                '{}', '{AUDIO}', '{TEXT}', NULL, true, NOW(), NOW()),
            ('sarvamai', 'bulbul:v3', 'tts',
                '{"voice": {"type": "enum", "default": "simran", "options": ["simran", "shubh", "roopa"], "description": "TTS voice."}}',
                '{TEXT}', '{AUDIO}', NULL, true, NOW(), NOW()),
            ('elevenlabs', 'scribe_v2', 'stt',
                '{}', '{AUDIO}', '{TEXT}', NULL, true, NOW(), NOW()),
            ('elevenlabs', 'eleven_v3', 'tts',
                '{"voice": {"type": "enum", "default": "Sarah", "options": ["Sarah", "George", "Callum", "Liam"], "description": "TTS voice."}}',
                '{TEXT}', '{AUDIO}', NULL, true, NOW(), NOW())
        ON CONFLICT (provider, model_name) DO NOTHING
        """
    )


def downgrade():
    op.execute("DROP INDEX IF EXISTS global.ix_model_config_output_modalities")
    op.execute("DROP INDEX IF EXISTS global.ix_model_config_input_modalities")
    op.execute("DROP INDEX IF EXISTS global.ix_model_config_provider_type_active")
    op.execute("DROP INDEX IF EXISTS global.ix_model_config_provider_active")

    op.execute(
        """
        ALTER TABLE global.model_config
            DROP COLUMN completion_type,
            ALTER COLUMN provider TYPE varchar USING provider::varchar
        """
    )

    op.execute("DROP TYPE IF EXISTS global.completion_type_enum")
    op.execute("DROP TYPE IF EXISTS global.provider_enum")
