"""add anthropic + google-vertex to provider_enum and seed test model_config rows

Revision ID: 064
Revises: 063
Create Date: 2026-05-28 00:00:00.000000

"""

from alembic import op


revision = "064"
down_revision = "063"
branch_labels = None
depends_on = None


def upgrade():
    # ALTER TYPE ... ADD VALUE cannot run inside a transaction block; use
    # autocommit per existing pattern (see migration 056). The added values
    # are visible to subsequent statements once the autocommit_block exits.
    with op.get_context().autocommit_block():
        op.execute(
            "ALTER TYPE global.provider_enum ADD VALUE IF NOT EXISTS 'anthropic'"
        )
        op.execute(
            "ALTER TYPE global.provider_enum ADD VALUE IF NOT EXISTS 'google-vertex'"
        )

    # Pass-through seed rows for testing. Pricing values are placeholders;
    # revise once real cost data is available.
    op.execute(
        """
        INSERT INTO global.model_config
            (provider, model_name, completion_type, config, input_modalities, output_modalities, pricing, is_active, inserted_at, updated_at)
        VALUES
            -- Anthropic text models
            ('anthropic', 'claude-opus-4-7', 'text',
                '{"temperature": {"type": "float", "default": 1.0, "min": 0.0, "max": 1.0, "description": "Sampling temperature."}}',
                '{TEXT,IMAGE,PDF}', '{TEXT}',
                '{"response": {"input_token_cost": 15.0, "output_token_cost": 75.0}, "batch": {"input_token_cost": 7.5, "output_token_cost": 37.5}}',
                true, NOW(), NOW()),
            ('anthropic', 'claude-sonnet-4-6', 'text',
                '{"temperature": {"type": "float", "default": 1.0, "min": 0.0, "max": 1.0, "description": "Sampling temperature."}}',
                '{TEXT,IMAGE,PDF}', '{TEXT}',
                '{"response": {"input_token_cost": 3.0, "output_token_cost": 15.0}, "batch": {"input_token_cost": 1.5, "output_token_cost": 7.5}}',
                true, NOW(), NOW()),
            ('anthropic', 'claude-haiku-4-5-20251001', 'text',
                '{"temperature": {"type": "float", "default": 1.0, "min": 0.0, "max": 1.0, "description": "Sampling temperature."}}',
                '{TEXT,IMAGE,PDF}', '{TEXT}',
                '{"response": {"input_token_cost": 1.0, "output_token_cost": 5.0}, "batch": {"input_token_cost": 0.5, "output_token_cost": 2.5}}',
                true, NOW(), NOW()),
            -- Google Vertex STT models (Gemini 3.x family — GA per
            -- https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/google-models)
            ('google-vertex', 'gemini-3.1-pro-preview', 'stt',
                '{"thinking_level": {"type": "enum", "default": "high", "options": ["low", "medium", "high"], "description": "Max reasoning depth before output. high = best quality, low = faster/cheaper."}}',
                '{AUDIO}', '{TEXT}',
                '{"response": {"input_token_cost": 2.0, "output_token_cost": 12.0}, "audio": {"input_token_cost": 3.5, "output_token_cost": 12.0}}',
                true, NOW(), NOW()),
            ('google-vertex', 'gemini-3-pro', 'stt',
                '{"thinking_level": {"type": "enum", "default": "high", "options": ["low", "medium", "high"], "description": "Max reasoning depth before output."}}',
                '{AUDIO}', '{TEXT}',
                '{"response": {"input_token_cost": 1.5, "output_token_cost": 10.0}, "audio": {"input_token_cost": 3.0, "output_token_cost": 10.0}}',
                true, NOW(), NOW()),
            ('google-vertex', 'gemini-3.5-flash', 'stt',
                '{"thinking_level": {"type": "enum", "default": "high", "options": ["minimal", "low", "medium", "high"], "description": "Max reasoning depth before output."}}',
                '{AUDIO}', '{TEXT}',
                '{"response": {"input_token_cost": 0.6, "output_token_cost": 3.5}, "audio": {"input_token_cost": 1.2, "output_token_cost": 3.5}}',
                true, NOW(), NOW()),
            ('google-vertex', 'gemini-3-flash-preview', 'stt',
                '{"thinking_level": {"type": "enum", "default": "high", "options": ["minimal", "low", "medium", "high"], "description": "Max reasoning depth before output."}}',
                '{AUDIO}', '{TEXT}',
                '{"response": {"input_token_cost": 0.5, "output_token_cost": 3.0}, "audio": {"input_token_cost": 1.0, "output_token_cost": 3.0}}',
                true, NOW(), NOW()),
            ('google-vertex', 'gemini-3.1-flash-lite', 'stt',
                '{"temperature": {"type": "float", "default": 0.0, "min": 0.0, "max": 2.0, "description": "Controls randomness. Lower = more deterministic."}}',
                '{AUDIO}', '{TEXT}',
                '{"response": {"input_token_cost": 0.1, "output_token_cost": 0.4}, "audio": {"input_token_cost": 0.3, "output_token_cost": 0.4}}',
                true, NOW(), NOW()),
            ('google-vertex', 'gemini-2.5-flash', 'stt',
                '{"temperature": {"type": "float", "default": 0.0, "min": 0.0, "max": 2.0, "description": "Controls randomness. Lower = more deterministic."}}',
                '{AUDIO}', '{TEXT}',
                '{"response": {"input_token_cost": 0.3, "output_token_cost": 2.5}, "audio": {"input_token_cost": 1.0, "output_token_cost": 2.5}}',
                true, NOW(), NOW()),
            ('google-vertex', 'gemini-2.5-pro', 'stt',
                '{"temperature": {"type": "float", "default": 0.0, "min": 0.0, "max": 2.0, "description": "Controls randomness. Lower = more deterministic."}}',
                '{AUDIO}', '{TEXT}',
                '{"response": {"input_token_cost": 1.25, "output_token_cost": 10.0}, "audio": {"input_token_cost": 3.5, "output_token_cost": 10.0}}',
                true, NOW(), NOW()),
            -- Google Vertex TTS models
            ('google-vertex', 'gemini-2.5-flash-preview-tts', 'tts',
                '{"voice": {"type": "enum", "default": "Kore", "options": ["Aoede", "Charon", "Fenrir", "Kore", "Puck"], "description": "TTS voice."}}',
                '{TEXT}', '{AUDIO}',
                '{"response": {"input_token_cost": 0.5, "output_token_cost": 10.0}, "audio": {"input_token_cost": 0.5, "output_token_cost": 10.0}}',
                true, NOW(), NOW()),
            ('google-vertex', 'gemini-2.5-pro-preview-tts', 'tts',
                '{"voice": {"type": "enum", "default": "Kore", "options": ["Aoede", "Charon", "Fenrir", "Kore", "Puck"], "description": "TTS voice."}}',
                '{TEXT}', '{AUDIO}',
                '{"response": {"input_token_cost": 1.0, "output_token_cost": 20.0}, "audio": {"input_token_cost": 1.0, "output_token_cost": 20.0}}',
                true, NOW(), NOW())
        ON CONFLICT (provider, model_name) DO NOTHING
        """
    )


def downgrade():
    op.execute(
        """
        DELETE FROM global.model_config
        WHERE provider IN ('anthropic', 'google-vertex')
        """
    )
    # Enum value removal requires rebuilding the type and re-pointing every
    # referencing column. Skipped — see migrations 035 / 056 for the same
    # convention.
