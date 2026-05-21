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
    op.execute(
        """
        INSERT INTO global.model_config
            (provider, model_name, config, input_modalities, output_modalities, pricing, is_active, inserted_at, updated_at)
        VALUES
            ('google',     'gemini-2.5-pro',                '{"temperature": {"type": "float", "default": 1.0, "min": 0.0, "max": 2.0, "description": "Controls randomness. Lower = more deterministic."}}', '{AUDIO}', '{TEXT}',  NULL, true, NOW(), NOW()),
            ('google',     'gemini-3.1-pro-preview',        '{"thinking_level": {"type": "enum", "default": "high", "options": ["low", "medium", "high"], "description": "Max reasoning depth before output. high = best quality, low = faster/cheaper."}}', '{AUDIO}', '{TEXT}',  NULL, true, NOW(), NOW()),
            ('google',     'gemini-3-flash-preview',        '{"thinking_level": {"type": "enum", "default": "high", "options": ["minimal", "low", "medium", "high"], "description": "Max reasoning depth before output."}}', '{AUDIO}', '{TEXT}',  NULL, true, NOW(), NOW()),
            ('google',     'gemini-2.5-flash',              '{"temperature": {"type": "float", "default": 1.0, "min": 0.0, "max": 2.0, "description": "Controls randomness. Lower = more deterministic."}}', '{AUDIO}', '{TEXT}',  NULL, true, NOW(), NOW()),
            ('google',     'gemini-2.5-flash-preview-tts',  '{"voice": {"type": "enum", "default": "Kore", "options": ["Kore", "Orus", "Leda", "Charon"], "description": "TTS voice."}}', '{TEXT}',  '{AUDIO}', NULL, true, NOW(), NOW()),
            ('google',     'gemini-2.5-pro-preview-tts',    '{"voice": {"type": "enum", "default": "Kore", "options": ["Kore", "Orus", "Leda", "Charon"], "description": "TTS voice."}}', '{TEXT}',  '{AUDIO}', NULL, true, NOW(), NOW()),
            ('sarvamai',   'saaras:v3',                     '{}', '{AUDIO}', '{TEXT}',  NULL, true, NOW(), NOW()),
            ('sarvamai',   'bulbul:v3',                     '{"voice": {"type": "enum", "default": "simran", "options": ["simran", "shubh", "roopa"], "description": "TTS voice."}}', '{TEXT}',  '{AUDIO}', NULL, true, NOW(), NOW()),
            ('elevenlabs', 'scribe_v2',                     '{}', '{AUDIO}', '{TEXT}',  NULL, true, NOW(), NOW()),
            ('elevenlabs', 'eleven_v3',                     '{"voice": {"type": "enum", "default": "Sarah", "options": ["Sarah", "George", "Callum", "Liam"], "description": "TTS voice."}}', '{TEXT}',  '{AUDIO}', NULL, true, NOW(), NOW())
        ON CONFLICT (provider, model_name) DO NOTHING
        """
    )


def downgrade():
    op.execute(
        """
        DELETE FROM global.model_config
        WHERE (provider, model_name) IN (
            ('google',     'gemini-2.5-pro'),
            ('google',     'gemini-3.1-pro-preview'),
            ('google',     'gemini-3-flash-preview'),
            ('google',     'gemini-2.5-flash'),
            ('google',     'gemini-2.5-flash-preview-tts'),
            ('google',     'gemini-2.5-pro-preview-tts'),
            ('sarvamai',   'saaras:v3'),
            ('sarvamai',   'bulbul:v3'),
            ('elevenlabs', 'scribe_v2'),
            ('elevenlabs', 'eleven_v3')
        )
        """
    )
