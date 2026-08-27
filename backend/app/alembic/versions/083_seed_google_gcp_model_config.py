"""seed google-gcp model_config rows, mirroring google and google-aistudio

Revision ID: 083
Revises: 082
Create Date: 2026-08-27 00:00:00.000000

077 first seeded these rows and added 'google-gcp' to global.provider_enum,
but 079 reverted the seed (while leaving the enum value in place, since
Postgres cannot drop an enum member) as part of unwinding an unrelated
feature. GoogleGCPProvider now supports text, stt, and tts (077 only mirrored
'google', which was stt/tts-only at the time), so this seeds from both
'google' and 'google-aistudio' to also pick up any text-capable rows.
"""

from alembic import op

revision = "083"
down_revision = "082"
branch_labels = None
depends_on = None


def upgrade():
    # Enum value already exists from 077 and survives forever (Postgres can't
    # drop enum members) — ADD VALUE IF NOT EXISTS kept only for a from-scratch
    # DB that never ran 077.
    with op.get_context().autocommit_block():
        op.execute(
            "ALTER TYPE global.provider_enum ADD VALUE IF NOT EXISTS 'google-gcp'"
        )

    op.execute(
        """
        INSERT INTO global.model_config
            (provider, model_name, completion_type, config, input_modalities,
             output_modalities, pricing, is_active, inserted_at, updated_at)
        SELECT 'google-gcp', model_name, completion_type, config, input_modalities,
               output_modalities, pricing, is_active, NOW(), NOW()
        FROM global.model_config
        WHERE provider = 'google'
        ON CONFLICT (provider, model_name) DO NOTHING
        """
    )
    op.execute(
        """
        INSERT INTO global.model_config
            (provider, model_name, completion_type, config, input_modalities,
             output_modalities, pricing, is_active, inserted_at, updated_at)
        SELECT 'google-gcp', model_name, completion_type, config, input_modalities,
               output_modalities, pricing, is_active, NOW(), NOW()
        FROM global.model_config
        WHERE provider = 'google-aistudio'
        ON CONFLICT (provider, model_name) DO NOTHING
        """
    )


def downgrade():
    # PostgreSQL cannot remove a value from an enum; drop only the seeded rows.
    op.execute("DELETE FROM global.model_config WHERE provider = 'google-gcp'")
