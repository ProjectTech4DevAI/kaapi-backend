"""add voice, usage and issues to tts_result

Revision ID: 092
Revises: 091
Create Date: 2026-11-05 00:00:00.000000

All columns are nullable with no backfill. Rows of runs created before per-model
voices keep a NULL voice and the services fall back to the model's default voice,
which is what those rows were synthesized with. Past runs have no recorded usage
and no issue tags. Usage is stored in each provider's own billing unit (Gemini
tokens, Sarvam characters, ElevenLabs credits) with no normalisation, so rows are
only summed within a model.
"""

import sqlalchemy as sa
import sqlmodel.sql.sqltypes
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "092"
down_revision = "091"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "tts_result",
        sa.Column(
            "voice",
            sqlmodel.sql.sqltypes.AutoString(length=100),
            nullable=True,
            comment=(
                "Voice used for synthesis (Gemini voice, Sarvam speaker or "
                "ElevenLabs voice name); NULL for rows created before per-model "
                "voices, which used the model's default voice"
            ),
        ),
    )
    op.add_column(
        "tts_result",
        sa.Column(
            "usage",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
            comment=(
                "Provider-native usage, always with a unit: "
                '{unit: "tokens", input_tokens, output_tokens, total_tokens} (Gemini), '
                '{unit: "characters", characters} (Sarvam), '
                '{unit: "credits", characters, credits, credit_multiplier} (ElevenLabs; '
                "credits computed, null if the multiplier was unavailable); "
                "NULL when not reported"
            ),
        ),
    )
    op.add_column(
        "tts_result",
        sa.Column(
            "issues",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
            comment=(
                "Human feedback issue tags: list of mispronunciation, robotic, "
                "wrong_language, cut_off, long_pause, background_noise"
            ),
        ),
    )


def downgrade():
    op.drop_column("tts_result", "issues")
    op.drop_column("tts_result", "usage")
    op.drop_column("tts_result", "voice")
