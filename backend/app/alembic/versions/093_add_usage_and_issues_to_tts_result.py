"""add usage and issues to tts_result

Revision ID: 093
Revises: 092
Create Date: 2026-11-25 00:00:00.000000

Usage is stored in each provider's own billing unit (Gemini tokens, Sarvam
characters, ElevenLabs credits) with no normalisation, so rows are only summed
within a model. Both columns are nullable with no backfill: past runs have no
recorded usage and no issue tags.
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "093"
down_revision = "092"
branch_labels = None
depends_on = None


def upgrade():
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
