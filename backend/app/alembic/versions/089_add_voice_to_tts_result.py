"""add voice to tts_result

Revision ID: 089
Revises: 088
Create Date: 2026-11-05 00:00:00.000000

TTS evaluation runs now pick a voice per model. Nullable with no backfill: rows of
runs created before this change keep NULL and the services fall back to the model's
default voice, which is what those rows were synthesized with.
"""

import sqlalchemy as sa
import sqlmodel.sql.sqltypes
from alembic import op

revision = "089"
down_revision = "088"
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


def downgrade():
    op.drop_column("tts_result", "voice")
