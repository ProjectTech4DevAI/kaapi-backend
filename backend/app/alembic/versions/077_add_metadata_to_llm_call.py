"""Add metadata to llm_call; add google-gcp to provider_enum

Revision ID: 077
Revises: 076
Create Date: 2026-08-10 00:00:00.000000

`llm_call.provider` records what the client asked for; platform routing
(GEMINI_DEFAULT_INFERENCE_ROUTE) can send "google" to a different backend, so
monitoring needs the resolved backend. Left nullable with no backfill: rows
predating routing metadata have no recoverable effective provider and are
grouped as unknown/legacy.

The model attribute is `call_metadata` because `metadata` is reserved on
SQLModel declarative classes; the column itself is plain `metadata`.

Also registers the `google-gcp` provider in `global.provider_enum` and mirrors
the `google` model catalog rows for it (google-gcp serves the same Gemini
models).
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "077"
down_revision = "076"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "llm_call",
        sa.Column(
            "metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
            comment=(
                "Platform metadata: {effective_provider} - backend actually used after "
                "routing (NULL for rows predating routing metadata)"
            ),
        ),
    )

    # A newly added enum value cannot be used in the same transaction,
    # so the ADD VALUE must commit before the seed INSERT below.
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


def downgrade():
    op.drop_column("llm_call", "metadata")
    # PostgreSQL cannot remove a value from an enum; drop only the seeded rows.
    op.execute("DELETE FROM global.model_config WHERE provider = 'google-gcp'")
