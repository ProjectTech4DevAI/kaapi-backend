"""Revert 077: drop llm_call.metadata and google-gcp model_config rows

Revision ID: 079
Revises: 078
Create Date: 2026-08-12 00:00:00.000000

Forward migration undoing 077's schema changes, needed because #982 (Gemini
Toggleable Inference Routing) was reverted after 078 was already chained on
top of 077. Deleting 077's file would orphan 078's down_revision, so 077 stays
and this migration mirrors its downgrade() instead.

The `google-gcp` value in `global.provider_enum` is left in place: PostgreSQL
cannot remove enum values. It's inert once no code references it.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "079"
down_revision = "078"
branch_labels = None
depends_on = None


def upgrade():
    op.drop_column("llm_call", "metadata")
    op.execute("DELETE FROM global.model_config WHERE provider = 'google-gcp'")


def downgrade():
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
