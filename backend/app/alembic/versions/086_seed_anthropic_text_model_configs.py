"""seed anthropic text model_config rows

Revision ID: 086
Revises: 085
Create Date: 2026-10-09 00:00:00.000000
"""

from alembic import op

revision = "086"
down_revision = "085"
branch_labels = None
depends_on = None


def upgrade():
    op.execute(
        """
        INSERT INTO global.model_config
            (provider, model_name, completion_type, config, input_modalities,
             output_modalities, pricing, is_active, inserted_at, updated_at)
        VALUES
            ('anthropic', 'claude-sonnet-4-6', '{text}',
                '{"effort": {"type": "enum", "default": "high", "options": ["low", "medium", "high", "max"], "description": "How long the model spends reasoning. Higher = better but slower."}}',
                '{TEXT,IMAGE,FILES}', '{TEXT}',
                '{"response": {"input_token_cost": 3, "output_token_cost": 15}, "batch": {"input_token_cost": 1.5, "output_token_cost": 7.5}}',
                true, NOW(), NOW()),
            ('anthropic', 'claude-opus-5-5', '{text}',
                '{"effort": {"type": "enum", "default": "high", "options": ["low", "medium", "high", "xhigh", "max"], "description": "How long the model spends reasoning. Higher = better but slower."}}',
                '{TEXT,IMAGE,FILES}', '{TEXT}',
                NULL,
                true, NOW(), NOW())
        ON CONFLICT (provider, model_name) DO NOTHING
        """
    )


def downgrade():
    op.execute(
        """
        DELETE FROM global.model_config
        WHERE provider = 'anthropic'
          AND model_name IN ('claude-sonnet-4-6', 'claude-opus-5-5')
        """
    )
