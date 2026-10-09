"""seed anthropic text model_config rows

Revision ID: 086
Revises: 085
Create Date: 2026-10-09 00:00:00.000000

`anthropic` has been a valid provider since 066 but had no model_config rows,
so it never showed up in /models/providers and config-version updates could
not tell its models are reasoning models. The `effort` key in `config` is what
`is_reasoning_model` keys on — it makes `_strip_unsupported_params` drop
`temperature`, which the Messages API rejects at non-default values.

`effort.default` is "high" because that is what the Messages API applies when
the param is omitted. Sonnet 4.6 does not expose the "xhigh" level.

claude-opus-5-5 pricing is left NULL until list prices are confirmed; cost
estimation returns None for NULL pricing rather than a wrong number.

Downgrade deletes the two rows by key, so a row that already existed before
upgrade (ON CONFLICT DO NOTHING skipped it) is removed as well.
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
