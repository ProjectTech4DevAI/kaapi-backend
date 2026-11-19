"""seed current-generation OpenAI/Anthropic model_config rows

Seeds text rows for gpt-6-luna, gpt-6.1-sol, gpt-5.6-luna, claude-sonnet-5-5,
claude-opus-5-5, claude-sonnet-4-6 and claude-opus-4-8.

Rows are upserted, so an existing row is repaired rather than skipped.

Revision ID: 090
Revises: 089
Create Date: 2026-11-19 00:00:00.000000

"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "090"
down_revision = "089"
branch_labels = None
depends_on = None

SEEDED_OPENAI = ("gpt-6-luna", "gpt-6.1-sol", "gpt-5.6-luna")
SEEDED_ANTHROPIC = (
    "claude-sonnet-5-5",
    "claude-opus-5-5",
    "claude-sonnet-4-6",
    "claude-opus-4-8",
)
_UPSERT = """
    ON CONFLICT (provider, model_name) DO UPDATE SET
        completion_type   = EXCLUDED.completion_type,
        config            = EXCLUDED.config,
        input_modalities  = EXCLUDED.input_modalities,
        output_modalities = EXCLUDED.output_modalities,
        pricing           = EXCLUDED.pricing,
        is_active         = EXCLUDED.is_active,
        updated_at        = NOW()
"""


def _sql_list(names: tuple[str, ...]) -> str:
    return ", ".join(f"'{name}'" for name in names)


def upgrade():
    op.execute(
        """
        INSERT INTO global.model_config
            (provider, model_name, completion_type, config, input_modalities,
             output_modalities, pricing, is_active, inserted_at, updated_at)
        VALUES
            ('openai', 'gpt-6-luna',
             ARRAY['text']::global.completion_type_enum[],
             '{"effort": {"type": "enum", "default": "medium", "options": ["none", "low", "medium", "high", "xhigh", "max"], "description": "How long the model spends reasoning. Higher = better but slower."}, "summary": {"type": "enum", "default": "auto", "options": ["auto", "detailed", "concise", "null"], "description": "Summarize the reasoning result."}}',
             '{TEXT,IMAGE}', '{TEXT}',
             '{"response": {"input_token_cost": 0.1, "output_token_cost": 0.5}, "batch": {"input_token_cost": 0.05, "output_token_cost": 0.25}}',
             true, NOW(), NOW()),

            -- gpt-6.1-sol has no "none" effort level; it always reasons.
            ('openai', 'gpt-6.1-sol',
             ARRAY['text']::global.completion_type_enum[],
             '{"effort": {"type": "enum", "default": "medium", "options": ["low", "medium", "high", "xhigh", "max"], "description": "How long the model spends reasoning. Higher = better but slower."}, "summary": {"type": "enum", "default": "auto", "options": ["auto", "detailed", "concise", "null"], "description": "Summarize the reasoning result."}}',
             '{TEXT,IMAGE}', '{TEXT}',
             '{"response": {"input_token_cost": 2, "output_token_cost": 10}, "batch": {"input_token_cost": 1, "output_token_cost": 5}}',
             true, NOW(), NOW()),

            ('openai', 'gpt-5.6-luna',
             ARRAY['text']::global.completion_type_enum[],
             '{"effort": {"type": "enum", "default": "medium", "options": ["none", "low", "medium", "high", "xhigh", "max"], "description": "How long the model spends reasoning. Higher = better but slower."}, "summary": {"type": "enum", "default": "auto", "options": ["auto", "detailed", "concise", "null"], "description": "Summarize the reasoning result."}}',
             '{TEXT,IMAGE}', '{TEXT}',
             '{"response": {"input_token_cost": 0.2, "output_token_cost": 1.2}, "batch": {"input_token_cost": 0.1, "output_token_cost": 0.6}}',
             true, NOW(), NOW())
        """
        + _UPSERT
    )

    # Anthropic rows carry neither `summary` nor `temperature`: the Messages API returns no
    # reasoning summaries and rejects non-default sampling, so the mapper drops both.
    op.execute(
        """
        INSERT INTO global.model_config
            (provider, model_name, completion_type, config, input_modalities,
             output_modalities, pricing, is_active, inserted_at, updated_at)
        VALUES
            ('anthropic', 'claude-sonnet-5-5',
             ARRAY['text']::global.completion_type_enum[],
             '{"effort": {"type": "enum", "default": "high", "options": ["low", "medium", "high", "xhigh", "max"], "description": "How long the model spends reasoning. Higher = better but slower."}}',
             '{TEXT,IMAGE}', '{TEXT}',
             '{"response": {"input_token_cost": 2, "output_token_cost": 10}, "batch": {"input_token_cost": 1, "output_token_cost": 5}}',
             true, NOW(), NOW()),

            ('anthropic', 'claude-opus-5-5',
             ARRAY['text']::global.completion_type_enum[],
             '{"effort": {"type": "enum", "default": "medium", "options": ["low", "medium", "high", "xhigh", "max"], "description": "How long the model spends reasoning. Higher = better but slower."}}',
             '{TEXT,IMAGE}', '{TEXT}',
             '{"response": {"input_token_cost": 4, "output_token_cost": 20}, "batch": {"input_token_cost": 2, "output_token_cost": 10}}',
             true, NOW(), NOW()),

            -- claude-sonnet-4-6 predates the `xhigh` level.
            ('anthropic', 'claude-sonnet-4-6',
             ARRAY['text']::global.completion_type_enum[],
             '{"effort": {"type": "enum", "default": "high", "options": ["low", "medium", "high", "max"], "description": "How long the model spends reasoning. Higher = better but slower."}}',
             '{TEXT,IMAGE}', '{TEXT}',
             '{"response": {"input_token_cost": 3, "output_token_cost": 15}, "batch": {"input_token_cost": 1.5, "output_token_cost": 7.5}}',
             true, NOW(), NOW()),

            ('anthropic', 'claude-opus-4-8',
             ARRAY['text']::global.completion_type_enum[],
             '{"effort": {"type": "enum", "default": "high", "options": ["low", "medium", "high", "xhigh", "max"], "description": "How long the model spends reasoning. Higher = better but slower."}}',
             '{TEXT,IMAGE}', '{TEXT}',
             '{"response": {"input_token_cost": 5, "output_token_cost": 25}, "batch": {"input_token_cost": 2.5, "output_token_cost": 12.5}}',
             true, NOW(), NOW())
        """
        + _UPSERT
    )


def downgrade():
    op.execute(
        f"""
        DELETE FROM global.model_config
        WHERE (provider = 'openai' AND model_name IN ({_sql_list(SEEDED_OPENAI)}))
           OR (provider = 'anthropic' AND model_name IN ({_sql_list(SEEDED_ANTHROPIC)}))
        """
    )
