"""create model_config table

Revision ID: 052
Revises: 051
Create Date: 2026-03-12 00:00:00.000000

"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "052"
down_revision = "051"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "model_config",
        sa.Column(
            "id",
            sa.Integer(),
            sa.Identity(always=False),
            nullable=False,
            comment="unique identifier for model config table",
        ),
        sa.Column(
            "provider",
            sa.String(),
            nullable=False,
            comment="provider name (e.g. openai, google)",
        ),
        sa.Column(
            "model_name",
            sa.String(),
            nullable=False,
            comment="model name (e.g. gpt-4o, gemini-3-flash-preview)",
        ),
        sa.Column(
            "config",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            comment="model adhoc configuration",
        ),
        sa.Column(
            "input_modalities",
            postgresql.ARRAY(sa.String()),
            nullable=False,
            server_default="{}",
            comment="supported input modalities: TEXT, IMAGE, FILES, AUDIO",
        ),
        sa.Column(
            "output_modalities",
            postgresql.ARRAY(sa.String()),
            nullable=False,
            server_default="{}",
            comment="supported output modalities: TEXT, AUDIO",
        ),
        sa.Column(
            "pricing",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
            comment=(
                "pricing per 1M tokens in USD. "
                "Structure: {response: {input_token_cost, output_token_cost}, batch: {input_token_cost, output_token_cost}}"
            ),
        ),
        sa.Column(
            "is_active",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("true"),
            comment="whether this model is available",
        ),
        sa.Column(
            "inserted_at",
            sa.DateTime(),
            nullable=False,
            comment="timestamp when model configuration was created",
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(),
            nullable=False,
            comment="timestamp when model configuration was updated",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("provider", "model_name"),
        schema="global",
    )

    # Seed default model configurations
    op.execute(
        """
        INSERT INTO global.model_config (id, provider, model_name, config, input_modalities, output_modalities, pricing, is_active, inserted_at, updated_at)
        VALUES
            (1, 'openai', 'gpt-4o-mini', '{"temperature": {"type": "float", "default": 1.0, "min": 0.0, "max": 2.0, "description": "Controls randomness. Lower = more deterministic."}, "top_p": {"type": "float", "default": 1.0, "min": 0.0, "max": 1.0, "description": "Nucleus sampling. Use either this or temperature, not both."}, "max_output_tokens": {"type": "int", "default": 2048, "min": 1, "max": 32768, "description": "Max tokens in the response."}}', '{TEXT,IMAGE}', '{TEXT}', '{"response": {"input_token_cost": 0.15, "output_token_cost": 0.6}, "batch": {"input_token_cost": 0.075, "output_token_cost": 0.3}}', true, NOW(), NOW()),
            (2, 'openai', 'gpt-4o', '{"temperature": {"type": "float", "default": 1.0, "min": 0.0, "max": 2.0, "description": "Controls randomness. Lower = more deterministic."}, "top_p": {"type": "float", "default": 1.0, "min": 0.0, "max": 1.0, "description": "Nucleus sampling. Use either this or temperature, not both."}, "max_output_tokens": {"type": "int", "default": 2048, "min": 1, "max": 32768, "description": "Max tokens in the response."}}', '{TEXT,IMAGE}', '{TEXT}', '{"response": {"input_token_cost": 2.5, "output_token_cost": 10}, "batch": {"input_token_cost": 1.25, "output_token_cost": 5}}', true, NOW(), NOW()),
            (3, 'openai', 'gpt-4.1', '{"temperature": {"type": "float", "default": 1.0, "min": 0.0, "max": 2.0, "description": "Controls randomness. Lower = more deterministic."}, "top_p": {"type": "float", "default": 1.0, "min": 0.0, "max": 1.0, "description": "Nucleus sampling. Use either this or temperature, not both."}, "max_output_tokens": {"type": "int", "default": 2048, "min": 1, "max": 32768, "description": "Max tokens in the response."}}', '{TEXT,IMAGE}', '{TEXT}', '{"response": {"input_token_cost": 2, "output_token_cost": 8}, "batch": {"input_token_cost": 1, "output_token_cost": 4}}', true, NOW(), NOW()),
            (4, 'openai', 'gpt-4.1-mini', '{"temperature": {"type": "float", "default": 1.0, "min": 0.0, "max": 2.0, "description": "Controls randomness. Lower = more deterministic."}, "top_p": {"type": "float", "default": 1.0, "min": 0.0, "max": 1.0, "description": "Nucleus sampling. Use either this or temperature, not both."}, "max_output_tokens": {"type": "int", "default": 2048, "min": 1, "max": 32768, "description": "Max tokens in the response."}}', '{TEXT,IMAGE}', '{TEXT}', '{"response": {"input_token_cost": 0.4, "output_token_cost": 1.6}, "batch": {"input_token_cost": 0.2, "output_token_cost": 0.8}}', true, NOW(), NOW()),
            (5, 'openai', 'gpt-4.1-nano', '{"temperature": {"type": "float", "default": 1.0, "min": 0.0, "max": 2.0, "description": "Controls randomness. Lower = more deterministic."}, "top_p": {"type": "float", "default": 1.0, "min": 0.0, "max": 1.0, "description": "Nucleus sampling. Use either this or temperature, not both."}, "max_output_tokens": {"type": "int", "default": 2048, "min": 1, "max": 32768, "description": "Max tokens in the response."}}', '{TEXT,IMAGE}', '{TEXT}', '{"response": {"input_token_cost": 0.1, "output_token_cost": 0.4}, "batch": {"input_token_cost": 0.05, "output_token_cost": 0.2}}', true, NOW(), NOW()),
            (6, 'openai', 'o3-mini', '{"effort": {"type": "enum", "default": "medium", "options": ["low", "medium", "high"], "description": "How long the model spends reasoning. Higher = better but slower."}, "summary": {"type": "enum", "default": "auto", "options": ["auto", "detailed", "concise"], "description": "Summarize the reasoning result."}}', '{TEXT,IMAGE}', '{TEXT}', '{"response": {"input_token_cost": 1.1, "output_token_cost": 4.4}, "batch": {"input_token_cost": 0.55, "output_token_cost": 2.2}}', true, NOW(), NOW()),
            (7, 'openai', 'o3', '{"effort": {"type": "enum", "default": "medium", "options": ["low", "medium", "high"], "description": "How long the model spends reasoning. Higher = better but slower."}, "summary": {"type": "enum", "default": "auto", "options": ["auto", "detailed", "concise", "null"], "description": "Summarize the reasoning result."}}', '{TEXT,IMAGE}', '{TEXT}', '{"response": {"input_token_cost": 2, "output_token_cost": 8}, "batch": {"input_token_cost": 1, "output_token_cost": 4}}', true, NOW(), NOW()),
            (8, 'openai', 'o4-mini', '{"effort": {"type": "enum", "default": "medium", "options": ["low", "medium", "high"], "description": "How long the model spends reasoning. Higher = better but slower."}, "summary": {"type": "enum", "default": "auto", "options": ["auto", "detailed", "concise", "null"], "description": "Summarize the reasoning result."}}', '{TEXT,IMAGE}', '{TEXT}', '{"response": {"input_token_cost": 1.1, "output_token_cost": 4.4}, "batch": {"input_token_cost": 0.55, "output_token_cost": 2.2}}', true, NOW(), NOW()),
            (9, 'openai', 'gpt-5', '{"effort": {"type": "enum", "default": "medium", "options": ["minimal", "low", "medium", "high"], "description": "How long the model spends reasoning. Higher = better but slower."}, "summary": {"type": "enum", "default": "auto", "options": ["auto", "detailed", "concise", "null"], "description": "Summarize the reasoning result."}}', '{TEXT,IMAGE}', '{TEXT}', '{"response": {"input_token_cost": 1.25, "output_token_cost": 10}, "batch": {"input_token_cost": 0.625, "output_token_cost": 5}}', true, NOW(), NOW()),
            (10, 'openai', 'gpt-5-mini', '{"effort": {"type": "enum", "default": "medium", "options": ["minimal", "low", "medium", "high"], "description": "How long the model spends reasoning. Higher = better but slower."}, "summary": {"type": "enum", "default": "auto", "options": ["auto", "detailed", "concise", "null"], "description": "Summarize the reasoning result."}}', '{TEXT,IMAGE}', '{TEXT}', '{"response": {"input_token_cost": 0.25, "output_token_cost": 2}, "batch": {"input_token_cost": 0.125, "output_token_cost": 1}}', true, NOW(), NOW()),
            (11, 'openai', 'gpt-5-nano', '{"effort": {"type": "enum", "default": "medium", "options": ["minimal", "low", "medium", "high"], "description": "How long the model spends reasoning. Higher = better but slower."}, "summary": {"type": "enum", "default": "auto", "options": ["auto", "detailed", "concise", "null"], "description": "Summarize the reasoning result."}}', '{TEXT,IMAGE}', '{TEXT}', '{"response": {"input_token_cost": 0.05, "output_token_cost": 0.4}, "batch": {"input_token_cost": 0.025, "output_token_cost": 0.2}}', true, NOW(), NOW()),
            (12, 'openai', 'gpt-5.1', '{"effort": {"type": "enum", "default": "medium", "options": ["none", "low", "medium", "high"], "description": "How long the model spends reasoning. Higher = better but slower."}, "summary": {"type": "enum", "default": "auto", "options": ["auto", "detailed", "concise", "null"], "description": "Summarize the reasoning result."}}', '{TEXT,IMAGE}', '{TEXT}', '{"response": {"input_token_cost": 1.25, "output_token_cost": 10}, "batch": {"input_token_cost": 0.625, "output_token_cost": 5}}', true, NOW(), NOW()),
            (13, 'openai', 'gpt-5.2', '{"effort": {"type": "enum", "default": "medium", "options": ["none", "low", "medium", "high", "xhigh"], "description": "How long the model spends reasoning. Higher = better but slower."}, "summary": {"type": "enum", "default": "auto", "options": ["auto", "detailed", "concise", "null"], "description": "Summarize the reasoning result."}}', '{TEXT,IMAGE}', '{TEXT}', '{"response": {"input_token_cost": 1.75, "output_token_cost": 14}, "batch": {"input_token_cost": 0.875, "output_token_cost": 7}}', true, NOW(), NOW()),
            (14, 'openai', 'gpt-5.2-pro', '{"summary": {"type": "enum", "default": "auto", "options": ["auto", "detailed", "concise", "null"], "description": "Summarize the reasoning result."}}', '{TEXT,IMAGE}', '{TEXT}', '{"response": {"input_token_cost": 21, "output_token_cost": 168}, "batch": {"input_token_cost": 10.5, "output_token_cost": 84}}', true, NOW(), NOW()),
            (15, 'openai', 'gpt-5.3-chat-latest', '{"summary": {"type": "enum", "default": "auto", "options": ["auto", "detailed", "concise", "null"], "description": "Summarize the reasoning result."}}', '{TEXT,IMAGE}', '{TEXT}', '{"response": {"input_token_cost": 1.75, "output_token_cost": 14}, "batch": {"input_token_cost": 0.875, "output_token_cost": 7}}', true, NOW(), NOW()),
            (16, 'openai', 'gpt-5.4', '{"effort": {"type": "enum", "default": "medium", "options": ["none", "low", "medium", "high", "xhigh"], "description": "How long the model spends reasoning. Higher = better but slower."}, "summary": {"type": "enum", "default": "auto", "options": ["auto", "detailed", "concise", "null"], "description": "Summarize the reasoning result."}}', '{TEXT,IMAGE}', '{TEXT}', '{"response": {"input_token_cost": 2.5, "output_token_cost": 15}, "batch": {"input_token_cost": 1.25, "output_token_cost": 7.5}}', true, NOW(), NOW()),
            (17, 'openai', 'gpt-5.4-mini', '{"effort": {"type": "enum", "default": "medium", "options": ["none", "low", "medium", "high", "xhigh"], "description": "How long the model spends reasoning. Higher = better but slower."}, "summary": {"type": "enum", "default": "auto", "options": ["auto", "detailed", "concise", "null"], "description": "Summarize the reasoning result."}}', '{TEXT,IMAGE}', '{TEXT}', '{"response": {"input_token_cost": 0.75, "output_token_cost": 4.5}, "batch": {"input_token_cost": 0.375, "output_token_cost": 2.25}}', true, NOW(), NOW()),
            (18, 'openai', 'gpt-5.4-nano', '{"effort": {"type": "enum", "default": "medium", "options": ["none", "low", "medium", "high", "xhigh"], "description": "How long the model spends reasoning. Higher = better but slower."}, "summary": {"type": "enum", "default": "auto", "options": ["auto", "detailed", "concise", "null"], "description": "Summarize the reasoning result."}}', '{TEXT,IMAGE}', '{TEXT}', '{"response": {"input_token_cost": 0.2, "output_token_cost": 1.25}, "batch": {"input_token_cost": 0.1, "output_token_cost": 0.625}}', true, NOW(), NOW()),
            (19, 'openai', 'gpt-5.4-pro', '{"effort": {"type": "enum", "default": "medium", "options": ["none", "low", "medium", "high", "xhigh"], "description": "How long the model spends reasoning. Higher = better but slower."}, "summary": {"type": "enum", "default": "auto", "options": ["auto", "detailed", "concise", "null"], "description": "Summarize the reasoning result."}}', '{TEXT,IMAGE}', '{TEXT}', '{"response": {"input_token_cost": 30, "output_token_cost": 180}, "batch": {"input_token_cost": 15, "output_token_cost": 90}}', true, NOW(), NOW()),
            (20, 'openai', 'text-embedding-3-large', '{}', '{TEXT}', '{}', '{"response": {"input_token_cost": 0.13, "output_token_cost": 0}, "batch": {"input_token_cost": 0.065, "output_token_cost": 0}}', true, NOW(), NOW()),
            (21, 'openai', 'text-embedding-3-small', '{}', '{TEXT}', '{}', '{"response": {"input_token_cost": 0.02, "output_token_cost": 0}, "batch": {"input_token_cost": 0.01, "output_token_cost": 0}}', true, NOW(), NOW()),
            (22, 'openai', 'text-embedding-ada-002', '{}', '{TEXT}', '{}', '{"response": {"input_token_cost": 0.1, "output_token_cost": 0}, "batch": {"input_token_cost": 0.05, "output_token_cost": 0}}', true, NOW(), NOW())
        """
    )

    # Reset the id sequence to continue after the last seeded id
    op.execute(
        "SELECT setval(pg_get_serial_sequence('global.model_config', 'id'), "
        "(SELECT MAX(id) FROM global.model_config))"
    )


def downgrade():
    op.drop_table("model_config", schema="global")
