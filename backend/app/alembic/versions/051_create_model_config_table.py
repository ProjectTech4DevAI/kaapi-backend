"""create model_config table

Revision ID: 051
Revises: 050
Create Date: 2026-03-12 00:00:00.000000

"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "051"
down_revision = "050"
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
            comment="supported input modalities: TEXT, IMAGE, PDF, AUDIO",
        ),
        sa.Column(
            "output_modalities",
            postgresql.ARRAY(sa.String()),
            nullable=False,
            server_default="{}",
            comment="supported output modalities: TEXT, AUDIO",
        ),
        sa.Column(
            "default_for",
            sa.String(),
            nullable=True,
            comment=(
                "completion types this model is the default for. "
                "e.g. [text, stt, tts]. "
                "NULL means not a default. "
                "Supported: text, stt, tts"
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
        INSERT INTO global.model_config (id, provider, model_name, config, input_modalities, output_modalities, default_for, is_active, inserted_at, updated_at)
        VALUES
            (1, 'openai', 'gpt-4o-mini', '{"temperature": {"type": "float", "default": 1.0, "min": 0.0, "max": 2.0, "description": "Controls randomness. Lower = more deterministic."}, "top_p": {"type": "float", "default": 1.0, "min": 0.0, "max": 1.0, "description": "Nucleus sampling. Use either this or temperature, not both."}, "max_output_tokens": {"type": "int", "default": 2048, "min": 1, "max": 32768, "description": "Max tokens in the response."}}', '{TEXT,IMAGE}', '{TEXT}', NULL, true, NOW(), NOW()),
            (2, 'openai', 'gpt-4o', '{"temperature": {"type": "float", "default": 1.0, "min": 0.0, "max": 2.0, "description": "Controls randomness. Lower = more deterministic."}, "top_p": {"type": "float", "default": 1.0, "min": 0.0, "max": 1.0, "description": "Nucleus sampling. Use either this or temperature, not both."}, "max_output_tokens": {"type": "int", "default": 2048, "min": 1, "max": 32768, "description": "Max tokens in the response."}}', '{TEXT,IMAGE}', '{TEXT}', 'text', true, NOW(), NOW()),
            (3, 'openai', 'gpt-4.1', '{"temperature": {"type": "float", "default": 1.0, "min": 0.0, "max": 2.0, "description": "Controls randomness. Lower = more deterministic."}, "top_p": {"type": "float", "default": 1.0, "min": 0.0, "max": 1.0, "description": "Nucleus sampling. Use either this or temperature, not both."}, "max_output_tokens": {"type": "int", "default": 2048, "min": 1, "max": 32768, "description": "Max tokens in the response."}}', '{TEXT,IMAGE}', '{TEXT}', NULL, true, NOW(), NOW()),
            (4, 'openai', 'gpt-4.1-mini', '{"temperature": {"type": "float", "default": 1.0, "min": 0.0, "max": 2.0, "description": "Controls randomness. Lower = more deterministic."}, "top_p": {"type": "float", "default": 1.0, "min": 0.0, "max": 1.0, "description": "Nucleus sampling. Use either this or temperature, not both."}, "max_output_tokens": {"type": "int", "default": 2048, "min": 1, "max": 32768, "description": "Max tokens in the response."}}', '{TEXT,IMAGE}', '{TEXT}', NULL, true, NOW(), NOW()),
            (5, 'openai', 'gpt-4.1-nano', '{"temperature": {"type": "float", "default": 1.0, "min": 0.0, "max": 2.0, "description": "Controls randomness. Lower = more deterministic."}, "top_p": {"type": "float", "default": 1.0, "min": 0.0, "max": 1.0, "description": "Nucleus sampling. Use either this or temperature, not both."}, "max_output_tokens": {"type": "int", "default": 2048, "min": 1, "max": 32768, "description": "Max tokens in the response."}}', '{TEXT,IMAGE}', '{TEXT}', NULL, true, NOW(), NOW()),
            (6, 'openai', 'o3-mini', '{"effort": {"type": "enum", "default": "medium", "options": ["low", "medium", "high"], "description": "How long the model spends reasoning. Higher = better but slower."}, "summary": {"type": "enum", "default": "auto", "options": ["auto", "detailed", "concise"], "description": "Summarize the reasoning result."}}', '{TEXT,IMAGE}', '{TEXT}', NULL, true, NOW(), NOW()),
            (7, 'openai', 'o3', '{"effort": {"type": "enum", "default": "medium", "options": ["low", "medium", "high"], "description": "How long the model spends reasoning. Higher = better but slower."}, "summary": {"type": "enum", "default": "auto", "options": ["auto", "detailed", "concise", "null"], "description": "Summarize the reasoning result."}}', '{TEXT,IMAGE}', '{TEXT}', NULL, true, NOW(), NOW()),
            (8, 'openai', 'o4-mini', '{"effort": {"type": "enum", "default": "medium", "options": ["low", "medium", "high"], "description": "How long the model spends reasoning. Higher = better but slower."}, "summary": {"type": "enum", "default": "auto", "options": ["auto", "detailed", "concise", "null"], "description": "Summarize the reasoning result."}}', '{TEXT,IMAGE}', '{TEXT}', NULL, true, NOW(), NOW()),
            (9, 'openai', 'gpt-5', '{"effort": {"type": "enum", "default": "medium", "options": ["minimal", "low", "medium", "high"], "description": "How long the model spends reasoning. Higher = better but slower."}, "summary": {"type": "enum", "default": "auto", "options": ["auto", "detailed", "concise", "null"], "description": "Summarize the reasoning result."}}', '{TEXT,IMAGE}', '{TEXT}', NULL, true, NOW(), NOW()),
            (10, 'openai', 'gpt-5-mini', '{"effort": {"type": "enum", "default": "medium", "options": ["minimal", "low", "medium", "high"], "description": "How long the model spends reasoning. Higher = better but slower."}, "summary": {"type": "enum", "default": "auto", "options": ["auto", "detailed", "concise", "null"], "description": "Summarize the reasoning result."}}', '{TEXT,IMAGE}', '{TEXT}', NULL, true, NOW(), NOW()),
            (11, 'openai', 'gpt-5-nano', '{"effort": {"type": "enum", "default": "medium", "options": ["minimal", "low", "medium", "high"], "description": "How long the model spends reasoning. Higher = better but slower."}, "summary": {"type": "enum", "default": "auto", "options": ["auto", "detailed", "concise", "null"], "description": "Summarize the reasoning result."}}', '{TEXT,IMAGE}', '{TEXT}', NULL, true, NOW(), NOW()),
            (12, 'openai', 'gpt-5.1', '{"effort": {"type": "enum", "default": "medium", "options": ["none", "low", "medium", "high"], "description": "How long the model spends reasoning. Higher = better but slower."}, "summary": {"type": "enum", "default": "auto", "options": ["auto", "detailed", "concise", "null"], "description": "Summarize the reasoning result."}}', '{TEXT,IMAGE}', '{TEXT}', NULL, true, NOW(), NOW()),
            (13, 'openai', 'gpt-5.1-chat-latest', '{"summary": {"type": "enum", "default": "auto", "options": ["auto", "detailed", "concise", "null"], "description": "Summarize the reasoning result."}}', '{TEXT,IMAGE}', '{TEXT}', NULL, true, NOW(), NOW()),
            (14, 'openai', 'gpt-5.2', '{"effort": {"type": "enum", "default": "medium", "options": ["none", "low", "medium", "high", "xhigh"], "description": "How long the model spends reasoning. Higher = better but slower."}, "summary": {"type": "enum", "default": "auto", "options": ["auto", "detailed", "concise", "null"], "description": "Summarize the reasoning result."}}', '{TEXT,IMAGE}', '{TEXT}', NULL, true, NOW(), NOW()),
            (15, 'openai', 'gpt-5.2-chat-latest', '{"summary": {"type": "enum", "default": "auto", "options": ["auto", "detailed", "concise", "null"], "description": "Summarize the reasoning result."}}', '{TEXT,IMAGE}', '{TEXT}', NULL, true, NOW(), NOW()),
            (16, 'openai', 'gpt-5.2-pro', '{"summary": {"type": "enum", "default": "auto", "options": ["auto", "detailed", "concise", "null"], "description": "Summarize the reasoning result."}}', '{TEXT,IMAGE}', '{TEXT}', NULL, true, NOW(), NOW()),
            (17, 'openai', 'gpt-5.3-chat-latest', '{"summary": {"type": "enum", "default": "auto", "options": ["auto", "detailed", "concise", "null"], "description": "Summarize the reasoning result."}}', '{TEXT,IMAGE}', '{TEXT}', NULL, true, NOW(), NOW()),
            (18, 'openai', 'gpt-5.4-2026-03-05', '{"effort": {"type": "enum", "default": "medium", "options": ["none", "low", "medium", "high", "xhigh"], "description": "How long the model spends reasoning. Higher = better but slower."}, "summary": {"type": "enum", "default": "auto", "options": ["auto", "detailed", "concise", "null"], "description": "Summarize the reasoning result."}}', '{TEXT,IMAGE}', '{TEXT}', NULL, true, NOW(), NOW()),
            (19, 'openai', 'gpt-5.4-pro', '{"effort": {"type": "enum", "default": "medium", "options": ["none", "low", "medium", "high", "xhigh"], "description": "How long the model spends reasoning. Higher = better but slower."}, "summary": {"type": "enum", "default": "auto", "options": ["auto", "detailed", "concise", "null"], "description": "Summarize the reasoning result."}}', '{TEXT,IMAGE}', '{TEXT}', NULL, true, NOW(), NOW())
        """
    )

    # Reset the id sequence to continue after the last seeded id
    op.execute(
        "SELECT setval(pg_get_serial_sequence('global.model_config', 'id'), "
        "(SELECT MAX(id) FROM global.model_config))"
    )


def downgrade():
    op.drop_table("model_config", schema="global")
