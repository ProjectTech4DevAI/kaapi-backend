"""add tag column to config table

Revision ID: 056
Revises: 055
Create Date: 2026-05-03 12:00:00.000000

"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "056"
down_revision = "055"
branch_labels = None
depends_on = None


CONFIG_TAG_VALUES = ("default", "ASSESSMENT")


def upgrade():
    config_tag = postgresql.ENUM(
        *CONFIG_TAG_VALUES,
        name="config_tag",
        create_type=False,
    )
    config_tag.create(op.get_bind(), checkfirst=True)

    op.add_column(
        "config",
        sa.Column(
            "tag",
            config_tag,
            nullable=True,
            comment=(
                "Optional tag classifying the config: "
                "'default' for general use, 'ASSESSMENT' for configs used in assessments. "
                "NULL means untagged (legacy)."
            ),
        ),
    )

    op.create_index(
        "idx_config_project_id_tag_active",
        "config",
        ["project_id", "tag", sa.text("updated_at DESC")],
        unique=False,
        postgresql_where=sa.text("deleted_at IS NULL"),
    )


def downgrade():
    op.drop_index("idx_config_project_id_tag_active", table_name="config")
    op.drop_column("config", "tag")
    sa.Enum(name="config_tag").drop(op.get_bind(), checkfirst=True)
