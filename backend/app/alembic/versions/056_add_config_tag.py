"""add tag column to config and document tables

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
DEFAULT_TAG_SERVER_DEFAULT = sa.text("'default'::config_tag")


def upgrade():
    config_tag = postgresql.ENUM(
        *CONFIG_TAG_VALUES,
        name="config_tag",
        create_type=False,
    )
    config_tag.create(op.get_bind(), checkfirst=True)

    with op.get_context().autocommit_block():
        op.execute("ALTER TYPE config_tag ADD VALUE IF NOT EXISTS 'default'")
        op.execute("ALTER TYPE config_tag ADD VALUE IF NOT EXISTS 'ASSESSMENT'")

    op.add_column(
        "config",
        sa.Column(
            "tag",
            config_tag,
            nullable=False,
            server_default=DEFAULT_TAG_SERVER_DEFAULT,
            comment=(
                "Tag classifying the config: "
                "'default' for general use, 'ASSESSMENT' for configs used in assessments. "
            ),
        ),
    )

    op.execute(
        """
        UPDATE config
        SET tag = 'ASSESSMENT'
        FROM (
            SELECT DISTINCT config_id
            FROM assessment_run
        ) AS assessment_configs
        WHERE config.id = assessment_configs.config_id
        """
    )

    op.add_column(
        "document",
        sa.Column(
            "tag",
            config_tag,
            nullable=False,
            server_default=DEFAULT_TAG_SERVER_DEFAULT,
            comment=(
                "Tag classifying the document: "
                "'default' for general use, 'ASSESSMENT' for documents used in assessments. "
            ),
        ),
    )

    with op.get_context().autocommit_block():
        op.create_index(
            "idx_config_project_id_tag_active",
            "config",
            ["project_id", "tag", sa.text("updated_at DESC")],
            unique=False,
            postgresql_where=sa.text("deleted_at IS NULL"),
            postgresql_concurrently=True,
        )
        op.create_index(
            "idx_document_project_id_tag_active",
            "document",
            ["project_id", "tag", sa.text("inserted_at DESC")],
            unique=False,
            postgresql_where=sa.text("is_deleted IS FALSE"),
            postgresql_concurrently=True,
        )


def downgrade():
    with op.get_context().autocommit_block():
        op.drop_index(
            "idx_document_project_id_tag_active",
            table_name="document",
            postgresql_concurrently=True,
        )
        op.drop_index(
            "idx_config_project_id_tag_active",
            table_name="config",
            postgresql_concurrently=True,
        )

    op.drop_column("document", "tag")
    op.drop_column("config", "tag")
    sa.Enum(name="config_tag").drop(op.get_bind(), checkfirst=True)
